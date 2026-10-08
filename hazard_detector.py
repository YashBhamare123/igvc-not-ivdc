#!/usr/bin/env python3
"""Live hazard circles from the RealSense D455 depth feed, for path planning.

Pipeline per frame:
  depth pixels -> 3D points -> ground frame (using camera height + tilt)
  -> keep points within range AND inside the obstacle height band (drops floor/ceiling)
  -> top-down occupancy grid -> connected blobs -> enclosing circle + safety margin
  -> tracked over time so circles stay stable.

Ground frame: origin on the floor directly below the camera,
  x = metres to the right, y = metres forward. Circles are (x, y, radius) in that frame.

Use from a planner:
    from hazard_detector import HazardConfig, stream_hazards
    for hazards in stream_hazards(HazardConfig(cam_height_m=0.45, cam_tilt_deg=10)):
        plan(hazards)          # list[Hazard], updated every frame

Keys (viewer): c  calibrate height/tilt from the floor (point camera at open floor)
               [ ]  shrink / grow the hazard range      p  print hazards      q / Esc  quit
"""

import argparse
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs

from depth_viewer import put_text


@dataclass
class HazardConfig:
    # Mounting — set these for your robot (or press 'c' in the viewer to measure them)
    cam_height_m: float = 0.30  # lens height above the floor
    cam_tilt_deg: float = 0.0  # pitch; positive = looking down
    # Thresholds that decide what counts as a hazard
    min_range_m: float = 0.30  # D455 can't see closer than ~0.4 m anyway
    max_range_m: float = 8.0  # planning goal is 10 m; keep depth useful out to ~8 m
    min_obstacle_height_m: float = 0.05  # below this = floor (or floor noise)
    max_obstacle_height_m: float = 1.50  # above this = overhead, robot passes under
    # Grid / noise rejection
    cell_m: float = 0.05
    min_points_per_cell: int = 3
    min_cells: int = 3  # blobs smaller than this are noise
    pixel_stride: int = 4  # sample every Nth pixel (speed vs detail)
    # Must match VehicleConfig.robot_radius_m — planning uses Hazard.radius as-is
    safety_margin_m: float = 0.35  # robot half-width (m), added to every radius
    # Tracking
    match_dist_m: float = 0.40
    smoothing: float = 0.5  # weight of the new measurement (1 = no smoothing)
    confirm_frames: int = 3  # seen this many times before it's reported
    drop_frames: int = 5  # kept (coasting) this many frames after last seen


@dataclass
class Hazard:
    id: int
    x: float  # metres right of the camera
    y: float  # metres ahead of the camera
    radius: float  # includes safety margin — use this for planning
    object_radius: float  # measured size without margin
    edge_distance: float  # camera to the object's nearest edge, metres
    visible: bool  # False while coasting on a missed frame


@dataclass
class _Track:
    id: int
    x: float
    y: float
    r: float
    hits: int = 1
    misses: int = 0
    label: int = 0  # blob label in the current frame (0 = not seen)


@dataclass
class _Detection:
    x: float
    y: float
    r: float
    label: int


class HazardDetector:
    def __init__(self, cfg, intrinsics):
        self.cfg = cfg
        s = cfg.pixel_stride
        v, u = np.mgrid[0 : intrinsics.height : s, 0 : intrinsics.width : s]
        self.u, self.v = u, v
        self.rx = (u - intrinsics.ppx) / intrinsics.fx
        self.ry = (v - intrinsics.ppy) / intrinsics.fy
        self.hfov = 2 * math.atan(intrinsics.width / 2 / intrinsics.fx)
        self.tracks = []
        self.next_id = 1
        # Latest frame's intermediates, for display
        self.occupancy = None
        self.pixel_labels = None

    def camera_points(self, depth_m):
        z = depth_m[:: self.cfg.pixel_stride, :: self.cfg.pixel_stride]
        return self.rx * z, self.ry * z, z  # RealSense axes: x right, y down, z forward

    def to_ground(self, x, y, z):
        t = math.radians(self.cfg.cam_tilt_deg)
        forward = z * math.cos(t) - y * math.sin(t)
        up = self.cfg.cam_height_m - (z * math.sin(t) + y * math.cos(t))
        return x, forward, up

    def update(self, depth_m):
        cfg = self.cfg
        lat, fwd, up = self.to_ground(*self.camera_points(depth_m))
        rng = np.hypot(lat, fwd)
        keep = (
            (depth_m[:: cfg.pixel_stride, :: cfg.pixel_stride] > 0)
            & (rng >= cfg.min_range_m)
            & (rng <= cfg.max_range_m)
            & (up >= cfg.min_obstacle_height_m)
            & (up <= cfg.max_obstacle_height_m)
        )

        # Top-down grid: columns span x in [-max_range, max_range], rows y in [0, max_range]
        nx = int(math.ceil(2 * cfg.max_range_m / cfg.cell_m))
        ny = int(math.ceil(cfg.max_range_m / cfg.cell_m))
        ix = ((lat + cfg.max_range_m) / cfg.cell_m).astype(np.int32)
        iy = (fwd / cfg.cell_m).astype(np.int32)
        keep &= (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        counts = np.bincount(iy[keep] * nx + ix[keep], minlength=nx * ny).reshape(ny, nx)
        occ = (counts >= cfg.min_points_per_cell).astype(np.uint8)
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # bridge small gaps

        n, labels, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
        detections = []
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < cfg.min_cells:
                labels[labels == k] = 0
                continue
            cy, cx = np.nonzero(labels == k)
            pts = np.stack(
                [(cx + 0.5) * cfg.cell_m - cfg.max_range_m, (cy + 0.5) * cfg.cell_m], axis=1
            ).astype(np.float32)
            (mx, my), r = cv2.minEnclosingCircle(pts)
            detections.append(_Detection(mx, my, r + cfg.cell_m / 2, k))

        pixel_labels = np.zeros(lat.shape, np.int32)
        pixel_labels[keep] = labels[iy[keep], ix[keep]]
        self.occupancy, self.pixel_labels = occ, pixel_labels

        self._track(detections)
        return self.hazards()

    def _track(self, detections):
        cfg, a = self.cfg, self.cfg.smoothing
        pairs = sorted(
            (math.hypot(t.x - d.x, t.y - d.y), ti, di)
            for ti, t in enumerate(self.tracks)
            for di, d in enumerate(detections)
        )
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            t, d = self.tracks[ti], detections[di]
            # Big objects' centres shift as more of them comes into view, so widen the gate
            if ti in used_t or di in used_d or dist > max(cfg.match_dist_m, 0.5 * t.r):
                continue
            t.x += a * (d.x - t.x)
            t.y += a * (d.y - t.y)
            t.r += a * (d.r - t.r)
            t.hits, t.misses, t.label = t.hits + 1, 0, d.label
            used_t.add(ti)
            used_d.add(di)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses, t.label = t.misses + 1, 0
        for di, d in enumerate(detections):
            if di not in used_d:
                self.tracks.append(_Track(self.next_id, d.x, d.y, d.r, label=d.label))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= cfg.drop_frames]

    def hazards(self):
        m = self.cfg.safety_margin_m
        return sorted(
            (
                Hazard(
                    t.id,
                    t.x,
                    t.y,
                    t.r + m,
                    t.r,
                    max(0.0, math.hypot(t.x, t.y) - t.r),
                    t.misses == 0,
                )
                for t in self.tracks
                if t.hits >= self.cfg.confirm_frames
            ),
            key=lambda h: h.edge_distance,
        )

    def calibrate_floor(self, depth_m, iters=200, tol=0.02):
        """Fit the floor plane in the lower part of the image -> (height_m, tilt_deg, roll_deg)."""
        x, y, z = self.camera_points(depth_m)
        lower = self.v >= self.v.max() * 0.5
        m = lower & (z > 0) & (z < 4)
        pts = np.stack([x[m], y[m], z[m]], axis=1)
        if len(pts) < 100:
            return None
        rng = np.random.default_rng()
        best = None
        for _ in range(iters):
            p = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            if np.linalg.norm(n) < 1e-6:
                continue
            n /= np.linalg.norm(n)
            inliers = np.abs((pts - p[0]) @ n) < tol
            if best is None or inliers.sum() > best.sum():
                best = inliers
        if best is None or best.mean() < 0.3:
            return None
        floor = pts[best]
        centroid = floor.mean(axis=0)
        n = np.linalg.svd(floor - centroid)[2][-1]
        if n[1] > 0:  # orient the normal "up", i.e. towards camera -y
            n = -n
        height = abs(n @ centroid)
        tilt = math.degrees(math.atan2(-n[2], -n[1]))
        roll = math.degrees(math.asin(np.clip(n[0], -1, 1)))
        return height, tilt, roll


# ---------------------------------------------------------------- display

PALETTE = [
    (80, 80, 255),
    (80, 200, 255),
    (255, 160, 60),
    (200, 90, 255),
    (90, 230, 120),
    (255, 90, 200),
    (60, 220, 220),
    (180, 180, 255),
]


def _color(hid):
    return PALETTE[hid % len(PALETTE)]


def draw_camera_view(img, det, hazards):
    """Tint the pixels counted as hazards and box each tracked object."""
    s = det.cfg.pixel_stride
    by_label = {t.label: t.id for t in det.tracks if t.label and t.hits >= det.cfg.confirm_frames}
    hazard_by_id = {h.id: h for h in hazards}
    mask = (
        np.isin(det.pixel_labels, list(by_label))
        if by_label
        else np.zeros_like(det.pixel_labels, bool)
    )
    big = cv2.resize(
        mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    tint = img.copy()
    tint[big > 0] = (0, 0, 255)
    cv2.addWeighted(tint, 0.35, img, 0.65, 0, dst=img)

    for label, hid in by_label.items():
        sel = det.pixel_labels == label
        us, vs = det.u[sel], det.v[sel]
        x0, y0, x1, y1 = us.min(), vs.min(), us.max() + s, vs.max() + s
        col = _color(hid)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
        # Label above the box, or just inside it when the box touches the panel title
        ty = y0 - 6 if y0 > 50 else y0 + 18
        put_text(img, f"#{hid}  {hazard_by_id[hid].edge_distance:.2f} m", (x0 + 4, ty), 0.5, col, 1)


def draw_top_down(det, hazards, w, h, path=None, goal=None, view_range_m=None):
    """Top-down panel. path: list[(x,y)] m; goal: (x,y) m; view_range_m sets scale."""
    cfg = det.cfg
    span = view_range_m or max(cfg.max_range_m, 10.0)
    panel = np.full((h, w, 3), 24, np.uint8)
    scale = min((h - 30) / span, (w / 2 - 10) / max(span, cfg.max_range_m))  # px per metre
    ox, oy = w // 2, h - 15

    def px(x, y):
        return int(round(ox + x * scale)), int(round(oy - y * scale))

    # Occupied cells
    if det.occupancy is not None:
        occ = np.flipud(det.occupancy) * 70
        gw = int(round(occ.shape[1] * cfg.cell_m * scale))
        gh = int(round(occ.shape[0] * cfg.cell_m * scale))
        grid = cv2.resize(occ, (gw, gh), interpolation=cv2.INTER_NEAREST)
        x0, y0 = ox - gw // 2, oy - gh
        gx0, gy0 = max(0, -x0), max(0, -y0)
        region = panel[max(0, y0) : y0 + gh, max(0, x0) : x0 + gw]
        region[:] = np.maximum(
            region, grid[gy0 : gy0 + region.shape[0], gx0 : gx0 + region.shape[1], None]
        )

    # Range rings and field of view
    step = 0.5 if span <= 4 else 1.0
    r = step
    while r <= span + 1e-6:
        cv2.ellipse(
            panel, (ox, oy), (int(r * scale),) * 2, 0, 180, 360, (60, 60, 60), 1, cv2.LINE_AA
        )
        put_text(panel, f"{r:g} m", (ox + 4, int(oy - r * scale) + 14), 0.38, (130, 130, 130))
        r += step
    for sgn in (-1, 1):
        a = det.hfov / 2
        cv2.line(
            panel,
            (ox, oy),
            px(sgn * cfg.max_range_m * math.sin(a), cfg.max_range_m * math.cos(a)),
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

    # Path (under hazards so circles stay readable)
    if path and len(path) >= 2:
        pts = np.array([px(x, y) for x, y in path], dtype=np.int32)
        cv2.polylines(panel, [pts], False, (80, 220, 80), 2, cv2.LINE_AA)
        for p in pts[:: max(1, len(pts) // 12)]:
            cv2.circle(panel, tuple(p), 3, (80, 220, 80), -1, cv2.LINE_AA)

    if goal is not None:
        gv = px(goal[0], goal[1])
        cv2.drawMarker(panel, gv, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        put_text(
            panel,
            f"goal ({goal[0]:+.1f}, {goal[1]:.1f}) m",
            (gv[0] + 8, gv[1] - 8),
            0.4,
            (0, 255, 255),
        )

    # Hazards: filled object, outline = planning radius incl. margin
    overlay = panel.copy()
    for hz in hazards:
        cv2.circle(
            overlay,
            px(hz.x, hz.y),
            max(1, int(hz.object_radius * scale)),
            _color(hz.id),
            -1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.45, panel, 0.55, 0, dst=panel)
    for hz in hazards:
        c = px(hz.x, hz.y)
        col = _color(hz.id)
        cv2.circle(
            panel, c, max(1, int(hz.radius * scale)), col, 2 if hz.visible else 1, cv2.LINE_AA
        )
        cv2.drawMarker(panel, c, col, cv2.MARKER_CROSS, 8, 1)
        put_text(
            panel,
            f"#{hz.id} ({hz.x:+.2f}, {hz.y:.2f}) r{hz.radius:.2f}",
            (c[0] + 8, c[1] - 6),
            0.4,
            col,
        )

    # Camera
    cv2.fillPoly(
        panel, [np.array([(ox, oy - 10), (ox - 7, oy + 4), (ox + 7, oy + 4)])], (255, 255, 255)
    )
    put_text(panel, "TOP-DOWN  (x right, y forward)", (10, 24), 0.55)
    return panel


# ---------------------------------------------------------------- live loop


@dataclass
class FrameContext:
    """One depth frame after hazard update. Used by the real-time navigator."""

    hazards: list
    depth_m: np.ndarray
    color: np.ndarray
    detector: HazardDetector
    key: int = -1  # last waitKey code when show=True, else -1


def stream_hazards(cfg=None, show=True, width=640, height=480, fps=30, frame_hook=None):
    """Yield list[Hazard] every frame. Stops when the viewer is closed.

    frame_hook(ctx: FrameContext) -> dict | None may return
      {"path": [(x,y), ...], "goal": (x,y), "status": str, "view_range_m": float}
    for top-down overlay. Hook runs before display each frame.
    """
    cfg = cfg or HazardConfig()
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    # No hole-filling here: it invents depth, which would invent obstacles
    filters = [rs.spatial_filter(), rs.temporal_filter()]
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    det = HazardDetector(cfg, intr)

    window = "Hazards"
    if show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window, 2 * width, height + 72)
    fps_est, last = 0.0, time.time()
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames(5000))
            depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            for f in filters:
                depth_frame = f.process(depth_frame)
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale
            color = np.asanyarray(color_frame.get_data()).copy()

            hazards = det.update(depth_m)
            ctx = FrameContext(hazards, depth_m, color, det)
            overlay = frame_hook(ctx) if frame_hook else None
            yield hazards

            if not show:
                continue
            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 / max(now - last, 1e-6)
            last = now

            draw_camera_view(color, det, hazards)
            put_text(color, "CAMERA  (red = counted as hazard)", (10, 24), 0.55)
            path = goal = status = None
            view_range = None
            if isinstance(overlay, dict):
                path = overlay.get("path")
                goal = overlay.get("goal")
                status = overlay.get("status")
                view_range = overlay.get("view_range_m")
            top = draw_top_down(
                det, hazards, width, height, path=path, goal=goal, view_range_m=view_range
            )

            header = np.full((44, 2 * width, 3), 32, np.uint8)
            put_text(
                header,
                f"{fps_est:4.1f} FPS  |  {len(hazards)} hazards  |  cam h {cfg.cam_height_m:.2f} m "
                f"tilt {cfg.cam_tilt_deg:.1f} deg  |  range {cfg.min_range_m:g}-{cfg.max_range_m:g} m  |  "
                f"height band {cfg.min_obstacle_height_m:g}-{cfg.max_obstacle_height_m:g} m  |  "
                f"margin {cfg.safety_margin_m:g} m",
                (12, 28),
                0.5,
            )
            footer = np.full((28, 2 * width, 3), 32, np.uint8)
            footer_msg = "c: calibrate floor   [ ]: range   p: print hazards   q: quit"
            if status:
                footer_msg = f"{status}   |   {footer_msg}"
            put_text(footer, footer_msg, (12, 19), 0.45, (190, 190, 190))
            cv2.imshow(window, np.vstack([header, np.hstack([color, top]), footer]))

            key = cv2.waitKey(1) & 0xFF
            ctx.key = key
            # WND_PROP_VISIBLE is always -1 on this GTK build; AUTOSIZE turns -1 once closed
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_AUTOSIZE) < 0:
                break
            elif key == ord("]"):
                cfg.max_range_m = min(cfg.max_range_m + 0.5, 10.0)
            elif key == ord("["):
                cfg.max_range_m = max(cfg.max_range_m - 0.5, 1.0)
            elif key == ord("p"):
                print_hazards(hazards)
            elif key == ord("c"):
                result = det.calibrate_floor(depth_m)
                if result is None:
                    print(
                        "calibration failed: not enough floor visible in the lower half of the image"
                    )
                else:
                    cfg.cam_height_m, cfg.cam_tilt_deg, roll = result
                    print(
                        f"floor fit -> cam_height_m={cfg.cam_height_m:.3f}, cam_tilt_deg={cfg.cam_tilt_deg:.1f}"
                        f"  (roll {roll:.1f} deg, not compensated)  — applied"
                    )
    finally:
        pipeline.stop()
        if show:
            cv2.destroyAllWindows()


def print_hazards(hazards):
    print(f"--- {len(hazards)} hazards ---")
    for h in hazards:
        print(
            f"#{h.id:<3} x {h.x:+.2f}  y {h.y:.2f}  radius {h.radius:.2f}  "
            f"(object {h.object_radius:.2f})  edge {h.edge_distance:.2f} m{'' if h.visible else '  [coasting]'}"
        )


def main():
    cfg = HazardConfig()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name, value in vars(cfg).items():
        ap.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    ap.add_argument("--headless", action="store_true", help="no window; print hazards every second")
    args = ap.parse_args()
    cfg = HazardConfig(**{k: getattr(args, k) for k in vars(cfg)})

    last_print = 0.0
    for hazards in stream_hazards(cfg, show=not args.headless):
        if args.headless and time.time() - last_print > 1.0:
            print_hazards(hazards)
            last_print = time.time()


if __name__ == "__main__":
    main()
