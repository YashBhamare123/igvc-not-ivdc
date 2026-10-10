#!/usr/bin/env python3
"""Basic reactive lane follower: depth camera + lane mask -> one heading vector per frame.

Every frame:
  1. top-down grid ahead of the camera: lane-mask pixels and depth points standing above
     the floor are blocked, grown by half the car width (so the car's centre line only has
     to stay on free cells)
  2. rays from the camera every --ray-step degrees across +/- --ray-max: free distance =
     how far each goes before a blocked cell
  3. heading = the ray with the longest free distance, mildly preferring straight ahead and
     last frame's choice. Between two lane lines the longest free ray runs along the lane,
     so bends are followed without special cases.
  4. drive toward it with moderate L/R differences (measured to steer correctly); sharper,
     or nothing free close ahead: turn in place toward it.
No IMU, no map memory, no path planning.

Usage:
    python3 lane_follow.py --dry-run --frames 30 --save DIR
    python3 lane_follow.py --max-dist 8 --save runs/lf1
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import types

import cv2
import numpy as np
import pyrealsense2 as rs

import avoid
import lane_detector as ld
import stop_ahead as sa

IN = 0.0254


class Follower:
    def __init__(self, a):
        self.a = a
        self.lane_args = types.SimpleNamespace(
            near=0.4, far=4.0, half_span=2.5, line_kernel_px=21, black_contrast=25, red_hue_lo=160, red_hue_hi=8,
            red_min_sat=256, red_contrast=999, above_floor_m=0.03, depth_noise_k=0.006, blocked_grow_px=5,
            per_pixel_floor=True, min_piece_m=0.4, min_thinness=4.0, under_object_m=0.35, tall_m=0.25,
            glare_l=225, black_max_ratio=0.62, floor_depth_win_px=31, min_floor_depth_frac=0.5, min_support=25,
            min_length_m=0.3, max_gap_m=0.3, max_segments=6, inlier_m=0.03, ransac_iters=60, max_points=1500)
        c = a.cell
        self.nx, self.ny = int(2 * a.grid_half_width / c), int(a.grid_ahead / c)
        # rays: cell indices along each heading, from the camera outward
        self.angles = np.arange(-a.ray_max, a.ray_max + 1e-6, a.ray_step)
        s = np.arange(0.0, a.grid_ahead, c / 2)
        self.ray_s = s
        self.ray_i, self.ray_j = [], []
        for ang in np.radians(self.angles):
            x, y = s * math.sin(ang), s * math.cos(ang)  # + angle = right
            self.ray_i.append(((x + a.grid_half_width) / c).astype(int))
            self.ray_j.append((y / c).astype(int))
        self.ray_i, self.ray_j = np.array(self.ray_i), np.array(self.ray_j)
        r = int(round((a.width / 2 + a.margin) / c))
        self.grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        self.prev = 0.0
        self.smooth = 0.0  # heading actually steered toward (deg, + = right)
        self.mode, self.until, self.commit, self.move_hall = "drive", 0.0, 0, None
        self.pivot = 0
        self.car = None
        self.t0 = time.time()
        self.n = 0
        self.last_print = self.last_save = -1
        self.moves = []

    # --- setup
    def start(self):
        a = self.a
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, sa.WIDTH, sa.HEIGHT, rs.format.z16, sa.FPS)
        cfg.enable_stream(rs.stream.color, sa.WIDTH, sa.HEIGHT, rs.format.bgr8, sa.FPS)
        prof = self.pipe.start(cfg)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)
        self.rx, self.ry, self.rows = sa.pixel_rays(self.intr)
        for _ in range(15):
            self.pipe.wait_for_frames()
        stack = [np.asanyarray(self.align.process(self.pipe.wait_for_frames()).get_depth_frame().get_data()).copy()
                 for _ in range(10)]
        plane = sa.fit_floor(sa.to_points(np.median(np.stack(stack), axis=0) * self.scale, self.rx, self.ry), self.rows)
        if plane is None:
            sys.exit("floor calibration failed: not enough flat floor in view")
        self.normal, self.offset = plane
        self.det = ld.LaneDetector(self.intr, self.normal, self.offset, self.lane_args)
        print(f"floor: camera height {self.offset:.3f} m, tilt {math.degrees(math.atan2(-self.normal[2], -self.normal[1])):.1f} deg down")
        if not a.dry_run:
            self.car = avoid.Car(a.port, a.m_per_count)
            self.car.set_origin()
        if a.save:
            os.makedirs(os.path.join(a.save, "masks"), exist_ok=True)

    def close(self):
        if self.car:
            self.car.close()
        self.pipe.stop()

    # --- one frame: grid, rays, heading
    def sense(self):
        a = self.a
        f = self.align.process(self.pipe.wait_for_frames(1000))
        self.color = np.asanyarray(f.get_color_frame().get_data()).copy()
        self.depth = np.asanyarray(f.get_depth_frame().get_data()) * self.scale
        if self.car:
            self.car.poll()
        self.n += 1
        pts = sa.to_points(self.depth, self.rx, self.ry)
        fit = sa.refine_floor(pts, self.rows, self.normal, self.offset)  # the tall mount pitches
        if fit is not None:
            self.normal, self.offset = fit
        fwd, right = sa.ground_axes(self.normal)
        # objects: depth points standing on the floor (threshold grows with depth noise)
        h = pts @ self.normal + self.offset
        y, x = pts @ fwd, pts @ right
        obj = (pts[..., 2] > 0) & (h > a.min_height + 0.006 * pts[..., 2] ** 2) & (h < a.max_height) & (y > 0.1) & (y < a.grid_ahead)
        # lanes: tape pixels projected onto the floor
        _, self.lane_mask = self.det.masks(self.color, self.depth)
        lx, ly = self.det.gx[self.lane_mask], self.det.gy[self.lane_mask]  # relative to the floor point below the camera
        # grid (camera frame: x right, y ahead)
        c = a.cell
        grid = np.zeros((self.ny, self.nx), np.uint8)
        for gx, gy in ((x[obj], y[obj]), (lx, ly)):
            i = ((gx + a.grid_half_width) / c).astype(int)
            j = (gy / c).astype(int)
            ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
            grid[j[ok], i[ok]] = 1
        self.raw_grid = grid
        self.blocked = cv2.dilate(grid, self.grow).astype(bool)
        # rays
        ok = (self.ray_i >= 0) & (self.ray_i < self.nx) & (self.ray_j >= 0) & (self.ray_j < self.ny)
        hit = np.zeros(self.ray_i.shape, bool)
        hit[ok] = self.blocked[self.ray_j[ok], self.ray_i[ok]]
        hit[~ok] = True  # off the grid's sides: treat as blocked
        hit[:, self.ray_s < a.ignore_near] = False  # the cell the car is in (and the blind band below the camera)
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), hit.shape[1])
        self.free = self.ray_s[np.minimum(first, len(self.ray_s) - 1)]
        self.free[~hit.any(axis=1)] = a.grid_ahead
        score = np.minimum(self.free, a.free_cap) - a.w_straight * np.abs(self.angles) / 45 - a.w_keep * np.abs(self.angles - self.prev) / 45
        best = int(np.argmax(score))
        self.heading, self.best_free = float(self.angles[best]), float(self.free[best])
        self.prev = self.heading

    # --- motors
    def command(self):
        """(L, R) toward the smoothed heading (+ = right): small corrections only, never a
        turn in place (those make the car skid and slide around)."""
        a = self.a
        self.smooth += a.heading_smooth * (self.heading - self.smooth)
        diff = max(-a.max_diff, min(a.max_diff, a.steer_gain * self.smooth))  # + = turn right
        return a.base + diff, a.base - diff

    def start_move(self, kind, now):
        self.mode, self.until = kind, now + (self.a.step_max_s if kind == "step" else self.a.nudge_max_s)
        self.move_hall = self.car.hall if self.car else None

    def stalled(self):
        """Commanded to move but the wheel counts haven't changed for --stall-s."""
        if not self.car or self.car.hall is None:
            return False
        now = time.time()
        self.moves = [m for m in self.moves if now - m[0] < self.a.stall_s] + [(now, self.car.hall)]
        if now - self.moves[0][0] < self.a.stall_s * 0.9:
            return False
        (h0l, h0r), (h1l, h1r) = self.moves[0][1], self.moves[-1][1]
        return abs(h1l - h0l) + abs(h1r - h0r) < 3

    # --- loop
    def run(self):
        a = self.a
        while True:
            self.sense()
            el = time.time() - self.t0
            if self.car and self.car.odo >= a.max_dist:
                return f"drove {a.max_dist:.1f} m"
            if el >= a.max_time:
                return f"{a.max_time:.0f} s limit"
            if a.dry_run and a.frames and self.n >= a.frames:
                return f"{self.n} frames"
            now = time.time()
            cmd = (0.0, 0.0)
            if self.mode == "drive":
                if abs(self.heading) > a.search_deg or self.best_free < a.search_free:
                    self.mode, self.until, self.commit = "look", now + a.look_s, 0
                    print(f"  {el:5.1f}s searching: best heading {self.heading:+.0f} deg, free {self.best_free:.2f} m")
                else:
                    cmd = self.command()
            elif self.mode == "look" and now >= self.until:
                h, fr = self.heading, self.best_free
                if abs(h) <= a.go_deg and fr >= a.go_free:
                    self.mode, self.smooth = "drive", h
                    print(f"  {el:5.1f}s way found: heading {h:+.0f} deg, free {fr:.2f} m")
                    cmd = self.command()
                elif abs(h) <= a.search_deg and fr >= a.step_free:
                    self.start_move("step", now)  # roughly ahead: creep a little, then look again
                else:
                    want = 1 if h > 0 else -1
                    side = self.free[self.angles * self.commit > 0] if self.commit else None
                    if not self.commit or (side is not None and side.max() < a.step_free):
                        self.commit = want  # pick a turning side and keep it while it has room
                    self.start_move("nudge", now)
            if self.mode in ("step", "nudge"):
                moved = 0.0
                if self.car and self.move_hall is not None and self.car.hall is not None:
                    moved = 0.5 * (abs(self.car.hall[0] - self.move_hall[0]) + abs(self.car.hall[1] - self.move_hall[1]))
                if self.mode == "step":
                    diff = max(-a.max_diff, min(a.max_diff, a.steer_gain * self.heading))
                    cmd, target = (a.base + diff, a.base - diff), a.step_counts
                else:
                    cmd, target = (self.commit * a.nudge_cmd, -self.commit * a.nudge_cmd), a.nudge_counts
                if moved >= target or now >= self.until:
                    self.mode, self.until, cmd = "look", now + a.look_s, (0.0, 0.0)
            if self.car:
                if cmd != (0.0, 0.0) and self.mode == "drive" and self.stalled():
                    print(f"  WARNING: L{cmd[0]:.0f} R{cmd[1]:.0f} but wheels not turning: zero for 0.4 s (motor stall cut-out)")
                    t = time.time()
                    while time.time() - t < 0.4:
                        self.car.drive(0, 0)
                        time.sleep(0.05)
                    self.moves = []
                    continue
                self.car.drive(*cmd)
            self.log(cmd, el)

    def stop(self):
        if self.car:
            self.car.drive(0, 0)

    # --- logging / images
    def log(self, cmd, el):
        a = self.a
        if int(el * 4) != self.last_print:
            self.last_print = int(el * 4)
            odo = f"odo {self.car.odo:5.2f} m | " if self.car else ""
            fan = " ".join(f"{f:.1f}" for f in self.free[:: max(1, len(self.free) // 9)])
            print(f"{el:5.1f}s {odo}heading {self.heading:+5.0f} deg free {self.best_free:.2f} m | "
                  f"{self.mode:5s} steer {self.smooth:+5.1f} | L{cmd[0]:.0f} R{cmd[1]:.0f} | free by angle: {fan}")
        if a.save and int(el * 2) != self.last_save:
            self.last_save = int(el * 2)
            cv2.imwrite(os.path.join(a.save, f"t{el:05.1f}.jpg"), np.hstack([self.overlay(), self.topdown()]))
            cv2.imwrite(os.path.join(a.save, "masks", f"t{el:05.1f}_lane_mask.png"), self.lane_mask.astype(np.uint8) * 255)

    def overlay(self):
        img = self.color.copy()
        img[self.lane_mask] = (0, 255, 255)
        cv2.putText(img, f"heading {self.heading:+.0f} deg, free {self.best_free:.2f} m", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img

    def topdown(self, size=480):
        a = self.a
        px = size / max(2 * a.grid_half_width, a.grid_ahead)
        img = np.full((size, size, 3), 30, np.uint8)
        to = lambda x, y: (int(size / 2 + x * px), int(size - 10 - y * px))
        blk = cv2.resize(self.blocked[::-1].astype(np.uint8) * 80, (int(self.nx * a.cell * px), int(self.ny * a.cell * px)), interpolation=cv2.INTER_NEAREST)
        raw = cv2.resize(self.raw_grid[::-1] * 255, blk.shape[::-1], interpolation=cv2.INTER_NEAREST)
        x0, y0 = to(-a.grid_half_width, a.grid_ahead)
        h_, w_ = blk.shape
        y1, x1 = min(size, y0 + h_), min(size, x0 + w_)
        roi = img[max(0, y0):y1, max(0, x0):x1]
        b = blk[: roi.shape[0], : roi.shape[1]]; r = raw[: roi.shape[0], : roi.shape[1]]
        roi[b > 0] = (60, 60, 120)
        roi[r > 0] = (0, 200, 255)
        for ang, fr in zip(self.angles, self.free):
            t = math.radians(ang)
            cv2.line(img, to(0, 0), to(fr * math.sin(t), fr * math.cos(t)), (90, 90, 90), 1)
        t = math.radians(self.heading)
        cv2.arrowedLine(img, to(0, 0), to(self.best_free * math.sin(t), self.best_free * math.cos(t)), (0, 255, 0), 3, tipLength=0.08)
        return img


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--width", type=float, default=32 * IN)
    ap.add_argument("--margin", type=float, default=0.03, help="extra clearance each side (m)")
    ap.add_argument("--m-per-count", type=float, default=0.0075)
    # grid / rays
    ap.add_argument("--cell", type=float, default=0.04)
    ap.add_argument("--grid-half-width", type=float, default=2.0)
    ap.add_argument("--grid-ahead", type=float, default=3.0)
    ap.add_argument("--min-height", type=float, default=0.08)
    ap.add_argument("--max-height", type=float, default=1.3)
    ap.add_argument("--ray-step", type=float, default=4.0)
    ap.add_argument("--ray-max", type=float, default=32.0)
    ap.add_argument("--ignore-near", type=float, default=0.3, help="rays start counting this far from the camera (m)")
    ap.add_argument("--free-cap", type=float, default=2.5, help="free distance beyond this is all equally good (m)")
    ap.add_argument("--w-straight", type=float, default=0.3, help="preference for straight ahead (m per 45 deg)")
    ap.add_argument("--w-keep", type=float, default=0.3, help="preference for last frame's heading (m per 45 deg)")
    # driving (measured: moderate L/R differences steer correctly; one wheel slow + other fast does not)
    ap.add_argument("--base", type=float, default=340.0)
    ap.add_argument("--steer-gain", type=float, default=3.0, help="L/R difference per degree of (smoothed) heading")
    ap.add_argument("--max-diff", type=float, default=40.0, help="largest +/- around the base: 300/380 at most (small turns only)")
    ap.add_argument("--heading-smooth", type=float, default=0.25, help="how fast the steered heading follows the chosen ray")
    ap.add_argument("--search-deg", type=float, default=16.0, help="best heading further off than this: stop and search")
    ap.add_argument("--search-free", type=float, default=0.7, help="less free than this ahead: stop and search (m)")
    ap.add_argument("--go-deg", type=float, default=8.0, help="back to driving when the best heading is within this...")
    ap.add_argument("--go-free", type=float, default=1.0, help="...and this much is free (m)")
    ap.add_argument("--look-s", type=float, default=0.3, help="stand still this long before judging (sharp image)")
    ap.add_argument("--step-free", type=float, default=0.5, help="searching: creep forward if this much is free roughly ahead (m)")
    ap.add_argument("--step-counts", type=float, default=12, help="one forward creep: wheel counts (~9 cm)")
    ap.add_argument("--step-max-s", type=float, default=1.5)
    ap.add_argument("--nudge-counts", type=float, default=3, help="one small search turn: wheel counts each side (~2 cm, a few degrees)")
    ap.add_argument("--nudge-max-s", type=float, default=0.6)
    ap.add_argument("--nudge-cmd", type=float, default=750.0, help="turn-in-place command (600 doesn't turn the car)")
    ap.add_argument("--stall-s", type=float, default=1.5)
    # run
    ap.add_argument("--max-dist", type=float, default=8.0)
    ap.add_argument("--max-time", type=float, default=120.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--save")
    a = ap.parse_args()
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        sys.stdout = Tee(sys.stdout, open(os.path.join(a.save, "log.txt"), "w"))
    f = Follower(a)
    try:
        f.start()
        why = f.run()
        f.stop()
        print(f"STOP: {why}" + (f"; travelled {f.car.odo:.2f} m" if f.car else ""))
    finally:
        f.close()


if __name__ == "__main__":
    main()
