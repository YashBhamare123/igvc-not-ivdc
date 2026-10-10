#!/usr/bin/env python3
"""Drive the course: A* path finding around objects between the lanes.

Inputs each frame (vehicle frame: x right, y forward, origin = vehicle centre):
  - circles (x, y, r): zones to avoid (objects from the depth camera, remembered with
    wheel odometry + IMU while they're in the camera's blind band next to the car)
  - lane points (x, y): black tape detected on the floor

States:
  1  no objects, lanes far away   -> drive forward along the lane direction (IMU heading hold)
  2  objects (or a lane near)      -> A* on a grid: cells holding an object (grown by half the
                                      car width) are opaque, cells near objects are heavily
                                      weighted; target = a far point (--goal-dist) along the lane
                                      direction (from the lane pieces in view, held with the IMU),
                                      clipped to the grid; if unreachable, the closest reachable cell.
                                      The 38 x 32 in rectangle along the path must not touch
                                      any object (checked; offending cells blocked, A* re-run).
                                      Re-planned every frame.
  3  objects, no path              -> log nearby objects + lanes, save images, exit with error
  4  no objects, no path           -> (should never happen) log, save images, exit with error

Usage:
    python3 pathfind.py --dry-run --frames 30 --save DIR   # perceive + plan, motors off
    python3 pathfind.py --save runs/pf1                     # drive
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
import threading
import time
import types

import cv2
import numpy as np
import pyrealsense2 as rs

import avoid
import lane_detector as ld
import stop_ahead as sa

IN = 0.0254


# ---------------------------------------------------------------- A* planner


class AStar:
    """8-connected A* on a vehicle-frame grid. Everything that doesn't change between
    frames (cell centres, neighbour steps, the car's outline) is computed once."""

    def __init__(self, a):
        self.a = a
        self.c = a.cell
        self.xs = np.arange(-a.grid_half_width, a.grid_half_width + 1e-6, a.cell)
        self.ys = np.arange(-a.grid_behind, a.grid_ahead + 1e-6, a.cell)
        self.nx, self.ny = len(self.xs), len(self.ys)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)  # (ny, nx)
        self.steps = [(di, dj, math.hypot(di, dj) * a.cell) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        hw, hl = a.width / 2, a.length / 2
        e = np.arange(-1, 1 + 1e-6, 0.04 / hw)
        f = np.arange(-1, 1 + 1e-6, 0.04 / hl)
        self.outline = np.concatenate([np.stack([e * hw, np.full_like(e, hl)], 1), np.stack([e * hw, np.full_like(e, -hl)], 1),
                                       np.stack([np.full_like(f, hw), f * hl], 1), np.stack([np.full_like(f, -hw), f * hl], 1)])

    def cell_of(self, x, y):
        return int(round((y + self.a.grid_behind) / self.c)), int(round((x + self.a.grid_half_width) / self.c))

    def costs(self, circles, lane_pts):
        """(blocked, weight) grids from the distance of each cell centre to the nearest
        object edge (lanes count as thin objects)."""
        a = self.a
        d = np.full((self.ny, self.nx), np.inf)
        self.d_lane = np.full((self.ny, self.nx), np.inf)
        for x, y, r in circles:
            d = np.minimum(d, np.hypot(self.X - x, self.Y - y) - r)
        if len(lane_pts):
            # lane tape: rasterise, then distance transform (many points, cheap this way)
            m = np.ones((self.ny, self.nx), np.uint8)
            j = np.round((lane_pts[:, 1] + a.grid_behind) / self.c).astype(int)
            i = np.round((lane_pts[:, 0] + a.grid_half_width) / self.c).astype(int)
            ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
            m[j[ok], i[ok]] = 0
            if ok.any():
                self.d_lane = cv2.distanceTransform(m, cv2.DIST_L2, 5) * self.c - a.lane_r
                d = np.minimum(d, self.d_lane)
        self.d = d
        clear = d - a.width / 2  # how far the car's side would be from the object, centred on this cell
        blocked = clear < a.hard_margin
        near = np.clip((a.soft_zone - clear) / a.soft_zone, 0.0, 1.0)
        weight = 1.0 + a.near_weight * near ** 2  # heavily weighted close to objects
        return blocked, weight

    def search(self, start, goal, blocked, weight):
        """Cells from start to goal, or (if the goal can't be reached) to the reached cell
        closest to it. Returns (cells, reached_goal)."""
        a = self.a
        ny, nx = self.ny, self.nx
        gj, gi = goal
        h = lambda j, i: math.hypot(j - gj, i - gi) * self.c
        g = {start: 0.0}
        parent = {start: None}
        heap = [(h(*start), 0.0, start)]
        best = (h(*start), start)
        closed = set()
        while heap:
            _, gc, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            closed.add(cur)
            hc = h(*cur)
            if hc < best[0]:
                best = (hc, cur)
            if cur == goal:
                break
            j, i = cur
            for di, dj, L in self.steps:
                jj, ii = j + dj, i + di
                if not (0 <= jj < ny and 0 <= ii < nx) or blocked[jj, ii] or (jj, ii) in closed:
                    continue
                if di and dj and (blocked[j, ii] or blocked[jj, i]):  # no squeezing past corners
                    continue
                ng = gc + L * 0.5 * (weight[j, i] + weight[jj, ii])
                if ng < g.get((jj, ii), np.inf):
                    g[(jj, ii)] = ng
                    parent[(jj, ii)] = cur
                    heapq.heappush(heap, (ng + h(jj, ii), ng, (jj, ii)))
        reached = goal in closed
        end = goal if reached else best[1]
        self.closed = closed
        cells = []
        while end is not None:
            cells.append(end)
            end = parent[end]
        return cells[::-1], reached

    def reach_along(self, ux, uy):
        """Furthest the explored (reachable) cells get along the direction (ux, uy)."""
        return max(self.xs[i] * ux + self.ys[j] * uy for j, i in self.closed)

    def poses(self, cells):
        """Path cells -> (N, 3) poses x, y, heading (0 = straight ahead, + = left)."""
        p = np.array([(self.xs[i], self.ys[j]) for j, i in cells], float)
        if len(p) < 2:
            return np.column_stack([p, np.zeros(len(p))])
        # smooth a little, then heading from the direction of travel
        k = 5
        ps = np.array([p[max(0, n - k): n + k + 1].mean(axis=0) for n in range(len(p))])
        ps[0] = p[0]
        d = np.gradient(ps, axis=0)
        th = np.arctan2(-d[:, 0], d[:, 1])
        th[0] = 0.0
        return np.column_stack([ps, th])

    def rectangle_hits(self, poses, circles, lane_pts):
        """Index of the first pose (beyond --check-from metres) where the car's rectangle
        touches an object or a lane, or None."""
        a = self.a
        if len(poses) == 0:
            return None
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poses[:, :2], axis=0).T))])
        idx = np.nonzero(s >= a.check_from)[0]
        if len(idx) == 0:
            return None
        P = poses[idx]
        c, sn = np.cos(P[:, 2:3]), np.sin(P[:, 2:3])
        ox, oy = self.outline[None, :, 0], self.outline[None, :, 1]
        pts = np.stack([P[:, :1] + ox * c - oy * sn, P[:, 1:2] + ox * sn + oy * c], -1)  # (n, K, 2)
        hit = np.zeros(len(P), bool)
        for x, y, r in circles:
            hit |= (np.hypot(pts[..., 0] - x, pts[..., 1] - y) < r + a.touch_margin).any(axis=1)
        if len(lane_pts):  # lanes: distance-to-tape grid, looked up at the outline points
            j = np.round((pts[..., 1] + a.grid_behind) / self.c).astype(int)
            i = np.round((pts[..., 0] + a.grid_half_width) / self.c).astype(int)
            ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
            dl = np.full(pts.shape[:2], np.inf)
            dl[ok] = self.d_lane[j[ok], i[ok]]
            hit |= (dl < self.c * 0.5).any(axis=1)
        k = np.nonzero(hit)[0]
        return int(idx[k[0]]) if len(k) else None

    def clip_goal(self, gx, gy):
        """The course end (vehicle frame) if it's on the grid, else the point where the line
        from the car toward it leaves the grid."""
        a = self.a
        xmax, ymax, ymin = a.grid_half_width - self.c, a.grid_ahead - self.c, -a.grid_behind + self.c
        t = 1.0
        if abs(gx) > xmax:
            t = min(t, xmax / abs(gx))
        if gy > ymax:
            t = min(t, ymax / gy)
        if gy < ymin:
            t = min(t, ymin / gy)
        return gx * t, gy * t

    def plan(self, circles, lane_pts, goal_xy, prev=None):
        """Path (poses) toward the far target `goal_xy` (vehicle frame); re-runs A* with
        offending cells blocked until the rectangle along the path is clear. Returns (poses, note)."""
        a = self.a
        blocked, weight = self.costs(circles, lane_pts)
        if prev is not None and len(prev):
            # stick to the side chosen last frame: cells near the previous path are cheaper, so
            # two near-equal routes around an object don't swap from frame to frame
            near = np.zeros((self.ny, self.nx), np.uint8)
            j = np.round((prev[:, 1] + a.grid_behind) / self.c).astype(int)
            i = np.round((prev[:, 0] + a.grid_half_width) / self.c).astype(int)
            ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
            near[j[ok], i[ok]] = 1
            near = cv2.dilate(near, np.ones((5, 5), np.uint8)).astype(bool)
            weight = np.where(near, weight * a.keep_path, weight)
        start = self.cell_of(0.0, 0.0)
        blocked[start] = False  # the car is where it is
        self.goal_xy = self.clip_goal(*goal_xy)
        goal = self.cell_of(*self.goal_xy)  # if it can't be reached, A* returns the path to the
        # reachable cell closest to it (at a bend: into the turn)
        note = ""
        for attempt in range(a.max_replans + 1):
            cells, reached = self.search(start, goal, blocked, weight)
            poses = self.poses(cells)
            length = float(np.hypot(*np.diff(poses[:, :2], axis=0).T).sum()) if len(poses) > 1 else 0.0
            if length < a.min_path:
                return None, f"no path (best reaches {length:.2f} m{note})"
            bad = self.rectangle_hits(poses, circles, lane_pts)
            tag = "to goal" if reached else "partial, toward goal"
            if bad is None:
                return poses, f"{tag} {length:.2f} m, {len(cells)} cells{note}"
            # The rectangle touches something further on (often where the path ends beside a
            # lane, the car's nose poking over it). It re-plans every frame: if the first
            # --commit-dist is clear, drive that part.
            s_bad = float(np.hypot(*np.diff(poses[: bad + 1, :2], axis=0).T).sum())
            if s_bad >= a.commit_dist:
                return poses[:bad], f"{tag}, rectangle clear for {s_bad:.2f} m of {length:.2f} m{note}"
            j, i = cells[min(bad, len(cells) - 1)]
            blocked[max(0, j - 1): j + 2, max(0, i - 1): i + 2] = True
            blocked[start] = False
            note = f", {attempt + 1} re-plan(s) for the rectangle"
        return None, f"no path: rectangle touches objects on every route{note}"


# ---------------------------------------------------------------- the driver


class Driver:
    def __init__(self, a):
        self.a = a
        self.lane_args = types.SimpleNamespace(
            near=0.4, far=4.0, half_span=2.5, line_kernel_px=21, black_contrast=25, red_hue_lo=160, red_hue_hi=8,
            red_min_sat=256, red_contrast=999, above_floor_m=0.03, depth_noise_k=0.006, blocked_grow_px=5,
            per_pixel_floor=True, min_piece_m=0.4, min_thinness=4.0, under_object_m=0.35, tall_m=0.25,
            glare_l=225, black_max_ratio=0.62, floor_depth_win_px=31, min_floor_depth_frac=0.5, min_support=25,
            min_length_m=0.3, max_gap_m=0.3, max_segments=6, inlier_m=0.03, ransac_iters=60, max_points=1500)
        self.planner = AStar(a)
        self.cam_ahead = a.length / 2  # camera at the very front, middle
        self.car = None
        self.X = self.Y = self.th = 0.0
        self.mem_circles = []  # world (x, y, r, t_seen)
        self.mem_lanes = np.zeros((0, 3))  # world (x, y, t_seen)
        self.course = 0.0  # lane direction, world heading (+ = left); the car starts along the lane
        self.t0 = time.time()
        self.n = 0
        self.last_print = self.last_save = -1
        self.path, self.note, self.state = None, "", 0
        self.prev_world = None  # last path, world frame
        self.pivot = 0  # +1 / -1 while turning in place
        self.pivot_since = 0.0
        self._hdg = []

    # --- setup
    def start(self):
        a = self.a
        result = {}
        if not a.dry_run:  # the Arduino resets when its port opens: connect while the camera settles
            def connect():
                try:
                    result["car"] = avoid.Car(a.port, a.m_per_count)
                except Exception as e:
                    result["car"] = e
            th = threading.Thread(target=connect, daemon=True)
            th.start()
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
        self.cfwd, self.cright = sa.ground_axes(self.normal)
        self.det = ld.LaneDetector(self.intr, self.normal, self.offset, self.lane_args)
        self.det_plane = plane
        self.rng = np.random.default_rng(0)
        if a.record:  # raw frames for replaying offline, written by a background thread
            import queue
            os.makedirs(a.record, exist_ok=True)
            self.rec_q = queue.Queue(maxsize=200)
            self.rec_csv = open(os.path.join(a.record, "frames.csv"), "w")
            self.rec_csv.write("n,t,X,Y,theta,nx,ny,nz,offset\n")
            np.save(os.path.join(a.record, "intrinsics.npy"),
                    np.array([self.intr.width, self.intr.height, self.intr.ppx, self.intr.ppy, self.intr.fx, self.intr.fy]))

            def writer():
                while True:
                    item = self.rec_q.get()
                    if item is None:
                        return
                    n, color, depth_mm = item
                    cv2.imwrite(os.path.join(a.record, f"{n:05d}_color.png"), color)
                    np.savez_compressed(os.path.join(a.record, f"{n:05d}_depth.npz"), depth_mm=depth_mm)
            self.rec_thread = threading.Thread(target=writer, daemon=True)
            self.rec_thread.start()
        print(f"floor: camera height {self.offset:.3f} m ({self.offset / IN:.1f} in), tilt {self.tilt():.1f} deg down")
        if not a.dry_run:
            th.join()
            if isinstance(result["car"], Exception):
                raise result["car"]
            self.car = result["car"]
            self.car.set_origin()

    def close(self):
        if self.car:
            self.car.close()
        if getattr(self, "rec_thread", None):
            self.rec_q.put(None)
            self.rec_thread.join(timeout=30)
            self.rec_csv.close()
            print(f"recorded {self.n} frames to {self.a.record}")
        self.pipe.stop()

    def tilt(self):
        return math.degrees(math.atan2(-self.normal[2], -self.normal[1]))

    # --- frames
    def to_world(self, xy):
        c, s = math.cos(self.th), math.sin(self.th)
        return np.stack([self.X + xy[:, 0] * c - xy[:, 1] * s, self.Y + xy[:, 0] * s + xy[:, 1] * c], 1)

    def to_vehicle(self, xy):
        d = xy - [self.X, self.Y]
        c, s = math.cos(self.th), math.sin(self.th)
        return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], 1)

    def in_view(self, v):
        """Vehicle-frame points the camera sees on the floor right now (so memory there is
        replaced by what's seen)."""
        yc = v[:, 1] - self.cam_ahead
        return (yc > self.a.view_near) & (yc < self.a.view_far) & (np.abs(v[:, 0]) < yc * math.tan(math.radians(42)))

    def sense(self):
        a = self.a
        f = self.align.process(self.pipe.wait_for_frames(1000))
        self.color = np.asanyarray(f.get_color_frame().get_data()).copy()
        self.depth = np.asanyarray(f.get_depth_frame().get_data()) * self.scale
        now = time.time()
        if self.car:
            self.car.poll()
            self.X, self.Y, self.th = -self.car.left, self.car.ahead, self.car.heading()
        self._hdg = [h for h in self._hdg if now - h[0] < 0.3] + [(now, self.th)]
        (t0, h0), (t1, h1) = self._hdg[0], self._hdg[-1]
        self.yaw_rate = math.degrees(avoid.wrap(h1 - h0)) / (t1 - t0) if t1 > t0 else 0.0  # deg/s, + = left
        self.n += 1
        now = time.time()
        if getattr(self, "rec_thread", None):  # floor plane as used for this frame (before the refit)
            self.rec_csv.write(f"{self.n},{now:.3f},{self.X:.4f},{self.Y:.4f},{self.th:.5f},"
                               f"{self.normal[0]:.6f},{self.normal[1]:.6f},{self.normal[2]:.6f},{self.offset:.5f}\n")
            try:
                self.rec_q.put_nowait((self.n, self.color, np.round(self.depth * 1000).astype(np.uint16)))
            except Exception:
                print("  (recording queue full: frame skipped)")
        pts = sa.to_points(self.depth, self.rx, self.ry)
        # floor plane follows the tall mount's pitch
        fit = sa.refine_floor(pts, self.rows, self.normal, self.offset)
        if fit is not None:
            n = self.normal + 0.5 * (fit[0] - self.normal)
            self.normal = n / np.linalg.norm(n)
            self.offset += 0.5 * (fit[1] - self.offset)
            self.cfwd, self.cright = sa.ground_axes(self.normal)
            n0, off0 = self.det_plane
            if math.degrees(math.acos(min(1.0, float(self.normal @ n0)))) > 0.4 or abs(self.offset - off0) > 0.01:
                self.det.set_plane(self.normal, self.offset)
                self.det_plane = (self.normal, self.offset)

        # objects: depth points standing on the floor -> clusters -> circles (vehicle frame)
        h = pts @ self.normal + self.offset
        fwd, lat = pts @ self.cfwd, pts @ self.cright
        # depth noise grows with distance squared: so does the height that counts as standing
        obj = ((pts[..., 2] > 0) & (h > a.min_height + a.height_noise_k * pts[..., 2] ** 2) & (h < a.max_height)
               & (fwd > 0.15) & (fwd < a.see_range))
        self.obj_mask = obj  # (rows, cols) at the point stride, for the images
        corridor = obj & (np.abs(lat) < a.width / 2)  # straight in front, as wide as the car
        self.front = None
        if corridor.sum() >= a.front_points:
            f = np.sort(fwd[corridor])
            k = a.front_points
            dense = np.nonzero(f[k - 1:] - f[: len(f) - k + 1] <= 0.15)[0]  # a cluster, not stray pixels
            if len(dense):
                self.front = float(f[dense[0]])
        vx, vy = lat[obj], fwd[obj] + self.cam_ahead
        circles = []
        if len(vx):
            c = a.cluster_cell
            i = np.floor((vx + 4) / c).astype(int)
            j = np.floor(vy / c).astype(int)
            ok = (i >= 0) & (i < int(8 / c)) & (j >= 0) & (j < int(5 / c))
            counts = np.zeros((int(5 / c), int(8 / c)), np.int32)
            np.add.at(counts, (j[ok], i[ok]), 1)
            occ = (counts >= a.min_cell_points).astype(np.uint8)
            occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))
            n_lab, lab = cv2.connectedComponents(occ)
            for k in range(1, n_lab):
                jj, ii = np.nonzero(lab == k)
                cy_m = (jj.mean() + 0.5) * c - self.cam_ahead  # metres ahead of the camera
                if counts[jj, ii].sum() < a.min_points * (1 + cy_m / 2):  # far clusters need more evidence
                    continue
                cx, cy = (ii + 0.5) * c - 4, (jj + 0.5) * c
                mx, my = cx.mean(), cy.mean()
                r = float(np.hypot(cx - mx, cy - my).max()) + c / 2 + a.object_pad
                circles.append((float(mx), float(my), max(r, a.min_radius)))
        # memory: what's seen now replaces memory in view; out of view (blind band, sides) is kept
        if self.mem_circles:
            mem = np.array(self.mem_circles)
            v = self.to_vehicle(mem[:, :2])
            keep = ~self.in_view(v) & (now - mem[:, 3] < a.memory_s) & (np.hypot(v[:, 0], v[:, 1]) < a.memory_range)
            self.mem_circles = [tuple(m) for m in mem[keep]]
        if circles:
            w = self.to_world(np.array([c[:2] for c in circles]))
            self.mem_circles += [(float(p[0]), float(p[1]), c[2], now) for p, c in zip(w, circles)]
        self.circles = []
        if self.mem_circles:
            mem = np.array(self.mem_circles)
            v = self.to_vehicle(mem[:, :2])
            self.circles = [(float(x), float(y), float(r)) for (x, y), r in zip(v, mem[:, 2])]
            # anything overlapping the car itself isn't there any more (it moved: a person
            # walking by): drop it from memory too
            hw, hl = a.width / 2, a.length / 2
            gone = [max(abs(x) - hw, 0.0) ** 2 + max(abs(y) - hl, 0.0) ** 2 < (r - a.overlap_ok) ** 2 for x, y, r in self.circles]
            self.mem_circles = [m for m, g in zip(self.mem_circles, gone) if not g]
            self.circles = merge_circles([c for c, g in zip(self.circles, gone) if not g])

        # lanes: black tape segments -> points (vehicle frame), same memory rule
        _, black = self.det.masks(self.color, self.depth)
        self.lane_mask = black
        self.segments = self.det.segments(black, self.rng)
        if abs(getattr(self, "yaw_rate", 0.0)) > a.blur_rate:
            # rotating fast: the image is smeared and tape comes out in fragments. Don't
            # learn lanes from it; the remembered lanes (moved with the IMU heading) carry on.
            self.segments = []
        lp, ext = [], []
        for sg in self.segments:
            t = np.linspace(0, 1, max(2, int(sg.length / 0.05) + 1))
            p = sg.start[None] + t[:, None] * (sg.end - sg.start)[None]
            lp.append(np.column_stack([p[:, 0], p[:, 1] + self.cam_ahead]))
            # The camera can't see the floor beside the car: a piece running along the car
            # continues back past it, so the lane has no gap A* could route out through.
            near, far = (sg.start, sg.end) if sg.start[1] < sg.end[1] else (sg.end, sg.start)
            d = (far - near) / max(sg.length, 1e-6)
            if abs(math.degrees(math.atan2(d[0], d[1]))) < a.extend_max_deg and sg.length >= a.extend_min_len:
                back = near[1] + self.cam_ahead + a.grid_behind + 0.1  # metres back to behind the car centre
                t = np.arange(0.05, back / max(d[1], 0.3) + 1e-6, 0.05)
                q = near[None] - t[:, None] * d[None]
                # only if it runs beside the car, not under it (a bend piece extended would)
                if np.abs(q[:, 0]).min() > a.width / 2 + a.extend_side_gap:
                    ext.append(np.column_stack([q[:, 0], q[:, 1] + self.cam_ahead]))  # this frame only
        if len(self.mem_lanes):
            v = self.to_vehicle(self.mem_lanes[:, :2])
            keep = ~self.in_view(v) & (now - self.mem_lanes[:, 2] < a.memory_s) & (np.hypot(v[:, 0], v[:, 1]) < a.memory_range)
            self.mem_lanes = self.mem_lanes[keep]
        if lp:
            p = np.concatenate(lp)
            self.mem_lanes = np.vstack([self.mem_lanes, np.column_stack([self.to_world(p), np.full(len(p), now)])])
            # one point per 5 cm cell (newest kept)
            key = np.round(self.mem_lanes[::-1, :2] / 0.05).astype(np.int64)
            _, first = np.unique(key[:, 0] * 100003 + key[:, 1], return_index=True)
            self.mem_lanes = self.mem_lanes[::-1][first]
        self.lanes = self.to_vehicle(self.mem_lanes[:, :2]) if len(self.mem_lanes) else np.zeros((0, 2))
        if len(self.lanes):  # tape under the car can't be a wall it has to avoid: drop it
            under = (np.abs(self.lanes[:, 0]) < a.width / 2) & (np.abs(self.lanes[:, 1]) < a.length / 2)
            if under.any():
                self.mem_lanes = self.mem_lanes[~under]
                self.lanes = self.lanes[~under]
        if ext:  # extensions are guesses: rebuilt every frame from what's seen, never remembered
            self.lanes = np.vstack([self.lanes, np.concatenate(ext)])

    # --- deciding
    def goal_vehicle(self):
        """Target: a far point (--goal-dist) along the lane direction, vehicle frame. Partial
        views of objects and lanes always leave a way toward it; at a bend the lane direction
        turns, and so does the target."""
        ang = avoid.wrap(self.course - self.th)  # lane direction relative to the car (+ = left)
        return -self.a.goal_dist * math.sin(ang), self.a.goal_dist * math.cos(ang)

    def update_course(self):
        """Lane direction (world heading, + = left), from the lane pieces in view; held with
        the IMU when there are none. Pieces far off the current estimate (tape across the
        lane, other tape on the floor) are ignored."""
        a = self.a
        angs, wts = [], []
        segs = [sg for sg in self.segments if sg.length >= a.course_min_len]
        near = [sg for sg in segs if min(sg.start[1], sg.end[1]) < a.course_near]
        for sg in near or segs:  # the lane the car is in now, not the bend further on
            d = sg.end - sg.start
            if d[1] < 0:
                d = -d
            world = self.th + math.atan2(-d[0], d[1])
            if abs(math.degrees(avoid.wrap(world - self.course))) < a.course_accept:
                angs.append(world)
                wts.append(sg.length)
        if angs:
            w = np.array(wts)
            mean = math.atan2(float(np.sum(w * np.sin(angs))), float(np.sum(w * np.cos(angs))))
            self.course += a.course_smooth * avoid.wrap(mean - self.course)

    def step(self):
        """One frame: choose the state, plan, return the motor command (L, R) or an exit
        reason string."""
        a = self.a
        hw = a.width / 2
        self.update_course()
        # hard safety stop: something about to touch the car (planning keeps a margin, but the
        # car can lag the path), or depth sees something right at the bumper
        hw, hl = a.width / 2, a.length / 2
        for x, y, r in self.circles:
            gap = math.hypot(max(abs(x) - hw, 0.0), max(abs(y) - hl, 0.0)) - r
            ahead = y > hl  # in front of the bumper: the car drives into it; beside: it drives past
            if gap < (a.stop_gap if ahead else a.side_stop_gap):
                self.state = 3
                return f"state 3: SAFETY STOP, object at ({x:+.2f},{y:+.2f}) r{r:.2f} is {gap:.2f} m from the car"
        if self.front is not None and self.front < a.front_stop:
            self.state = 3
            return f"state 3: SAFETY STOP, depth sees something {self.front:.2f} m in front of the bumper"
        # --lanes-only: plan between the lanes only (the safety stop above still watches objects)
        objects = [] if a.lanes_only else [c for c in self.circles if 0 < math.hypot(c[0], c[1]) < a.object_range]
        L = self.lanes
        lane_near = bool(len(L)) and bool(((np.abs(L[:, 0]) < hw + a.lane_near) & (L[:, 1] > 0) & (L[:, 1] < a.lane_near_ahead)).any())
        if not objects and not lane_near:
            self.state, self.path, self.note = 1, None, "clear: straight ahead"
            err = -math.degrees(avoid.wrap(self.course - self.th))  # + = lane direction is to the right
            return self.wheels(err, a.heading_gain, objects)
        t = time.time()
        prev = self.to_vehicle(self.prev_world) if self.prev_world is not None else None
        poses, note = self.planner.plan(objects, L, self.goal_vehicle(), prev)
        self.plan_ms = 1000 * (time.time() - t)
        self.path, self.note = poses, note
        self.prev_world = self.to_world(poses[:, :2]) if poses is not None else None
        if poses is None:
            self.state = 3 if objects else 4
            return f"state {self.state}: {note}"
        self.state = 2
        # follow: aim at the path point --pursuit ahead of the car
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poses[:, :2], axis=0).T))])
        k = int(np.searchsorted(s, a.pursuit))
        x, y = poses[min(k, len(poses) - 1), :2]
        err = math.degrees(math.atan2(x, max(y, 0.05)))  # + = aim is to the right
        return self.wheels(err, a.steer_gain, objects)

    def wheels(self, err, gain, objects):
        """L/R commands to head `err` degrees (+ = right). The motors don't turn below ~230
        and a car with one wheel stopped often doesn't turn at all (skid steer), so: steer
        with a difference while both wheels keep driving forward; anything sharper (or a
        stall) turns in place with balanced commands, if the rectangle can swing round."""
        a = self.a
        # where the heading will be once the car stops rotating (it coasts ~--coast-s)
        err = err + getattr(self, "yaw_rate", 0.0) * a.coast_s  # turning left moves an aim to the right further right
        diff = min(abs(gain * err), a.diff_max)
        stalled = False  # stalls are handled in run() (motor controller latch)
        now = time.time()
        want = -1 if err > 0 else 1  # +1 = turn left
        if not self.pivot and gain * abs(err) > 2 * a.max_diff_drive and now - self.pivot_since > a.pivot_hold:
            self.pivot, self.pivot_since = want, now  # sharper than driving can steer: turn in place
        if self.pivot:  # keep turning until the predicted heading is close, without flip-flopping
            if want != self.pivot or abs(err) < a.pivot_exit:
                self.pivot = 0
        elif (abs(err) > a.pivot_deg or stalled) and now - self.pivot_since > a.pivot_hold:
            self.pivot, self.pivot_since = want, now
        if self.pivot:
            if self.can_rotate(self.pivot, objects):
                cmd = a.pivot_cmd * (1.2 if stalled else 1.0)
                return -self.pivot * cmd, self.pivot * cmd
            self.pivot = 0
        # Moderate differences only: measured, L250/R450 turns left as asked, but one wheel near
        # the deadband with the other above ~550 (L260/R650-800) crawls and turns the WRONG way.
        # Sharper turns are balanced turns in place (above).
        inner = max(a.base - diff, a.inner_min)
        outer = min(a.base + diff, inner + a.max_diff_drive)
        return (outer, inner) if err > 0 else (inner, outer)  # err > 0: turn right (left wheel faster)

    def stalled(self):
        """Commanded to move, but neither position nor heading has changed for --stall-s."""
        now = time.time()
        hist = getattr(self, "_moves", [])
        hist = [h for h in hist if now - h[0] < a_stall(self.a)] + [(now, self.X, self.Y, self.th)]
        self._moves = hist
        if not self.car or now - hist[0][0] < a_stall(self.a) * 0.9:
            return False
        t0, x0, y0, th0 = hist[0]
        still = math.hypot(self.X - x0, self.Y - y0) < 0.02 and abs(math.degrees(avoid.wrap(self.th - th0))) < 1.0
        return still

    def can_rotate(self, sgn, circles, deg=12.0):
        """Can the car's rectangle turn `deg` in place (sgn + = left) without touching an object?"""
        out = self.planner.outline
        for d in np.radians(np.arange(3.0, deg + 1e-6, 3.0)) * sgn:
            c, s = math.cos(d), math.sin(d)
            pts = np.stack([out[:, 0] * c - out[:, 1] * s, out[:, 0] * s + out[:, 1] * c], 1)
            for x, y, r in circles:
                if (np.hypot(pts[:, 0] - x, pts[:, 1] - y) < r).any():
                    return False
        return True

    # --- logging / images
    def log(self, cmd):
        el = time.time() - self.t0
        if int(el * 4) == self.last_print:
            return
        self.last_print = int(el * 4)
        near = sorted(self.circles, key=lambda c: math.hypot(c[0], c[1]) - c[2])[:4]
        objs = " ".join(f"({x:+.2f},{y:.2f} r{r:.2f})" for x, y, r in near)
        pl = f" {self.plan_ms:.0f} ms" if self.state == 2 else ""
        cm = f" | L{cmd[0]:.0f} R{cmd[1]:.0f}" if isinstance(cmd, tuple) else ""
        print(f"{el:5.1f}s state {self.state} | pos ({self.X:+.2f},{self.Y:.2f}) hdg {math.degrees(self.th):+6.1f} | "
              f"objects {len(self.circles)} {objs} | lane pts {len(self.lanes)} | {self.note}{pl}{cm}")

    def project(self, xy_vehicle):
        """Vehicle-frame floor points -> image pixels (None where behind the camera)."""
        foot = -self.offset * self.normal
        out = []
        for x, y in xy_vehicle:
            p = foot + x * self.cright + (y - self.cam_ahead) * self.cfwd
            if p[2] <= 0.05:
                out.append(None)
                continue
            out.append((int(self.intr.fx * p[0] / p[2] + self.intr.ppx), int(self.intr.fy * p[1] / p[2] + self.intr.ppy)))
        return out

    def mark(self, img):
        """Objects (mask + circles + labels) and lanes (lines) drawn on an image."""
        img = img.copy()
        m = cv2.resize(self.obj_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        img[m] = (0.5 * img[m] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        for sg in self.segments:  # lanes as lines
            a_, b_ = self.project([(sg.start[0], sg.start[1] + self.cam_ahead), (sg.end[0], sg.end[1] + self.cam_ahead)])
            if a_ and b_:
                cv2.line(img, a_, b_, (0, 255, 0), 3)
        for n, (x, y, r) in enumerate(self.circles):  # objects: floor circle + label
            ring = self.project([(x + r * math.cos(t), y + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 24)])
            ring = [p for p in ring if p]
            if len(ring) > 2:
                cv2.polylines(img, [np.array(ring, np.int32)], True, (0, 165, 255), 2)
            c = self.project([(x, y)])[0]
            if c:
                cv2.putText(img, f"#{n} {math.hypot(x, y):.2f}m", (c[0] - 30, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        if self.path is not None:
            pp = [p for p in self.project(self.path[:, :2]) if p]
            if len(pp) > 1:
                cv2.polylines(img, [np.array(pp, np.int32)], False, (255, 0, 255), 2)
        cv2.putText(img, f"state {self.state}: {self.note}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return img

    def topdown(self, size=480, span=5.0):
        a = self.a
        px = size / span
        img = np.full((size, size, 3), 30, np.uint8)
        to = lambda x, y: (int(size / 2 + x * px), int(size * 0.85 - y * px))
        for x, y, r in self.circles:
            cv2.circle(img, to(x, y), max(2, int(r * px)), (0, 165, 255), 2)
            cv2.circle(img, to(x, y), max(2, int((r + a.width / 2) * px)), (60, 60, 120), 1)
        for x, y in self.lanes:
            cv2.circle(img, to(x, y), 1, (255, 255, 0), -1)
        if self.path is not None:
            cv2.polylines(img, [np.array([to(x, y) for x, y in self.path[:, :2]], np.int32)], False, (0, 255, 0), 2)
        hw, hl = a.width / 2, a.length / 2
        cv2.polylines(img, [np.array([to(-hw, -hl), to(hw, -hl), to(hw, hl), to(-hw, hl)], np.int32)], True, (255, 255, 255), 2)
        gx, gy = self.goal_vehicle()  # far target (may be off the picture) and the A* target
        cv2.drawMarker(img, to(gx, gy), (0, 0, 255), cv2.MARKER_STAR, 18, 2)
        if hasattr(self.planner, "goal_xy"):
            cv2.drawMarker(img, to(*self.planner.goal_xy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, f"lane direction {math.degrees(avoid.wrap(self.course - self.th)):+.0f} deg", (8, size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return img

    def depth_image(self):
        d8 = cv2.applyColorMap(cv2.convertScaleAbs(self.depth, alpha=255 / 5.0), cv2.COLORMAP_JET)
        d8[self.depth == 0] = 0
        return d8

    def save_periodic(self):
        a = self.a
        el = time.time() - self.t0
        if a.save and int(el * 2) != self.last_save:
            self.last_save = int(el * 2)
            cv2.imwrite(os.path.join(a.save, f"t{el:05.1f}.jpg"), np.hstack([self.mark(self.color), self.topdown()]))
            os.makedirs(os.path.join(a.save, "masks"), exist_ok=True)  # what the lane detector saw, black/white
            cv2.imwrite(os.path.join(a.save, "masks", f"t{el:05.1f}_lane_mask.png"), self.lane_mask.astype(np.uint8) * 255)

    def save_failure(self, tag):
        out = self.a.save or os.path.join("runs", time.strftime("fail_%Y%m%d_%H%M%S"))
        os.makedirs(out, exist_ok=True)
        files = {
            "depth image": os.path.join(out, f"{tag}_depth.png"),
            "depth image, objects + lanes marked": os.path.join(out, f"{tag}_depth_marked.png"),
            "colour image, objects + lanes marked": os.path.join(out, f"{tag}_color_marked.png"),
            "object mask": os.path.join(out, f"{tag}_object_mask.png"),
            "lane mask": os.path.join(out, f"{tag}_lane_mask.png"),
            "top-down map + path": os.path.join(out, f"{tag}_topdown.png"),
        }
        cv2.imwrite(files["depth image"], self.depth_image())
        cv2.imwrite(files["depth image, objects + lanes marked"], self.mark(self.depth_image()))
        cv2.imwrite(files["colour image, objects + lanes marked"], self.mark(self.color))
        cv2.imwrite(files["object mask"], cv2.resize(self.obj_mask.astype(np.uint8) * 255, (sa.WIDTH, sa.HEIGHT), interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(files["lane mask"], self.lane_mask.astype(np.uint8) * 255)
        cv2.imwrite(files["top-down map + path"], self.topdown())
        return files

    def report_failure(self, why):
        print(f"\nERROR: {why}")
        print(f"  car at ({self.X:+.2f},{self.Y:.2f}) heading {math.degrees(self.th):+.1f} deg")
        print(f"  nearby objects (vehicle frame: x right, y ahead of the car centre; distance to edge):")
        for n, (x, y, r) in enumerate(sorted(self.circles, key=lambda c: math.hypot(c[0], c[1]) - c[2])):
            print(f"    #{n}: centre ({x:+.2f}, {y:+.2f}) m, radius {r:.2f} m, edge {math.hypot(x, y) - r:.2f} m away")
        if not self.circles:
            print("    none")
        print(f"  lanes: {len(self.segments)} tape segment(s) in view, {len(self.lanes)} lane points remembered")
        for sg in self.segments:
            print(f"    ({sg.start[0]:+.2f},{sg.start[1] + self.cam_ahead:.2f}) -> ({sg.end[0]:+.2f},{sg.end[1] + self.cam_ahead:.2f}) m, "
                  f"{sg.length:.2f} m long")
        files = self.save_failure(f"state{self.state}")
        print("  saved:")
        for k, v in files.items():
            print(f"    {k}: {v}")

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
            cmd = self.step()
            if isinstance(cmd, tuple) and self.car and self.stalled():
                # The Roboteq cuts a motor that draws current without turning, and keeps it cut
                # until the command goes back to zero: release it, then carry on
                print(f"  WARNING: commanded L{cmd[0]:.0f} R{cmd[1]:.0f} but not moving for {a.stall_s:.1f} s: "
                      f"sending zero for {a.stall_release:.1f} s to release the motor controller's stall cut-out")
                t_rel = time.time()
                while time.time() - t_rel < a.stall_release:
                    self.car.drive(0, 0)
                    time.sleep(0.05)
                self._moves = []
                continue
            if isinstance(cmd, str) and cmd.startswith("done"):
                self.stop()
                return cmd
            if isinstance(cmd, str):
                self.stop()
                self.report_failure(cmd)
                return cmd
            if self.car:
                self.car.drive(*cmd)
            self.log(cmd)
            self.save_periodic()

    def stop(self):
        if self.car:
            self.car.drive(0, 0)


def a_stall(a):
    return a.stall_s


def merge_circles(circles):
    """Merge circles whose centres are closer than their radii (same object seen twice)."""
    out = []
    for c in sorted(circles, key=lambda c: -c[2]):
        if all(math.hypot(c[0] - o[0], c[1] - o[1]) > max(c[2], o[2]) * 0.8 for o in out):
            out.append(c)
    return out


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
    # vehicle (camera at the very front, middle, 42 in up)
    ap.add_argument("--length", type=float, default=38 * IN, help="m (38 in)")
    ap.add_argument("--width", type=float, default=32 * IN, help="m (32 in)")
    ap.add_argument("--m-per-count", type=float, default=0.0075)
    # perception
    ap.add_argument("--see-range", type=float, default=3.5)
    ap.add_argument("--view-far", type=float, default=5.0, help="memory this far ahead of the camera, in view, is replaced (m)")
    ap.add_argument("--view-near", type=float, default=0.70, help="camera sees the floor from this far ahead of it (m)")
    ap.add_argument("--stop-gap", type=float, default=0.12, help="safety stop: object this close to the car's rectangle (m)")
    ap.add_argument("--side-stop-gap", type=float, default=0.03, help="safety stop: object beside the car this close (m)")
    ap.add_argument("--front-stop", type=float, default=0.25, help="safety stop: depth sees something this close in front of the camera/bumper (m)")
    ap.add_argument("--front-points", type=int, default=30)
    ap.add_argument("--min-height", type=float, default=0.08)
    ap.add_argument("--height-noise-k", type=float, default=0.006, help="standing threshold grows by this x depth^2 (m)")
    ap.add_argument("--max-height", type=float, default=1.3)
    ap.add_argument("--cluster-cell", type=float, default=0.05)
    ap.add_argument("--min-cell-points", type=int, default=3)
    ap.add_argument("--min-points", type=int, default=30, help="depth points for a cluster to be an object")
    ap.add_argument("--object-pad", type=float, default=0.05, help="added to each object's radius (m)")
    ap.add_argument("--min-radius", type=float, default=0.30, help="objects at least this big: a stool's base hides under its seat (m)")
    ap.add_argument("--memory-s", type=float, default=6.0, help="remember objects/lanes out of view this long (s)")
    ap.add_argument("--overlap-ok", type=float, default=0.05, help="a remembered object overlapping the car more than this is dropped (m)")
    ap.add_argument("--memory-range", type=float, default=2.0, help="remember out-of-view things only this close (m)")
    # planning
    ap.add_argument("--cell", type=float, default=0.05, help="A* grid cell (m)")
    ap.add_argument("--grid-half-width", type=float, default=1.4,
                    help="A* grid reaches this far to each side (m): the car can't route outside the lane lines")
    ap.add_argument("--grid-behind", type=float, default=0.05, help="grid starts this far behind the car centre: no routes behind it (m)")
    ap.add_argument("--grid-ahead", type=float, default=4.0)
    ap.add_argument("--goal-dist", type=float, default=8.0,
                    help="target: this far along the lane direction (beyond the grid: aimed at its edge) (m)")
    ap.add_argument("--course-min-len", type=float, default=0.6, help="lane pieces this long set the lane direction (m)")
    ap.add_argument("--course-near", type=float, default=1.5, help="lane direction from pieces starting this close to the camera (m)")
    ap.add_argument("--course-accept", type=float, default=30.0, help="...if within this of the current direction (deg)")
    ap.add_argument("--course-smooth", type=float, default=0.08)
    ap.add_argument("--hard-margin", type=float, default=0.02, help="car side must stay this clear of objects (m)")
    ap.add_argument("--soft-zone", type=float, default=0.5, help="cells within this of the car touching are weighted (m)")
    ap.add_argument("--near-weight", type=float, default=25.0, help="extra cost right next to an object")
    ap.add_argument("--lane-r", type=float, default=0.0, help="lane tape counted this thick around its centre line (m)")
    ap.add_argument("--touch-margin", type=float, default=0.02, help="rectangle check: minimum gap to objects (m)")
    ap.add_argument("--check-from", type=float, default=0.3, help="rectangle check starts this far along the path (m)")
    ap.add_argument("--blocking-halfwidth", type=float, default=1.2, help="objects within this of the centre line are 'in the way' (m)")
    ap.add_argument("--extend-side-gap", type=float, default=0.1, help="extended lane must pass this far beside the car (m)")
    ap.add_argument("--extend-max-deg", type=float, default=20.0, help="lane pieces within this of straight ahead are extended back past the car")
    ap.add_argument("--extend-min-len", type=float, default=0.5, help="only lane pieces at least this long are extended (m)")
    ap.add_argument("--commit-dist", type=float, default=1.0,
                    help="a path whose rectangle is clear this far is driven up to where it isn't (m)")
    ap.add_argument("--max-replans", type=int, default=6)
    ap.add_argument("--min-path", type=float, default=0.3, help="shorter best path = no path (m)")
    ap.add_argument("--object-range", type=float, default=3.5, help="objects this close count (m)")
    ap.add_argument("--lane-near", type=float, default=0.25, help="lane this close to the car's side = 'near' (m)")
    ap.add_argument("--lane-near-ahead", type=float, default=1.5)
    # driving (motors don't turn below ~230; 300/300 rolls ~0.15 m/s)
    ap.add_argument("--base", type=float, default=340.0, help="forward L/R command")
    ap.add_argument("--diff-max", type=float, default=550.0, help="inner wheel may run backwards: skid-steer needs it to turn")
    ap.add_argument("--steer-gain", type=float, default=6.0, help="L/R difference per degree off the path")
    ap.add_argument("--heading-gain", type=float, default=8.0, help="state 1: per degree off the held heading")
    ap.add_argument("--pursuit", type=float, default=1.3, help="aim at the path point this far along (m)")
    ap.add_argument("--pivot-deg", type=float, default=35.0, help="aim further off than this: turn in place")
    ap.add_argument("--pivot-cmd", type=float, default=820.0, help="in-place turns need ~800 (600 stalls the motors)")
    ap.add_argument("--pivot-exit", type=float, default=12.0, help="stop turning in place within this of the aim (deg)")
    ap.add_argument("--pivot-hold", type=float, default=1.0, help="no new in-place turn within this of the last (s)")
    ap.add_argument("--coast-s", type=float, default=0.45, help="the car keeps rotating this long after the command changes")
    ap.add_argument("--keep-path", type=float, default=0.6, help="cost factor near last frame's path (sticks to a side)")
    ap.add_argument("--deadband", type=float, default=230.0, help="motor command below which a wheel doesn't turn")
    ap.add_argument("--max-cmd", type=float, default=850.0)
    ap.add_argument("--inner-min", type=float, default=300.0, help="slower wheel while steering on the move")
    ap.add_argument("--max-diff-drive", type=float, default=200.0, help="largest L/R difference while driving (bigger: turn in place)")
    ap.add_argument("--stall-release", type=float, default=0.4, help="zero command this long to clear a motor stall cut-out (s)")
    ap.add_argument("--stall-s", type=float, default=1.5, help="no movement this long while commanded = stalled (s)")
    # run
    ap.add_argument("--max-dist", type=float, default=15.0)
    ap.add_argument("--max-time", type=float, default=180.0)
    ap.add_argument("--record", help="save every raw frame (colour, depth, pose, floor plane) here for offline replay")
    ap.add_argument("--blur-rate", type=float, default=30.0, help="ignore lane detections while turning faster than this (deg/s)")
    ap.add_argument("--lanes-only", action="store_true", help="drive between the lanes, ignoring objects for planning")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--save", help="directory for log.txt, images every 0.5 s, and failure images")
    a = ap.parse_args()
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        sys.stdout = Tee(sys.stdout, open(os.path.join(a.save, "log.txt"), "w"))
    d = Driver(a)
    code = 0
    try:
        d.start()
        why = d.run()
        d.stop()
        if d.car:
            d.car.settle()
        print(f"STOP: {why}; pos ({d.X:+.2f},{d.Y:.2f}) heading {math.degrees(d.th):+.1f} deg"
              + (f"; travelled {d.car.odo:.2f} m" if d.car else ""))
        code = 1 if why.startswith("state") else 0
    finally:
        d.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
