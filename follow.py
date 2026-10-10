#!/usr/bin/env python3
"""Drive the taped course around obstacles: one simple reactive loop.

Every camera frame:
  1. Map (world grid, kept with wheel odometry + IMU heading): depth points standing above
     the floor mark cells occupied, floor seen in view clears them; what's out of view (the
     blind band right in front, beside, behind) is remembered. Black tape cells are walls.
  2. Goal: the lane centre --lookahead ahead (from whichever lanes are visible), else the
     last lane direction (held with the IMU).
  3. Steer: a fan of arcs (straight to tight) is checked with the car's outline against the
     map; each gets a free distance. Take the arc, free for at least --min-free, that best
     points at the goal (preferring long free distance and small steering changes). Speed
     drops as the free distance shrinks.
  4. No arc free: turn slowly in place toward the side with more room, re-checking every
     frame; drive as soon as an arc opens. Only if it can't even turn, reverse a little.

Usage:
    python3 follow.py --dry-run --frames 60 --save DIR   # perceive + choose, motors off
    python3 follow.py --save runs/follow1                 # drive (log.txt + images in DIR)
"""

from __future__ import annotations

import argparse
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
from lane_drive import LaneCentre


# ---------------------------------------------------------------- geometry


def outline(hw, hl, step=0.06):
    """Points on the car's rectangle, car frame (x right, y forward), turning centre at 0."""
    xs = np.linspace(-hw, hw, int(2 * hw / step) + 1)
    ys = np.linspace(-hl, hl, int(2 * hl / step) + 1)
    return np.concatenate([np.stack([xs, np.full_like(xs, hl)], 1), np.stack([xs, np.full_like(xs, -hl)], 1),
                           np.stack([np.full_like(ys, hw), ys], 1), np.stack([np.full_like(ys, -hw), ys], 1)])


def arc_poses(k, length, step=0.05):
    """Poses (N, 3) along an arc of curvature k (+ = left) from the origin, car frame."""
    s = np.arange(0.0, length + 1e-6, step)
    th = k * s
    if abs(k) < 1e-6:
        x, y = np.zeros_like(s), s
    else:
        x, y = -(1 - np.cos(th)) / k, np.sin(th) / k
    return np.stack([x, y, th], 1)


def place(poses, pts, X, Y, theta):
    """Car-frame outline `pts` at car-frame `poses`, moved to the world pose (X, Y, theta).
    Returns world points (N, K, 2)."""
    th = theta + poses[:, 2:3]
    c, s = np.cos(theta), np.sin(theta)
    px = X + poses[:, 0:1] * c - poses[:, 1:2] * s  # car frame (x right, y fwd) -> world
    py = Y + poses[:, 0:1] * s + poses[:, 1:2] * c
    ox, oy = pts[None, :, 0], pts[None, :, 1]
    return np.stack([px + ox * np.cos(th) - oy * np.sin(th), py + ox * np.sin(th) + oy * np.cos(th)], -1)


# ---------------------------------------------------------------- world map


class Grid:
    """World map, kept with odometry + IMU. World: x right, y ahead of the start.
    Two layers for the planner:
      hard: objects as seen (never driven, turned or reversed into)
      soft: a margin around objects, and the tape lines (lane edges: not driven across,
            but turning in place or reversing over them is fine)"""

    def __init__(self, a):
        self.a = a
        self.cell = a.cell
        self.x0, self.y0 = -a.map_size / 2, -a.map_size / 4
        self.n = int(a.map_size / a.cell)
        self.occ = np.zeros((self.n, self.n), np.int16)  # evidence: + seen standing, - seen floor
        self.tape = np.zeros((self.n, self.n), np.int16)  # evidence: + seen tape, - seen bare floor
        r = int(round(a.object_pad / a.cell))
        self.pad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        self.hard = np.zeros((self.n, self.n), bool)
        self.soft = self.hard
        self.seen = self.hard

    def ij(self, xy):
        i = ((xy[..., 0] - self.x0) / self.cell).astype(np.int32)
        j = ((xy[..., 1] - self.y0) / self.cell).astype(np.int32)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        return i, j, ok

    def counts(self, xy):
        i, j, ok = self.ij(xy)
        c = np.zeros((self.n, self.n), np.int32)
        np.add.at(c, (j[ok], i[ok]), 1)
        return c

    def update(self, obstacle_xy, floor_xy, tape_xy, near_floor_xy):
        """One frame: objects, floor seen (clears objects), tape seen (and bare floor near
        enough to judge tape on, which fades tape that isn't there any more)."""
        a = self.a
        hit = self.counts(obstacle_xy) >= a.min_cell_points
        seen = self.counts(floor_xy) > 0
        self.occ[hit] = np.minimum(self.occ[hit] + 2, a.occ_max)
        clear = seen & ~hit
        self.occ[clear] = np.maximum(self.occ[clear] - 1, 0)
        obj = self.occ >= 2
        obj_pad = cv2.dilate(obj.astype(np.uint8), self.pad).astype(bool)
        if tape_xy is not None:
            tape = (self.counts(tape_xy) > 0) & ~obj_pad  # stool legs and rings are black too
            tape = cv2.dilate(tape.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            self.tape[tape] = np.minimum(self.tape[tape] + 3, a.tape_max)
            bare = (self.counts(near_floor_xy) > 0) & ~tape
            self.tape[bare] = np.maximum(self.tape[bare] - 1, 0)
        self.hard = obj
        self.soft = obj_pad | (self.tape >= 2)

    def hits(self, world_pts, layer):
        """Bool per point: on a cell of `layer` (outside the map counts as blocked)."""
        g = self.hard if layer == "hard" else self.soft
        i, j, ok = self.ij(world_pts)
        out = np.ones(world_pts.shape[:-1], bool)
        out[ok] = g[j[ok], i[ok]]
        return out


# ---------------------------------------------------------------- the driver


class Follower:
    def __init__(self, a):
        self.a = a
        self.lane_args = types.SimpleNamespace(
            near=0.4, far=5.0, half_span=2.5, line_kernel_px=21, black_contrast=25, red_hue_lo=160, red_hue_hi=8,
            red_min_sat=256, red_contrast=999, above_floor_m=0.03, depth_noise_k=0.006, blocked_grow_px=15,
            glare_l=225, black_max_ratio=0.62, floor_depth_win_px=31, min_floor_depth_frac=0.5, min_support=25,
            min_length_m=0.3, max_gap_m=0.3, max_segments=6, inlier_m=0.03, ransac_iters=60, max_points=1500)
        self.centre_args = types.SimpleNamespace(lane_width=a.lane_width, left_lane="red", lookahead=a.lookahead,
                                                 width_smooth=0.1, target_smooth=0.4)
        self.out = outline(a.half_width, a.half_length)
        self.ks = np.linspace(-1 / a.min_radius, 1 / a.min_radius, a.arcs)
        self.arcs = [arc_poses(k, a.arc_length) for k in self.ks]
        self.grid = Grid(a)
        self.car = None
        self.X = self.Y = self.th = 0.0
        self.goal_heading = 0.0  # world heading of the course when no lane is visible
        self.k_prev = 0.0
        self.spin = 0  # +1 / -1 while turning in place to find a way
        self.cmd = 0.0
        self.t0 = time.time()
        self.n = 0
        self.last_print = self.last_save = -1

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
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)
        self.rx, self.ry, self.rows = sa.pixel_rays(intr)
        for _ in range(15):
            self.pipe.wait_for_frames()
        stack = [np.asanyarray(self.align.process(self.pipe.wait_for_frames()).get_depth_frame().get_data()).copy()
                 for _ in range(10)]
        plane = sa.fit_floor(sa.to_points(np.median(np.stack(stack), axis=0) * self.scale, self.rx, self.ry), self.rows)
        if plane is None:
            sys.exit("floor calibration failed: not enough flat floor in view")
        self.normal, self.offset = plane
        self.cfwd, self.cright = sa.ground_axes(self.normal)
        self.det = ld.LaneDetector(intr, self.normal, self.offset, self.lane_args)
        self.det_plane = plane
        self.centre = LaneCentre(self.centre_args)
        self.rng = np.random.default_rng(0)
        print(f"floor: camera height {self.offset:.3f} m, tilt {self.tilt():.1f} deg down")
        if not a.dry_run:
            th.join()
            if isinstance(result["car"], Exception):
                raise result["car"]
            self.car = result["car"]
            self.car.set_origin()

    def close(self):
        if self.car:
            self.car.close()
        self.pipe.stop()

    def tilt(self):
        return math.degrees(math.atan2(-self.normal[2], -self.normal[1]))

    # --- frames
    def cam_to_world(self, lat, fwd):
        """Ground coordinates relative to the camera (lat right, fwd ahead) -> world (N, 2)."""
        c, s = math.cos(self.th), math.sin(self.th)
        cx, cy = self.X - self.a.cam_ahead * s, self.Y + self.a.cam_ahead * c
        return np.stack([cx + lat * c - fwd * s, cy + lat * s + fwd * c], -1)

    def update_pose(self):
        if self.car:
            self.car.poll()
            self.X, self.Y, self.th = -self.car.left, self.car.ahead, self.car.heading()

    def sense(self):
        a = self.a
        f = self.align.process(self.pipe.wait_for_frames(1000))
        color = np.asanyarray(f.get_color_frame().get_data())
        depth = np.asanyarray(f.get_depth_frame().get_data()) * self.scale
        self.update_pose()
        self.n += 1
        pts = sa.to_points(depth, self.rx, self.ry)
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
        # objects + floor into the map
        valid = pts[..., 2] > 0
        h = pts @ self.normal + self.offset
        fwd, lat = pts @ self.cfwd, pts @ self.cright
        inrange = valid & (fwd > 0.2) & (fwd < a.see_range)
        obj = inrange & (h > a.min_height) & (h < a.max_height)
        floor = inrange & (np.abs(h) < 0.03)
        near_floor = floor & (fwd < a.tape_clear_range)
        # lanes: split by side of the car, tape into the map
        red, black = self.det.masks(color, depth)
        segs = self.det.segments(black, self.rng)
        lanes = {"red": [], "black": []}  # "red" = left lane, "black" = right lane (both black tape)
        tape_pts = []
        for sg in segs:
            p = sg.start if sg.start[1] < sg.end[1] else sg.end
            (lanes["red"] if p[0] < 0 else lanes["black"]).append(sg)
            d = (sg.end - sg.start) / max(sg.length, 1e-6)
            if d[1] < 0:
                d = -d
            near_end = sg.start if sg.start[1] < sg.end[1] else sg.end
            # the tape continues toward the car through the camera's blind band
            # (only pieces running roughly along the car: a bend extended would cross the lane)
            back = a.tape_extend if abs(math.degrees(math.atan2(d[0], d[1]))) < 30 else 0.0
            t = np.arange(-back, sg.length + 1e-6, 0.03)
            p = near_end + t[:, None] * d
            p = p[np.hypot(p[:, 0], p[:, 1]) < a.tape_range]  # far tape is placed less accurately
            tape_pts.append(p)
        tape_xy = None
        if tape_pts:
            p = np.concatenate(tape_pts)
            tape_xy = self.cam_to_world(p[:, 0], p[:, 1])
        self.grid.update(self.cam_to_world(lat[obj], fwd[obj]), self.cam_to_world(lat[floor], fwd[floor]),
                         tape_xy, self.cam_to_world(lat[near_floor], fwd[near_floor]))
        target, self.how = self.centre.update(lanes)
        # The course heading lives in the world (IMU): a lane-centre point only counts if it
        # lies roughly along it, and it only nudges it. Turning the car can't drag it around.
        self.goal = None
        if target is not None:
            g = self.cam_to_world(np.array([target[0]]), np.array([target[1]]))[0]
            bearing = math.atan2(-(g[0] - self.X), g[1] - self.Y)
            if abs(math.degrees(avoid.wrap(bearing - self.goal_heading))) < a.goal_accept:
                self.goal = g
                self.goal_heading += a.goal_smooth * avoid.wrap(bearing - self.goal_heading)
        self._frame = (color, red, black, lanes)

    # --- choosing a move
    def free_distances(self):
        """Free distance (m) along each arc before the car's outline touches the map. If the
        car already overlaps the soft layer (margin, tape), that layer is ignored for the
        first --escape metres, so it can drive off the line / out of the margin."""
        a = self.a
        here = place(np.zeros((1, 3)), self.out, self.X, self.Y, self.th)
        self.on_soft = bool(self.grid.hits(here, "soft").any())
        free = np.empty(len(self.arcs))
        for i, poses in enumerate(self.arcs):
            pts = place(poses, self.out, self.X, self.Y, self.th)
            soft = self.grid.hits(pts, "soft").any(axis=1)
            if self.on_soft:
                soft[: int(a.escape / 0.05) + 1] = False
            hit = soft | self.grid.hits(pts, "hard").any(axis=1)
            hit[0] = False  # judged by the move, not where the car stands now
            free[i] = (np.argmax(hit) - 1) * 0.05 if hit.any() else a.arc_length
        return free

    def choose(self):
        """(curvature, free distance, note) of the best arc, or None if nothing is free."""
        a = self.a
        free = self.free_distances()
        self.free = free
        # escaping a line/margin: the arc must get fully off it, then still have room
        ok = free >= a.min_free + (a.escape if self.on_soft else 0.0)
        if not ok.any():
            return None
        # how well each arc points at the goal: heading at the end of its free part
        s = np.minimum(free, a.lookahead)
        end_th = self.th + self.ks * s
        err = np.abs(np.degrees(np.arctan2(np.sin(end_th - self.goal_heading), np.cos(end_th - self.goal_heading))))
        if self.goal is not None:  # and how close its end gets to the lane-centre point
            c, sn = math.cos(self.th), math.sin(self.th)
            kk = np.where(np.abs(self.ks) < 1e-6, 1e-6, self.ks)
            ex, ey = -(1 - np.cos(kk * s)) / kk, np.sin(kk * s) / kk
            wx, wy = self.X + ex * c - ey * sn, self.Y + ex * sn + ey * c
            err = err * 0.5 + 0.5 * np.degrees(np.hypot(wx - self.goal[0], wy - self.goal[1]))  # ~1 deg per cm
        score = a.w_free * np.minimum(free, a.arc_length) - a.w_goal * err / 45 - a.w_smooth * np.abs(self.ks - self.k_prev)
        score[~ok] = -np.inf
        best = int(np.argmax(score))
        return float(self.ks[best]), float(free[best]), "escaping" if self.on_soft else ""

    def _sweep_ok(self, poses):
        """Do these poses keep the car's outline off objects it isn't already touching?"""
        now = self.grid.hits(place(np.zeros((1, 3)), self.out, self.X, self.Y, self.th), "hard")[0]
        return not (self.grid.hits(place(poses, self.out, self.X, self.Y, self.th), "hard") & ~now[None]).any()

    def can_rotate(self, sign, deg=12.0):
        """Can the car turn `deg` in place (+ = left)? Only objects block; tape doesn't."""
        d = np.radians(np.arange(3.0, deg + 1e-6, 3.0)) * sign
        return self._sweep_ok(np.stack([np.zeros_like(d), np.zeros_like(d), d], 1))

    def can_reverse(self, dist=0.12):
        y = -np.linspace(0.03, dist, 4)
        return self._sweep_ok(np.stack([np.zeros(4), y, np.zeros(4)], 1))

    def room(self, sign):
        """Free space on one side: mean free distance of that side's arcs."""
        side = self.ks * sign > 0
        return float(self.free[side].mean())

    # --- the loop
    def run(self):
        a = self.a
        cmd = float(a.speed)
        spin_cmd = float(a.spin_cmd)
        last = time.time()
        rev_until = None
        stuck_since = None
        while True:
            self.sense()
            now = time.time()
            dt, last = now - last, now
            el = now - self.t0
            if self.car and self.car.odo >= a.max_dist:
                return f"drove {a.max_dist:.1f} m"
            if el >= a.max_time:
                return f"{a.max_time:.0f} s limit"
            if a.dry_run and a.frames and self.n >= a.frames:
                return f"{self.n} frames"

            pick = self.choose()
            if rev_until is not None:  # finishing a short reverse
                action = "reverse"
                if self.car.odo <= rev_until or not self.can_reverse():
                    rev_until = None
            elif pick is not None and (self.spin == 0 or pick[1] >= a.min_free + a.spin_exit):
                action = "drive"
                self.spin = 0
            else:
                if self.spin == 0:  # turn toward the course heading (the other way only if objects block it)
                    off = math.degrees(avoid.wrap(self.goal_heading - self.th))
                    if abs(off) > 5:
                        self.spin = 1 if off > 0 else -1
                    else:  # already along the course: toward the side with more room
                        self.spin = 1 if self.room(1) >= self.room(-1) else -1
                off = math.degrees(math.atan2(math.sin(self.th - self.goal_heading), math.cos(self.th - self.goal_heading)))
                if off * self.spin > a.max_off_course:  # turned far enough this way: look the other way
                    self.spin = -self.spin
                if self.can_rotate(self.spin):
                    action = "spin"
                elif self.can_rotate(-self.spin):
                    self.spin = -self.spin
                    action = "spin"
                elif self.car and self.can_reverse():
                    action = "reverse"
                    rev_until = self.car.odo - a.reverse_step
                else:
                    action = "stuck"

            if action == "stuck":
                stuck_since = stuck_since or now
                if now - stuck_since > a.stuck_time:
                    self.stop()
                    return "stuck: can't drive, turn or reverse"
            else:
                stuck_since = None

            if self.car:
                if action == "drive":
                    k, free, _ = pick
                    want = a.cruise * min(1.0, max(0.5, (free - a.min_free) / a.slow_range))
                    if self.car.speed < 0.03:  # standing: it needs a push to start rolling
                        cmd = max(cmd, a.speed_start)
                    cmd += a.speed_gain * (want - self.car.speed) * dt
                    cmd = max(a.speed_min, min(a.speed_max, cmd))
                    self.cmd = cmd
                    left, right = self.wheels(cmd, k)
                    self.car.drive(left, right)
                    self.k_prev = k
                elif action == "spin":
                    rate = math.degrees(self.car_rate()) * self.spin
                    spin_cmd += a.spin_gain * (a.spin_rate - rate) * dt
                    spin_cmd = max(a.spin_min, min(a.spin_max, spin_cmd))
                    self.car.drive(-self.spin * spin_cmd, self.spin * spin_cmd)
                    self.cmd = spin_cmd
                    cmd = float(a.speed)
                elif action == "reverse":
                    self.car.drive(-a.reverse_cmd, -a.reverse_cmd)
                    cmd = float(a.speed)
                else:
                    self.stop()
                if action != "spin":
                    spin_cmd = float(a.spin_cmd)
            self.log(action, pick)

    def wheels(self, cmd, k):
        """L/R commands that drive the arc of curvature k (+ = left) at forward command `cmd`.
        The motors don't turn below ~--deadband (measured: 200/200 and 100/300 don't move
        the car), so each wheel gets the deadband plus its share of the speed: both wheels
        always really drive, and the car follows the arc the planner picked."""
        a = self.a
        eff = max(0.0, cmd - a.deadband)
        out = []
        for w in (eff * (1 - k * a.track / 2), eff * (1 + k * a.track / 2)):
            out.append(0.0 if abs(w) < 1e-6 else math.copysign(a.deadband + abs(w), w))
        return out

    def car_rate(self):
        """Yaw rate (rad/s) from the last ~0.3 s of IMU heading."""
        now = time.time()
        self._hist = [(t, h) for t, h in getattr(self, "_hist", []) if now - t < 0.3] + [(now, self.th)]
        (t0, h0), (t1, h1) = self._hist[0], self._hist[-1]
        return avoid.wrap(h1 - h0) / (t1 - t0) if t1 > t0 else 0.0

    def stop(self):
        if self.car:
            self.car.drive(0, 0)

    # --- logging
    def log(self, action, pick):
        a = self.a
        el = time.time() - self.t0
        if int(el * 4) != self.last_print:
            self.last_print = int(el * 4)
            mv = f"k {pick[0]:+.2f} free {pick[1]:.2f} {pick[2]}" if pick else "no free arc"
            spd = f" speed {self.car.speed:.2f} cmd {self.cmd:.0f}" if self.car else ""
            print(f"{el:5.1f}s pos ({self.X:+.2f},{self.Y:.2f}) hdg {math.degrees(self.th):+6.1f} goal {math.degrees(self.goal_heading):+6.1f} "
                  f"| lanes {self.how:5s} | {action:7s} {mv}{spd} | tilt {self.tilt():.1f}")
        if a.save and int(el * 2) != self.last_save:
            self.last_save = int(el * 2)
            color, red, black, lanes = self._frame
            cam = self.det.debug_image(color, red, black, lanes)[:, : color.shape[1]]
            cv2.imwrite(os.path.join(a.save, f"t{el:05.1f}.jpg"), np.hstack([cam, self.map_image(pick)]))

    def map_image(self, pick, size=480, span=6.0):
        g = self.grid
        px = size / span
        img = np.full((size, size, 3), 30, np.uint8)
        # cells around the car, car near the bottom centre, heading up
        v, u = np.mgrid[0:size, 0:size]
        fx, fy = (u - size / 2) / px, (size * 0.8 - v) / px  # car frame
        c, s = math.cos(self.th), math.sin(self.th)
        wx, wy = self.X + fx * c - fy * s, self.Y + fx * s + fy * c
        i, j, ok = g.ij(np.stack([wx, wy], -1))
        occ = np.zeros((size, size), bool)
        tape = np.zeros((size, size), bool)
        pad = np.zeros((size, size), bool)
        occ[ok] = g.occ[j[ok], i[ok]] >= 2
        tape[ok] = g.tape[j[ok], i[ok]] >= 2
        pad[ok] = g.soft[j[ok], i[ok]]
        img[pad] = (40, 60, 110)
        img[occ] = (0, 140, 255)
        img[tape] = (255, 255, 0)

        def to_px(x, y):
            return int(size / 2 + x * px), int(size * 0.8 - y * px)

        for kk, poses, fr in zip(self.ks, self.arcs, self.free):
            pts = poses[: int(round(fr / 0.05)) + 1]
            for p0, p1 in zip(pts[:-1], pts[1:]):
                cv2.line(img, to_px(*p0[:2]), to_px(*p1[:2]), (90, 90, 90), 1)
        if pick:
            poses = arc_poses(pick[0], pick[1])
            for p0, p1 in zip(poses[:-1], poses[1:]):
                cv2.line(img, to_px(*p0[:2]), to_px(*p1[:2]), (0, 255, 0), 2)
        corners = np.array([[-self.a.half_width, -self.a.half_length], [self.a.half_width, -self.a.half_length],
                            [self.a.half_width, self.a.half_length], [-self.a.half_width, self.a.half_length]])
        cv2.polylines(img, [np.array([to_px(*p) for p in corners], np.int32)], True, (255, 255, 255), 2)
        if self.goal is not None:
            d = self.goal - [self.X, self.Y]
            gx, gy = d[0] * c + d[1] * s, -d[0] * s + d[1] * c
            cv2.drawMarker(img, to_px(gx, gy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
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
    # car
    ap.add_argument("--half-width", type=float, default=0.42, help="car half-width (m): 33 in wide")
    ap.add_argument("--half-length", type=float, default=0.48, help="car half-length (m): 38 in long")
    ap.add_argument("--cam-ahead", type=float, default=0.3, help="camera distance ahead of the turning centre (m)")
    ap.add_argument("--m-per-count", type=float, default=0.0075)
    # map
    ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--map-size", type=float, default=24.0)
    ap.add_argument("--see-range", type=float, default=3.5, help="use depth up to this far (m)")
    ap.add_argument("--min-height", type=float, default=0.08, help="standing above the floor more than this = object (m)")
    ap.add_argument("--max-height", type=float, default=1.2)
    ap.add_argument("--min-cell-points", type=int, default=4, help="depth points in a cell, one frame, to mark it")
    ap.add_argument("--occ-max", type=int, default=8, help="evidence cap: cells clear after ~this many floor views")
    ap.add_argument("--object-pad", type=float, default=0.15,
                    help="objects grow by this: margin + the stool base hidden under the seat from above (m)")
    ap.add_argument("--tape-range", type=float, default=2.5, help="map tape only this close (m): placed better")
    ap.add_argument("--tape-clear-range", type=float, default=1.6,
                    help="tape mapped earlier fades if this close and not seen again (m)")
    ap.add_argument("--tape-max", type=int, default=15, help="tape evidence cap (frames it survives unseen)")
    ap.add_argument("--escape", type=float, default=0.4,
                    help="on a tape line / in an object's margin: arcs may stay on it this far to get off (m)")
    ap.add_argument("--tape-extend", type=float, default=0.8, help="tape assumed to continue this far toward the car (m)")
    # lanes / goal
    ap.add_argument("--lane-width", type=float, default=2.1)
    ap.add_argument("--goal-accept", type=float, default=40.0, help="lane-centre points this far off the course heading are ignored (deg)")
    ap.add_argument("--goal-smooth", type=float, default=0.1, help="how fast the course heading follows the lanes")
    ap.add_argument("--lookahead", type=float, default=1.2, help="lane-centre goal this far ahead (m)")
    # arcs
    ap.add_argument("--arcs", type=int, default=25)
    ap.add_argument("--min-radius", type=float, default=0.6, help="tightest arc (m)")
    ap.add_argument("--arc-length", type=float, default=1.5)
    ap.add_argument("--min-free", type=float, default=0.35, help="an arc must be free this far to drive it (m)")
    ap.add_argument("--spin-exit", type=float, default=0.15, help="while turning to look: need this much more (m)")
    ap.add_argument("--w-free", type=float, default=1.0)
    ap.add_argument("--w-goal", type=float, default=1.5)
    ap.add_argument("--w-smooth", type=float, default=0.3)
    # driving
    ap.add_argument("--cruise", type=float, default=0.15, help="m/s")
    ap.add_argument("--slow-range", type=float, default=0.6, help="full speed once free this much beyond min-free (m)")
    ap.add_argument("--speed", type=int, default=330, help="first forward command")
    ap.add_argument("--speed-start", type=int, default=330, help="command to get rolling from standstill (300/300 rolls)")
    ap.add_argument("--speed-min", type=int, default=270)
    ap.add_argument("--speed-max", type=int, default=480)
    ap.add_argument("--speed-gain", type=float, default=500.0)
    ap.add_argument("--deadband", type=float, default=230.0, help="motor command below which a wheel doesn't turn")
    ap.add_argument("--track", type=float, default=0.70, help="effective wheel track for arc steering (m)")
    ap.add_argument("--spin-rate", type=float, default=15.0, help="deg/s when turning in place")
    ap.add_argument("--spin-cmd", type=float, default=800.0)
    ap.add_argument("--spin-min", type=float, default=550.0)
    ap.add_argument("--spin-max", type=float, default=950.0)
    ap.add_argument("--spin-gain", type=float, default=8.0)
    ap.add_argument("--max-off-course", type=float, default=100.0, help="turning in place: at most this far off the course (deg)")
    ap.add_argument("--reverse-cmd", type=float, default=330.0)
    ap.add_argument("--reverse-step", type=float, default=0.12, help="reverse this far when it can't even turn (m)")
    ap.add_argument("--stuck-time", type=float, default=5.0)
    # run
    ap.add_argument("--max-dist", type=float, default=15.0)
    ap.add_argument("--max-time", type=float, default=180.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--save", help="directory for log.txt and debug images")
    a = ap.parse_args()
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        sys.stdout = Tee(sys.stdout, open(os.path.join(a.save, "log.txt"), "w"))
    f = Follower(a)
    try:
        f.start()
        why = f.run()
        f.stop()
        if f.car:
            f.car.settle()
            f.update_pose()
        print(f"STOP: {why}; pos ({f.X:+.2f},{f.Y:.2f}) heading {math.degrees(f.th):+.1f} deg"
              + (f"; travelled {f.car.odo:.2f} m" if f.car else ""))
    finally:
        f.close()


if __name__ == "__main__":
    main()
