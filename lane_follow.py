#!/usr/bin/env python3
"""Basic reactive lane follower: depth camera + lane mask -> one heading vector per frame.

Every frame:
  1. two clean image masks: lanes (tape) and objects (depth pixels standing above the
     floor, speckle and tiny blobs removed), projected onto the floor
  2. candidate headings every --ray-step degrees across +/- --ray-max, each a straight
     corridor as wide as the car plus --margin each side (--object-margin for objects). Free distance = how far the
     corridor runs before --min-points obstacle points are inside it (a stray depth pixel
     or two doesn't block anything)
  3. heading = the corridor with the longest free distance, mildly preferring straight
     ahead and last frame's choice. Between two lane lines the longest corridor runs along
     the lane, so bends are followed without special cases; objects are avoided the same way.
  4. drive toward it with small L/R differences. Too far off, or too little free: stop and
     search with small forward steps and small turns in place.
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
import stream

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
        self.angles = np.arange(-a.ray_max, a.ray_max + 1e-6, a.ray_step)  # + = right
        self.sin, self.cos = np.sin(np.radians(self.angles)), np.cos(np.radians(self.angles))
        self.half = a.width / 2 + a.margin  # corridor half width for lane points
        self.half_obj = a.width / 2 + a.object_margin  # ...and for object points: keep well clear
        self.lane_counts, self.held = [], 0
        self.prev = 0.0
        self.smooth = 0.0  # heading actually steered toward (deg, + = right)
        self.mode, self.until, self.commit, self.move_hall = "drive", 0.0, 0, None
        self.pivot = 0
        self.car = None
        self.t0 = time.time()
        self.n = 0
        self.last_print = self.last_save = -1
        self.moves = []
        self.streamer = stream.Streamer(a.stream, a.stream_port) if a.stream else None

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
        if self.streamer:
            self.streamer.close()
        if self.car:
            self.car.close()
        self.pipe.stop()

    # --- one frame: obstacle points, corridors, heading
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
        raw = (pts[..., 2] > 0) & (h > a.min_height + 0.006 * pts[..., 2] ** 2) & (h < a.max_height) & (y > 0.1) & (y < a.look_ahead)
        obj = self.object_mask(raw, pts[..., 2])
        self.obj_mask = obj  # at the depth point stride
        # lanes: tape pixels projected onto the floor
        _, lane_mask = self.det.masks(self.color, self.depth)
        lane_mask = self.steady(lane_mask)
        self.bridge_mask = self.bridges(lane_mask)
        self.lane_mask = lane_mask | self.bridge_mask
        lane = self.lane_mask[::sa.STRIDE, ::sa.STRIDE]  # same density as the depth points
        gx, gy = self.det.gx[::sa.STRIDE, ::sa.STRIDE], self.det.gy[::sa.STRIDE, ::sa.STRIDE]
        # obstacle points on the floor (x right, y ahead of the camera)
        self.obj_xy = (x[obj], y[obj])
        self.lane_xy = (gx[lane], gy[lane])
        self.free = np.minimum(self.corridors(x[obj], y[obj], self.half_obj), self.corridors(gx[lane], gy[lane], self.half))
        score = np.minimum(self.free, a.free_cap) - a.w_straight * np.abs(self.angles) / 45 - a.w_keep * np.abs(self.angles - self.prev) / 45
        best = int(np.argmax(score))
        self.heading, self.best_free = float(self.angles[best]), float(self.free[best])
        self.prev = self.heading

    def object_mask(self, raw, z):
        """Clean object mask, the way the lane mask is cleaned: drop speckle, then drop blobs
        smaller than --min-object-m across (size from pixel area and the blob's depth)."""
        a = self.a
        m = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return m.astype(bool)
        zsum = np.bincount(labels.ravel(), weights=np.where(m > 0, z, 0).ravel(), minlength=n)
        area = stats[:, cv2.CC_STAT_AREA]
        across = np.sqrt(area) * sa.STRIDE * (zsum / np.maximum(area, 1)) / self.intr.fx  # metres
        keep = across >= a.min_object_m
        keep[0] = False  # background
        return keep[labels]

    def steady(self, mask):
        """Lane mask flicker guard: a mask that suddenly falls below 30% of its recent size
        is replaced by the last good one, for at most --lane-hold frames (then it's believed:
        the tape really left the view). The raw frame is saved to <save>/drops/ for diagnosis."""
        a = self.a
        n = int(mask.sum())
        recent = np.median(self.lane_counts) if len(self.lane_counts) >= 5 else 0
        if recent > 500 and n < 0.3 * recent and self.held < a.lane_hold:
            self.held += 1
            print(f"  lane mask dropped to {n} px (recent {recent:.0f}): holding the last one ({self.held}/{a.lane_hold})")
            if a.save:
                os.makedirs(os.path.join(a.save, "drops"), exist_ok=True)
                base = os.path.join(a.save, "drops", f"f{self.n:05d}")
                cv2.imwrite(base + "_color.png", self.color)
                np.save(base + "_depth.npy", (self.depth * 1000).astype(np.uint16))
            return self.good_mask
        self.held = 0
        self.good_mask = mask
        self.lane_counts = (self.lane_counts + [n])[-10:]
        return mask

    def bridges(self, mask):
        """Join tape pieces across clear gaps: two piece ends that point at each other (within
        --bridge-deg) and are at most --bridge-m apart on the floor get a line drawn between
        them. Redone every frame, so real tape replaces a bridge as soon as it's seen."""
        a = self.a
        out = np.zeros(mask.shape, np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        ends = []  # (pixel (u, v), floor point, outward unit direction, piece)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < 30:
                continue
            vs, us = np.nonzero(labels == k)
            fl = np.stack([self.det.gx[vs, us], self.det.gy[vs, us]], axis=1)
            ok = np.isfinite(fl).all(axis=1)
            vs, us, fl = vs[ok], us[ok], fl[ok]
            if len(fl) < 30:
                continue
            c = fl - fl.mean(axis=0)
            axis = np.linalg.svd(c, full_matrices=False)[2][0]
            t = c @ axis
            for i in (int(np.argmin(t)), int(np.argmax(t))):
                near = fl[np.linalg.norm(fl - fl[i], axis=1) < 0.3]  # the piece's last 30 cm
                d = fl[i] - near.mean(axis=0)
                if np.linalg.norm(d) < 0.03:
                    continue
                ends.append(((int(us[i]), int(vs[i])), fl[i], d / np.linalg.norm(d), k))
        cos = math.cos(math.radians(a.bridge_deg))
        pairs = []
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                (pi, fi, di, ki), (pj, fj, dj, kj) = ends[i], ends[j]
                gap = fj - fi
                dist = float(np.linalg.norm(gap))
                if ki == kj or dist > a.bridge_m or dist < 1e-3:
                    continue
                g = gap / dist
                if di @ g > cos and dj @ -g > cos:
                    pairs.append((dist, i, j))
        used = set()
        for dist, i, j in sorted(pairs):
            if i in used or j in used:
                continue
            used |= {i, j}
            cv2.line(out, ends[i][0], ends[j][0], 1, 6)
        return out.astype(bool) & ~mask

    def corridors(self, x, y, half):
        """Free distance along each heading's car-wide corridor (m)."""
        a = self.a
        free = np.full(len(self.angles), a.look_ahead)
        if len(x) < a.min_points:
            return free
        along = np.outer(self.sin, x) + np.outer(self.cos, y)  # headings x points
        side = np.abs(np.outer(self.cos, x) - np.outer(self.sin, y))
        inside = (side < half) & (along > a.ignore_near) & (along < a.look_ahead)
        along = np.where(inside, along, np.inf)
        kth = np.partition(along, a.min_points - 1, axis=1)[:, a.min_points - 1]
        return np.minimum(free, kth)

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
            if self.streamer and self.n % a.stream_every == 0:
                self.streamer.send(self.view(cmd))

    def view(self, cmd):
        """2x2 live view: camera + detections, depth, lane mask, top-down corridors with the heading."""
        cam = self.overlay()
        objs = self.full_obj_mask().astype(bool)
        cam[objs] = (0.5 * cam[objs] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        depth = cv2.applyColorMap(cv2.convertScaleAbs(self.depth, alpha=255 / 5.0), cv2.COLORMAP_JET)
        depth[self.depth == 0] = 0
        mask = cv2.cvtColor(self.lane_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        mask[self.bridge_mask] = (0, 140, 255)
        mask[objs] = (0, 0, 255)
        state = f"{self.mode} L{cmd[0]:.0f} R{cmd[1]:.0f}" if self.car else "motors off"
        return stream.grid([(cam, f"camera: lanes yellow/orange, objects red | {state}"), (depth, "depth (0-5 m)"),
                            (mask, "masks: lanes white, bridged gaps orange, objects red"), (self.topdown(), f"top-down: heading {self.heading:+.0f} deg, free {self.best_free:.1f} m")])

    def full_obj_mask(self):
        return cv2.resize(self.obj_mask.astype(np.uint8), self.color.shape[1::-1], interpolation=cv2.INTER_NEAREST)

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
            cv2.imwrite(os.path.join(a.save, "masks", f"t{el:05.1f}_object_mask.png"), self.full_obj_mask() * 255)

    def overlay(self):
        img = self.color.copy()
        img[self.lane_mask] = (0, 255, 255)
        img[self.bridge_mask] = (0, 140, 255)  # bridged gaps
        cv2.putText(img, f"heading {self.heading:+.0f} deg, free {self.best_free:.2f} m", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img

    def topdown(self, size=480):
        a = self.a
        px = (size - 20) / a.look_ahead
        img = np.full((size, size, 3), 30, np.uint8)
        to = lambda x, y: (int(size / 2 + x * px), int(size - 10 - y * px))
        # chosen corridor
        t = math.radians(self.heading)
        d, n = np.array([math.sin(t), math.cos(t)]), np.array([math.cos(t), -math.sin(t)])
        poly = [to(*(e * d + k * self.half * n)) for e, k in ((0, -1), (self.best_free, -1), (self.best_free, 1), (0, 1))]
        cv2.fillPoly(img, [np.array(poly)], (40, 80, 40))
        for ang, fr in zip(self.angles, self.free):
            r = math.radians(ang)
            cv2.line(img, to(0, 0), to(fr * math.sin(r), fr * math.cos(r)), (90, 90, 90), 1)
        for (xs, ys), colour in ((self.lane_xy, (0, 255, 255)), (self.obj_xy, (0, 0, 255))):
            u, v = (size / 2 + xs * px).astype(int), (size - 10 - ys * px).astype(int)
            ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
            img[v[ok], u[ok]] = colour
        cv2.arrowedLine(img, to(0, 0), to(*(self.best_free * d)), (0, 255, 0), 3, tipLength=0.08)
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
    ap.add_argument("--margin", type=float, default=0.03, help="extra clearance each side from lane tape (m)")
    ap.add_argument("--object-margin", type=float, default=0.25, help="extra clearance each side from objects (m)")
    ap.add_argument("--bridge-m", type=float, default=1.0, help="join tape pieces across gaps up to this long (m)")
    ap.add_argument("--bridge-deg", type=float, default=35.0, help="...if their ends point at each other within this angle")
    ap.add_argument("--lane-hold", type=int, default=3, help="frames to keep the last lane mask when it suddenly drops out")
    ap.add_argument("--m-per-count", type=float, default=0.0075)
    # corridors
    ap.add_argument("--look-ahead", type=float, default=3.0, help="ignore anything further than this (m)")
    ap.add_argument("--min-points", type=int, default=3, help="obstacle points inside a corridor before it counts as blocked")
    ap.add_argument("--min-height", type=float, default=0.08)
    ap.add_argument("--min-object-m", type=float, default=0.05, help="object blobs smaller than this across are noise (m)")
    ap.add_argument("--max-height", type=float, default=1.3)
    ap.add_argument("--ray-step", type=float, default=4.0)
    ap.add_argument("--ray-max", type=float, default=32.0)
    ap.add_argument("--ignore-near", type=float, default=0.3, help="corridors start this far from the camera (m)")
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
    ap.add_argument("--stream", metavar="HOST", help="stream a live view (H.264 over RTP/UDP) to this machine")
    ap.add_argument("--stream-port", type=int, default=5000)
    ap.add_argument("--stream-every", type=int, default=2, help="send every Nth frame (camera runs at 30)")
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
