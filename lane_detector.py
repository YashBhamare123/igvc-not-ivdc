#!/usr/bin/env python3
"""Detect the red and black tape lanes on the floor with the D455 colour camera.

1. Calibrate the floor plane from depth (as in stop_ahead.py).
2. Find tape pixels in the colour image (tape is a few pixels wide there):
     black: thin and darker than the floor right around it (black-hat filter);
            wide dark patches like stains don't pass
     red:   red hue with some saturation (the floor is blue), or thin and redder
            than the floor around it (top-hat on Lab a*) for the faint far tape
3. Keep only pixels that are on the floor: their ray must hit the floor within
   range, and depth must not show something standing there (stools, feet).
4. Project them onto the floor (x = metres right of the camera, y = metres ahead)
   and fit each lane as straight segments (sequential RANSAC), since the tape bends.

Usage:
    python3 lane_detector.py                 # print lanes
    python3 lane_detector.py --save DIR      # also save debug images every second
    python3 lane_detector.py --show          # live window (needs a display)
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs

import stop_ahead as sa

COLOURS = {"red": (0, 0, 255), "black": (255, 255, 0)}  # BGR for drawing


@dataclass
class Segment:
    start: np.ndarray  # (x, y) metres, end nearer the car
    end: np.ndarray
    support: int  # pixels on it

    @property
    def length(self):
        return float(np.hypot(*(self.end - self.start)))

    @property
    def angle_deg(self):
        """Direction relative to straight ahead (+ = leaning right)."""
        d = self.end - self.start
        return math.degrees(math.atan2(d[0], d[1]))


class LaneDetector:
    def __init__(self, intr, normal, offset, args):
        self.args = args
        self.intr = intr
        v, u = np.mgrid[0 : intr.height, 0 : intr.width]
        self.rays = np.stack([(u - intr.ppx) / intr.fx, (v - intr.ppy) / intr.fy, np.ones(u.shape)], axis=-1)
        self.set_plane(normal, offset)

    def set_plane(self, normal, offset):
        """(Re)compute per-pixel floor geometry for this floor plane."""
        args, intr = self.args, self.intr
        self.normal, self.offset = normal, offset
        self.fwd, self.right = sa.ground_axes(normal)
        self.foot = -offset * normal  # floor point straight below the camera

        # Where each pixel's ray meets the floor, and the depth the camera should read there
        rays = self.rays
        down = rays @ -normal  # > 0 if the ray goes toward the floor
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(down > 1e-6, offset / down, np.nan)
        pts = rays * t[..., None]
        self.floor_z = pts[..., 2]  # expected depth (z) of floor at each pixel
        self.gx = (pts - self.foot) @ self.right
        self.gy = (pts - self.foot) @ self.fwd
        self.on_floor_range = np.isfinite(t) & (self.gy > args.near) & (self.gy < args.far) & (np.abs(self.gx) < args.half_span)

        k = args.line_kernel_px | 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))  # square: much faster than round
        self.red_on = args.red_min_sat <= 255 or args.red_contrast < 255

        # Only rows showing floor in range need processing (skips walls, ceiling)
        floor_rows = np.nonzero(self.on_floor_range.any(axis=1))[0]
        self.row0 = max(0, int(floor_rows[0]) - k) if floor_rows.size else 0
        self.floor_z_c = self.floor_z[self.row0 :]
        self.gx_c, self.gy_c = self.gx[self.row0 :], self.gy[self.row0 :]
        self.on_floor_c = self.on_floor_range[self.row0 :]
        # Depth noise grows with distance squared, so the floor tolerance does too
        self.tol_c = args.above_floor_m + args.depth_noise_k * np.nan_to_num(self.floor_z_c, nan=0.0) ** 2

    def masks(self, color, depth_m):
        """Red and black tape masks (full image size)."""
        red_c, black_c = self._masks(color[self.row0 :], depth_m[self.row0 :])
        red = np.zeros(color.shape[:2], bool)
        black = np.zeros(color.shape[:2], bool)
        red[self.row0 :], black[self.row0 :] = red_c, black_c
        return red, black

    def _masks(self, color, depth_m):
        a = self.args
        lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        if self.red_on:
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            h, s, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            red_hue = ((h >= a.red_hue_lo) | (h <= a.red_hue_hi)) & (s >= a.red_min_sat) & (val >= 35)
            redder = cv2.morphologyEx(lab[..., 1], cv2.MORPH_TOPHAT, self.kernel) > a.red_contrast
            red = red_hue | (redder & (h >= 140))  # top-hat alone also fires on yellow/brown
        else:
            red = np.zeros(color.shape[:2], bool)
        dark = cv2.morphologyEx(lab[..., 0], cv2.MORPH_BLACKHAT, self.kernel) > a.black_contrast
        # Floor glare is a speckle of very bright spots whose gaps look like dark lines
        glare = cv2.dilate((lab[..., 0] > a.glare_l).astype(np.uint8), self.kernel).astype(bool)
        dark &= ~glare

        # Floor only: in range, and depth doesn't show something standing there
        tol, floor_z = self.tol_c, self.floor_z_c
        blocked = (depth_m > 0) & (depth_m < floor_z - tol)
        # Thin dark parts (stool rings, legs) often have no depth at all: also drop
        # the area around anything found standing
        blocked = cv2.dilate(blocked.astype(np.uint8), np.ones((a.blocked_grow_px,) * 2, np.uint8)).astype(bool)
        floor = self.on_floor_c & ~blocked
        on_floor_depth = (depth_m > 0) & (np.abs(depth_m - floor_z) < tol)
        if getattr(a, "per_pixel_floor", False):
            # each tape pixel must itself read floor depth (or no depth): tape running between
            # stool legs survives, the legs and rings (above the floor) don't
            floor &= on_floor_depth | (depth_m == 0)
        red = self._floor_blobs(red & floor, on_floor_depth)
        # Tape is much darker than the floor around it (ratio ~0.5); stains (~0.7) and
        # the gaps between glare spots aren't. Relative, so it follows the lighting.
        local_floor = cv2.dilate(lab[..., 0], self.kernel).astype(np.float32)
        black = dark & (lab[..., 0] < a.black_max_ratio * local_floor)
        black = self._floor_blobs(black & floor & ~red, on_floor_depth)
        if getattr(a, "under_object_m", 0) > 0:
            black &= ~self._under_objects(depth_m)
        if getattr(a, "min_piece_m", 0) > 0:
            black = self._line_shapes(black)
        return red, black

    def _under_objects(self, depth_m, step=4, cell=0.05):
        """Pixels whose floor point lies within --under-object-m of something standing
        clearly above the floor (a stool's column or seat): the floor there is under its
        legs, and dark legs and feet there aren't tape."""
        a = self.args
        rays = self.rays[self.row0 :: step, :: step]
        z = depth_m[::step, ::step]
        p = rays * z[..., None]
        h = p @ self.normal + self.offset
        tall = (z > 0) & (h > a.tall_m) & (h < 1.5)
        if not tall.any():
            return np.zeros(depth_m.shape, bool)
        q = p[tall] - self.foot
        tx, ty = q @ self.right, q @ self.fwd
        nx, ny = int(2 * a.half_span / cell), int(a.far / cell)
        grid = np.zeros((ny, nx), np.uint8)
        i = ((tx + a.half_span) / cell).astype(int)
        j = (ty / cell).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        grid[j[ok], i[ok]] = 1
        r = int(round(a.under_object_m / cell))
        grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        gx, gy = self.gx_c, self.gy_c
        i = np.nan_to_num((gx + a.half_span) / cell, nan=-1).astype(int)
        j = np.nan_to_num(gy / cell, nan=-1).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        out = np.zeros(depth_m.shape, bool)
        out[ok] = grid[j[ok], i[ok]].astype(bool)
        return out

    def _line_shapes(self, mask):
        """Keep only blobs shaped like tape: at least --min-piece-m long on the floor and
        thin (length >= --min-thinness x width; length from the outline, so bends and
        curves pass). Casters, stool feet, stains and specks don't."""
        a = self.args
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < 15:
                continue
            comp = (lab[y:y + h, x:x + w] == k).astype(np.uint8)
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            length_px = max(cv2.arcLength(max(cs, key=len), True), 2.0) / 2  # thin shape: perimeter ~ 2 x length
            if length_px / max(area / length_px, 1.0) < a.min_thinness:
                continue
            ys, xs = np.nonzero(comp)
            gx, gy = self.gx_c[y + ys, x + xs], self.gy_c[y + ys, x + xs]
            ok = np.isfinite(gx) & np.isfinite(gy)
            if ok.sum() < 5:
                continue
            keep[k] = np.hypot(gx[ok].max() - gx[ok].min(), gy[ok].max() - gy[ok].min()) >= a.min_piece_m
        return keep[lab] & mask

    def _floor_blobs(self, mask, on_floor_depth):
        """Keep mask pixels whose neighbourhood mostly reads floor-level depth. Tape lies
        on the floor (~90% of its pixels read floor depth); black metal (stool rings,
        legs) returns little depth, so it and the leg shadows next to it are dropped.
        Judged per neighbourhood, not per blob, so tape touching a stool far away
        isn't thrown out with it."""
        m = mask.astype(np.float32)
        win = (self.args.floor_depth_win_px,) * 2
        near_mask = cv2.boxFilter(m, -1, win, normalize=False)
        near_floor = cv2.boxFilter(m * on_floor_depth, -1, win, normalize=False)
        return mask & (near_floor >= self.args.min_floor_depth_frac * np.maximum(near_mask, 1))

    def segments(self, mask, rng):
        """Fit up to --max-segments straight pieces to the mask's floor points."""
        a = self.args
        pts = np.stack([self.gx[mask], self.gy[mask]], axis=1)
        if len(pts) > a.max_points:
            pts = pts[rng.choice(len(pts), a.max_points, replace=False)]
        found = []
        while len(pts) >= a.min_support and len(found) < a.max_segments:
            # RANSAC, all candidate lines at once: random point pairs -> line normals ->
            # inlier counts for every candidate in one (iters x points) step
            ij = rng.integers(0, len(pts), size=(a.ransac_iters, 2))
            p, q = pts[ij[:, 0]], pts[ij[:, 1]]
            d = q - p
            length = np.hypot(d[:, 0], d[:, 1])
            good = length >= 0.1
            if not good.any():
                break
            p, d, length = p[good], d[good], length[good]
            normal = np.stack([-d[:, 1], d[:, 0]], 1) / length[:, None]
            dist = np.abs((pts[None, :, :] - p[:, None, :]) @ normal[:, :, None])[..., 0]
            counts = (dist < a.inlier_m).sum(axis=1)
            best = dist[int(np.argmax(counts))] < a.inlier_m
            if best.sum() < a.min_support:
                break
            inl = pts[best]
            centre = inl.mean(axis=0)
            # principal direction of a 2x2 covariance (closed form, no SVD)
            cx, cy = (inl - centre).T
            sxx, syy, sxy = cx @ cx, cy @ cy, cx @ cy
            ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
            direction = np.array([math.cos(ang), math.sin(ang)])
            along = (inl - centre) @ direction
            # Keep the longest run without gaps (one tape piece, not two collinear ones)
            order = np.sort(along)
            breaks = np.nonzero(np.diff(order) > a.max_gap_m)[0]
            runs = np.split(order, breaks + 1)
            run = max(runs, key=lambda r: r[-1] - r[0])
            in_run = best & False
            idx = np.nonzero(best)[0]
            keep = (along >= run[0]) & (along <= run[-1])
            in_run[idx[keep]] = True
            pts_run = pts[in_run]
            if run[-1] - run[0] >= a.min_length_m and len(pts_run) >= a.min_support:
                e0, e1 = centre + run[0] * direction, centre + run[-1] * direction
                if e0[1] > e1[1]:
                    e0, e1 = e1, e0
                found.append(Segment(e0, e1, len(pts_run)))
                pts = pts[~in_run]
            else:
                pts = pts[~best]  # a scatter, not a tape piece: drop and keep looking
        return sorted(found, key=lambda sg: sg.start[1])

    def ground_to_pixel(self, x, y):
        p = self.foot + x * self.right + y * self.fwd
        return int(self.intr.fx * p[0] / p[2] + self.intr.ppx), int(self.intr.fy * p[1] / p[2] + self.intr.ppy)

    def debug_image(self, color, red, black, lanes):
        a = self.args
        cam = color.copy()
        cam[black] = (255, 255, 0)
        cam[red] = (0, 0, 255)
        # top-down plot: 1 px = 1 cm, car at the bottom centre
        sc = 100
        W, H = int(2 * a.half_span * sc), int(a.far * sc)
        top = np.full((H, W, 3), 40, np.uint8)
        to_top = lambda x, y: (int((x + a.half_span) * sc), int(H - y * sc))
        for y in range(1, int(a.far) + 1):
            cv2.line(top, (0, H - y * sc), (W, H - y * sc), (80, 80, 80), 1)
            cv2.putText(top, f"{y} m", (4, H - y * sc - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.line(top, (W // 2, 0), (W // 2, H), (80, 80, 80), 1)
        for name, mask in (("red", red), ("black", black)):
            col = COLOURS[name]
            for x, y in zip(self.gx[mask][::3], self.gy[mask][::3]):
                cv2.circle(top, to_top(x, y), 1, col, -1)
            for sg in lanes[name]:
                cv2.line(top, to_top(*sg.start), to_top(*sg.end), (0, 255, 0), 3)
                cv2.line(cam, self.ground_to_pixel(*sg.start), self.ground_to_pixel(*sg.end), (0, 255, 0), 2)
        cv2.circle(top, (W // 2, H), 8, (255, 255, 255), -1)
        top = cv2.resize(top, (int(W * color.shape[0] / H), color.shape[0]))
        return np.hstack([cam, top])


def describe(name, segs):
    if not segs:
        return f"{name}: not found"
    parts = [f"({s.start[0]:+.2f},{s.start[1]:.2f})->({s.end[0]:+.2f},{s.end[1]:.2f}) {s.angle_deg:+.0f}deg"
             for s in segs]
    return f"{name}: " + " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--near", type=float, default=0.4, help="ignore floor closer than this (m)")
    ap.add_argument("--far", type=float, default=5.0, help="ignore floor farther than this (m)")
    ap.add_argument("--half-span", type=float, default=2.5, help="ignore floor further to the side (m)")
    ap.add_argument("--line-kernel-px", type=int, default=11, help="tape must be narrower than this in the image")
    ap.add_argument("--black-contrast", type=float, default=25, help="how much darker than the floor around it (Lab L)")
    ap.add_argument("--red-hue-lo", type=int, default=160, help="red hue range (OpenCV 0-179 wraps)")
    ap.add_argument("--red-hue-hi", type=int, default=8)
    ap.add_argument("--red-min-sat", type=int, default=70, help="floor saturation stays below ~65")
    ap.add_argument("--red-contrast", type=float, default=8, help="how much redder than the floor around it (Lab a*)")
    ap.add_argument("--above-floor-m", type=float, default=0.03, help="depth this much short of the floor = an object")
    ap.add_argument("--black-max-ratio", type=float, default=0.62, help="tape brightness / floor around it (tape ~0.5, stains ~0.7)")
    ap.add_argument("--floor-depth-win-px", type=int, default=31, help="neighbourhood for the floor-depth check")
    ap.add_argument("--min-floor-depth-frac", type=float, default=0.6, help="blob pixels that must read floor depth")
    ap.add_argument("--depth-noise-k", type=float, default=0.006, help="extra tolerance per m^2 of distance")
    ap.add_argument("--blocked-grow-px", type=int, default=15, help="also ignore this far around standing objects")
    ap.add_argument("--glare-l", type=float, default=225, help="Lab L above this = glare (floor is ~140)")
    ap.add_argument("--smooth", type=float, default=0.5, help="weight of the new frame when averaging colour (noise)")
    ap.add_argument("--min-support", type=int, default=25, help="pixels needed for a segment")
    ap.add_argument("--min-length-m", type=float, default=0.3, help="segments shorter than this are ignored")
    ap.add_argument("--max-gap-m", type=float, default=0.3, help="split a segment at gaps longer than this")
    ap.add_argument("--max-segments", type=int, default=4, help="per lane")
    ap.add_argument("--inlier-m", type=float, default=0.03)
    ap.add_argument("--ransac-iters", type=int, default=60)
    ap.add_argument("--max-points", type=int, default=1500, help="subsample tape pixels to this many before fitting")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until Ctrl+C)")
    ap.add_argument("--save", help="directory for debug images (one per second)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, sa.WIDTH, sa.HEIGHT, rs.format.z16, sa.FPS)
    cfg.enable_stream(rs.stream.color, sa.WIDTH, sa.HEIGHT, rs.format.bgr8, sa.FPS)
    prof = pipe.start(cfg)
    try:
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        align = rs.align(rs.stream.color)
        rx, ry, rows = sa.pixel_rays(intr)

        for _ in range(30):
            pipe.wait_for_frames()
        stack = []
        for _ in range(15):
            f = align.process(pipe.wait_for_frames())
            stack.append(np.asanyarray(f.get_depth_frame().get_data()).copy())
        plane = sa.fit_floor(sa.to_points(np.median(np.stack(stack), axis=0) * scale, rx, ry), rows)
        if plane is None:
            raise SystemExit("floor calibration failed: not enough flat floor in view")
        normal, offset = plane
        print(f"floor: camera height {offset:.3f} m, tilt {math.degrees(math.atan2(-normal[2], -normal[1])):.1f} deg down")

        det = LaneDetector(intr, normal, offset, args)
        rng = np.random.default_rng(0)
        if args.save:
            os.makedirs(args.save, exist_ok=True)
        n, t0, last_save, avg = 0, time.time(), -1, None
        while True:
            f = align.process(pipe.wait_for_frames(1000))
            raw = np.asanyarray(f.get_color_frame().get_data()).astype(np.float32)
            avg = raw if avg is None else args.smooth * raw + (1 - args.smooth) * avg
            color = avg.astype(np.uint8)
            depth = np.asanyarray(f.get_depth_frame().get_data()) * scale
            red, black = det.masks(color, depth)
            lanes = {"red": det.segments(red, rng), "black": det.segments(black, rng)}
            n += 1
            el = time.time() - t0
            if args.save and int(el) != last_save:
                last_save = int(el)
                cv2.imwrite(os.path.join(args.save, f"lanes_{last_save:03d}.jpg"), det.debug_image(color, red, black, lanes))
            print(f"{el:5.1f}s {n / el if el else 0:4.1f}fps  {describe('red', lanes['red'])}  |  {describe('black', lanes['black'])}")
            if args.show:
                cv2.imshow("lanes", det.debug_image(color, red, black, lanes))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.frames and n >= args.frames:
                break
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
