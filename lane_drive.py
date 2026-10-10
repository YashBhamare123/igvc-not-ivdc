#!/usr/bin/env python3
"""Drive between the tape lanes, stopping for obstacles.

Each frame: detect the red and black lanes (lane_detector.py) and anything in the car's
path (depth). Steer toward the lane centre --lookahead metres ahead: midway between the
two lanes, or half a lane width from the one that's visible (width learned whenever both
are seen). With no lane in view, hold the heading (BNO085 on the Arduino). Stop for
anything in the path closer than --stop-dist, after --max-dist metres or --max-time
seconds, on any error, or on Ctrl+C.

Usage:
    python3 lane_drive.py --dry-run          # detect + show steering, no motor commands
    python3 lane_drive.py --save DIR         # drive, saving debug images twice a second
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


def lane_point(segments, dist):
    """Point on a lane about `dist` metres from the camera, and the lane's direction there
    (unit vector pointing away from the car). x right, y ahead."""
    if not segments:
        return None
    best = None
    for sg in segments:
        d = sg.end - sg.start
        length = float(np.hypot(*d))
        if length < 1e-3:
            continue
        u = d / length
        # where the segment crosses the circle of radius `dist` (or its closest end)
        b = float(sg.start @ u)
        c = float(sg.start @ sg.start) - dist * dist
        disc = b * b - c
        if disc >= 0:
            t = min(max(-b + math.sqrt(disc), 0.0), length)
        else:
            t = 0.0 if abs(np.hypot(*sg.start) - dist) < abs(np.hypot(*sg.end) - dist) else length
        p = sg.start + t * u
        miss = abs(float(np.hypot(*p)) - dist)
        if best is None or miss < best[0]:
            best = (miss, p, u)
    return None if best is None else (best[1], best[2])


class LaneCentre:
    """Lane centre about `lookahead` metres ahead. Each lane colour has a fixed side
    (--left-lane), and the centre is half a lane width inward from a lane, measured
    across the lane's own direction, so it follows bends."""

    def __init__(self, args):
        self.args = args
        self.width = args.lane_width
        self.sides = {args.left_lane: "left", ("black" if args.left_lane == "red" else "red"): "right"}
        self.target = None

    def update(self, lanes):
        a = self.args
        estimates, points = [], {}
        for colour, side in self.sides.items():
            hit = lane_point(lanes[colour], a.lookahead)
            if hit is None:
                continue
            p, u = hit
            inward = np.array([u[1], -u[0]]) if side == "left" else np.array([-u[1], u[0]])  # toward the lane centre
            points[side] = (p, inward)
            estimates.append(p + inward * self.width / 2)
        if not estimates:
            self.target = None
            return None, "none"
        if len(points) == 2:
            (pl, nl), (pr, nr) = points["left"], points["right"]
            # only measure the width where the lanes run parallel: across a bend it comes out short
            parallel = abs(float(nl @ -nr)) > math.cos(math.radians(20))
            w = float((pr - pl) @ nl)  # right lane's distance from the left lane, across it
            if parallel and 1.0 < w < 4.0:
                self.width += a.width_smooth * (w - self.width)
        centre = np.mean(estimates, axis=0)
        # Smooth the target so steering doesn't jitter with detection noise
        self.target = centre if self.target is None else self.target + a.target_smooth * (centre - self.target)
        return self.target, "both" if len(points) == 2 else next(iter(points))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--stop-dist", type=float, default=0.6, help="stop for anything in the path this close (m)")
    ap.add_argument("--max-dist", type=float, default=3.0, help="stop after driving this far (m)")
    ap.add_argument("--max-time", type=float, default=8.0, help="stop after this long (s)")
    ap.add_argument("--cruise", type=float, default=0.4, help="driving speed (m/s)")
    ap.add_argument("--speed", type=int, default=700, help="starting forward L/R command")
    ap.add_argument("--speed-min", type=int, default=450)
    ap.add_argument("--speed-max", type=int, default=850)
    ap.add_argument("--speed-gain", type=float, default=400.0, help="command change per (m/s error x s)")
    ap.add_argument("--lookahead", type=float, default=1.2, help="steer toward the lane centre this far ahead (m)")
    ap.add_argument("--left-lane", choices=("red", "black"), default="red", help="which tape is on the left")
    ap.add_argument("--lane-width", type=float, default=2.1, help="starting guess (m); updated when both lanes are seen")
    ap.add_argument("--width-smooth", type=float, default=0.1)
    ap.add_argument("--target-smooth", type=float, default=0.4)
    ap.add_argument("--steer-gain", type=float, default=10.0, help="L/R difference per degree off the target")
    ap.add_argument("--steer-max", type=float, default=300.0)
    ap.add_argument("--m-per-count", type=float, default=0.0075, help="metres per hall count (measured)")
    ap.add_argument("--dry-run", action="store_true", help="detect and print steering; never send motor commands")
    ap.add_argument("--frames", type=int, default=0, help="dry run: stop after N frames")
    ap.add_argument("--save", help="directory for debug images (twice a second)")
    args = ap.parse_args()

    # lane detector and obstacle check settings (the defaults tuned in lane_detector.py / stop_ahead.py)
    lane_args = types.SimpleNamespace(
        near=0.4, far=5.0, half_span=2.5, line_kernel_px=11, black_contrast=25, red_hue_lo=160, red_hue_hi=8,
        red_min_sat=70, red_contrast=8, above_floor_m=0.03, depth_noise_k=0.006, blocked_grow_px=15, glare_l=225,
        black_max_ratio=0.62, floor_depth_win_px=31, min_floor_depth_frac=0.6, min_support=25, min_length_m=0.3,
        max_gap_m=0.3, max_segments=4, inlier_m=0.03, ransac_iters=60, max_points=1500)
    obstacle_args = types.SimpleNamespace(half_width=0.40, min_height=0.08, max_height=1.5, max_range=6.0, min_points=30)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, sa.WIDTH, sa.HEIGHT, rs.format.z16, sa.FPS)
    cfg.enable_stream(rs.stream.color, sa.WIDTH, sa.HEIGHT, rs.format.bgr8, sa.FPS)
    prof = pipe.start(cfg)
    car = None
    try:
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        align = rs.align(rs.stream.color)
        rx, ry, rows = sa.pixel_rays(intr)
        for _ in range(30):
            pipe.wait_for_frames()
        stack = [np.asanyarray(align.process(pipe.wait_for_frames()).get_depth_frame().get_data()).copy() for _ in range(15)]
        plane = sa.fit_floor(sa.to_points(np.median(np.stack(stack), axis=0) * scale, rx, ry), rows)
        if plane is None:
            sys.exit("floor calibration failed: not enough flat floor in view")
        normal, offset = plane
        fwd, right = sa.ground_axes(normal)
        print(f"floor: camera height {offset:.3f} m, tilt {math.degrees(math.atan2(-normal[2], -normal[1])):.1f} deg down")
        det = ld.LaneDetector(intr, normal, offset, lane_args)
        centre = LaneCentre(args)
        rng = np.random.default_rng(0)
        if args.save:
            os.makedirs(args.save, exist_ok=True)

        def sense():
            f = align.process(pipe.wait_for_frames(1000))
            color = np.asanyarray(f.get_color_frame().get_data())
            depth = np.asanyarray(f.get_depth_frame().get_data()) * scale
            near = sa.nearest_obstacle(sa.to_points(depth, rx, ry), normal, offset, fwd, right, obstacle_args)
            red, black = det.masks(color, depth)
            return color, red, black, {"red": det.segments(red, rng), "black": det.segments(black, rng)}, near

        color, red, black, lanes, near = sense()
        print(f"start: path {'clear' if near is None else f'{near:.2f} m'} | {ld.describe('red', lanes['red'])} | "
              f"{ld.describe('black', lanes['black'])}")
        if near is not None and near < args.stop_dist:
            sys.exit(f"NOT DRIVING: object {near:.2f} m ahead in the path")

        if not args.dry_run:
            car = avoid.Car(args.port, args.m_per_count)
            car.set_origin()
        hold = 0.0  # heading to keep when no lane is visible
        cmd = float(args.speed)
        t0 = last = time.time()
        hits, n, last_save, why = 0, 0, -1, None
        while why is None:
            color, red, black, lanes, near = sense()
            now = time.time()
            dt, last = now - last, now
            el = now - t0
            n += 1
            hits = hits + 1 if (near is not None and near < args.stop_dist) else 0
            target, how = centre.update(lanes)

            if hits >= 2:
                why = f"object {near:.2f} m in the path"
            elif car and car.ahead >= args.max_dist:
                why = f"drove {args.max_dist:.1f} m"
            elif el >= args.max_time:
                why = f"{args.max_time:.0f} s limit"
            elif args.dry_run and args.frames and n >= args.frames:
                why = f"{n} frames"

            if target is not None:
                err_deg = math.degrees(math.atan2(target[0], target[1]))  # + = centre is to the right
                steer = -args.steer_gain * err_deg  # + steer = turn left
                if car:
                    hold = car.heading() - math.radians(err_deg)  # heading that points at the centre
            elif car:
                err_deg = math.degrees(avoid.wrap(car.heading() - hold))
                steer = -args.steer_gain * err_deg
            else:
                err_deg, steer = 0.0, 0.0
            steer = max(-args.steer_max, min(args.steer_max, steer))

            if car and why is None:
                car.poll()
                cmd += args.speed_gain * (args.cruise - car.speed) * dt
                cmd = max(args.speed_min, min(args.speed_max, cmd))
                car.drive(cmd - steer, cmd + steer)

            pos = f"ahead {car.ahead:4.2f} left {car.left:+.2f} heading {math.degrees(car.heading()):+5.1f} speed {car.speed:.2f} | " if car else ""
            tgt = "--" if target is None else f"({target[0]:+.2f},{target[1]:.2f})"
            near_txt = "clear" if near is None else f"{near:.2f}"
            print(f"{el:4.1f}s {n / el if el else 0:4.1f}fps {pos}lanes {how:5s} centre {tgt} -> steer {steer:+4.0f} "
                  f"(width {centre.width:.2f}) path {near_txt}")
            if args.save and int(el * 2) != last_save:
                last_save = int(el * 2)
                cv2.imwrite(os.path.join(args.save, f"t{el:04.1f}.jpg"), det.debug_image(color, red, black, lanes))

        if car:
            car.settle()
            print(f"STOP: {why}; final ahead {car.ahead:.2f} m, left {car.left:+.2f} m, "
                  f"heading {math.degrees(car.heading()):+.1f} deg")
        else:
            print(f"done: {why}")
    finally:
        if car:
            car.close()
        pipe.stop()


if __name__ == "__main__":
    main()
