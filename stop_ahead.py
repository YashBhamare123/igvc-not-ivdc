#!/usr/bin/env python3
"""Drive straight ahead; stop when the D455 sees an object in the path, or after a timeout.

1. Calibrate: fit the floor plane from depth (point the car at open, flat floor).
2. Drive: send L<speed> R<speed> to the Arduino every frame.
3. Stop: when the nearest object inside the robot's corridor is closer than
   --stop-dist, or after --timeout seconds, or on any error / Ctrl+C.

Usage:
    python3 stop_ahead.py --dry-run            # detection only, no motor commands
    python3 stop_ahead.py                      # drive (car on blocks first time!)
    python3 stop_ahead.py --show               # also show the colour image + overlay
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 30
STRIDE = 4  # use every 4th pixel: plenty for "is something ahead"


# ---------------------------------------------------------------- geometry


CLUSTER_M = 0.15  # an object = min_points obstacle points within this depth span

# The D4xx left image edge has an invalid-depth band whose border gives spurious
# near readings (seen as phantom obstacles); ignore the outer image columns.
EDGE_LEFT, EDGE_RIGHT = int(WIDTH * 0.10), int(WIDTH * 0.03)


def pixel_rays(intr):
    v, u = np.mgrid[0:HEIGHT:STRIDE, 0:WIDTH:STRIDE]
    return (u - intr.ppx) / intr.fx, (v - intr.ppy) / intr.fy, v


def to_points(depth_m, rx, ry):
    z = depth_m[::STRIDE, ::STRIDE].copy()
    z[:, : EDGE_LEFT // STRIDE] = 0
    z[:, (WIDTH - EDGE_RIGHT) // STRIDE :] = 0
    return np.stack([rx * z, ry * z, z], axis=-1)  # camera frame: x right, y down, z forward


def fit_floor(points, rows, iters=300, tol=0.02):
    """RANSAC plane through the lower half of the image. Returns (normal, offset) with
    height_above_floor(p) = normal @ p + offset, positive above the floor."""
    lower = rows >= rows.max() * 0.5
    z = points[..., 2]
    pts = points[lower & (z > 0.3) & (z < 4.0)]
    if len(pts) < 200:
        return None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        inliers = np.abs((pts - a) @ n) < tol
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.mean() < 0.4:
        return None
    floor = pts[best]
    centroid = floor.mean(axis=0)
    n = np.linalg.svd(floor - centroid)[2][-1]
    if n[1] > 0:  # camera y points down, so "up" has negative y
        n = -n
    return n, -(n @ centroid)


def ground_axes(normal):
    """Unit forward (camera z projected onto the floor) and right vectors."""
    fwd = np.array([0.0, 0.0, 1.0]) - normal[2] * normal
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, normal)  # y-down camera frame: fwd x up = right
    return fwd, right


def obstacle_points(points, normal, offset, fwd, right, args, half_width=None):
    """(forward, lateral) metres of obstacle points inside the corridor ahead."""
    half_width = args.half_width if half_width is None else half_width
    h = points @ normal + offset
    f = points @ fwd
    lat = points @ right
    m = (
        (points[..., 2] > 0)
        & (h > args.min_height)
        & (h < args.max_height)
        & (np.abs(lat) < half_width)
        & (f > 0.2)
        & (f < args.max_range)
    )
    return f[m], lat[m]


def nearest_obstacle(points, normal, offset, fwd, right, args, half_width=None):
    """Forward distance (m) to the nearest object in the corridor, or None if clear."""
    ahead, _ = obstacle_points(points, normal, offset, fwd, right, args, half_width)
    # Nearest distance with at least min_points within CLUSTER_M of it. Floor glare
    # gives a few isolated pixels with bogus short depths; a real object is a cluster.
    k = args.min_points
    if ahead.size < k:
        return None
    ahead = np.sort(ahead)
    dense = np.nonzero(ahead[k - 1 :] - ahead[: ahead.size - k + 1] <= CLUSTER_M)[0]
    return float(ahead[dense[0]]) if dense.size else None


# ---------------------------------------------------------------- arduino


class Arduino:
    def __init__(self, port, baud=115200):
        import serial

        self.ser = serial.Serial(port, baud, timeout=0.05)
        buf, t = "", time.time()
        while "SYSTEM READY" not in buf:  # opening the port resets the Mega
            if time.time() - t > 12:
                raise RuntimeError(f"Arduino on {port} never printed SYSTEM READY")
            buf = (buf + self.ser.read(4096).decode("ascii", "replace"))[-4000:]
        self.boot_log = buf
        self.stop()

    def drive(self, left, right):
        self.ser.write(f"L{int(left)} R{int(right)}\n".encode())
        self.ser.reset_input_buffer()  # we don't parse feedback; don't let it pile up

    def stop(self):
        for _ in range(3):
            self.ser.write(b"STOP\n")
            time.sleep(0.02)

    def close(self):
        try:
            self.stop()
        finally:
            self.ser.close()


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--speed", type=int, default=550, help="L/R command, -1000..1000 (~500 to start rolling)")
    ap.add_argument("--stop-dist", type=float, default=1.0, help="stop when an object is this close (m)")
    ap.add_argument("--timeout", type=float, default=3.0, help="stop after this many seconds of driving")
    ap.add_argument("--half-width", type=float, default=0.40, help="corridor half-width (m): robot half-width + margin")
    ap.add_argument("--min-height", type=float, default=0.08, help="ignore anything lower (floor noise)")
    ap.add_argument("--max-height", type=float, default=1.50, help="ignore anything higher (overhead)")
    ap.add_argument("--max-range", type=float, default=6.0)
    ap.add_argument("--min-points", type=int, default=30, help="points needed to call it an object")
    ap.add_argument("--dry-run", action="store_true", help="detect only; never send motor commands")
    ap.add_argument("--show", action="store_true", help="show colour image with overlay")
    args = ap.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    if args.show:
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    prof = pipe.start(cfg)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    rx, ry, rows = pixel_rays(intr)
    align = rs.align(rs.stream.color) if args.show else None
    if args.show:
        import cv2

    arduino = None
    try:
        # --- calibrate the floor from the median of a few frames (less noise)
        print("calibrating floor: keep the area in front of the car clear...")
        for _ in range(30):  # let auto-exposure settle
            pipe.wait_for_frames()
        # .copy(): holding views into frames starves librealsense's frame pool
        stack = [
            np.asanyarray(pipe.wait_for_frames().get_depth_frame().get_data()).copy()
            for _ in range(15)
        ]
        depth = np.median(np.stack(stack), axis=0) * scale
        plane = fit_floor(to_points(depth, rx, ry), rows)
        if plane is None:
            sys.exit("floor calibration failed: not enough flat floor in the lower half of the image")
        normal, offset = plane
        fwd, right = ground_axes(normal)
        tilt = np.degrees(np.arctan2(-normal[2], -normal[1]))
        print(f"floor: camera height {offset:.3f} m, tilt {tilt:.1f} deg down")

        if not args.dry_run:
            print(f"connecting to Arduino on {args.port}...")
            arduino = Arduino(args.port)

        print(
            f"{'DRY RUN: ' if args.dry_run else ''}driving at {args.speed}; "
            f"stop at {args.stop_dist:.2f} m or after {args.timeout:.1f} s"
        )
        t0 = time.time()
        reason = None
        while reason is None:
            frames = pipe.wait_for_frames(1000)  # raises if the camera stalls -> finally stops
            if align:
                frames = align.process(frames)
            depth = np.asanyarray(frames.get_depth_frame().get_data()) * scale
            nearest = nearest_obstacle(to_points(depth, rx, ry), normal, offset, fwd, right, args)
            elapsed = time.time() - t0

            if nearest is not None and nearest < args.stop_dist:
                reason = f"object at {nearest:.2f} m"
            elif elapsed >= args.timeout:
                reason = f"timeout ({args.timeout:.1f} s)"
            elif arduino:
                arduino.drive(args.speed, args.speed)

            near_txt = "clear" if nearest is None else f"{nearest:.2f} m"
            print(f"{elapsed:5.2f}s  nearest ahead: {near_txt}")

            if args.show:
                img = np.asanyarray(frames.get_color_frame().get_data()).copy()
                col = (0, 0, 255) if reason else (0, 200, 0)
                cv2.putText(img, f"ahead: {near_txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                cv2.imshow("stop_ahead", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    reason = "quit"

        if arduino:
            arduino.stop()
        print(f"STOP: {reason}")
    finally:
        if arduino:
            arduino.close()
        pipe.stop()


if __name__ == "__main__":
    main()
