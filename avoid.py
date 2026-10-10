#!/usr/bin/env python3
"""Drive ahead and steer smoothly around the first object in the way, then rejoin the
original straight line, without stopping.

The car keeps rolling at --cruise m/s the whole time. When an object comes within
--plan-dist, it plans an S-shaped path from the depth image: ease sideways (cosine ramp
over --ramp metres) far enough that the object is --margin clear of the car, hold that
offset until past the object, ease back onto the original line, and carry on straight
for --after-dist metres. A pure-pursuit controller follows the path with small,
continuous left/right speed differences.

Position is dead-reckoned from the wheel hall counts (distance) and the BNO085 on the
Arduino (heading). Anything in the car's path closer than --stop-dist, any error, stale
Arduino feedback or Ctrl+C sends STOP.

Usage:
    python3 avoid.py --dry-run     # calibrate + show the plan, no motor commands
    python3 avoid.py
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time

import numpy as np
import pyrealsense2 as rs

import stop_ahead as sa

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stop_logs")
FEEDBACK = re.compile(r"FEEDBACK: L(-?\d+) R(-?\d+) RPM \| HALL L(-?\d+) R(-?\d+) \| IMU (.*)")
MAX_COUNTS_PER_FB = 60  # 50 ms feedback: ~9 m/s at 7.5 mm/count, far beyond the car


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Car(sa.Arduino):
    """Arduino link that also tracks hall counts and BNO085 yaw from FEEDBACK lines."""

    def __init__(self, port, m_per_count):
        super().__init__(port)
        if "BNO08x Found" not in self.boot_log:
            raise RuntimeError("Arduino IMU (BNO08x) did not start; heading is needed to turn back")
        self.m_per_count = m_per_count
        self.hall = None
        self.yaw = None
        self.last_fb = 0.0
        self._buf = ""
        # Dead reckoning in the frame set by set_origin(): metres left / ahead of the start
        self.yaw_ref = None
        self.left = self.ahead = self.odo = 0.0
        self.speed = 0.0
        self._prev_hall = None
        self._odo_hist = []
        # The BNO085 reports its default (1, 0, 0, 0) until the first rotation vector arrives
        t = time.time()
        while self.yaw is None or self._quat == (1.0, 0.0, 0.0, 0.0):
            if time.time() - t > 3:
                raise RuntimeError("no IMU heading in the Arduino FEEDBACK")
            self.poll()
            time.sleep(0.02)

    def poll(self):
        # Only what's already arrived: read(n) would block for the port timeout (50 ms)
        waiting = self.ser.in_waiting
        if waiting:
            self._buf += self.ser.read(waiting).decode("ascii", "replace")
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            m = FEEDBACK.match(line.strip())
            if not m:
                continue
            try:
                imu = [float(v) for v in m.group(5).split()]
                w, x, y, z = imu[:4]
            except ValueError:
                continue  # line garbled on the wire
            self._quat = (w, x, y, z)
            self.hall = (int(m.group(3)), int(m.group(4)))
            self.yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            self.last_fb = time.time()
            self._odometry()
        if self.yaw is not None and time.time() - self.last_fb > 0.5:
            raise RuntimeError("Arduino feedback stopped")

    def _odometry(self):
        if self._prev_hall is not None:
            dl, dr = self.hall[0] - self._prev_hall[0], self.hall[1] - self._prev_hall[1]
            # the first counts after the Arduino boots jump from 0 to the controller's running total
            if max(abs(dl), abs(dr)) > MAX_COUNTS_PER_FB:
                dl = dr = 0
            d = 0.5 * (dl + dr) * self.m_per_count
            self.odo += d
            if self.yaw_ref is not None:
                h = self.heading()  # + = turned left
                self.left += d * math.sin(h)
                self.ahead += d * math.cos(h)
        self._prev_hall = self.hall
        now = time.time()
        self._odo_hist.append((now, self.odo))
        while len(self._odo_hist) > 2 and now - self._odo_hist[0][0] > 0.3:
            self._odo_hist.pop(0)
        (ta, oa), (tb, ob) = self._odo_hist[0], self._odo_hist[-1]
        self.speed = abs(ob - oa) / (tb - ta) if tb > ta else 0.0

    def set_origin(self):
        """Current position and heading become (0, 0), straight ahead."""
        self.yaw_ref = self.yaw
        self.left = self.ahead = self.odo = 0.0
        self._odo_hist.clear()

    def heading(self):
        """Radians turned since set_origin(); + = left (the BNO085 yaw is counter-clockwise)."""
        return wrap(self.yaw - self.yaw_ref)

    def drive(self, left, right):
        self.ser.write(f"L{int(left)} R{int(right)}\n".encode())
        self.poll()

    def distance(self, start_hall):
        """Metres driven since start_hall (mean of both wheels)."""
        dl = self.hall[0] - start_hall[0]
        dr = self.hall[1] - start_hall[1]
        return 0.5 * (dl + dr) * self.m_per_count

    def settle(self, max_wait=2.0):
        """STOP and wait until the wheels stop turning."""
        self.stop()
        t, last, still_since = time.time(), None, time.time()
        while time.time() - t < max_wait:
            self.ser.write(b"STOP\n")
            time.sleep(0.05)
            self.poll()
            if self.hall != last:
                last, still_since = self.hall, time.time()
            elif time.time() - still_since > 0.3:
                return


class Eye:
    """D455 depth + floor calibration; answers 'what is ahead of the car'."""

    def __init__(self, args):
        self.args = args
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, sa.WIDTH, sa.HEIGHT, rs.format.z16, sa.FPS)
        cfg.enable_stream(rs.stream.color, sa.WIDTH, sa.HEIGHT, rs.format.bgr8, sa.FPS)  # for stop logs
        self.frames = None
        prof = self.pipe.start(cfg)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        intr = prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.rx, self.ry, rows = sa.pixel_rays(intr)

        for _ in range(30):  # let auto-exposure settle
            self.pipe.wait_for_frames()
        stack = [
            np.asanyarray(self.pipe.wait_for_frames().get_depth_frame().get_data()).copy()
            for _ in range(15)
        ]
        depth = np.median(np.stack(stack), axis=0) * self.scale
        plane = sa.fit_floor(sa.to_points(depth, self.rx, self.ry), rows)
        if plane is None:
            self.pipe.stop()
            sys.exit("floor calibration failed: not enough flat floor in the lower half of the image")
        self.normal, self.offset = plane
        self.fwd, self.right = sa.ground_axes(self.normal)
        tilt = math.degrees(math.atan2(-self.normal[2], -self.normal[1]))
        print(f"floor: camera height {self.offset:.3f} m, tilt {tilt:.1f} deg down")

    def _points(self):
        frames = self.pipe.wait_for_frames(1000)
        # Frames queue up whenever we stop reading (e.g. while settling); use the newest
        while True:
            newer = self.pipe.poll_for_frames()
            if not newer:
                break
            frames = newer
        self.frames = frames
        depth = np.asanyarray(frames.get_depth_frame().get_data()) * self.scale
        return sa.to_points(depth, self.rx, self.ry)

    def save_last(self, label):
        """Save the colour + depth view of the last frame used, for checking stops later."""
        import cv2

        if self.frames is None:
            return
        os.makedirs(LOG_DIR, exist_ok=True)
        color = np.asanyarray(self.frames.get_color_frame().get_data()).copy()
        depth = np.asanyarray(self.frames.get_depth_frame().get_data()) * self.scale
        dv = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth, 0, 6), alpha=255 / 6), cv2.COLORMAP_JET)
        dv[depth == 0] = 0
        # Mark the pixels that counted as obstacles in the robot's path (white dots)
        pts = sa.to_points(depth, self.rx, self.ry)
        h, f, lat = pts @ self.normal + self.offset, pts @ self.fwd, pts @ self.right
        a = self.args
        hit = (
            (pts[..., 2] > 0) & (h > a.min_height) & (h < a.max_height)
            & (np.abs(lat) < a.half_width) & (f > 0.2) & (f < a.max_range)
        )
        for v, u in zip(*np.nonzero(hit)):
            cv2.circle(dv, (int(u * sa.STRIDE), int(v * sa.STRIDE)), 3, (255, 255, 255), -1)
        cv2.putText(color, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        path = os.path.join(LOG_DIR, time.strftime("%Y%m%d-%H%M%S-") + label.split()[0] + ".jpg")
        cv2.imwrite(path, np.hstack([color, dv]))
        print(f"  saved {path}")

    def nearest(self, half_width=None):
        return sa.nearest_obstacle(
            self._points(), self.normal, self.offset, self.fwd, self.right, self.args, half_width
        )

    def detour(self, dist, wide, depth_window=0.6, scan=2.5):
        """How far the car centre must move sideways to pass the object at `dist` with `wide`
        clearance from its centre line: returns (metres left, metres right)."""
        f, lat = sa.obstacle_points(
            self._points(), self.normal, self.offset, self.fwd, self.right, self.args, scan
        )
        lat = lat[f < dist + depth_window]  # the object's front, not things behind it
        if lat.size < self.args.min_points:
            return None
        # 1st/99th percentile: a few glare pixels can't widen the object
        return -np.percentile(lat, 1) + wide, np.percentile(lat, 99) + wide

    def close(self):
        self.pipe.stop()


# ---------------------------------------------------------------- path following


class Detour:
    """Target 'left' offset (m) as a function of distance ahead: 0 -> offset -> 0 with
    cosine ramps, so the path's heading and curvature change smoothly."""

    def __init__(self, start, offset, ramp, hold_until):
        self.start, self.offset, self.ramp = start, offset, ramp
        self.back = max(hold_until, start + ramp)  # where the ramp back to the line begins
        self.end = self.back + ramp

    def left_at(self, s):
        if s <= self.start or s >= self.end:
            return 0.0
        if s < self.start + self.ramp:
            return self.offset * (1 - math.cos(math.pi * (s - self.start) / self.ramp)) / 2
        if s <= self.back:
            return self.offset
        return self.offset * (1 + math.cos(math.pi * (s - self.back) / self.ramp)) / 2


def steer_for(car, args, path):
    """Pure pursuit: L/R difference that turns the car toward the path point --lookahead ahead."""
    s = car.ahead + args.lookahead
    target_left = path.left_at(s) if path else 0.0
    desired = math.atan2(target_left - car.left, args.lookahead)  # + = left
    steer = args.steer_gain * math.degrees(wrap(desired - car.heading()))
    return max(-args.steer_max, min(args.steer_max, steer))


def run(car, eye, args):
    wide = args.half_width + args.margin
    path = None
    cmd = float(args.speed)  # forward command, adjusted to hold --cruise m/s
    t0 = last = time.time()
    hits = 0
    last_print = -1
    while True:
        near = eye.nearest()
        car.poll()
        now = time.time()
        dt, last = now - last, now

        # Emergency stop: something in the car's current path, two frames in a row
        hits = hits + 1 if (near is not None and near < args.stop_dist) else 0
        if hits >= 2:
            eye.save_last(f"object {near:.2f}m at left {car.left:+.2f} ahead {car.ahead:.2f}")
            return f"STOPPED: object {near:.2f} m in the path"

        # Plan the detour once, as soon as an object is close enough
        if path is None and near is not None and near < args.plan_dist:
            plan = eye.detour(near, wide)
            if plan is None:
                return "STOPPED: lost sight of the object while planning"
            side = 1 if plan[0] <= plan[1] else -1  # +1 = pass on the left
            obj_front = car.ahead + args.cam_ahead + near
            # Finish easing sideways a little before the object's front
            ramp = max(0.8, min(args.ramp, obj_front - car.ahead - args.ramp_clear))
            path = Detour(car.ahead, side * min(plan), ramp,
                          hold_until=obj_front + args.object_depth + args.pass_extra)
            print(f"{now - t0:5.1f}s object {near:.2f} m ahead -> pass on the "
                  f"{'left' if side == 1 else 'right'}, {abs(path.offset):.2f} m sideways; "
                  f"curve {path.start:.2f}-{path.start + ramp:.2f} m, back to the line "
                  f"{path.back:.2f}-{path.end:.2f} m")

        if path is not None and car.ahead >= path.end + args.after_dist:
            return "done"
        if path is None and now - t0 > args.approach_time:
            return f"nothing in the way for {args.approach_time:.0f} s"

        # Speed: nudge the forward command to hold --cruise m/s
        cmd += args.speed_gain * (args.cruise - car.speed) * dt
        cmd = max(args.speed_min, min(args.speed_max, cmd))
        steer = steer_for(car, args, path)
        car.drive(cmd - steer, cmd + steer)

        if int((now - t0) * 2) != last_print:  # log twice a second
            last_print = int((now - t0) * 2)
            target = path.left_at(car.ahead) if path else 0.0
            print(f"{now - t0:5.1f}s ahead {car.ahead:5.2f} m  left {car.left:+.2f} "
                  f"(path {target:+.2f})  heading {math.degrees(car.heading()):+5.1f}  "
                  f"speed {car.speed:.2f} m/s  cmd {cmd:4.0f} steer {steer:+4.0f}"
                  + ("" if near is None else f"  nearest {near:.2f} m"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--cruise", type=float, default=0.5, help="driving speed (m/s)")
    ap.add_argument("--speed", type=int, default=700, help="starting forward L/R command")
    ap.add_argument("--speed-min", type=int, default=450)
    ap.add_argument("--speed-max", type=int, default=850)
    ap.add_argument("--speed-gain", type=float, default=400.0, help="command change per (m/s error x s)")
    ap.add_argument("--plan-dist", type=float, default=2.6, help="plan the detour when an object is this close (m)")
    ap.add_argument("--stop-dist", type=float, default=0.7, help="emergency stop for anything in the path this close (m)")
    ap.add_argument("--margin", type=float, default=0.10, help="extra side clearance past the object (m)")
    ap.add_argument("--ramp", type=float, default=1.8, help="length of each sideways curve (m)")
    ap.add_argument("--ramp-clear", type=float, default=0.3, help="finish the outward curve this far before the object (m)")
    ap.add_argument("--object-depth", type=float, default=0.4, help="assumed object size front-to-back (m); the camera only sees its front")
    ap.add_argument("--pass-extra", type=float, default=0.3, help="hold the offset this much longer past the object (m)")
    ap.add_argument("--after-dist", type=float, default=1.0, help="drive this far along the line after rejoining it (m)")
    ap.add_argument("--cam-ahead", type=float, default=0.3, help="camera distance ahead of the car's turning centre (m)")
    ap.add_argument("--lookahead", type=float, default=0.6, help="pure-pursuit lookahead (m)")
    ap.add_argument("--steer-gain", type=float, default=10.0, help="L/R difference per degree of heading error")
    ap.add_argument("--steer-max", type=float, default=350.0, help="largest L/R steering difference")
    ap.add_argument("--approach-time", type=float, default=20.0, help="give up if nothing is in the way after this long (s)")
    ap.add_argument("--m-per-count", type=float, default=0.0075, help="metres per hall count (measured)")
    ap.add_argument("--half-width", type=float, default=0.40, help="car half-width + small margin (m)")
    ap.add_argument("--min-height", type=float, default=0.08)
    ap.add_argument("--max-height", type=float, default=1.50)
    ap.add_argument("--max-range", type=float, default=6.0)
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="show the plan; never send motor commands")
    args = ap.parse_args()

    eye = Eye(args)
    car = None
    try:
        if args.dry_run:
            near = eye.nearest()
            print(f"nearest ahead: {'clear' if near is None else f'{near:.2f} m'}")
            plan = eye.detour(near, args.half_width + args.margin) if near is not None else None
            if plan:
                print(f"sideways move needed: left {plan[0]:.2f} m, right {plan[1]:.2f} m "
                      f"-> would pass on the {'left' if plan[0] <= plan[1] else 'right'}")
            return

        car = Car(args.port, args.m_per_count)
        car.set_origin()
        result = run(car, eye, args)
        car.settle()
        print(f"{result}; final: left {car.left:+.2f} m  ahead {car.ahead:.2f} m  "
              f"heading {math.degrees(car.heading()):+.1f} deg")
    finally:
        if car:
            car.close()
        eye.close()


if __name__ == "__main__":
    main()
