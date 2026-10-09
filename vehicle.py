#!/usr/bin/env python3
"""Differential-drive vehicle interface and pure-pursuit follower (SI units).

Velocities are metres/second and radians/second. Pose is in the same ground
frame as hazard_detector (x right, y forward, yaw from +y toward +x = CW when
looking down — here yaw=0 means facing +y / forward).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class VehicleConfig:
    # Footprint / safety (metres). safety_margin_m in HazardConfig should match
    # robot_radius_m so planning circles already clear the body.
    robot_radius_m: float = 0.35
    emergency_stop_m: float = 0.45  # stop if object edge closer than this
    slow_distance_m: float = 1.20  # start scaling speed below this clearance
    # Limits
    max_linear_m_s: float = 0.60
    max_angular_rad_s: float = 0.80
    min_linear_m_s: float = 0.05  # below this, treat as stopped when turning
    # Pure pursuit
    lookahead_m: float = 0.80
    goal_tolerance_m: float = 0.25
    # Integration when no external odometry is available (open-loop dead reckoning)
    track_pose: bool = True


@dataclass
class Twist:
    linear_m_s: float = 0.0
    angular_rad_s: float = 0.0


@dataclass
class Pose:
    x: float = 0.0  # metres
    y: float = 0.0  # metres
    yaw: float = 0.0  # radians; 0 = facing +y (forward)


class Vehicle:
    """Sends twist commands in SI units (m/s, rad/s).

    Default send() only dead-reckons pose for the local planner. Override send()
    (or assign vehicle.send) to forward Twist to your drive base / ROS / serial;
    always call super().send(twist) or _integrate(twist) if you still want pose.
    """

    def __init__(self, cfg: VehicleConfig | None = None):
        self.cfg = cfg or VehicleConfig()
        self.pose = Pose()
        self._last_cmd = Twist()
        self._last_t = time.monotonic()
        self.stopped_reason: str | None = None

    def send(self, twist: Twist):
        """Write twist to actuators. Default: integrate pose only (no motors)."""
        self._integrate(twist)
        self._last_cmd = twist

    def stop(self, reason: str = "stop"):
        self.stopped_reason = reason
        self.send(Twist(0.0, 0.0))

    def drive(self, linear_m_s: float, angular_rad_s: float, reason: str | None = None):
        self.stopped_reason = reason
        cfg = self.cfg
        v = max(-cfg.max_linear_m_s, min(cfg.max_linear_m_s, linear_m_s))
        w = max(-cfg.max_angular_rad_s, min(cfg.max_angular_rad_s, angular_rad_s))
        self.send(Twist(v, w))

    def set_last_cmd(self, linear_m_s: float, angular_rad_s: float):
        """Record a twist for UI/logging without sending it to the motors."""
        self._last_cmd = Twist(linear_m_s, angular_rad_s)

    def _integrate(self, twist: Twist):
        if not self.cfg.track_pose:
            self._last_t = time.monotonic()
            return
        now = time.monotonic()
        dt = max(0.0, min(now - self._last_t, 0.2))  # clamp for stalls
        self._last_t = now
        if dt <= 0.0:
            return
        # Unicycle: yaw=0 faces +y
        self.pose.yaw += twist.angular_rad_s * dt
        self.pose.yaw = (self.pose.yaw + math.pi) % (2 * math.pi) - math.pi
        self.pose.x += twist.linear_m_s * math.sin(self.pose.yaw) * dt
        self.pose.y += twist.linear_m_s * math.cos(self.pose.yaw) * dt

    @property
    def last_cmd(self) -> Twist:
        return self._last_cmd


class SerialVehicle(Vehicle):
    """Vehicle that forwards each Twist to a microcontroller over serial.

    Line protocol (ASCII, newline-terminated), one line per send():
        V <linear_m_s> <angular_rad_s>\n      e.g. "V 0.350 -0.120\n"
    The firmware should stop the motors if no line arrives for ~0.5 s.
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        cfg: VehicleConfig | None = None,
        connect_wait_s: float = 2.0,
    ):
        import serial  # pyserial; imported lazily so dry runs don't need it

        super().__init__(cfg)
        self.ser = serial.serial_for_url(port, baudrate=baud, timeout=0, write_timeout=0.1)
        # Opening the port resets most Arduinos; wait for the bootloader to finish
        time.sleep(connect_wait_s)
        self.ser.reset_input_buffer()

    def send(self, twist: Twist):
        super().send(twist)
        line = f"V {twist.linear_m_s:.3f} {twist.angular_rad_s:.3f}\n"
        self.ser.write(line.encode("ascii"))

    def close(self):
        if self.ser.is_open:
            try:
                self.send(Twist(0.0, 0.0))
                self.ser.flush()
            finally:
                self.ser.close()


def min_edge_clearance_m(hazards) -> float:
    """Nearest physical-object edge distance (metres). inf if none."""
    if not hazards:
        return math.inf
    return min(h.edge_distance for h in hazards)


def speed_scale_for_clearance(clearance_m: float, cfg: VehicleConfig) -> float:
    if clearance_m <= cfg.emergency_stop_m:
        return 0.0
    if clearance_m >= cfg.slow_distance_m:
        return 1.0
    span = cfg.slow_distance_m - cfg.emergency_stop_m
    if span <= 1e-6:
        return 0.0
    return max(0.0, (clearance_m - cfg.emergency_stop_m) / span)


class PurePursuit:
    """Follow a polyline path in the *current* robot/camera ground frame.

    Paths from the local planner are expressed with the robot at the origin
    facing +y each frame, so the follower assumes pose ≈ (0, 0, 0) in that frame.
    """

    def __init__(self, cfg: VehicleConfig | None = None):
        self.cfg = cfg or VehicleConfig()

    def compute(
        self, path_points: list[tuple[float, float]], hazards=(), pose: Pose | None = None
    ) -> Twist:
        cfg = self.cfg
        pose = pose or Pose()
        clearance = min_edge_clearance_m(hazards)
        if clearance <= cfg.emergency_stop_m:
            return Twist(0.0, 0.0)

        if not path_points or len(path_points) < 2:
            return Twist(0.0, 0.0)

        goal = path_points[-1]
        dist_goal = math.hypot(goal[0] - pose.x, goal[1] - pose.y)
        if dist_goal <= cfg.goal_tolerance_m:
            return Twist(0.0, 0.0)

        target = self._lookahead_point(path_points, pose, cfg.lookahead_m)
        # Transform target into robot frame (x right, y forward)
        dx = target[0] - pose.x
        dy = target[1] - pose.y
        # world -> body with yaw=0 facing +y
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        x_b = c * dx - s * dy
        y_b = s * dx + c * dy

        # Curvature for unicycle pure pursuit: x forward in many texts; we use y forward
        ld2 = x_b * x_b + y_b * y_b
        if ld2 < 1e-6:
            return Twist(0.0, 0.0)
        curvature = 2.0 * x_b / ld2

        scale = speed_scale_for_clearance(clearance, cfg)
        # Slow in sharp turns
        turn_scale = 1.0 / (1.0 + 2.5 * abs(curvature))
        v = cfg.max_linear_m_s * scale * turn_scale
        # If the lookahead is behind / beside, prefer turning in place
        if y_b < 0.05:
            w = math.copysign(
                min(cfg.max_angular_rad_s, 1.2 * abs(x_b) + 0.3), x_b if x_b != 0 else 1.0
            )
            return Twist(0.0 if abs(w) > 0.15 else v, w)

        w = curvature * v
        w = max(-cfg.max_angular_rad_s, min(cfg.max_angular_rad_s, w))
        if abs(w) >= cfg.max_angular_rad_s and abs(v) > cfg.min_linear_m_s:
            v = abs(w) / max(abs(curvature), 1e-6)
            v = min(v, cfg.max_linear_m_s * scale)
        return Twist(v, w)

    def _lookahead_point(self, points, pose: Pose, lookahead_m: float):
        # Find the point on the polyline at least lookahead_m from the robot
        px, py = pose.x, pose.y
        best = points[-1]
        travelled = 0.0
        # First: closest segment, then walk forward
        closest_i, closest_t, closest_d = 0, 0.0, math.inf
        for i in range(len(points) - 1):
            ax, ay = points[i]
            bx, by = points[i + 1]
            dx, dy = bx - ax, by - ay
            seg = math.hypot(dx, dy)
            if seg < 1e-9:
                continue
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (seg * seg)))
            qx, qy = ax + t * dx, ay + t * dy
            d = math.hypot(px - qx, py - qy)
            if d < closest_d:
                closest_d, closest_i, closest_t = d, i, t

        i = closest_i
        ax, ay = points[i]
        bx, by = points[i + 1]
        seg_len = math.hypot(bx - ax, by - ay)
        along = closest_t * seg_len
        need = lookahead_m
        # Remaining distance on current segment
        remain = seg_len - along
        if remain >= need:
            t = (along + need) / seg_len
            return ax + t * (bx - ax), ay + t * (by - ay)
        need -= remain
        for j in range(i + 1, len(points) - 1):
            ax, ay = points[j]
            bx, by = points[j + 1]
            seg_len = math.hypot(bx - ax, by - ay)
            if seg_len >= need:
                t = need / seg_len
                return ax + t * (bx - ax), ay + t * (by - ay)
            need -= seg_len
        return points[-1]
