#!/usr/bin/env python3
"""Real-time depth → hazards → path → vehicle loop.

Streams the RealSense D455, builds hazard circles (metres), plans around them
with grid A*, and commands the vehicle with pure pursuit plus hard safety stops.

Mission goal (default): a fixed point exactly 10 m ahead of the robot at start,
expressed in the ground frame (x right, y forward). Hazards stay in the live
camera frame; the goal is transformed into that frame using open-loop pose.

Usage:
    python navigator.py
    python navigator.py --dry-run          # plan + view, always send zero twist
    python navigator.py --goal-ahead-m 10
"""

from __future__ import annotations

import argparse
import math

from hazard_detector import HazardConfig, stream_hazards
from pathfinder import Pathfinder, PathfinderConfig
from vehicle import Pose, PurePursuit, Vehicle, VehicleConfig, min_edge_clearance_m

GOAL_AHEAD_M = 10.0


def world_to_body(wx: float, wy: float, pose: Pose) -> tuple[float, float]:
    """Point in start/world frame → current robot/camera ground frame."""
    dx, dy = wx - pose.x, wy - pose.y
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return c * dx - s * dy, s * dx + c * dy


class Navigator:
    def __init__(
        self,
        hazard_cfg: HazardConfig,
        vehicle_cfg: VehicleConfig,
        path_cfg: PathfinderConfig,
        goal_ahead_m: float = GOAL_AHEAD_M,
        dry_run: bool = False,
    ):
        if abs(hazard_cfg.safety_margin_m - vehicle_cfg.robot_radius_m) > 1e-6:
            raise ValueError(
                f"HazardConfig.safety_margin_m ({hazard_cfg.safety_margin_m}) must equal "
                f"VehicleConfig.robot_radius_m ({vehicle_cfg.robot_radius_m}) so units stay consistent"
            )
        self.hazard_cfg = hazard_cfg
        self.vehicle_cfg = vehicle_cfg
        self.pathfinder = Pathfinder(path_cfg)
        self.vehicle = Vehicle(vehicle_cfg)
        self.follower = PurePursuit(vehicle_cfg)
        self.goal_ahead_m = goal_ahead_m
        self.dry_run = dry_run
        # Fixed world goal: 10 m straight ahead at mission start
        self.world_goal = (0.0, goal_ahead_m)
        self.mission_done = False
        self._last_status = ""
        self._last_path = []
        self._last_goal_body = (0.0, goal_ahead_m)

    def on_frame(self, ctx):
        hazards = ctx.hazards
        pose = self.vehicle.pose
        goal_body = world_to_body(self.world_goal[0], self.world_goal[1], pose)
        self._last_goal_body = goal_body

        clearance = min_edge_clearance_m(hazards)
        dist_goal = math.hypot(goal_body[0], goal_body[1])

        if self.mission_done or dist_goal <= self.vehicle_cfg.goal_tolerance_m:
            self.mission_done = True
            self._command(0.0, 0.0, "mission complete")
            self._last_path = [(0.0, 0.0), goal_body]
            return self._overlay("DONE — reached 10 m goal")

        if clearance <= self.vehicle_cfg.emergency_stop_m:
            self._command(0.0, 0.0, "emergency stop")
            self._last_path = []
            return self._overlay(f"E-STOP  edge {clearance:.2f} m")

        result = self.pathfinder.plan((0.0, 0.0), goal_body, hazards)
        self._last_path = result.points

        if result.blocked or len(result.points) < 2:
            self._command(0.0, 0.0, "no path")
            return self._overlay("BLOCKED — no safe path")

        # Follow in the current camera frame (robot at origin facing +y)
        twist = self.follower.compute(result.points, hazards, pose=Pose())
        if (
            twist.linear_m_s == 0.0
            and twist.angular_rad_s == 0.0
            and clearance <= self.vehicle_cfg.slow_distance_m
        ):
            self._command(0.0, 0.0, "holding")
            return self._overlay(f"HOLD  clear {clearance:.2f} m")

        self._command(twist.linear_m_s, twist.angular_rad_s, None)
        cmd = self.vehicle.last_cmd
        return self._overlay(
            f"v {cmd.linear_m_s:+.2f} m/s  w {cmd.angular_rad_s:+.2f} rad/s  "
            f"clear {clearance:.2f} m  goal {dist_goal:.1f} m"
        )

    def _command(self, v, w, reason):
        if self.dry_run:
            self.vehicle.stop(reason or "dry-run")
            self.vehicle.set_last_cmd(v, w)  # show planned twist without moving
            return
        if reason:
            self.vehicle.stop(reason)
        else:
            self.vehicle.drive(v, w)

    def _overlay(self, status: str) -> dict:
        self._last_status = status
        return {
            "path": self._last_path,
            "goal": self._last_goal_body,
            "status": status,
            "view_range_m": max(self.goal_ahead_m + 1.0, self.hazard_cfg.max_range_m),
        }

    def run(self, show=True):
        print(
            f"navigator: goal = (0, {self.goal_ahead_m:.1f}) m in start frame  "
            f"| robot_radius = {self.vehicle_cfg.robot_radius_m:.2f} m  "
            f"| dry_run = {self.dry_run}"
        )
        try:
            for _ in stream_hazards(self.hazard_cfg, show=show, frame_hook=self.on_frame):
                if self.mission_done:
                    break
        finally:
            self.vehicle.stop("shutdown")
            print(
                f"navigator stopped ({self.vehicle.stopped_reason})  "
                f"pose x={self.vehicle.pose.x:+.2f} y={self.vehicle.pose.y:.2f} m"
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--goal-ahead-m",
        type=float,
        default=GOAL_AHEAD_M,
        help="mission goal distance straight ahead at start (metres)",
    )
    ap.add_argument(
        "--robot-radius-m",
        type=float,
        default=0.35,
        help="robot half-width / planning inflation (metres)",
    )
    ap.add_argument("--max-linear-m-s", type=float, default=0.60)
    ap.add_argument("--max-angular-rad-s", type=float, default=0.80)
    ap.add_argument("--emergency-stop-m", type=float, default=0.45)
    ap.add_argument("--cam-height-m", type=float, default=0.30)
    ap.add_argument("--cam-tilt-deg", type=float, default=0.0)
    ap.add_argument(
        "--max-range-m",
        type=float,
        default=8.0,
        help="hazard detection range (metres); goal may lie beyond this",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="run perception + planning but always command zero velocity",
    )
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    r = args.robot_radius_m
    hazard_cfg = HazardConfig(
        cam_height_m=args.cam_height_m,
        cam_tilt_deg=args.cam_tilt_deg,
        max_range_m=args.max_range_m,
        safety_margin_m=r,
        confirm_frames=1,  # report hazards immediately for e-stop / planning
    )
    vehicle_cfg = VehicleConfig(
        robot_radius_m=r,
        emergency_stop_m=args.emergency_stop_m,
        max_linear_m_s=args.max_linear_m_s,
        max_angular_rad_s=args.max_angular_rad_s,
    )
    # Map must contain the goal even after the robot turns (body-frame planning)
    span = args.goal_ahead_m + 2.0
    path_cfg = PathfinderConfig(
        x_min_m=-span,
        x_max_m=span,
        y_min_m=-span,
        y_max_m=span,
        cell_m=0.10,
        extra_clearance_m=0.05,
    )
    Navigator(
        hazard_cfg, vehicle_cfg, path_cfg, goal_ahead_m=args.goal_ahead_m, dry_run=args.dry_run
    ).run(show=not args.headless)


if __name__ == "__main__":
    main()
