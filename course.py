#!/usr/bin/env python3
"""Complete the tape-lane course autonomously: follow the lanes, go around obstacles,
find a way when blocked, keep going.

World map, anchored where the car starts: X metres right, Y metres ahead. The car's pose
comes from the wheel hall counts and the BNO085 heading (avoid.Car), so everything the
camera sees is placed on the map and remembered after it leaves the view:

  * Objects: obstacle points (depth, standing above the floor) are grouped into blobs
    and matched to tracks: confirmed after --confirm sightings, smoothed as they move,
    kept while out of view (FOV sides, front blind band) until the camera sees them
    again; dropped if repeatedly missing where they should be visible, if they pass
    beside/under the car, fall too far behind, or exceed --memory-s without a re-sight.
  * Lanes: red (left) / black (right) tape builds a lane map; each lane is a wall on
    its outer side. Not seeing a lane is fine: the car carries on in the course
    direction (start heading, updated from the lane direction whenever tape is seen).
  * Planning: clearance-based shortest path (car half-width + margin) on a fine grid
    around the car, to the lane centre ahead, or else to the first reachable goal in a
    fan around the course direction; margins are relaxed if nothing fits.
  * Blocked (object in front, or no path): stop, then rotate in place to scan headings
    around the course direction, look, and replan, until a way is found. Every rotation
    is checked against the car's rectangle and the tracked objects first.

Stops: no feasible path after scanning, --max-dist, --max-time, any error, Ctrl+C.

Usage:
    python3 course.py --dry-run --frames 60     # perceive + plan, no motor commands
    python3 course.py --save DIR                # run, saving debug images twice a second
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import threading
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
from lane_drive import LaneCentre
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

LANE_SIDE = {"red": "left", "black": "right"}


# ---------------------------------------------------------------- pose / frames


class Pose:
    """Car pose in the world map, and camera <-> world conversion."""

    def __init__(self, cam_ahead):
        self.cam_ahead = cam_ahead
        self.set(0.0, 0.0, 0.0)

    def set(self, X, Y, theta):
        self.X, self.Y, self.theta = X, Y, theta  # theta + = turned left
        self.fwd = np.array([-math.sin(theta), math.cos(theta)])
        self.right = np.array([math.cos(theta), math.sin(theta)])
        self.cam = np.array([X, Y]) + self.cam_ahead * self.fwd

    def cam_to_world(self, pts):
        """(N, 2) camera-frame points (x right, y ahead) -> world (N, 2)."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        return self.cam + pts[:, :1] * self.right + pts[:, 1:] * self.fwd

    def world_to_cam(self, pts):
        d = np.asarray(pts, float).reshape(-1, 2) - self.cam
        return np.stack([d @ self.right, d @ self.fwd], axis=1)


# ---------------------------------------------------------------- object tracking


class Track:
    _next = 1

    def __init__(self, pos, radius, now=None):
        self.id = Track._next
        Track._next += 1
        self.pos = np.array(pos, float)
        self.radius = float(radius)
        self.hits = 1
        self.misses = 0
        self.last_seen = time.time() if now is None else now

    def __repr__(self):
        return f"#{self.id}({self.pos[0]:+.2f},{self.pos[1]:.2f} r{self.radius:.2f})"


class Tracker:
    """Persistent object tracks in the world frame.

    Matched detections update a track. Unmatched tracks stay when they leave the camera
    view (FOV edge or front blind band) until they are seen again; they are dropped only
    when missing inside the view cone, when they pass beside/under the car, when they fall
    behind, or when --memory-s elapses without a re-sight.
    """

    def __init__(self, args):
        self.args = args
        self.tracks: list[Track] = []

    def detections(self, f, lat, pose):
        """Group obstacle points (camera frame) into blobs; return [(world centre, radius)]."""
        a = self.args
        if f.size < a.min_points:
            return []
        world = pose.cam_to_world(np.stack([lat, f], axis=1))
        cell = a.track_cell_m
        ij = np.floor(world / cell).astype(int)
        lo = ij.min(axis=0)
        ij -= lo
        grid = np.zeros((ij[:, 1].max() + 1, ij[:, 0].max() + 1), np.uint8)
        np.add.at(grid, (ij[:, 1], ij[:, 0]), 1)
        occ = (grid >= a.points_per_cell).astype(np.uint8)
        occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))  # join pieces of one object (legs, ring)
        n, labels = cv2.connectedComponents(occ, connectivity=8)
        found = []
        lab_of_pt = labels[ij[:, 1], ij[:, 0]]
        for k in range(1, n):
            p = world[lab_of_pt == k]
            if len(p) < a.min_points:
                continue
            c = p.mean(axis=0)
            r = float(np.percentile(np.hypot(*(p - c).T), 98)) + cell / 2
            found.append((c, min(max(r, a.min_object_radius), a.max_object_radius)))
        return found

    def in_view(self, pos, pose):
        """Floor spot the depth camera should be seeing now (miss-count region)."""
        a = self.args
        x, y = pose.world_to_cam(pos)[0]
        if not (a.view_near < y < a.view_far):
            return False
        return abs(math.degrees(math.atan2(x, y))) < a.view_half_angle

    def beside_car(self, pos, pose):
        """Next to the car body on either side (not the front blind band, not far-off FOV).

        Used to expire lateral near-band ghosts that are neither in the miss-count cone nor
        far enough behind to hit --forget-behind. True out-of-view objects farther away are
        left alone until re-seen or --memory-s.
        """
        a = self.args
        d = np.asarray(pos, float) - [pose.X, pose.Y]
        xc, yc = float(d @ pose.right), float(d @ pose.fwd)
        side = a.half_width + a.beside_clear
        return side < abs(xc) < side + a.beside_band and abs(yc) < a.half_length + a.beside_along

    def overlaps_car(self, t, pose):
        """Object disk overlaps the car's rectangle enough that it isn't there any more
        (passed under / beside, or a person who walked away)."""
        a = self.args
        d = t.pos - np.array([pose.X, pose.Y])
        xc, yc = float(d @ pose.right), float(d @ pose.fwd)
        clear = max(abs(xc) - a.half_width, 0.0) ** 2 + max(abs(yc) - a.half_length, 0.0) ** 2
        return clear < max(t.radius - a.overlap_ok, 0.0) ** 2

    def update(self, dets, pose, now=None):
        a = self.args
        now = time.time() if now is None else now
        unmatched = list(range(len(dets)))
        for t in sorted(self.tracks, key=lambda t: -t.hits):
            best, best_d = None, None
            for i in unmatched:
                d = float(np.hypot(*(dets[i][0] - t.pos)))
                if d < a.match_dist + t.radius and (best_d is None or d < best_d):
                    best, best_d = i, d
            if best is not None:
                c, r = dets[best]
                t.pos += a.track_smooth * (c - t.pos)
                t.radius += a.track_smooth * (max(r, t.radius * 0.8) - t.radius)
                t.hits += 1
                t.misses = 0
                t.last_seen = now
                unmatched.remove(best)
            elif self.in_view(t.pos, pose):
                t.misses += 1  # should be visible: count toward drop
            elif self.beside_car(t.pos, pose):
                # lateral near-band ghosts used to live forever (not in_view, not behind);
                # expire them on the same miss budget as an in-view miss
                t.misses += 1
            # else: true out-of-view / front blind band — keep until re-seen or aged out
        for i in unmatched:
            self.tracks.append(Track(*dets[i], now=now))

        kept = []
        for t in self.tracks:
            if self.overlaps_car(t, pose):
                continue
            y = float(pose.world_to_cam(t.pos)[0][1])
            if y < -a.forget_behind:
                continue
            miss_lim = a.drop_misses if t.hits >= a.confirm else a.drop_misses_new
            if t.misses >= miss_lim:
                continue
            max_age = a.memory_s if t.hits >= a.confirm else a.memory_s_new
            if now - t.last_seen > max_age:
                continue
            kept.append(t)
        self.tracks = kept

    def confirmed(self):
        return [t for t in self.tracks if t.hits >= self.args.confirm]


# ---------------------------------------------------------------- lane map


class LaneMap:
    """World cells where each lane's tape has been seen, with the tape's direction."""

    def __init__(self, args):
        self.args = args
        self.cells = {"red": {}, "black": {}}  # (i, j) -> [count, direction]
        self.latest = {"red": [], "black": []}  # last seen segments, world (start, end)
        self.latest_at = {"red": None, "black": None}
        self.start_centre = args.start_centre  # the car is placed in the lane: corridor centred on it

    def add(self, lanes, pose):
        c = self.args.lane_cell_m
        for colour, segs in lanes.items():
            if segs:
                self.latest[colour] = [(pose.cam_to_world([sg.start])[0], pose.cam_to_world([sg.end])[0]) for sg in segs]
                self.latest_at[colour] = pose.Y
            for sg in segs:
                n = max(2, int(sg.length / c) + 1)
                pts = pose.cam_to_world(np.linspace(sg.start, sg.end, n))
                d = pose.cam_to_world([sg.end])[0] - pose.cam_to_world([sg.start])[0]
                d /= max(np.hypot(*d), 1e-6)
                for p in pts:
                    key = (int(p[0] // c), int(p[1] // c))
                    cell = self.cells[colour].setdefault(key, [0, d])
                    cell[0] += 1
                    cell[1] = d

    def walls(self, lane_width=None, near=None, radius=None):
        """Lane wall points as an (N, 3) array [x, y, r]: each seen lane cell plus a thick wall
        on the lane's outer side, the start corridor, and extended / mirrored lane pieces.
        With `near`/`radius`, only points within `radius` of `near`."""
        a, c = self.args, self.args.lane_cell_m
        ks = np.arange(0.0, a.wall_thickness + 1e-6, a.wall_step)
        chunks = []

        def wall(P, D, colour):  # P, D: (n, 2) points and unit directions along the lane
            if len(P) == 0:
                return
            out = np.stack([-D[:, 1], D[:, 0]], 1) if LANE_SIDE[colour] == "left" else np.stack([D[:, 1], -D[:, 0]], 1)
            chunks.append((P[:, None, :] + ks[None, :, None] * out[:, None, :]).reshape(-1, 2))

        for colour, cells in self.cells.items():
            seen = [(k, d) for k, (count, d) in cells.items() if count >= a.lane_min_seen]
            if seen:
                P = (np.array([k for k, _ in seen], float) + 0.5) * c
                wall(P, np.array([d for _, d in seen]), colour)
        # The car starts in the lane, pointing along it: until the tape there is seen,
        # assume straight lane edges half a lane width either side of the start centre
        # (only on a side where no tape has been seen near the start, at the configured width)
        if self.start_centre is not None and a.start_corridor > 0:
            ys = np.arange(-0.5, a.start_corridor + 1e-6, c)
            for colour, side in LANE_SIDE.items():
                # the camera can't see the floor right in front of the car: wall the stretch
                # below the nearest tape seen on this side (all of it if none seen)
                seen = [((i + 0.5) * c, (j + 0.5) * c) for (i, j), (count, _) in self.cells[colour].items()
                        if count >= a.lane_min_seen and -0.5 <= (j + 0.5) * c <= a.start_corridor]
                if seen:  # straight back from the nearest tape seen
                    x, y_end = min(seen, key=lambda p: p[1])
                else:
                    x = self.start_centre + (a.lane_width / 2 if side == "right" else -a.lane_width / 2)
                    y_end = a.start_corridor
                yy = ys[ys < y_end]
                wall(np.stack([np.full_like(yy, x), yy], 1), np.tile([0.0, 1.0], (len(yy), 1)), colour)
        # Tape is often hidden (stools, feet): extend the latest lane pieces both ways so the
        # lane edge stays continuous; if one lane isn't in view, mirror the other a lane width
        # across. Only pieces running roughly along the course: a bend extended in a straight
        # line would wall off the lane.
        along = lambda seg: abs(math.degrees(math.atan2(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]))) <= a.extend_max_angle
        latest = {colour: [sg for sg in segs if along(sg)] for colour, segs in self.latest.items()}
        if lane_width:
            for colour, other in (("red", "black"), ("black", "red")):
                if not latest[colour] and latest[other]:
                    shifted = []
                    for p0, p1 in latest[other]:
                        d = (p1 - p0) / max(float(np.hypot(*(p1 - p0))), 1e-6)
                        inward = np.array([d[1], -d[0]]) if LANE_SIDE[other] == "left" else np.array([-d[1], d[0]])
                        shifted.append((p0 + lane_width * inward, p1 + lane_width * inward))
                    latest[colour] = shifted
        for colour, segs in latest.items():
            synthetic = not any(along(sg) for sg in self.latest[colour])
            for p0, p1 in segs:
                length = float(np.hypot(*(p1 - p0)))
                if length < 1e-3:
                    continue
                d = (p1 - p0) / length
                ts = np.arange(-a.lane_extend, length + a.lane_extend + 1e-6, c)
                if not synthetic:
                    ts = ts[(ts < 0) | (ts > length)]  # the seen part is already in the cell walls
                wall(p0 + ts[:, None] * d, np.tile(d, (len(ts), 1)), colour)
        if not chunks:
            return np.zeros((0, 3))
        pts = np.concatenate(chunks)
        if near is not None:
            pts = pts[np.hypot(pts[:, 0] - near[0], pts[:, 1] - near[1]) < radius]
        return np.column_stack([pts, np.full(len(pts), a.lane_line_r)])


# ---------------------------------------------------------------- planning


def outline_points(args, per_side=7):
    """Points on the car's rectangle, car frame (x right, y forward), turning centre at 0."""
    hw, hl = args.half_width, args.half_length
    e = np.linspace(-1, 1, per_side)
    return np.concatenate([np.stack([e * hw, np.full(per_side, hl)], 1), np.stack([e * hw, np.full(per_side, -hl)], 1),
                           np.stack([np.full(per_side, hw), e * hl], 1), np.stack([np.full(per_side, -hw), e * hl], 1)])


def rollout(X, Y, th, segments, step=0.05, direction=1.0):
    """Poses (N, 3) along arcs (curvature + = left, length m); direction -1 = reversing."""
    out = [(X, Y, th)]
    for k, L in segments:
        for _ in range(max(1, int(round(L / step)))):
            ds = direction * step
            X += -math.sin(th) * ds
            Y += math.cos(th) * ds
            th += k * ds
            out.append((X, Y, th))
    return np.array(out)


class ClearanceMap:
    """Signed distance (m) from each cell to the nearest obstacle, around the car:
    negative inside an obstacle (how deep), so "don't get any deeper" can be checked."""

    def __init__(self, args):
        self.args = args

    def build(self, pose, circles):
        a = self.args
        cell, span = a.map_cell_m, a.map_half_span
        self.x0, self.y0, self.cell = pose.X - span, pose.Y - span, cell
        n = self.n = int(2 * span / cell)
        occ = np.zeros((n, n), np.uint8)
        circles = np.asarray(circles, float).reshape(-1, 3)
        i = ((circles[:, 0] - self.x0) / cell).astype(int)
        j = ((circles[:, 1] - self.y0) / cell).astype(int)
        rr = np.round(circles[:, 2] / cell).astype(int)
        small = rr <= 1  # lane wall points: set pixels directly, grow once
        sel = small & (i >= 0) & (i < n) & (j >= 0) & (j < n)
        occ[j[sel], i[sel]] = 1
        if sel.any():
            occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))
        for ii, jj, r in zip(i[~small], j[~small], rr[~small]):
            if -r <= ii < n + r and -r <= jj < n + r:
                cv2.circle(occ, (int(ii), int(jj)), int(r), 1, -1)
        self.clear = (ndimage.distance_transform_edt(occ == 0) - ndimage.distance_transform_edt(occ)) * cell

    def at(self, pts):
        """Clearance at world points (..., 2); outside the map counts as blocked (-0.5)."""
        i = ((pts[..., 0] - self.x0) / self.cell).astype(np.int32)
        j = ((pts[..., 1] - self.y0) / self.cell).astype(np.int32)
        inside = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        out = np.full(pts.shape[:-1], -0.5)
        out[inside] = self.clear[j[inside], i[inside]]
        return out

    def footprint(self, poses, outline):
        """Smallest clearance of the car's outline at each pose (N,)."""
        return self.outline_clearance(poses, outline).min(axis=1)

    def outline_clearance(self, poses, outline):
        """Clearance of each outline point at each pose (N, K)."""
        th = poses[:, 2:3]
        fx, fy, rx, ry = -np.sin(th), np.cos(th), np.cos(th), np.sin(th)
        ox, oy = outline[None, :, 0], outline[None, :, 1]
        pts = np.stack([poses[:, :1] + ox * rx + oy * fx, poses[:, 1:2] + ox * ry + oy * fy], axis=-1)
        return self.at(pts)


class ArcPlanner:
    """Local planner: short, car-like candidate moves from the current pose (arcs with a
    curvature limit, and S-shaped sidesteps), checked with the car's rectangle against the
    clearance map, scored for progress along the course. Candidate shapes are computed
    once; each frame they're moved to the car's pose and checked in one vectorised step."""

    def __init__(self, args, cmap):
        self.last_clr = None  # clearance of the last chosen move
        self.args = args
        self.cmap = cmap
        self.outline = outline_points(args)
        kmax = 1.0 / args.min_turn_radius
        cands = [[(float(k), float(L))] for k in np.linspace(-kmax, kmax, args.arc_count) for L in args.arc_lengths]
        for k in (kmax, kmax / 2, -kmax / 2, -kmax):  # sidestep: curve out, curve back, straight on
            for L1 in (0.4, 0.7):
                cands.append([(k, L1), (-k, L1), (0.0, 0.6)])
        self.candidates = cands
        rel = [rollout(0.0, 0.0, 0.0, segs, step=0.08) for segs in cands]
        self.rel = np.concatenate(rel)  # (M, 3) in the car frame: x right, y forward, heading
        self.cand_of = np.concatenate([np.full(len(r), i) for i, r in enumerate(rel)])
        starts = np.cumsum([0] + [len(r) for r in rel[:-1]])
        self.starts, self.ends = starts, starts + np.array([len(r) for r in rel]) - 1
        self.skip = np.zeros(len(self.rel), bool)  # the car's current spot may already be tight
        for st in starts:
            self.skip[st: st + 2] = True
        self.first_k = np.array([segs[0][0] for segs in cands])

    def choose(self, pose, course_heading, lane_goal, prev_k, eager=False):
        """Best candidate: (poses, info) or (None, reason). `eager`: when stuck, take any
        move that touches nothing (no margin) and makes a little progress."""
        a = self.args
        margin, min_progress = (a.eager_margin, a.eager_progress) if eager else (a.arc_margin, a.min_progress)
        th0 = pose.theta
        R = np.array([[math.cos(th0), -math.sin(th0)], [math.sin(th0), math.cos(th0)]])  # car frame -> world
        xy = np.array([pose.X, pose.Y]) + self.rel[:, :2] @ R.T
        poses = np.column_stack([xy, th0 + self.rel[:, 2]])
        clr_k = self.cmap.outline_clearance(poses, self.outline)  # (M, K): every point of the outline
        clr = clr_k.min(axis=1)
        clr[self.skip] = np.inf
        cmin = np.minimum.reduceat(clr, self.starts)
        if eager:
            # Already touching something (drifted onto a lane edge, bumped a stool): each
            # point of the car's outline must keep the margin, or at least not get any closer
            # to whatever it's near than it is now. Judged per point, so touching the lane
            # on the right doesn't excuse driving the front into a stool.
            now_k = self.cmap.outline_clearance(np.array([[pose.X, pose.Y, th0]]), self.outline)[0]
            good = ((clr_k >= margin) | (clr_k >= now_k[None, :] - 0.01)).all(axis=1) | self.skip
            ok = np.logical_and.reduceat(good, self.starts)
        else:
            ok = cmin >= margin
        end = poses[self.ends]
        course = np.array([-math.sin(course_heading), math.cos(course_heading)])
        progress = (end[:, :2] - [pose.X, pose.Y]) @ course
        off = np.degrees(np.abs((end[:, 2] - course_heading + np.pi) % (2 * np.pi) - np.pi))
        score = (a.w_progress * progress
                 - a.w_heading * np.maximum(0.0, off - a.heading_free) / 45.0
                 + a.w_clear * np.minimum(cmin, a.prefer_clear) / a.prefer_clear
                 - a.w_smooth * np.abs(self.first_k - prev_k))
        if lane_goal is not None:
            score -= a.w_lane * np.hypot(end[:, 0] - lane_goal[0], end[:, 1] - lane_goal[1])
        score[~ok | (progress < min_progress)] = -np.inf
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            self.last_clr = None
            return None, f"no safe forward move ({int(ok.sum())}/{len(ok)} clear, none making progress)"
        segs = self.candidates[best]
        self.last_clr = float(cmin[best])
        shape = "sidestep" if len(segs) > 1 else f"arc r={'inf' if abs(segs[0][0]) < 1e-6 else f'{1 / segs[0][0]:+.1f}'}m"
        info = f"{'eager ' if eager else ''}{shape} L={sum(L for _, L in segs):.1f}m prog {progress[best]:+.2f} clr {cmin[best]:.2f} ({int(ok.sum())} clear)"
        return poses[self.starts[best]: self.ends[best] + 1], info


class ManeuverPlanner:
    """Search (Hybrid A*) over small moves for tight spots: --search-step forward pieces
    (straight, or curving exactly one heading step), rotations in place of one heading step,
    and reverses (cost x2). Headings live on a --search-rot lattice relative to the start,
    so each move's footprint points are precomputed per heading: checking a move is one
    array lookup. Goal: --search-ahead further along the course, facing within
    --search-heading of the course direction."""

    def __init__(self, args, cmap):
        self.args = args
        self.cmap = cmap
        outline = outline_points(args, per_side=5)
        self.outline = outline
        L, rot = args.search_step, math.radians(args.search_rot)
        self.nb = int(round(360 / args.search_rot))
        k = rot / L  # a curving piece turns exactly one heading step
        prims = [("fwd", [(0.0, L)], 1.0, 0, L), ("fwdL", [(k, L)], 1.0, 1, L * 1.05), ("fwdR", [(-k, L)], 1.0, -1, L * 1.05),
                 ("back", [(0.0, L)], -1.0, 0, L * 2.0), ("rotL", None, 0, 1, 0.12), ("rotR", None, 0, -1, 0.12)]
        self.names = [p[0] for p in prims]
        self.cost = np.array([p[4] for p in prims])
        self.db = [p[3] for p in prims]
        self.offsets, self.seg_starts, self.end_dxy, self.rel_poses = [], [], [], []
        for b in range(self.nb):
            th0 = b * rot  # relative to the search's start heading
            offs, starts, ends, rels = [], [], [], []
            for name, segs, direction, db, _ in prims:
                if segs is None:
                    ths = th0 + np.linspace(0, db * rot, 6)
                    poses = np.stack([np.zeros(6), np.zeros(6), ths], 1)
                else:
                    poses = rollout(0.0, 0.0, th0, segs, step=0.05, direction=direction)
                rels.append(poses)
                th = poses[1:, 2:3]
                ox, oy = outline[None, :, 0], outline[None, :, 1]
                pts = np.stack([poses[1:, :1] + ox * np.cos(th) + oy * -np.sin(th),
                                poses[1:, 1:2] + ox * np.sin(th) + oy * np.cos(th)], -1).reshape(-1, 2)
                starts.append(sum(len(o) for o in offs))
                offs.append(pts)
                ends.append(poses[-1, :2])
            self.offsets.append(np.concatenate(offs))
            self.seg_starts.append(np.array(starts))
            self.end_dxy.append(np.array(ends))
            self.rel_poses.append(rels)

    def search(self, pose, course_heading):
        """List of (name, poses) moves, or None."""
        a = self.args
        th_start = pose.theta
        course = np.array([-math.sin(course_heading), math.cos(course_heading)])
        origin = np.array([pose.X, pose.Y])
        cell, nb = a.search_cell, self.nb
        heading_off = lambda b: abs(math.degrees(avoid.wrap(th_start + b * math.radians(a.search_rot) - course_heading)))
        start = (pose.X, pose.Y, 0)
        key = lambda st: (int(round(st[0] / cell)), int(round(st[1] / cell)), st[2])
        heap = [(a.search_ahead, 0.0, 0, start)]
        best_g = {key(start): 0.0}
        parent = {key(start): None}
        # Starting tighter than --arc-margin (e.g. stopped close to a stool): allow moves that
        # don't get tighter than where they start, until the margin is reached again
        c0 = float(self.cmap.footprint(np.array([[pose.X, pose.Y, th_start]]), self.outline)[0])
        need = {key(start): min(a.arc_margin, c0 - 0.005)}
        t0, n, tie = time.time(), 0, 0
        rot_world = np.array([[math.cos(th_start), -math.sin(th_start)], [math.sin(th_start), math.cos(th_start)]])
        offsets = [o @ rot_world.T for o in self.offsets]  # lattice frame -> world, done once
        end_dxy = [e @ rot_world.T for e in self.end_dxy]
        while heap and time.time() - t0 < a.search_time:
            _, g, _, st = heapq.heappop(heap)
            n += 1
            X, Y, b = st
            if (np.array([X, Y]) - origin) @ course >= a.search_ahead and heading_off(b) <= a.search_heading:
                moves, k = [], key(st)
                while parent[k] is not None:
                    pk, pi, (PX, PY, pb) = parent[k]
                    rel = self.rel_poses[pb][pi]
                    xy = np.array([PX, PY]) + rel[:, :2] @ rot_world.T
                    moves.append((self.names[pi], np.column_stack([xy, th_start + rel[:, 2]])))
                    k = pk
                moves.reverse()
                self.note = f"{len(moves)} moves, {n} expansions, {time.time() - t0:.2f} s"
                return moves
            clr = self.cmap.at(offsets[b] + [X, Y])
            move_clr = np.minimum.reduceat(clr, self.seg_starts[b])
            ok = move_clr >= need.get(key(st), a.arc_margin)
            for pi in np.nonzero(ok)[0]:
                dx, dy = end_dxy[b][pi]
                ns = (X + dx, Y + dy, (b + self.db[pi]) % nb)
                kk = key(ns)
                ng = g + self.cost[pi]
                if ng < best_g.get(kk, 1e9):
                    best_g[kk] = ng
                    parent[kk] = (key(st), pi, st)
                    need[kk] = min(a.arc_margin, float(move_clr[pi]) - 0.005)
                    tie += 1
                    prog = (np.array(ns[:2]) - origin) @ course
                    h = max(0.0, a.search_ahead - prog) + 0.1 * max(0.0, heading_off(ns[2]) - a.search_heading) / 15
                    heapq.heappush(heap, (ng + h, ng, tie, ns))
        self.note = f"no sequence ({n} expansions, {time.time() - t0:.1f} s)"
        return None


# ---------------------------------------------------------------- following


def pursuit_point(path, pos, lookahead):
    """Point on the polyline `path` about `lookahead` beyond the closest point to `pos`."""
    pts = np.asarray(path, float)
    if len(pts) < 2:
        return pts[-1]
    best_i, best_t, best_d = 0, 0.0, None
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, float((pos - a) @ ab) / L2))
        d = float(np.hypot(*(a + t * ab - pos)))
        if best_d is None or d < best_d:
            best_i, best_t, best_d = i, t, d
    need = lookahead
    a, b = pts[best_i], pts[best_i + 1]
    p = a + best_t * (b - a)
    for i in range(best_i, len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        start = p if i == best_i else a
        seg = float(np.hypot(*(b - start)))
        if seg >= need:
            return start + (b - start) * (need / max(seg, 1e-9))
        need -= seg
    return pts[-1]


def debug_map(args, pose, tracker, lanemap, path, goal, scale=60, size=(480, 480)):
    W, H = size
    img = np.full((H, W, 3), 30, np.uint8)
    ox, oy = W // 2, H - 60  # car at bottom centre-ish; map scrolls with the car
    to_px = lambda P: (int(ox + (P[0] - pose.X) * scale), int(oy - (P[1] - pose.Y) * scale))
    c = args.lane_cell_m
    for colour, cells in lanemap.cells.items():
        col = (0, 220, 255) if colour == "red" else (255, 255, 0)  # left lane yellow, right cyan (both are black tape)
        for (i, j), (count, _) in cells.items():
            if count >= args.lane_min_seen:
                cv2.circle(img, to_px(((i + 0.5) * c, (j + 0.5) * c)), 2, col, -1)
    now = time.time()
    for t in tracker.tracks:
        fresh = now - t.last_seen < 0.4
        if t.hits >= args.confirm:
            col = (0, 165, 255) if fresh else (200, 200, 80)  # orange = seen now; muted = held out of view
        else:
            col = (90, 90, 90)
        cv2.circle(img, to_px(t.pos), max(2, int(t.radius * scale)), col, 2)
        cv2.circle(img, to_px(t.pos), max(2, int((t.radius + args.half_width + args.obstacle_margin) * scale)), (60, 60, 120), 1)
        cv2.putText(img, f"#{t.id}", to_px(t.pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if path:
        pts = [to_px(p) for p in path]
        for p, q in zip(pts, pts[1:]):
            cv2.line(img, p, q, (0, 255, 0), 2)
    if goal is not None:
        cv2.drawMarker(img, to_px(goal), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
    car = to_px((pose.X, pose.Y))
    tip = to_px((pose.X + 0.4 * pose.fwd[0], pose.Y + 0.4 * pose.fwd[1]))
    cv2.circle(img, car, int(args.half_width * scale), (255, 255, 255), 2)
    cv2.line(img, car, tip, (255, 255, 255), 2)
    return img


# ---------------------------------------------------------------- footprint


def footprint(args, step=0.05):
    """Points on the car's outline, car frame (x right, y forward), turning centre at 0."""
    return outline_points(args, per_side=int(round(2 * args.half_width / step)) + 1)


def safe_reverse(args, pose, dist, circles, step=0.04):
    """How far (m) the car can back straight up before its outline would touch a circle."""
    outline = footprint(args)
    circles = np.asarray(circles, float).reshape(-1, 3)
    if len(circles) == 0:
        return dist
    cx, cy, cr = circles[:, 0], circles[:, 1], circles[:, 2]

    def clearance(d):  # per object: closest the outline gets, backed up by d
        centre = np.array([pose.X, pose.Y]) - d * pose.fwd
        pts = centre + outline[:, :1] * pose.right + outline[:, 1:] * pose.fwd
        return (np.hypot(pts[:, None, 0] - cx[None], pts[:, None, 1] - cy[None]) - cr[None]).min(axis=0)

    now = clearance(0.0)
    ok = 0.0
    for d in np.arange(step, dist + 1e-6, step):
        c = clearance(d)
        # blocked if it gets inside the margin of an object, unless it's already there and moving away
        if ((c < args.sweep_margin) & (c < now - 0.005)).any():
            break
        ok = d
    return ok


def safe_rotation(args, pose, delta_deg, circles, step_deg=3.0):
    """How far (deg, same sign as delta) the car can rotate in place before its outline
    would touch any circle (x, y, r) in the world map."""
    outline = footprint(args)
    centre = np.array([pose.X, pose.Y])
    circles = np.asarray(circles, float).reshape(-1, 3)
    if len(circles) == 0:
        return delta_deg
    cx, cy, cr = circles[:, 0], circles[:, 1], circles[:, 2]
    near = np.hypot(cx - centre[0], cy - centre[1]) - cr < math.hypot(args.half_width, args.half_length) + 0.1
    if not near.any():
        return delta_deg
    cx, cy, cr = cx[near], cy[near], cr[near]
    sign = 1 if delta_deg >= 0 else -1

    def clearance(deg):  # per object: closest the outline gets, rotated by deg
        th = pose.theta + math.radians(deg)
        fwd = np.array([-math.sin(th), math.cos(th)]); right = np.array([math.cos(th), math.sin(th)])
        pts = centre + outline[:, :1] * right + outline[:, 1:] * fwd
        return (np.hypot(pts[:, None, 0] - cx[None], pts[:, None, 1] - cy[None]) - cr[None]).min(axis=0)

    now = clearance(0.0)
    ok = 0.0
    for a in np.arange(step_deg, abs(delta_deg) + 1e-6, step_deg):
        c = clearance(sign * a)
        # blocked if it gets inside the margin of an object, unless it's already there and moving away
        if ((c < args.sweep_margin) & (c < now - 0.005)).any():
            break
        ok = a
    return sign * ok


# ---------------------------------------------------------------- the course run


class Course:
    def __init__(self, args):
        self.args = args
        black_only = args.tape == "black"
        self.lane_args = types.SimpleNamespace(
            near=0.4, far=5.0, half_span=2.5,
            line_kernel_px=21 if black_only else 11,  # thick black tape is wider than 11 px up close
            black_contrast=25, red_hue_lo=160, red_hue_hi=8,
            red_min_sat=256 if black_only else 70,  # no red tape: switch red detection off
            red_contrast=999 if black_only else 8,
            above_floor_m=0.03, depth_noise_k=0.006, blocked_grow_px=15, glare_l=225, black_max_ratio=0.62,
            floor_depth_win_px=31, min_floor_depth_frac=0.5 if black_only else 0.6, min_support=25, min_length_m=0.3,
            max_gap_m=0.3, max_segments=6 if black_only else 4, inlier_m=0.03, ransac_iters=60, max_points=1500)
        # obstacle points: wide view for tracking; just-narrower-than-the-car corridor for the emergency stop
        self.scan_args = types.SimpleNamespace(half_width=3.0, min_height=0.08, max_height=1.5,
                                               max_range=args.obstacle_range, min_points=args.min_points)
        self.stop_args = types.SimpleNamespace(half_width=args.half_width - 0.05, min_height=0.08, max_height=1.5,
                                               max_range=3.0, min_points=args.min_points)
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, sa.WIDTH, sa.HEIGHT, rs.format.z16, sa.FPS)
        cfg.enable_stream(rs.stream.color, sa.WIDTH, sa.HEIGHT, rs.format.bgr8, sa.FPS)
        self.prof = self.pipe.start(cfg)
        self.car = None
        self.t0 = time.time()
        self.n = 0
        self.last_save = -1
        self.last_print = -1
        self.path, self.goal, self.plan_note = [], None, "-"
        self.course_heading = 0.0  # direction the course goes, world (+ = left of the start heading)
        self.hits = 0
        self.near = None
        self.how = "none"
        self.ack_near = None  # front distance when the last front stop was pinned in the map
        self.bumps = []  # world spots where the front check stopped the car: kept as obstacles
        self.front_pt = None

    # --- setup / teardown
    def calibrate(self):
        a = self.args
        # The Arduino resets when its port opens (2-3 s): connect while the camera calibrates
        self._car_thread, self._car_result = None, None
        if not a.dry_run:
            def connect():
                try:
                    self._car_result = avoid.Car(a.port, a.m_per_count)
                except Exception as e:  # re-raised in the main thread
                    self._car_result = e
            self._car_thread = threading.Thread(target=connect, daemon=True)
            self._car_thread.start()
        self.scale = self.prof.get_device().first_depth_sensor().get_depth_scale()
        intr = self.prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)
        self.rx, self.ry, rows = sa.pixel_rays(intr)
        self.rows = rows
        for _ in range(15):  # let auto-exposure settle
            self.pipe.wait_for_frames()
        stack = [np.asanyarray(self.align.process(self.pipe.wait_for_frames()).get_depth_frame().get_data()).copy() for _ in range(10)]
        plane = sa.fit_floor(sa.to_points(np.median(np.stack(stack), axis=0) * self.scale, self.rx, self.ry), rows)
        if plane is None:
            sys.exit("floor calibration failed: not enough flat floor in view")
        self.normal, self.offset = plane
        self.cfwd, self.cright = sa.ground_axes(self.normal)
        print(f"floor: camera height {self.offset:.3f} m, tilt {self.tilt():.1f} deg down")
        self.det = ld.LaneDetector(intr, self.normal, self.offset, self.lane_args)
        self.det_plane = (self.normal, self.offset)  # plane the lane detector's geometry was built for
        self.floor_misses = 0
        self.centre = LaneCentre(a)
        self.tracker = Tracker(a)
        self.lanemap = LaneMap(a)
        self.cmap = ClearanceMap(a)
        self.planner = ArcPlanner(a, self.cmap)
        self.maneuvers = ManeuverPlanner(a, self.cmap)
        self.prev_k = 0.0
        self.pose = Pose(a.cam_ahead)
        self.rng = np.random.default_rng(0)
        if self._car_thread:
            self._car_thread.join()
            if isinstance(self._car_result, Exception):
                raise self._car_result
            self.car = self._car_result
            self.car.set_origin()
        if a.save:
            os.makedirs(a.save, exist_ok=True)

    def close(self):
        if self.car:
            self.car.close()
        self.pipe.stop()

    @property
    def travelled(self):
        return self.car.odo if self.car else 0.0

    # --- one camera frame: update pose, objects, lanes, course direction
    def sense(self):
        a = self.args
        f = self.align.process(self.pipe.wait_for_frames(1000))
        color = np.asanyarray(f.get_color_frame().get_data())
        depth = np.asanyarray(f.get_depth_frame().get_data()) * self.scale
        if self.car:
            self.car.poll()
            self.pose.set(-self.car.left, self.car.ahead, self.car.heading())
        self.n += 1
        pts = sa.to_points(depth, self.rx, self.ry)
        self.track_floor(pts)
        of, olat = sa.obstacle_points(pts, self.normal, self.offset, self.cfwd, self.cright, self.scan_args)
        self.tracker.update(self.tracker.detections(of, olat, self.pose), self.pose)
        self.near = sa.nearest_obstacle(pts, self.normal, self.offset, self.cfwd, self.cright, self.stop_args)
        self.front_pt = None
        if self.near is not None:  # where it is (camera frame x right, y ahead), to remember it
            f, lat = sa.obstacle_points(pts, self.normal, self.offset, self.cfwd, self.cright, self.stop_args)
            m = (f >= self.near) & (f < self.near + 0.15)
            if m.any():
                self.front_pt = (float(np.median(lat[m])), self.near + 0.05)
        close = self.near is not None and self.near < a.stop_dist
        if not close:
            self.ack_near = None
        # once a front stop is pinned in the map (remember_front), the planner steers around
        # it; only stop again if it gets closer than when it was pinned
        if close and self.ack_near is not None and self.near >= self.ack_near - 0.03:
            close = False
        self.hits = self.hits + 1 if close else 0
        red, black = self.det.masks(color, depth)
        if a.tape == "black":
            lanes = self.split_sides(self.drop_near_objects(self.det.segments(black, self.rng)))
        else:
            lanes = {"red": self.det.segments(red, self.rng), "black": self.det.segments(black, self.rng)}
        self.lanemap.add(lanes, self.pose)
        target, self.how = self.centre.update(lanes)
        self.lane_goal = self.pose.cam_to_world([target])[0] if target is not None else None
        # course direction follows lane pieces running roughly along it
        dirs = []
        for segs in lanes.values():
            for sg in segs:
                w0, w1 = self.pose.cam_to_world([sg.start])[0], self.pose.cam_to_world([sg.end])[0]
                th = math.atan2(-(w1[0] - w0[0]), w1[1] - w0[1])
                if abs(math.degrees(avoid.wrap(th - self.course_heading))) < a.course_max_angle and sg.length > 0.5:
                    dirs.append(th)
        if dirs:
            mean = math.atan2(np.mean(np.sin(dirs)), np.mean(np.cos(dirs)))
            self.course_heading += a.course_smooth * avoid.wrap(mean - self.course_heading)
        self._frame = (color, red, black, lanes)
        self.log()
        return color

    def track_floor(self, pts):
        """Follow the floor plane frame to frame: the tall mount pitches as the car moves,
        and a stale plane turns stool legs into "tape" and floor into "obstacles"."""
        fit = sa.refine_floor(pts, self.rows, self.normal, self.offset)
        if fit is None:
            self.floor_misses += 1
            return
        n, off = fit
        n = self.normal + 0.5 * (n - self.normal)
        self.normal = n / np.linalg.norm(n)
        self.offset += 0.5 * (off - self.offset)
        self.cfwd, self.cright = sa.ground_axes(self.normal)
        n0, off0 = self.det_plane  # the lane detector's per-pixel geometry costs ~25 ms: only on real change
        if math.degrees(math.acos(min(1.0, float(self.normal @ n0)))) > 0.4 or abs(self.offset - off0) > 0.01:
            self.det.set_plane(self.normal, self.offset)
            self.det_plane = (self.normal, self.offset)

    def tilt(self):
        return math.degrees(math.atan2(-self.normal[2], -self.normal[1]))

    def drop_near_objects(self, segments):
        """Stool legs and base rings are black too: drop tape pieces (or the parts of them)
        within --tape-object-gap of a tracked object's edge."""
        objs = [(t.pos, t.radius + self.args.tape_object_gap) for t in self.tracker.tracks if t.hits >= 2]
        if not objs:
            return segments
        kept = []
        for sg in segments:
            n = max(2, int(sg.length / 0.05) + 1)
            t = np.linspace(0.0, 1.0, n)
            w = self.pose.cam_to_world(sg.start[None] + t[:, None] * (sg.end - sg.start)[None])
            free = np.ones(n, bool)
            for c, r in objs:
                free &= np.hypot(*(w - c).T) > r
            if free.all():
                kept.append(sg)
                continue
            # keep the longest free run, if it's still a usable piece
            best, run, i0 = (0, 0, 0), 0, 0
            for i, f in enumerate(free):
                run = run + 1 if f else 0
                if run > best[0]:
                    best = (run, i - run + 1, i)
            if best[0] >= 2:
                p0 = sg.start + t[best[1]] * (sg.end - sg.start)
                p1 = sg.start + t[best[2]] * (sg.end - sg.start)
                if np.hypot(*(p1 - p0)) >= self.lane_args.min_length_m:
                    kept.append(dataclasses.replace(sg, start=p0, end=p1))
        return kept

    def split_sides(self, segments):
        """Both lanes are black: a piece is the left or right lane by which side of the
        course line (through the car, along the course direction) its middle lies on.
        Returned under the keys the rest of the code uses: "red" = left, "black" = right."""
        left, right = [], []
        course_dir = np.array([-math.sin(self.course_heading), math.cos(self.course_heading)])
        car = np.array([self.pose.X, self.pose.Y])
        for sg in segments:
            mid = self.pose.cam_to_world([(sg.start + sg.end) / 2])[0] - car
            side = course_dir[0] * mid[1] - course_dir[1] * mid[0]  # > 0: left of the course line
            (left if side > 0 else right).append(sg)
        return {"red": left, "black": right}

    # --- planning: lane-centre goal first, then a fan around the course direction
    def obstacles(self):
        """(tracked objects, objects + lane walls) as (N, 3) arrays [x, y, r], near the car."""
        objs = np.array([(t.pos[0], t.pos[1], t.radius) for t in self.tracker.confirmed()] + self.bumps,
                        float).reshape(-1, 3)
        walls = self.lanemap.walls(lane_width=self.centre.width, near=(self.pose.X, self.pose.Y),
                                   radius=self.args.map_half_span * 1.5)
        return objs, np.vstack([objs, walls])

    def plan(self, eager=False):
        """Pick the best short move from here (arc planner); False if none is safe."""
        _, circles = self.obstacles()
        t = time.time()
        self.cmap.build(self.pose, circles)
        poses, info = self.planner.choose(self.pose, self.course_heading, self.lane_goal, self.prev_k, eager)
        ms = 1000 * (time.time() - t)
        if poses is None:
            self.path, self.plan_note = [], f"{info} ({ms:.0f} ms)"
            return False
        self.path = [tuple(p[:2]) for p in poses]
        self.goal = poses[-1][:2]
        self.plan_note = f"{info} ({ms:.0f} ms)"
        return True

    # --- motion primitives
    def remember_front(self):
        """The front check stopped the car on something the map may not hold (low parts,
        things in the camera's near blind band): pin it in the world map for the run,
        so the planner doesn't send the car straight back into it."""
        if self.front_pt is None:
            return
        # Closer to the camera than the bumper means it overhangs the car (a seat): pin it
        # just beyond the bumper, never inside the car (that would block every move)
        bumper = self.args.half_length - self.args.cam_ahead
        x, y = self.front_pt
        w = self.pose.cam_to_world([(x, max(y, bumper + self.args.bump_r + 0.02))])[0]
        self.ack_near = self.near
        if all(math.hypot(w[0] - b[0], w[1] - b[1]) > 0.08 for b in self.bumps):
            self.bumps.append((float(w[0]), float(w[1]), self.args.bump_r))
            print(f"  remembered obstacle at ({w[0]:+.2f},{w[1]:.2f})")

    def stop(self):
        if self.car:
            self.car.drive(0, 0)

    def rotate_to(self, heading):
        """Rotate in place to world `heading`, limited to what the car's outline can sweep
        without touching a tracked object. Returns degrees actually turned."""
        a = self.args
        if not self.car:
            return 0.0
        objects, _ = self.obstacles()
        want = math.degrees(avoid.wrap(heading - self.pose.theta))
        allowed = safe_rotation(a, self.pose, want, objects)
        if abs(allowed) < 5:
            print(f"  rotate {want:+.0f} deg: not enough room (objects within the car's sweep)")
            return 0.0
        if abs(allowed) < abs(want) - 1:
            print(f"  rotate {want:+.0f} deg: limited to {allowed:+.0f} deg by nearby objects")
        start = self.pose.theta
        target = start + math.radians(allowed)
        direction = 1 if allowed > 0 else -1
        t0 = time.time()
        hist = [(t0, self.pose.theta)]
        try:
            while time.time() - t0 < 8.0:
                self.car.poll()
                self.pose.set(-self.car.left, self.car.ahead, self.car.heading())
                now = time.time()
                hist.append((now, self.pose.theta))
                while len(hist) > 2 and now - hist[0][0] > 0.2:
                    hist.pop(0)
                rate = abs(math.degrees(avoid.wrap(hist[-1][1] - hist[0][1]))) / max(hist[-1][0] - hist[0][0], 1e-3)
                remaining = direction * math.degrees(avoid.wrap(target - self.pose.theta))
                if remaining <= max(3.0, rate * a.turn_coast_s):
                    break
                if now - t0 > 2.0 and abs(math.degrees(avoid.wrap(self.pose.theta - start))) < 3:
                    print("  rotate: heading not changing (stalled?)")
                    break
                self.car.drive(-direction * a.turn_speed, direction * a.turn_speed)
                time.sleep(0.02)
        finally:
            self.car.drive(0, 0)
        self.car.settle()
        self.pose.set(-self.car.left, self.car.ahead, self.car.heading())
        return math.degrees(avoid.wrap(self.pose.theta - start))

    def back_up(self, dist):
        """Reverse straight up to `dist` metres (holding the heading), as far as the car's
        outline stays clear of tracked objects. The camera can't see behind, but the car
        just came from there. Returns metres actually reversed."""
        a = self.args
        if not self.car:
            return 0.0
        objects, _ = self.obstacles()
        allowed = safe_reverse(a, self.pose, dist, objects)
        if allowed < 0.05:
            print("  back up: no room behind (tracked objects)")
            return 0.0
        start = np.array([self.pose.X, self.pose.Y])
        hold = self.pose.theta
        t0 = time.time()
        try:
            while time.time() - t0 < 6.0:
                self.car.poll()
                self.pose.set(-self.car.left, self.car.ahead, self.car.heading())
                moved = float(np.hypot(*(np.array([self.pose.X, self.pose.Y]) - start)))
                if moved >= allowed - 0.05:  # it coasts a few cm
                    break
                steer = max(-200.0, min(200.0, -a.steer_gain * math.degrees(avoid.wrap(self.pose.theta - hold))))
                self.car.drive(-a.reverse_speed + steer, -a.reverse_speed - steer)
                time.sleep(0.02)
        finally:
            self.car.drive(0, 0)
        self.car.settle()
        self.pose.set(-self.car.left, self.car.ahead, self.car.heading())
        return float(np.hypot(*(np.array([self.pose.X, self.pose.Y]) - start)))

    def look(self, seconds):
        """Hold still and keep perceiving, so the map fills in from this viewpoint."""
        t = time.time()
        while time.time() - t < seconds:
            self.stop()
            self.sense()

    def find_a_way(self):
        """Blocked: back off a little and turn slowly, looking for a way on every frame.
        1. look; take any forward move that touches nothing (eager).
        2. turn continuously toward the course side (then the other side), re-planning from
           where the car will stop each frame; stop the moment a move exists.
        3. neither side: back up --backup-dist (at most --max-backup in all), go to 2."""
        a = self.args
        print(f"{self.el():5.1f}s BLOCKED ({'object in front' if self.hits >= 2 else self.plan_note}) -> looking for an opening")
        backed, turn_first = 0.0, False
        for attempt in range(1, a.recover_steps + 1):
            self.look(a.scan_pause)
            if self.hits >= 2:
                self.remember_front()
            if not turn_first and self.plan(eager=True) and self.hits < 2:
                print(f"  way found: {self.plan_note}")
                return True
            for sign in self.scan_sides():
                if self.scan_turn(sign):
                    return True
            if backed >= a.max_backup - 0.05:
                print(f"  no opening within {a.max_rotate:.0f} deg either side, already backed up {backed:.2f} m")
                return False
            moved = self.back_up(min(a.backup_dist, a.max_backup - backed))
            backed += moved
            print(f"  backed up {moved:.2f} m")
            if moved < 0.05:
                return False
            turn_first = True  # straight ahead from here is what just failed
        return False

    def scan_sides(self):
        """Turn directions to try (+1 left, -1 right): back toward the course direction
        first; if already along it, toward the side with more room."""
        off = math.degrees(avoid.wrap(self.pose.theta - self.course_heading))
        if abs(off) > 10:
            first = -1 if off > 0 else 1
        else:
            _, circles = self.obstacles()
            self.cmap.build(self.pose, circles)
            room = lambda sg: float(self.cmap.at(np.array([[
                self.pose.X + d * -math.sin(self.pose.theta + sg * math.radians(45)),
                self.pose.Y + d * math.cos(self.pose.theta + sg * math.radians(45))] for d in (0.6, 1.0, 1.4)])).min())
            first = 1 if room(1) >= room(-1) else -1
        return (first, -first)

    def scan_turn(self, sign):
        """Rotate in place slowly (--scan-rate deg/s), sensing every frame. Each frame, plan an
        eager move from the heading the car will stop at (coast included); stop as soon as
        one exists. Stops at --max-rotate from the course direction, or if the car's sweep
        would touch a tracked object. True when a move was found."""
        a = self.args
        if not self.car:
            return False
        side = "left" if sign > 0 else "right"
        test = Pose(a.cam_ahead)
        cmd = float(a.scan_cmd)
        hist = []
        t0 = last = time.time()
        start = self.pose.theta
        recheck_at = 0.0  # degrees turned before a found move is acted on again (after a false alarm)
        try:
            while time.time() - t0 < a.scan_timeout:
                self.sense()
                now = time.time()
                dt, last = now - last, now
                hist.append((now, self.pose.theta))
                while len(hist) > 2 and now - hist[0][0] > 0.3:
                    hist.pop(0)
                rate = (math.degrees(avoid.wrap(hist[-1][1] - hist[0][1])) / max(hist[-1][0] - hist[0][0], 1e-3)
                        if len(hist) > 1 else 0.0) * sign  # deg/s in the turning direction
                objects, circles = self.obstacles()
                self.cmap.build(self.pose, circles)
                stop_at = self.pose.theta + sign * math.radians(max(rate, 0.0) * a.turn_coast_s)
                test.set(self.pose.X, self.pose.Y, stop_at)
                poses, info = self.planner.choose(test, self.course_heading, self.lane_goal, 0.0, eager=True)
                turned = math.degrees(avoid.wrap(self.pose.theta - start))
                if poses is not None and abs(turned) >= recheck_at:
                    self.stop()
                    self.car.settle()
                    self.sense()
                    if self.plan(eager=True) and self.hits < 2:
                        print(f"  turning {side}: way found after {turned:+.0f} deg: {self.plan_note}")
                        return True
                    # coasted past it, or the front check disagrees: turn on a bit before re-checking
                    turned = math.degrees(avoid.wrap(self.pose.theta - start))
                    recheck_at = abs(turned) + 8.0
                    hist, last = [], time.time()
                off = sign * math.degrees(avoid.wrap(self.pose.theta - self.course_heading))
                if off >= a.max_rotate:
                    print(f"  turning {side}: nothing within {a.max_rotate:.0f} deg of the course ({turned:+.0f} deg turned)")
                    return False
                if abs(safe_rotation(a, self.pose, sign * 8.0, objects)) < 7.0:
                    print(f"  turning {side}: would touch an object ({turned:+.0f} deg turned)")
                    return False
                if now - t0 > 2.5 and abs(turned) < 3:
                    print(f"  turning {side}: not turning (stalled at command {cmd:.0f})")
                    return False
                # slow, steady turn: adjust the command to hold --scan-rate (in-place turns need
                # a big push to start, much less to keep going)
                cmd += a.scan_gain * (a.scan_rate - rate) * dt
                cmd = max(a.scan_cmd_min, min(a.scan_cmd_max, cmd))
                self.car.drive(-sign * cmd, sign * cmd)
            print(f"  turning {side}: gave up after {a.scan_timeout:.0f} s")
            return False
        finally:
            self.stop()
            self.car.settle()
            self.pose.set(-self.car.left, self.car.ahead, self.car.heading())

    def objects_ahead(self, dist):
        """Any confirmed object within `dist` metres ahead along the course, inside the lane?"""
        course = np.array([-math.sin(self.course_heading), math.cos(self.course_heading)])
        side = np.array([course[1], -course[0]])
        car = np.array([self.pose.X, self.pose.Y])
        for t in self.tracker.confirmed():
            d = t.pos - car
            if 0.0 < d @ course < dist and abs(d @ side) < self.centre.width / 2 + t.radius:
                return True
        return False

    def manoeuvre(self):
        """Search for a sequence of small moves through a tight spot and carry it out piece
        by piece, re-checking the rest against the updated map after each piece."""
        a = self.args
        self.look(a.scan_pause)
        _, circles = self.obstacles()
        self.cmap.build(self.pose, circles)
        moves = self.maneuvers.search(self.pose, self.course_heading)
        print(f"  search: {self.maneuvers.note}")
        if not moves or not self.car:
            return False
        # group consecutive moves of the same kind
        chunks = []
        for name, poses in moves:
            kind = "rot" if name.startswith("rot") else ("back" if name == "back" else "drive")
            if chunks and chunks[-1][0] == kind:
                chunks[-1][1].append(poses)
            else:
                chunks.append([kind, [poses]])
        print("  manoeuvre: " + ", ".join(
            f"{k} {math.degrees(sum(p[-1][2] - p[0][2] for p in ps)):+.0f}deg" if k == "rot" else
            f"{k} {sum(float(np.hypot(*(p[-1][:2] - p[0][:2]))) for p in ps):.2f}m" for k, ps in chunks))
        for kind, ps in chunks:
            if kind == "rot":
                self.rotate_to(ps[-1][-1][2])
            elif kind == "back":
                self.back_up(sum(float(np.hypot(*(p[-1][:2] - p[0][:2]))) for p in ps))
            else:
                if not self.follow(np.concatenate(ps)):
                    return False
            self.sense()
        return True

    def follow(self, poses):
        """Drive slowly along planned world poses (N, 3) (pure pursuit). Stops when the rest
        of the path, checked with the car's outline against the live map, would hit
        something, or anything is right in front (--hard-stop). True when the end is reached.
        (The straight-ahead stop check would fire on stools the curving path passes.)"""
        a = self.args
        pts = poses[:, :2]
        end = pts[-1]
        cmd = float(a.speed)
        t0 = last = time.time()
        bad = 0
        while time.time() - t0 < 8.0:
            self.sense()
            now = time.time()
            dt, last = now - last, now
            pos = np.array([self.pose.X, self.pose.Y])
            i = int(np.argmin(np.hypot(*(pts - pos).T)))
            _, circles = self.obstacles()
            self.cmap.build(self.pose, circles)
            clr = float(self.cmap.footprint(poses[i:], self.maneuvers.outline).min())
            bad = bad + 1 if clr < a.follow_min_clr else 0
            if bad >= 2 or (self.near is not None and self.near < a.hard_stop):
                self.stop()
                print(f"  {self.el():5.1f}s stopped: {'path blocked (clr %.2f)' % clr if bad >= 2 else 'object %.2f m in front' % self.near}")
                return False
            if np.hypot(*(end - pos)) < 0.08 or (end - pos) @ self.pose.fwd < 0:
                break
            aim = pursuit_point(pts, pos, a.manoeuvre_pursuit)
            to = self.pose.world_to_cam([aim])[0] + np.array([0.0, a.cam_ahead])
            err = math.degrees(math.atan2(to[0], max(to[1], 0.05)))
            steer = max(-a.steer_max, min(a.steer_max, -a.steer_gain * err))
            cmd += a.speed_gain * (a.manoeuvre_speed - self.car.speed) * dt
            cmd = max(a.speed_min, min(a.speed_max, cmd))
            self.car.drive(cmd - steer, cmd + steer)
        self.stop()
        self.car.settle()
        return True

    # --- logging
    def el(self):
        return time.time() - self.t0

    def log(self, steer=None):
        a = self.args
        el = self.el()
        if int(el * 4) != self.last_print:
            self.last_print = int(el * 4)
            conf = self.tracker.confirmed()
            print(f"{el:5.1f}s pos ({self.pose.X:+.2f},{self.pose.Y:.2f}) hdg {math.degrees(self.pose.theta):+5.1f} "
                  f"course {math.degrees(self.course_heading):+5.1f} | lanes {self.how:5s} | objects {len(conf)} {conf} | "
                  f"path {len(self.path)} pts ({self.plan_note}) | front {'clear' if self.near is None else f'{self.near:.2f}'} | tilt {self.tilt():.1f}")
        if a.save and int(el * 2) != self.last_save and hasattr(self, "_frame"):
            self.last_save = int(el * 2)
            color, red, black, lanes = self._frame
            cam = self.det.debug_image(color, red, black, lanes)[:, : color.shape[1]]
            cv2.imwrite(os.path.join(a.save, f"t{el:05.1f}.jpg"),
                        np.hstack([cam, debug_map(a, self.pose, self.tracker, self.lanemap, self.path, self.goal)]))

    # --- the run
    def run(self):
        a = self.args
        cmd = float(a.speed)
        last = time.time()
        last_plan = -1e9
        last_search = -1e9
        blocked_since = None
        while True:
            self.sense()
            now = time.time()
            dt, last = now - last, now
            if self.travelled >= a.max_dist:
                return f"drove {a.max_dist:.0f} m"
            if self.el() >= a.max_time:
                return f"{a.max_time:.0f} s limit"
            if a.dry_run and a.frames and self.n >= a.frames:
                self.plan()
                return f"{self.n} frames"
            # Objects coming up: search a manoeuvre through them before driving in, so the
            # car doesn't nose into a pocket that only the search could have avoided
            arcs_tight = self.planner.last_clr is None or self.planner.last_clr < a.search_skip_clr
            if (self.car and arcs_tight and self.travelled - last_search >= a.search_every
                    and self.objects_ahead(a.search_trigger)):
                last_search = self.travelled
                self.stop()
                print(f"{self.el():5.1f}s objects within {a.search_trigger:.1f} m ahead -> planning a manoeuvre through them")
                if self.manoeuvre():
                    last_plan, last, blocked_since = -1e9, time.time(), None
                    continue
            if now - last_plan > a.replan_s or not self.path:
                last_plan = now
                if self.plan() or self.plan(eager=True):
                    seg = np.diff(np.array(self.path[:3]), axis=0)
                    h0, h1 = (math.atan2(-d[0], d[1]) for d in seg)
                    self.prev_k = avoid.wrap(h1 - h0) / max(float(np.hypot(*seg[1])), 1e-3)

            if self.hits >= 2 or not self.path:
                if self.hits >= 2:
                    self.remember_front()
                self.stop()
                blocked_since = blocked_since or now
                if now - blocked_since > a.blocked_wait and not a.dry_run:
                    if not self.find_a_way():
                        return "no feasible path after scanning around"
                    blocked_since, last_plan, last = None, time.time(), time.time()
                continue
            blocked_since = None

            aim = pursuit_point(self.path, np.array([self.pose.X, self.pose.Y]), a.pursuit)
            to = self.pose.world_to_cam([aim])[0] + np.array([0.0, a.cam_ahead])  # from the car centre
            err = math.degrees(math.atan2(to[0], max(to[1], 0.05)))  # + = aim is to the right
            steer = max(-a.steer_max, min(a.steer_max, -a.steer_gain * err))
            if self.car:
                want = a.cruise * (0.6 if abs(err) > 25 else 1.0)  # slow down for sharp turns
                cmd += a.speed_gain * (want - self.car.speed) * dt
                cmd = max(a.speed_min, min(a.speed_max, cmd))
                self.car.drive(cmd - steer, cmd + steer)


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
    # car + safety
    ap.add_argument("--half-width", type=float, default=0.42, help="car half-width (m): 33 in wide")
    ap.add_argument("--margins", type=float, nargs="+", default=[0.05, 0.02, 0.0],
                    help="clearance margins to try in order; the first that gives a path is used (m)")
    ap.add_argument("--obstacle-margin", type=float, default=0.05, help="(map drawing only)")
    ap.add_argument("--lane-line-r", type=float, default=0.03, help="lane tape drawn this thick (m)")
    ap.add_argument("--cam-ahead", type=float, default=0.3, help="camera distance ahead of the car's turning centre (m)")
    ap.add_argument("--map-cell-m", type=float, default=0.03, help="clearance map resolution (m)")
    ap.add_argument("--map-half-span", type=float, default=4.5, help="clearance map extends this far around the car (m)")
    ap.add_argument("--plan-behind", type=float, default=0.4, help="planning window starts this far behind the car (m)")
    ap.add_argument("--plan-ahead", type=float, default=4.5)
    ap.add_argument("--prefer-room", type=float, default=0.3, help="extra room beyond the minimum that's preferred (m)")
    ap.add_argument("--room-weight", type=float, default=3.0, help="cost of having no extra room")
    ap.add_argument("--stop-dist", type=float, default=0.30, help="emergency stop: anything directly in front of the camera this close (m)")
    ap.add_argument("--half-length", type=float, default=0.48, help="car half-length (m): 38 in long")
    ap.add_argument("--sweep-margin", type=float, default=0.02, help="clearance kept when rotating in place (m)")
    ap.add_argument("--blocked-wait", type=float, default=0.3, help="blocked this long -> start looking for a way (s)")
    ap.add_argument("--scan-pause", type=float, default=0.3, help="look this long before each recovery step (s)")
    ap.add_argument("--recover-steps", type=int, default=8, help="recovery steps before giving up")
    ap.add_argument("--recover-turn", type=float, default=15.0, help="recovery: rotations are tried in steps of this (deg)")
    # manoeuvre search (tight spots)
    ap.add_argument("--search-step", type=float, default=0.2, help="length of each small move (m)")
    ap.add_argument("--search-rot", type=float, default=15.0, help="rotation in place per move (deg)")
    ap.add_argument("--search-cell", type=float, default=0.05, help="position resolution of the search (m)")
    ap.add_argument("--search-ahead", type=float, default=4.0, help="search until this far further along the course (m)")
    ap.add_argument("--search-heading", type=float, default=30.0, help="...facing within this of the course direction (deg)")
    ap.add_argument("--search-time", type=float, default=15.0, help="give up searching after this long (s)")
    ap.add_argument("--search-skip-clr", type=float, default=0.10,
                    help="no pre-entry search while the smooth arc plan keeps this much clearance (m)")
    ap.add_argument("--follow-min-clr", type=float, default=-0.02,
                    help="manoeuvre: stop when the rest of the path has less clearance than this (m)")
    ap.add_argument("--hard-stop", type=float, default=0.25, help="manoeuvre: stop for anything this close in front (m)")
    ap.add_argument("--search-trigger", type=float, default=0.0,
                    help="search a long manoeuvre when objects are this close ahead (m); 0 = off (reactive only)")
    ap.add_argument("--scan-rate", type=float, default=15.0, help="recovery: turn this slowly while looking (deg/s)")
    ap.add_argument("--scan-cmd", type=float, default=800.0, help="recovery: starting L/R command for the slow turn")
    ap.add_argument("--scan-cmd-min", type=float, default=550.0)
    ap.add_argument("--scan-cmd-max", type=float, default=950.0)
    ap.add_argument("--scan-gain", type=float, default=8.0, help="command change per (deg/s error x s)")
    ap.add_argument("--scan-timeout", type=float, default=15.0, help="recovery: longest one slow turn may take (s)")
    ap.add_argument("--bump-r", type=float, default=0.12, help="radius of a remembered front-stop obstacle (m)")
    ap.add_argument("--max-rotate", type=float, default=90.0, help="recovery: largest rotation in place considered (deg)")
    ap.add_argument("--max-backup", type=float, default=0.4, help="recovery: back up at most this far in all (m)")
    ap.add_argument("--eager-margin", type=float, default=0.01, help="stuck: clearance a move still needs (m)")
    ap.add_argument("--eager-progress", type=float, default=0.05, help="stuck: a move need only gain this much (m)")
    ap.add_argument("--tape-object-gap", type=float, default=0.15, help="ignore tape this close to an object's edge (m)")
    ap.add_argument("--search-every", type=float, default=2.0, help="at most one such search per this many metres driven")
    ap.add_argument("--manoeuvre-speed", type=float, default=0.12, help="m/s while carrying out a manoeuvre")
    ap.add_argument("--manoeuvre-pursuit", type=float, default=0.25, help="pursuit lookahead during manoeuvres (m)")
    # arc planner
    ap.add_argument("--min-turn-radius", type=float, default=0.8, help="tightest planned turn (m)")
    ap.add_argument("--arc-count", type=int, default=15, help="curvatures sampled")
    ap.add_argument("--arc-lengths", type=float, nargs="+", default=[0.6, 1.2, 1.8])
    ap.add_argument("--arc-margin", type=float, default=0.03, help="car outline must stay this clear of everything (m)")
    ap.add_argument("--prefer-clear", type=float, default=0.25, help="clearance beyond which there's no extra credit (m)")
    ap.add_argument("--min-progress", type=float, default=0.15, help="a move must gain this much along the course (m)")
    ap.add_argument("--heading-free", type=float, default=20.0, help="no penalty within this angle of the course direction (deg)")
    ap.add_argument("--w-progress", type=float, default=1.0)
    ap.add_argument("--w-heading", type=float, default=1.0)
    ap.add_argument("--w-clear", type=float, default=0.4)
    ap.add_argument("--w-smooth", type=float, default=0.3)
    ap.add_argument("--w-lane", type=float, default=0.3)
    ap.add_argument("--turn-speed", type=int, default=800, help="L/R command for rotating in place")
    ap.add_argument("--turn-coast-s", type=float, default=0.45, help="the car keeps rotating ~this long after STOP")
    ap.add_argument("--course-smooth", type=float, default=0.1, help="how fast the course direction follows the lanes")
    ap.add_argument("--course-max-angle", type=float, default=30.0, help="lane pieces this far off the course direction don't steer it (deg)")
    ap.add_argument("--backups", type=int, default=3, help="when stuck, back up this many times at most")
    ap.add_argument("--backup-dist", type=float, default=0.2, help="metres per back-up")
    ap.add_argument("--reverse-speed", type=int, default=650, help="L/R command when reversing")
    ap.add_argument("--max-dist", type=float, default=15.0)
    ap.add_argument("--max-time", type=float, default=120.0)
    # driving
    ap.add_argument("--cruise", type=float, default=0.15, help="m/s")
    ap.add_argument("--speed", type=int, default=550, help="starting forward L/R command")
    ap.add_argument("--speed-min", type=int, default=450)
    ap.add_argument("--speed-max", type=int, default=850)
    ap.add_argument("--speed-gain", type=float, default=400.0)
    ap.add_argument("--pursuit", type=float, default=0.4, help="pure-pursuit lookahead (m)")
    ap.add_argument("--steer-gain", type=float, default=9.0, help="L/R difference per degree off the path")
    ap.add_argument("--steer-max", type=float, default=350.0)
    ap.add_argument("--replan-s", type=float, default=0.0, help="replan this often (s); 0 = every frame")
    # goal / lanes
    ap.add_argument("--goal-ahead", type=float, default=2.5, help="plan to the lane centre this far ahead (m)")
    ap.add_argument("--lookahead", type=float, default=2.5, help="(for LaneCentre) same as --goal-ahead")
    ap.add_argument("--tape", choices=("black", "red-black"), default="black",
                    help="black: both lanes black (sides by position); red-black: red left, black right")
    ap.add_argument("--left-lane", choices=("red", "black"), default="red", help="(red-black tape only)")
    ap.add_argument("--lane-width", type=float, default=2.0)
    ap.add_argument("--width-smooth", type=float, default=0.1)
    ap.add_argument("--target-smooth", type=float, default=0.5)
    ap.add_argument("--lane-cell-m", type=float, default=0.1)
    ap.add_argument("--lane-min-seen", type=int, default=2, help="frames a lane cell must be seen in")
    ap.add_argument("--wall-thickness", type=float, default=0.8, help="lane wall extends this far outward (m)")
    ap.add_argument("--wall-step", type=float, default=0.2)
    ap.add_argument("--start-centre", type=float, default=0.0, help="lane centre at the start, metres right of the car")
    ap.add_argument("--start-corridor", type=float, default=2.5, help="assume straight lanes this far from the start (m)")
    ap.add_argument("--extend-max-angle", type=float, default=35.0, help="only extend lane pieces this close to straight ahead (deg)")
    ap.add_argument("--lane-extend", type=float, default=1.5, help="extend seen lane pieces this far both ways (m)")
    # object tracking
    ap.add_argument("--track-cell-m", type=float, default=0.05)
    ap.add_argument("--points-per-cell", type=int, default=2)
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--max-object-radius", type=float, default=0.8)
    ap.add_argument("--min-object-radius", type=float, default=0.30,
                    help="objects are at least this big: from 1 m up the seat hides a stool's wider base (m)")
    ap.add_argument("--match-dist", type=float, default=0.5)
    ap.add_argument("--track-smooth", type=float, default=0.4)
    ap.add_argument("--confirm", type=int, default=3)
    ap.add_argument("--drop-misses", type=int, default=10, help="confirmed track: frames unseen while in view (or beside the car)")
    ap.add_argument("--drop-misses-new", type=int, default=3)
    ap.add_argument("--forget-behind", type=float, default=1.5, help="drop tracks this far behind the camera (m)")
    ap.add_argument("--beside-clear", type=float, default=0.15,
                    help="outside the car half-width by this much starts the beside-expire band (m)")
    ap.add_argument("--beside-band", type=float, default=1.5,
                    help="width of the beside-expire band beyond --beside-clear (m)")
    ap.add_argument("--beside-along", type=float, default=0.5,
                    help="beside-expire band extends this far past the car half-length (m)")
    ap.add_argument("--overlap-ok", type=float, default=0.05,
                    help="drop a track whose disk overlaps the car rectangle more than this (m)")
    ap.add_argument("--memory-s", type=float, default=20.0,
                    help="confirmed track: max time held out of view without a re-sight (s)")
    ap.add_argument("--memory-s-new", type=float, default=2.0,
                    help="unconfirmed track: max time held out of view (s)")
    ap.add_argument("--view-near", type=float, default=0.7)
    ap.add_argument("--view-far", type=float, default=3.5)
    ap.add_argument("--view-half-angle", type=float, default=42.0,
                    help="miss-count cone half-angle; tracks outside are kept until re-seen (deg)")
    ap.add_argument("--obstacle-range", type=float, default=4.0, help="track objects up to this far (m)")
    ap.add_argument("--m-per-count", type=float, default=0.0075)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="dry run: stop after N frames")
    ap.add_argument("--save", help="directory for debug images (twice a second)")
    args = ap.parse_args()
    args.lookahead = args.goal_ahead
    if args.save:  # keep the text log with the images
        os.makedirs(args.save, exist_ok=True)
        sys.stdout = Tee(sys.stdout, open(os.path.join(args.save, "log.txt"), "w"))
    course = Course(args)
    try:
        course.calibrate()
        why = course.run()
        if course.car:
            course.car.settle()
            course.pose.set(-course.car.left, course.car.ahead, course.car.heading())
        print(f"STOP: {why}; final pos ({course.pose.X:+.2f},{course.pose.Y:.2f}) heading "
              f"{math.degrees(course.pose.theta):+.1f} deg; travelled {course.travelled:.2f} m; "
              f"objects tracked: {course.tracker.confirmed()}")
        if args.save:
            cv2.imwrite(os.path.join(args.save, "final_map.jpg"),
                        debug_map(args, course.pose, course.tracker, course.lanemap, course.path, course.goal))
    finally:
        course.close()


if __name__ == "__main__":
    main()
