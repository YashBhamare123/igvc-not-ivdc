#!/usr/bin/env python3
"""World-frame state of obstacles and lane tape for the taped course.

Why this exists
---------------
Earlier pipelines either:
  * kept hazards only in the live camera frame and forgot them after a few frames
    (navigator / hazard_detector), so stools that left the FOV vanished from the plan, or
  * stored world tracks but treated "in view" as a cone narrower than the detector, so
    side detections never aged and coasted forever as ghosts (course.Tracker).

Rules used here
---------------
1. Canonical store is the world map (start-anchored: X right, Y ahead). Pose from
   hall counts + IMU moves the camera through that map; tracks do not slide with the image.
2. The replace frustum matches the depth detector (range × lateral half-width), not a
   tighter "miss" cone. Inside it, live detections own the state; unmatched confirmed
   tracks take misses and free-space clears. Outside it, confirmed tracks coast.
3. Coasting has a time TTL and a range limit from the car, and anything that ends up
   overlapping the car footprint is dropped (person walked past / bad near hit).
4. Floor points seen empty inside a track's footprint are negative evidence.
5. Lane cells use the same replace / coast / TTL idea, plus bare-floor decay in view.
6. Front-stop bumps share the object store with a TTL instead of living forever.

Pose duck-type: needs .cam_to_world((N,2)), .world_to_cam((N,2)), .X, .Y, .fwd, .right,
and .cam (camera origin in world).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

LANE_SIDE = {"red": "left", "black": "right"}


# ---------------------------------------------------------------- config / state


@dataclass
class TrackerConfig:
    # blob clustering (camera → world)
    track_cell_m: float = 0.05
    points_per_cell: int = 2
    min_points: int = 30
    min_object_radius: float = 0.25
    max_object_radius: float = 0.8
    # association
    match_dist: float = 0.55
    track_smooth: float = 0.45
    confirm_hits: int = 3
    # detection frustum (= replace / miss region). Must match scan corridor.
    view_near: float = 0.35
    view_far: float = 4.0
    view_half_width: float = 3.0  # metres lateral; preferred over a tight angle
    view_half_angle: float = 50.0  # deg fallback if a point is past half_width check edge
    # life cycle
    drop_misses: int = 8
    drop_misses_new: int = 3
    coast_s: float = 6.0
    coast_range: float = 3.5
    forget_behind: float = 1.2
    # free space (floor seen empty inside a track)
    free_band_m: float = 0.04
    free_clear_frames: int = 4
    free_min_points: int = 12
    # car footprint cull
    half_width: float = 0.42
    half_length: float = 0.48
    overlap_ok: float = 0.05
    # bumps (front e-stop pins)
    bump_r: float = 0.12
    bump_ttl_s: float = 8.0
    bump_merge_m: float = 0.25
    # lanes
    lane_cell_m: float = 0.1
    lane_min_seen: int = 2
    lane_coast_s: float = 10.0
    lane_coast_range: float = 5.0
    lane_line_r: float = 0.03
    wall_thickness: float = 0.8
    wall_step: float = 0.2
    start_centre: float = 0.0
    start_corridor: float = 2.5
    lane_width: float = 2.0
    lane_extend: float = 1.5
    extend_max_angle: float = 35.0
    blur_yaw_rate: float = 0.6  # rad/s: skip learning lanes while spinning
    # skip associating while yaw rate this high (depth smear)
    blur_skip_objects: float = 1.2


@dataclass
class ObjectState:
    id: int
    pos: np.ndarray  # world (2,)
    radius: float
    hits: int = 1
    misses: int = 0
    free_hits: int = 0
    last_seen: float = 0.0
    in_fov: bool = True
    source: str = "depth"  # depth | bump
    status: str = "tentative"  # tentative | confirmed | coasting

    def __repr__(self):
        return (f"#{self.id}({self.pos[0]:+.2f},{self.pos[1]:.2f} "
                f"r{self.radius:.2f} {self.status})")


@dataclass
class _LaneCell:
    count: int = 0
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    last_seen: float = 0.0


# ---------------------------------------------------------------- tracker


class StateTracker:
    """Fuses depth blobs + lane segments into world object / lane state."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.tracks: list[ObjectState] = []
        self._next_id = 1
        self.cells = {"red": {}, "black": {}}  # (i,j) -> _LaneCell
        self.latest = {"red": [], "black": []}
        self.latest_at = {"red": None, "black": None}
        self.now = 0.0

    # --- public API used by Course -------------------------------------------

    def confirmed(self) -> list[ObjectState]:
        return [t for t in self.tracks if t.hits >= self.cfg.confirm_hits or t.source == "bump"]

    def obstacle_circles(self) -> np.ndarray:
        """(N, 3) world [x, y, r] for planning: confirmed objects + live bumps."""
        rows = [(t.pos[0], t.pos[1], t.radius) for t in self.confirmed()]
        return np.array(rows, float).reshape(-1, 3)

    def remember_bump(self, world_xy, now: Optional[float] = None):
        """Pin a front e-stop hit into the map (TTL). Merges with a nearby track if any."""
        a = self.cfg
        now = self.now if now is None else now
        p = np.asarray(world_xy, float).reshape(2)
        for t in self.tracks:
            if float(np.hypot(*(t.pos - p))) < a.bump_merge_m + t.radius * 0.5:
                t.pos = t.pos + 0.5 * (p - t.pos)
                t.radius = max(t.radius, a.bump_r)
                t.hits = max(t.hits, a.confirm_hits)
                t.misses = 0
                t.free_hits = 0
                t.last_seen = now
                t.source = "bump"
                t.status = "confirmed"
                return t
        t = ObjectState(
            id=self._alloc_id(), pos=p.copy(), radius=a.bump_r,
            hits=a.confirm_hits, last_seen=now, source="bump", status="confirmed",
        )
        self.tracks.append(t)
        return t

    def update(
        self,
        pose,
        now: float,
        obstacle_f=None,
        obstacle_lat=None,
        floor_f=None,
        floor_lat=None,
        lanes=None,
        yaw_rate: float = 0.0,
        *,
        update_objects: bool = True,
        update_lanes: bool = True,
    ):
        """One frame: cluster depth → associate → coast / clear; update lane cells.

        `update_objects` / `update_lanes` let the caller refresh depth tracks first, then
        filter tape against those tracks, then commit lanes (Course.sense does this).
        """
        self.now = now
        a = self.cfg

        if update_objects and obstacle_f is not None and obstacle_lat is not None:
            spinning = abs(yaw_rate) > a.blur_skip_objects
            dets = [] if spinning else self.cluster_detections(obstacle_f, obstacle_lat, pose)
            self._associate(dets, pose, now)
            if floor_f is not None and floor_lat is not None and len(floor_f):
                self._clear_with_floor(floor_f, floor_lat, pose)
            self._cull(pose, now)

        if update_lanes and lanes is not None and abs(yaw_rate) <= a.blur_yaw_rate:
            self._update_lanes(lanes, pose, now)
        if update_lanes:
            self._decay_lanes(pose, now, floor_f, floor_lat)

    # --- detection / FOV -----------------------------------------------------

    def cluster_detections(self, f, lat, pose) -> list[tuple[np.ndarray, float]]:
        a = self.cfg
        f = np.asarray(f, float).reshape(-1)
        lat = np.asarray(lat, float).reshape(-1)
        if f.size < a.min_points:
            return []
        world = pose.cam_to_world(np.stack([lat, f], axis=1))
        cell = a.track_cell_m
        ij = np.floor(world / cell).astype(int)
        lo = ij.min(axis=0)
        ij = ij - lo
        h, w = int(ij[:, 1].max()) + 1, int(ij[:, 0].max()) + 1
        grid = np.zeros((h, w), np.int32)
        np.add.at(grid, (ij[:, 1], ij[:, 0]), 1)
        occ = (grid >= a.points_per_cell).astype(np.uint8)
        occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))
        n, labels = cv2.connectedComponents(occ, connectivity=8)
        lab_of_pt = labels[ij[:, 1], ij[:, 0]]
        found = []
        for k in range(1, n):
            p = world[lab_of_pt == k]
            if len(p) < a.min_points:
                continue
            c = p.mean(axis=0)
            r = float(np.percentile(np.hypot(*(p - c).T), 98)) + cell / 2
            r = min(max(r, a.min_object_radius), a.max_object_radius)
            found.append((c, r))
        return found

    def in_frustum(self, world_xy, pose) -> bool:
        """True if this world point sits inside the depth detector's replace region."""
        a = self.cfg
        x, y = pose.world_to_cam(np.asarray(world_xy, float).reshape(1, 2))[0]
        if not (a.view_near < y < a.view_far):
            return False
        if abs(x) > a.view_half_width:
            return False
        # also require a plausible bearing so far-corner cells are not "in view"
        if y > 1e-3 and abs(math.degrees(math.atan2(x, y))) > a.view_half_angle:
            return False
        return True

    def in_frustum_many(self, world_xy, pose) -> np.ndarray:
        pts = np.asarray(world_xy, float).reshape(-1, 2)
        if len(pts) == 0:
            return np.zeros(0, bool)
        cam = pose.world_to_cam(pts)
        x, y = cam[:, 0], cam[:, 1]
        a = self.cfg
        ang = np.degrees(np.arctan2(x, np.maximum(y, 1e-6)))
        return (
            (y > a.view_near) & (y < a.view_far)
            & (np.abs(x) <= a.view_half_width)
            & (np.abs(ang) <= a.view_half_angle)
        )

    # --- object association / life cycle -------------------------------------

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _associate(self, dets, pose, now: float):
        a = self.cfg
        for t in self.tracks:
            t.in_fov = self.in_frustum(t.pos, pose)

        if not self.tracks and not dets:
            return
        if not self.tracks:
            for c, r in dets:
                self.tracks.append(ObjectState(
                    id=self._alloc_id(), pos=np.array(c, float), radius=float(r),
                    last_seen=now, in_fov=True, status="tentative",
                ))
            return
        if not dets:
            for t in self.tracks:
                if t.in_fov:
                    t.misses += 1
                    t.status = "tentative" if t.hits < a.confirm_hits else "confirmed"
                else:
                    t.status = "coasting" if t.hits >= a.confirm_hits else "tentative"
            return

        n_t, n_d = len(self.tracks), len(dets)
        cost = np.full((n_t, n_d), 1e3)
        for i, t in enumerate(self.tracks):
            gate = a.match_dist + t.radius
            for j, (c, r) in enumerate(dets):
                d = float(np.hypot(*(c - t.pos)))
                if d < gate:
                    cost[i, j] = d
        ri, ci = linear_sum_assignment(cost)
        matched_t, matched_d = set(), set()
        for i, j in zip(ri, ci):
            if cost[i, j] >= 1e3 - 1:
                continue
            t = self.tracks[i]
            c, r = dets[j]
            t.pos = t.pos + a.track_smooth * (c - t.pos)
            # soften radius growth; do not ratchet up forever on one fat frame
            target_r = 0.7 * r + 0.3 * t.radius
            t.radius += a.track_smooth * (target_r - t.radius)
            t.radius = min(max(t.radius, a.min_object_radius), a.max_object_radius)
            t.hits += 1
            t.misses = 0
            t.free_hits = 0
            t.last_seen = now
            t.in_fov = True
            t.source = "depth"
            t.status = "confirmed" if t.hits >= a.confirm_hits else "tentative"
            matched_t.add(i)
            matched_d.add(j)

        for i, t in enumerate(self.tracks):
            if i in matched_t:
                continue
            if t.in_fov:
                t.misses += 1
                if t.hits >= a.confirm_hits:
                    t.status = "confirmed"
            else:
                t.status = "coasting" if t.hits >= a.confirm_hits else "tentative"

        for j, (c, r) in enumerate(dets):
            if j not in matched_d:
                self.tracks.append(ObjectState(
                    id=self._alloc_id(), pos=np.array(c, float), radius=float(r),
                    last_seen=now, in_fov=True, status="tentative",
                ))

    def _clear_with_floor(self, floor_f, floor_lat, pose):
        """Negative evidence: empty floor inside a track that is currently in view."""
        a = self.cfg
        if len(floor_f) < a.free_min_points:
            return
        floor_w = pose.cam_to_world(np.stack([floor_lat, floor_f], axis=1))
        for t in self.tracks:
            if not t.in_fov or t.hits < 2:
                continue
            # only count floor near the object's centre (not the whole corridor)
            d = np.hypot(*(floor_w - t.pos).T)
            inside = d < max(0.12, t.radius * 0.65)
            if int(inside.sum()) >= a.free_min_points:
                t.free_hits += 1
            else:
                t.free_hits = max(0, t.free_hits - 1)

    def _cull(self, pose, now: float):
        a = self.cfg
        car = np.array([pose.X, pose.Y], float)
        kept = []
        for t in self.tracks:
            body = t.pos - car
            bx, by = float(body @ pose.right), float(body @ pose.fwd)
            # overlapping the chassis → not a fixed obstacle anymore (walked past / bad near hit)
            if max(abs(bx) - a.half_width, 0.0) ** 2 + max(abs(by) - a.half_length, 0.0) ** 2 < (
                max(t.radius - a.overlap_ok, 0.0) ** 2
            ):
                continue
            # Only forget "behind" when we have actually driven past it on our path —
            # not when it sits beside us just aft of the camera plane (that must coast).
            rear_of_car = by < -(a.half_length + 0.1)
            in_rear_corridor = abs(bx) < a.half_width + t.radius
            if rear_of_car and in_rear_corridor:
                continue
            cam = pose.world_to_cam(t.pos)[0]
            if cam[1] < -a.forget_behind and abs(cam[0]) < a.half_width + t.radius:
                continue
            # free-space killed it
            if t.free_hits >= a.free_clear_frames:
                continue
            # in FOV: miss budget
            miss_lim = a.drop_misses if t.hits >= a.confirm_hits else a.drop_misses_new
            if t.in_fov and t.misses >= miss_lim:
                continue
            # out of FOV: only confirmed (or bumps) may coast, and only with TTL / range
            if not t.in_fov:
                if t.hits < a.confirm_hits and t.source != "bump":
                    continue
                age = now - t.last_seen
                ttl = a.bump_ttl_s if t.source == "bump" else a.coast_s
                if age > ttl:
                    continue
                if float(np.hypot(*(t.pos - car))) > a.coast_range:
                    continue
                t.status = "coasting"
            kept.append(t)
        self.tracks = kept

    # --- lanes ---------------------------------------------------------------

    def _update_lanes(self, lanes: dict, pose, now: float):
        a = self.cfg
        c = a.lane_cell_m
        for colour, segs in lanes.items():
            if colour not in self.cells:
                continue
            if segs:
                self.latest[colour] = [
                    (pose.cam_to_world([sg.start])[0], pose.cam_to_world([sg.end])[0])
                    for sg in segs
                ]
                self.latest_at[colour] = pose.Y
            for sg in segs:
                n = max(2, int(sg.length / c) + 1)
                pts = pose.cam_to_world(np.linspace(sg.start, sg.end, n))
                d = pose.cam_to_world([sg.end])[0] - pose.cam_to_world([sg.start])[0]
                d = d / max(float(np.hypot(*d)), 1e-6)
                for p in pts:
                    key = (int(p[0] // c), int(p[1] // c))
                    cell = self.cells[colour].get(key)
                    if cell is None:
                        cell = _LaneCell(count=0, direction=d.copy(), last_seen=now)
                        self.cells[colour][key] = cell
                    cell.count = min(cell.count + 1, 20)
                    cell.direction = d
                    cell.last_seen = now

    def _decay_lanes(self, pose, now: float, floor_f, floor_lat):
        """In frustum: bare floor fades tape; everywhere: TTL / range drop."""
        a = self.cfg
        c = a.lane_cell_m
        car = np.array([pose.X, pose.Y], float)

        # bare floor under a cell that is in view → decay count
        if floor_f is not None and len(floor_f):
            floor_w = pose.cam_to_world(np.stack([floor_lat, floor_f], axis=1))
            # subsample for speed
            if len(floor_w) > 800:
                floor_w = floor_w[:: max(1, len(floor_w) // 800)]
            keys_hit = set()
            for p in floor_w:
                keys_hit.add((int(p[0] // c), int(p[1] // c)))
            fov_keys = {
                k for colour in self.cells for k in self.cells[colour]
                if self.in_frustum(((k[0] + 0.5) * c, (k[1] + 0.5) * c), pose)
            }
            for colour in self.cells:
                for key in list(self.cells[colour]):
                    if key in fov_keys and key in keys_hit:
                        # only decay if we did not reinforce this frame
                        cell = self.cells[colour][key]
                        if now - cell.last_seen > 1e-3:
                            cell.count = max(0, cell.count - 1)

        for colour in self.cells:
            keep = {}
            for key, cell in self.cells[colour].items():
                p = np.array([(key[0] + 0.5) * c, (key[1] + 0.5) * c])
                cam = pose.world_to_cam(p)[0]
                if cam[1] < -a.forget_behind:
                    continue
                if cell.count <= 0:
                    continue
                in_view = self.in_frustum(p, pose)
                if in_view:
                    keep[key] = cell
                    continue
                if now - cell.last_seen > a.lane_coast_s:
                    continue
                if float(np.hypot(*(p - car))) > a.lane_coast_range:
                    continue
                keep[key] = cell
            self.cells[colour] = keep

    def walls(self, lane_width=None, near=None, radius=None) -> np.ndarray:
        """Lane wall points as (N, 3) [x, y, r] for the clearance map."""
        a = self.cfg
        c = a.lane_cell_m
        lane_width = a.lane_width if lane_width is None else lane_width
        ks = np.arange(0.0, a.wall_thickness + 1e-6, a.wall_step)
        chunks = []

        def wall(P, D, colour):
            if len(P) == 0:
                return
            out = (
                np.stack([-D[:, 1], D[:, 0]], 1)
                if LANE_SIDE[colour] == "left"
                else np.stack([D[:, 1], -D[:, 0]], 1)
            )
            chunks.append((P[:, None, :] + ks[None, :, None] * out[:, None, :]).reshape(-1, 2))

        for colour, cells in self.cells.items():
            seen = [(k, cell.direction) for k, cell in cells.items() if cell.count >= a.lane_min_seen]
            if seen:
                P = (np.array([k for k, _ in seen], float) + 0.5) * c
                wall(P, np.array([d for _, d in seen]), colour)

        if a.start_corridor > 0:
            ys = np.arange(-0.5, a.start_corridor + 1e-6, c)
            for colour, side in LANE_SIDE.items():
                seen = [
                    ((i + 0.5) * c, (j + 0.5) * c)
                    for (i, j), cell in self.cells[colour].items()
                    if cell.count >= a.lane_min_seen and -0.5 <= (j + 0.5) * c <= a.start_corridor
                ]
                if seen:
                    x, y_end = min(seen, key=lambda p: p[1])
                else:
                    x = a.start_centre + (lane_width / 2 if side == "right" else -lane_width / 2)
                    y_end = a.start_corridor
                yy = ys[ys < y_end]
                wall(np.stack([np.full_like(yy, x), yy], 1), np.tile([0.0, 1.0], (len(yy), 1)), colour)

        along = lambda seg: abs(math.degrees(math.atan2(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]))) <= a.extend_max_angle
        latest = {colour: [sg for sg in segs if along(sg)] for colour, segs in self.latest.items()}
        if lane_width:
            for colour, other in (("red", "black"), ("black", "red")):
                if not latest[colour] and latest[other]:
                    shifted = []
                    for p0, p1 in latest[other]:
                        d = (p1 - p0) / max(float(np.hypot(*(p1 - p0))), 1e-6)
                        inward = (
                            np.array([d[1], -d[0]]) if LANE_SIDE[other] == "left"
                            else np.array([-d[1], d[0]])
                        )
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
                    ts = ts[(ts < 0) | (ts > length)]
                wall(p0 + ts[:, None] * d, np.tile(d, (len(ts), 1)), colour)

        if not chunks:
            return np.zeros((0, 3))
        pts = np.concatenate(chunks)
        if near is not None:
            pts = pts[np.hypot(pts[:, 0] - near[0], pts[:, 1] - near[1]) < radius]
        return np.column_stack([pts, np.full(len(pts), a.lane_line_r)])


def config_from_args(args) -> TrackerConfig:
    """Build TrackerConfig from course.py argparse namespace."""
    return TrackerConfig(
        track_cell_m=args.track_cell_m,
        points_per_cell=args.points_per_cell,
        min_points=args.min_points,
        min_object_radius=args.min_object_radius,
        max_object_radius=args.max_object_radius,
        match_dist=args.match_dist,
        track_smooth=args.track_smooth,
        confirm_hits=args.confirm,
        view_near=args.view_near,
        view_far=args.view_far,
        view_half_width=args.view_half_width,
        view_half_angle=args.view_half_angle,
        drop_misses=args.drop_misses,
        drop_misses_new=args.drop_misses_new,
        coast_s=args.coast_s,
        coast_range=args.coast_range,
        forget_behind=args.forget_behind,
        free_clear_frames=args.free_clear_frames,
        half_width=args.half_width,
        half_length=args.half_length,
        bump_r=args.bump_r,
        bump_ttl_s=args.bump_ttl_s,
        lane_cell_m=args.lane_cell_m,
        lane_min_seen=args.lane_min_seen,
        lane_coast_s=args.lane_coast_s,
        lane_line_r=args.lane_line_r,
        wall_thickness=args.wall_thickness,
        wall_step=args.wall_step,
        start_centre=args.start_centre,
        start_corridor=args.start_corridor,
        lane_width=args.lane_width,
        lane_extend=args.lane_extend,
        extend_max_angle=args.extend_max_angle,
    )


def floor_points(points, normal, offset, fwd, right, half_width, max_range, band=0.04):
    """(forward, lateral) of points that look like open floor (for negative evidence)."""
    h = points @ normal + offset
    f = points @ fwd
    lat = points @ right
    m = (
        (points[..., 2] > 0)
        & (np.abs(h) < band)
        & (np.abs(lat) < half_width)
        & (f > 0.25)
        & (f < max_range)
    )
    return f[m], lat[m]
