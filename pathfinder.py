#!/usr/bin/env python3
"""Grid A* pathfinder around circular hazards (metres, ground frame).

Ground frame matches hazard_detector: origin under the camera, x right, y forward.
Hazard.radius already includes HazardConfig.safety_margin_m (robot half-width);
this planner only adds a small extra clearance.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PathfinderConfig:
    cell_m: float = 0.10
    x_min_m: float = -5.0
    x_max_m: float = 5.0
    y_min_m: float = -0.5  # slight rear so the start cell is inside the map
    y_max_m: float = 12.0  # past a 10 m forward goal
    extra_clearance_m: float = 0.05
    allow_diagonal: bool = True
    # If the start sits inside an inflated hazard (noise / close object), carve a
    # temporary free bubble so planning can still produce an escape path.
    start_clear_radius_m: float = 0.25


@dataclass
class PathResult:
    points: list[tuple[float, float]]  # metres, ground frame; includes start
    reached_goal: bool
    blocked: bool  # True when no path exists


class Pathfinder:
    def __init__(self, cfg: PathfinderConfig | None = None):
        self.cfg = cfg or PathfinderConfig()
        c = self.cfg
        self.nx = int(math.ceil((c.x_max_m - c.x_min_m) / c.cell_m))
        self.ny = int(math.ceil((c.y_max_m - c.y_min_m) / c.cell_m))
        self._neigh = self._build_neighbors()

    def _build_neighbors(self):
        steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
        if self.cfg.allow_diagonal:
            d = math.sqrt(2.0)
            steps += [(-1, -1, d), (-1, 1, d), (1, -1, d), (1, 1, d)]
        return steps

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        c = self.cfg
        ix = int((x - c.x_min_m) / c.cell_m)
        iy = int((y - c.y_min_m) / c.cell_m)
        return ix, iy

    def cell_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        c = self.cfg
        return (c.x_min_m + (ix + 0.5) * c.cell_m, c.y_min_m + (iy + 0.5) * c.cell_m)

    def _in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def build_occupancy(self, hazards) -> np.ndarray:
        """True = blocked. Inflates each hazard by extra_clearance_m only."""
        c = self.cfg
        occ = np.zeros((self.ny, self.nx), dtype=bool)
        if not hazards:
            return occ
        # Cell centres
        xs = c.x_min_m + (np.arange(self.nx) + 0.5) * c.cell_m
        ys = c.y_min_m + (np.arange(self.ny) + 0.5) * c.cell_m
        xx, yy = np.meshgrid(xs, ys)
        for h in hazards:
            r = h.radius + c.extra_clearance_m
            occ |= (xx - h.x) ** 2 + (yy - h.y) ** 2 <= r * r
        return occ

    def _nearest_free(self, occ: np.ndarray, ix: int, iy: int) -> tuple[int, int] | None:
        if self._in_bounds(ix, iy) and not occ[iy, ix]:
            return ix, iy
        best = None
        best_d = None
        for r in range(1, max(self.nx, self.ny)):
            for dy in range(-r, r + 1):
                for dx in (-r, r):
                    jx, jy = ix + dx, iy + dy
                    if self._in_bounds(jx, jy) and not occ[jy, jx]:
                        d = dx * dx + dy * dy
                        if best_d is None or d < best_d:
                            best, best_d = (jx, jy), d
            for dx in range(-r + 1, r):
                for dy in (-r, r):
                    jx, jy = ix + dx, iy + dy
                    if self._in_bounds(jx, jy) and not occ[jy, jx]:
                        d = dx * dx + dy * dy
                        if best_d is None or d < best_d:
                            best, best_d = (jx, jy), d
            if best is not None:
                return best
        return None

    def plan(self, start: tuple[float, float], goal: tuple[float, float], hazards) -> PathResult:
        c = self.cfg
        occ = self.build_occupancy(hazards)

        sx, sy = start
        # Clear a small bubble around the robot so tracking noise cannot trap it
        clear_r = c.start_clear_radius_m
        if clear_r > 0:
            xs = c.x_min_m + (np.arange(self.nx) + 0.5) * c.cell_m
            ys = c.y_min_m + (np.arange(self.ny) + 0.5) * c.cell_m
            xx, yy = np.meshgrid(xs, ys)
            occ &= (xx - sx) ** 2 + (yy - sy) ** 2 > clear_r * clear_r

        six, siy = self.world_to_cell(*start)
        gix, giy = self.world_to_cell(*goal)
        # Clamp out-of-map cells to the border so a turned goal still plans
        six = max(0, min(self.nx - 1, six))
        siy = max(0, min(self.ny - 1, siy))
        gix = max(0, min(self.nx - 1, gix))
        giy = max(0, min(self.ny - 1, giy))

        start_cell = self._nearest_free(occ, six, siy)
        goal_cell = self._nearest_free(occ, gix, giy)
        if start_cell is None or goal_cell is None:
            return PathResult([start], reached_goal=False, blocked=True)

        path_cells = self._astar(occ, start_cell, goal_cell)
        if path_cells is None:
            return PathResult([start], reached_goal=False, blocked=True)

        points = [start] + [self.cell_to_world(ix, iy) for ix, iy in path_cells[1:]]
        # Snap final point to the exact goal when the goal cell is free enough
        if not occ[goal_cell[1], goal_cell[0]]:
            points[-1] = goal
        points = self.simplify(points, hazards)
        reached = math.hypot(points[-1][0] - goal[0], points[-1][1] - goal[1]) < c.cell_m
        return PathResult(points, reached_goal=reached, blocked=False)

    def _astar(self, occ, start_cell, goal_cell):
        six, siy = start_cell
        gix, giy = goal_cell
        goal_i = giy * self.nx + gix

        def h(ix, iy):
            return math.hypot(ix - gix, iy - giy)

        open_h = [(h(six, siy), 0.0, six, siy)]
        gscore = {siy * self.nx + six: 0.0}
        parent = {siy * self.nx + six: None}

        while open_h:
            _, g, ix, iy = heapq.heappop(open_h)
            idx = iy * self.nx + ix
            if idx == goal_i:
                cells = []
                cur = idx
                while cur is not None:
                    cells.append((cur % self.nx, cur // self.nx))
                    cur = parent[cur]
                cells.reverse()
                return cells
            if g > gscore.get(idx, math.inf):
                continue
            for dx, dy, cost in self._neigh:
                jx, jy = ix + dx, iy + dy
                if not self._in_bounds(jx, jy) or occ[jy, jx]:
                    continue
                # Corner-cutting: both orthogonal neighbours must be free
                if dx != 0 and dy != 0 and (occ[iy, jx] or occ[jy, ix]):
                    continue
                jdx = jy * self.nx + jx
                ng = g + cost
                if ng < gscore.get(jdx, math.inf):
                    gscore[jdx] = ng
                    parent[jdx] = idx
                    heapq.heappush(open_h, (ng + h(jx, jy), ng, jx, jy))
        return None

    def line_free(self, a: tuple[float, float], b: tuple[float, float], hazards) -> bool:
        """True if segment a->b clears every inflated hazard circle."""
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return True
        extra = self.cfg.extra_clearance_m
        for h in hazards:
            r = h.radius + extra
            # Distance from circle centre to the segment
            t = max(0.0, min(1.0, ((h.x - ax) * dx + (h.y - ay) * dy) / (length * length)))
            px, py = ax + t * dx, ay + t * dy
            if math.hypot(px - h.x, py - h.y) <= r:
                return False
        return True

    def simplify(self, points: list[tuple[float, float]], hazards) -> list[tuple[float, float]]:
        if len(points) <= 2:
            return points
        out = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.line_free(points[i], points[j], hazards):
                j -= 1
            out.append(points[j])
            i = j
        return out
