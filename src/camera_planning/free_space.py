"""Bounded, point-free voxel graph. No body-size or action assumptions.

Edges are checked against geometry, including thin walls missed by center tests.
This is a discretized geometric neighborhood, NOT a feasible human volume.
"""

import heapq
import itertools

import numpy as np


class FreeGrid:
    def __init__(self, region, step, geometry, max_cells, clearance=0.0):
        self.low = np.asarray(region.minimum, dtype=float) + clearance
        self.high = np.asarray(region.maximum, dtype=float) - clearance
        if np.any(self.high <= self.low):
            raise ValueError("region too small for declared clearance")
        self.shape = np.maximum(np.ceil((self.high - self.low) / step).astype(int), 1)
        count = int(np.prod(self.shape))
        if count > max_cells:
            raise ValueError(
                f"voxel grid needs {count} cells; cap={max_cells}; increase voxel_size_m"
            )
        self.spacing = (self.high - self.low) / self.shape
        self.points = self.low + (np.indices(self.shape).reshape(3, -1).T + 0.5) * self.spacing
        self.geometry = geometry
        self.clearance = clearance
        self.validation_step = 0.05
        self.free = ~geometry.occupied(self.points, max(clearance, 1e-6))
        self.edges = {}

    def near(self, point):
        cell = np.floor((np.asarray(point) - self.low) / self.spacing).astype(int)
        cell = np.clip(cell, 0, self.shape - 1)
        candidates = []
        for delta in itertools.product((-1, 0, 1), repeat=3):
            index = cell + delta
            if np.all(index >= 0) and np.all(index < self.shape):
                flat = int(np.ravel_multi_index(tuple(index), self.shape))
                if self.free[flat]:
                    candidates.append(flat)
        return sorted(candidates, key=lambda i: (np.linalg.norm(self.points[i] - point), i))

    def segment_clear(self, a, b):
        ends = np.asarray([a, b], float)
        if np.any(ends < self.low - 1e-9) or np.any(ends > self.high + 1e-9):
            return False
        if self.geometry.blocked(np.asarray(a), np.asarray(b)[None])[0]:
            return False
        if self.clearance <= 0:
            return True
        # Distance to a closed set is 1-Lipschitz: spacing/2 extra clearance
        # at samples conservatively certifies the whole connecting segment.
        length = float(np.linalg.norm(np.asarray(b) - a))
        n = max(1, int(np.ceil(length / min(float(self.spacing.min()) / 4, self.validation_step))))
        points = np.linspace(a, b, n + 1)
        return not self.geometry.occupied(points, self.clearance + length / (2 * n)).any()

    def neighbors(self, flat):
        index = np.array(np.unravel_index(flat, self.shape))
        for axis in range(3):
            for sign in (-1, 1):
                other = index.copy()
                other[axis] += sign
                if other[axis] < 0 or other[axis] >= self.shape[axis]:
                    continue
                j = int(np.ravel_multi_index(tuple(other), self.shape))
                if not self.free[j]:
                    continue
                key = (min(flat, j), max(flat, j))
                if key not in self.edges:
                    self.edges[key] = self.segment_clear(self.points[flat], self.points[j])
                if self.edges[key]:
                    yield j, float(self.spacing[axis])


def furniture_domain(request, geometry):
    settings = request.observation
    grid = FreeGrid(
        settings.region,
        settings.voxel_size_m,
        geometry,
        settings.max_grid_cells,
        clearance=settings.occupancy_clearance_m,
    )
    surface, normals = geometry.sample_surface(
        request.target.object_id,
        max(settings.surface_samples, 512),
        settings.seed,
        request.up_index,
        settings.underside_normal_threshold,
    )
    distances = np.full(len(grid.points), np.inf)
    frontier = []
    max_seed_distance = float(np.linalg.norm(grid.spacing)) * 1.5
    for point, normal in zip(surface, normals):
        origin = point + normal * 1e-5
        if geometry.occupied(origin[None], 0.0)[0]:
            continue
        for cell in grid.near(origin):
            distance = float(np.linalg.norm(grid.points[cell] - point))
            if distance > max_seed_distance or distance > settings.neighborhood_radius_m:
                continue
            if distance < distances[cell] and grid.segment_clear(origin, grid.points[cell]):
                distances[cell] = distance
                heapq.heappush(frontier, (distance, cell))
    seed_count = int(np.isfinite(distances).sum())
    while frontier:
        distance, cell = heapq.heappop(frontier)
        if distance != distances[cell]:
            continue
        for neighbor, cost in grid.neighbors(cell):
            value = distance + cost
            if value <= settings.neighborhood_radius_m and value < distances[neighbor]:
                distances[neighbor] = value
                heapq.heappush(frontier, (value, neighbor))
    cells = np.flatnonzero(np.isfinite(distances))
    if len(cells) < 2:
        raise ValueError(
            "empty/undersampled furniture free-space domain; inspect region, grid and mesh"
        )
    total = len(cells)
    if total > settings.spatial_samples:
        cells = np.sort(
            np.random.default_rng(settings.seed + 1).choice(
                cells,
                settings.spatial_samples,
                replace=False,
            )
        )
    volume = total * float(np.prod(grid.spacing))
    weights = np.full(len(cells), volume / len(cells))
    return (
        grid.points[cells],
        weights,
        {
            "mode": "furniture_free_space",
            "grid_cells": len(grid.points),
            "seed_cells": seed_count,
            "domain_cells": total,
            "sample_count": len(cells),
            "estimated_volume_m3": volume,
            "spacing_m": grid.spacing.tolist(),
            "radius_m": settings.neighborhood_radius_m,
            "occupancy_clearance_m": settings.occupancy_clearance_m,
            "semantics": (
                "room-clipped geometric neighborhood with isotropic obstacle clearance; "
                "not a full human pose-feasibility model"
            ),
            "distance": "six-neighbor graph distance with mesh-tested edges; resolution-dependent",
            "human_scale_used": bool(settings.occupancy_clearance_m > 0),
        },
    )
