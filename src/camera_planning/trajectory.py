"""Open obstacle-aware polygonal camera tour; no unconstrained spline overshoot."""

import heapq
import itertools

import numpy as np

from .free_space import FreeGrid


def connect(grid, start, end, max_expansions):
    start, end = np.asarray(start), np.asarray(end)
    if grid.segment_clear(start, end):
        return [start.tolist(), end.tolist()]
    starts = [i for i in grid.near(start) if grid.segment_clear(start, grid.points[i])]
    goals = {
        i: float(np.linalg.norm(grid.points[i] - end))
        for i in grid.near(end)
        if grid.segment_clear(grid.points[i], end)
    }
    if not starts or not goals:
        return None
    distance, parent, queue = {}, {}, []
    for i in starts:
        cost = float(np.linalg.norm(start - grid.points[i]))
        distance[i], parent[i] = cost, None
        heapq.heappush(queue, (cost + float(np.linalg.norm(grid.points[i] - end)), cost, i))
    expansions = 0
    while queue and expansions < max_expansions:
        _, cost, i = heapq.heappop(queue)
        if cost != distance.get(i):
            continue
        if i == -1:
            chain = [end.tolist()]
            i = parent[-1]
            while i is not None:
                chain.append(grid.points[i].tolist())
                i = parent[i]
            chain.append(start.tolist())
            chain.reverse()
            result, current = [chain[0]], 0
            while current < len(chain) - 1:
                nxt = len(chain) - 1
                while nxt > current + 1 and not grid.segment_clear(chain[current], chain[nxt]):
                    nxt -= 1
                result.append(chain[nxt])
                current = nxt
            return result
        expansions += 1
        neighbors = list(grid.neighbors(i))
        if i in goals:
            neighbors.append((-1, goals[i]))
        for j, delta in neighbors:
            value = cost + delta
            if value < distance.get(j, np.inf):
                distance[j], parent[j] = value, i
                heuristic = 0 if j == -1 else float(np.linalg.norm(grid.points[j] - end))
                heapq.heappush(queue, (value + heuristic, value, j))
    return None


def open_order(costs):
    n = len(costs)
    if n <= 12:
        # Exact order on computed pairwise connections, NOT exact camera selection.
        dp = {(1 << i, i): (0.0, (i,)) for i in range(n)}
        for size in range(1, n):
            for (mask, last), (cost, path) in list(dp.items()):
                if mask.bit_count() != size:
                    continue
                for j in range(n):
                    if mask & (1 << j) or not np.isfinite(costs[last, j]):
                        continue
                    key, value = (mask | (1 << j), j), cost + costs[last, j]
                    if value < dp.get(key, (np.inf, ()))[0]:
                        dp[key] = value, path + (j,)
        complete = [value for (mask, _), value in dp.items() if mask == (1 << n) - 1]
        return list(min(complete)[1]) if complete else None
    best = None
    for start in range(n):
        path = [start]
        while len(path) < n:
            rest = [i for i in range(n) if i not in path]
            nxt = min(rest, key=lambda i: costs[path[-1], i])
            if not np.isfinite(costs[path[-1], nxt]):
                break
            path.append(nxt)
        if len(path) == n:
            value = sum(costs[a, b] for a, b in itertools.pairwise(path))
            if best is None or value < best[0]:
                best = value, path
    return best[1] if best else None


def plan_open_path(request, geometry, cameras):
    settings = request.path
    grid = FreeGrid(
        request.allowed_camera_region,
        settings.voxel_size_m,
        geometry,
        settings.max_grid_cells,
        request.candidates.camera_clearance_m,
    )
    grid.validation_step = settings.validation_step_m
    costs, links = np.full((len(cameras), len(cameras)), np.inf), {}
    for i, j in itertools.combinations(range(len(cameras)), 2):
        points = connect(grid, cameras[i].center, cameras[j].center, settings.max_expansions)
        if points:
            links[i, j], links[j, i] = points, points[::-1]
            costs[i, j] = costs[j, i] = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    order = open_order(costs)
    base = {
        "format_version": "camera_open_path_v1",
        "closed": False,
        "length_unit": "meter",
        "geometry_backend": geometry.name,
        "safety_model": "segment LOS plus Lipschitz-inflated clearance samples",
        "selection_coupling": "post-selection feasibility, not joint global optimization",
        "limitations": [
            "conditional on supplied static geometry and numerical precision",
            "no dynamic obstacles or human self-occlusion",
            "target continuity is not yet constrained along all transitions",
            "unreachable may mean voxel/search budget limitation, not physical impossibility",
        ],
    }
    if order is None:
        return {**base, "status": "blocked", "reason": "no_connected_open_tour_on_search_graph"}
    length = float(sum(costs[a, b] for a, b in itertools.pairwise(order)))
    legs = [
        {
            "from": cameras[a].camera_id,
            "to": cameras[b].camera_id,
            "points": links[a, b],
            "length_m": float(costs[a, b]),
        }
        for a, b in itertools.pairwise(order)
    ]
    if settings.max_length_m is not None and length > settings.max_length_m:
        return {**base, "status": "blocked", "reason": "path_length_budget", "length_m": length}
    return {
        **base,
        "status": "planned",
        "length_m": length,
        "ordered_camera_ids": [cameras[i].camera_id for i in order],
        "legs": legs,
    }
