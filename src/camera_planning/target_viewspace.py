"""Target-centred view generation and bounded, path-aware open-tour selection.

This module never samples hypothetical human positions. Visibility is a sampled
surface proxy supplied by the geometry backend, not rendered segmentation area.
"""

import itertools
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np

from .artifacts import json_hash, new_output, write_json
from .free_space import FreeGrid
from .geometry import inside_box, load_geometry, look_at, project, unit
from .trajectory import connect


def corners_for(request):
    if request.target_obb is not None:
        box = request.target_obb
        signs = np.array(list(itertools.product((-1, 1), repeat=3)))
        return np.asarray(box.center) + (signs * box.half_extents) @ np.asarray(box.axes).T
    lo, hi = request.target.minimum, request.target.maximum
    return np.array(list(itertools.product(*zip(lo, hi))), float)


def ray_interval(origin, direction, lo, hi):
    near, far = -np.inf, np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < lo[axis] or origin[axis] > hi[axis]:
                return None
        else:
            ends = sorted(((lo[axis] - origin[axis]) / direction[axis],
                           (hi[axis] - origin[axis]) / direction[axis]))
            near, far = max(near, ends[0]), min(far, ends[1])
    return (near, far) if far > near else None


class ViewEvaluator:
    def __init__(self, request, geometry):
        self.request, self.geometry, self.s = request, geometry, request.target_view
        self.corners = corners_for(request)
        self.center = self.corners.mean(axis=0)
        self.up = np.eye(3)[request.up_index]
        self.scale = max(float(np.linalg.norm(np.ptp(self.corners, axis=0))), 1e-3)
        height = float(np.ptp(self.corners[:, request.up_index]))
        self.focus = self.center + self.up * height * self.s.aim_height_ratio
        self.surface, self.normals = geometry.sample_surface(
            request.target.object_id, self.s.visibility_samples, 17, request.up_index, -1.0
        )
        self.rejections = Counter()

    def evaluate(self, position):
        position = np.asarray(position, float)
        req, s = self.request, self.s
        clearance = req.candidates.camera_clearance_m
        height = float(position[req.up_index] - req.floor_height_m)
        if height < s.min_camera_height_m or height > s.max_camera_height_m:
            return None, "camera_height_out_of_range"
        distance = float(np.linalg.norm(position - self.focus))
        if distance > s.max_camera_distance_m:
            return None, "camera_too_far"
        if not inside_box(position, req.allowed_camera_region, -clearance):
            return None, "outside_camera_region"
        if self.geometry.occupied(position[None], clearance)[0]:
            return None, "camera_collision"
        if abs(np.dot(unit(self.focus - position), self.up)) > 0.995:
            return None, "vertical_look_at_singularity"
        camera = look_at(position, self.focus, self.up, req.intrinsics, "pending")
        uv, depth = project(camera, self.corners)
        if (depth <= 0.01).any():
            return None, "target_behind_near_plane"
        xy = uv / [camera.width, camera.height]
        low, high = xy.min(axis=0), xy.max(axis=0)
        wh = high - low
        if (low < [s.side_margin_ratio, s.top_margin_ratio]).any() or (
            high > [1 - s.side_margin_ratio, 1 - s.bottom_margin_ratio]
        ).any() or (wh > [s.max_width_ratio, s.max_height_ratio]).any():
            return None, "composition_too_tight"
        if max(wh) < s.min_extent_ratio:
            return None, "target_too_small"
        facing = np.einsum("ij,ij->i", self.normals, position - self.surface) > 1e-8
        if not facing.any():
            return None, "no_front_facing_samples"
        points = self.surface[facing]
        # Offset toward the camera to avoid an intersection with the hit surface itself.
        endpoints = points + (position - points) * 1e-6
        pix, z = project(camera, points)
        in_frame = (z > 0.01) & np.all((pix >= 0) & (pix <= [camera.width, camera.height]), axis=1)
        visible = in_frame & ~self.geometry.blocked(position, endpoints)
        visibility = float(visible.mean())
        if visibility < s.min_visibility_fraction:
            return None, "target_occluded"
        fill = min(1.0, max(wh[0] / s.max_width_ratio, wh[1] / s.max_height_ratio))
        proximity = max(0.0, 1.0 - distance / s.max_camera_distance_m)
        remaining = 1.0 - s.distance_preference_weight
        composition = remaining * (0.6 * visibility + 0.4 * fill) + s.distance_preference_weight * proximity
        if composition < s.min_composition_score:
            return None, "composition_score_too_low"
        return {
            "camera": camera,
            "direction": unit(position - self.center),
            "bbox": [*low.tolist(), *high.tolist()],
            "visibility": visibility,
            "fill": float(fill),
            "distance_m": distance,
            "height_above_floor_m": height,
            "composition": float(composition),
        }, None

    def fit_interval(self, direction):
        """Sufficient framing interval from linear pinhole inequalities.

        Orbit origin is the fixed aim point; orientation stays constant along a ray.
        Do NOT binary-search collision/visibility, which are non-monotone.
        """
        req, s = self.request, self.s
        if abs(direction @ self.up) > 0.995:
            return None
        probe = look_at(self.focus + direction, self.focus, self.up, req.intrinsics, "probe")
        local = (self.corners - self.focus) @ np.asarray(probe.R).T
        k = req.intrinsics
        left = max(s.side_margin_ratio * k.width, k.cx - s.max_width_ratio * k.width / 2)
        right = min((1 - s.side_margin_ratio) * k.width, k.cx + s.max_width_ratio * k.width / 2)
        top = max(s.top_margin_ratio * k.height, k.cy - s.max_height_ratio * k.height / 2)
        bottom = min((1 - s.bottom_margin_ratio) * k.height, k.cy + s.max_height_ratio * k.height / 2)
        if not (left < k.cx < right and top < k.cy < bottom):
            raise ValueError("principal point must lie within composition margins")
        rho = max(0.01, float(np.max(0.011 - local[:, 2])))
        for axis, focal, principal, lower, upper in (
            (0, k.fx, k.cx, left, right), (1, k.fy, k.cy, top, bottom)
        ):
            bound = np.maximum(focal * local[:, axis] / (upper - principal),
                               -focal * local[:, axis] / (principal - lower)) - local[:, 2]
            rho = max(rho, float(bound.max()))
        margin = req.candidates.camera_clearance_m
        interval = ray_interval(self.focus, direction,
                                np.array(req.allowed_camera_region.minimum) + margin,
                                np.array(req.allowed_camera_region.maximum) - margin)
        if interval is None:
            return None
        lo = max(rho * (1 + 1e-6), interval[0], 0.01)
        hi = min(interval[1], s.max_camera_distance_m)
        return (lo, hi) if hi > lo else None

    def radial_intervals(self, direction):
        interval = self.fit_interval(direction)
        if interval is None:
            return []
        intervals = [interval]
        # Exact interval subtraction for the selected AABB proxy backend.
        # Mesh backends retain adaptive occupied/visibility tests below.
        if hasattr(self.geometry, "boxes"):
            margin = self.request.candidates.camera_clearance_m + 1e-6
            for box in self.geometry.boxes:
                cut = ray_interval(self.focus, direction, np.array(box.minimum) - margin,
                                   np.array(box.maximum) + margin)
                if cut is None:
                    continue
                pieces = []
                for lo, hi in intervals:
                    if cut[1] <= lo or cut[0] >= hi:
                        pieces.append((lo, hi))
                    else:
                        if lo < cut[0]:
                            pieces.append((lo, cut[0]))
                        if hi > cut[1]:
                            pieces.append((cut[1], hi))
                intervals = pieces
        return intervals


def circular_distance(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def azimuth_bin_indices(records, centers, count, per_bin):
    """Round-robin azimuth retention; quality breaks ties inside equal-occupancy bins."""
    if not records or count <= 0:
        return [], np.empty(0, dtype=int)
    angles = np.asarray([r["azimuth_rad"] for r in records])
    quality = np.asarray([r["composition"] for r in records])
    bins = np.asarray([int(np.argmin([circular_distance(a, c) for c in centers])) for a in angles])
    selected, occupancy = [], np.zeros(len(centers), dtype=int)
    available = np.ones(len(records), dtype=bool)
    nearest = np.full(len(records), math.pi)
    while len(selected) < min(count, len(records)):
        eligible = available & (occupancy[bins] < per_bin)
        if not eligible.any():
            break
        level = occupancy[bins[eligible]].min()
        eligible &= occupancy[bins] == level
        gain = nearest / math.pi + 0.15 * quality if selected else quality.copy()
        gain[~eligible] = -np.inf
        index = int(np.argmax(gain))
        selected.append(index)
        occupancy[bins[index]] += 1
        available[index] = False
        nearest = np.minimum(nearest, np.asarray([circular_distance(a, angles[index]) for a in angles]))
    return selected, bins


def generate_views(evaluator):
    """Uniform azimuths outside, elevation/radius search inside each vertical half-plane."""
    s, req = evaluator.s, evaluator.request
    records, direction_cache, position_cache, trace = [], {}, {}, []
    radial_queries = local_queries = 0
    horizontal_axes = [i for i in range(3) if i != req.up_index]
    azimuth_centers = np.arange(s.azimuth_count) * 2 * math.pi / s.azimuth_count
    initial_elevations = np.linspace(math.radians(s.elevation_min_degrees),
                                     math.radians(s.elevation_max_degrees), s.elevation_count)

    def vector(azimuth, elevation):
        horizontal = np.zeros(3)
        horizontal[horizontal_axes[0]] = math.cos(azimuth)
        horizontal[horizontal_axes[1]] = math.sin(azimuth)
        return math.cos(elevation) * horizontal + math.sin(elevation) * evaluator.up

    def annotate_angles(record):
        delta = record["camera"].center - evaluator.center
        horizontal = delta[horizontal_axes]
        record["azimuth_rad"] = math.atan2(horizontal[1], horizontal[0]) % (2 * math.pi)
        record["elevation_rad"] = math.atan2(delta[req.up_index], np.linalg.norm(horizontal))

    def evaluate_position(position, local=False):
        nonlocal radial_queries, local_queries
        key = tuple(np.round(position, 10))
        if key in position_cache:
            return position_cache[key], True
        if len(position_cache) >= s.max_position_evaluations:
            return None, False
        value, reason = evaluator.evaluate(position)
        position_cache[key] = value
        if local:
            local_queries += 1
        else:
            radial_queries += 1
        if reason:
            evaluator.rejections[reason] += 1
        return value, True

    def sample_direction(azimuth, elevation, stage):
        azimuth %= 2 * math.pi
        key = (round(azimuth, 10), round(elevation, 10))
        if key in direction_cache:
            return direction_cache[key]
        if len(direction_cache) >= s.max_direction_evaluations:
            return None
        direction = vector(azimuth, elevation)
        legal, radial_cache = [], {}

        def at(radius):
            if radius not in radial_cache:
                if len(radial_cache) >= s.max_radial_evaluations_per_direction:
                    return None, False
                value, tested = evaluate_position(evaluator.focus + radius * direction)
                if not tested:
                    return None, False
                radial_cache[radius] = value
                if value is not None:
                    annotate_angles(value)
                    legal.append(value)
            return radial_cache[radius], True

        intervals = evaluator.radial_intervals(direction)
        queue = deque((lo, hi, 0) for lo, hi in intervals)
        truncated = False
        while queue:
            lo, hi, depth = queue.popleft()
            mid = (lo + hi) / 2
            samples = [at(r) for r in (lo, mid, hi)]
            if not all(tested for _, tested in samples):
                truncated = True
                break
            valid = [value for value, _ in samples if value is not None]
            changing = len(valid) not in (0, 3)
            if len(valid) > 1:
                changing |= np.ptp([v["visibility"] for v in valid]) > s.max_visibility_change
                changing |= np.max(np.ptp([v["bbox"] for v in valid], axis=0)) > s.max_bbox_change
            if (depth < s.radial_max_depth
                    and hi - lo > s.min_radial_step_target_ratio * evaluator.scale
                    and (depth < s.radial_min_depth or changing)):
                queue.extend(((lo, mid, depth + 1), (mid, hi, depth + 1)))

        kept = []
        if legal:
            nearest = min(legal, key=lambda r: r["distance_m"])
            best = max(legal, key=lambda r: r["composition"])
            for record in (best, nearest):
                if not any(np.allclose(record["camera"].center, other["camera"].center) for other in kept):
                    kept.append(record)
            records.extend(kept)
        best = max(kept, key=lambda r: r["composition"]) if kept else None
        summary = {"legal": best is not None,
                   "composition": best["composition"] if best else 0.0,
                   "visibility": best["visibility"] if best else 0.0,
                   "distance_m": best["distance_m"] if best else None}
        direction_cache[key] = summary
        trace.append({"stage": stage, "azimuth_degrees": math.degrees(azimuth),
                      "elevation_degrees": math.degrees(elevation),
                      "framing_free_intervals_m": intervals, "tested_radii": len(radial_cache),
                      "radial_budget_truncated": truncated, "retained": len(kept), **summary})
        return summary

    def probe_vertical_plane(azimuth, stage):
        summaries = {}
        for elevation in initial_elevations:
            if len(direction_cache) >= s.max_direction_evaluations:
                break
            summaries[float(elevation)] = sample_direction(azimuth, float(elevation), stage)
        for round_index in range(s.elevation_refinement_rounds):
            options = []
            ordered = sorted(summaries)
            for lo, hi in itertools.pairwise(ordered):
                left, right = summaries[lo], summaries[hi]
                if left is None or right is None:
                    continue
                mixed = left["legal"] != right["legal"]
                quality = max(left["composition"], right["composition"])
                change = abs(left["visibility"] - right["visibility"])
                priority = 2.0 * float(mixed) + quality + change + (hi - lo) / math.pi
                options.append((priority, lo, hi))
            if not options or len(direction_cache) >= s.max_direction_evaluations:
                break
            _, lo, hi = max(options)
            mid = (lo + hi) / 2
            summaries[mid] = sample_direction(azimuth, mid, f"{stage}_elevation_refine_{round_index}")
        valid = [v for v in summaries.values() if v and v["legal"]]
        return {"legal": bool(valid),
                "best_composition": max((v["composition"] for v in valid), default=0.0)}

    # Complete uniform horizontal sweep before any azimuth refinement.
    plane_summaries = {}
    for azimuth in azimuth_centers:
        plane_summaries[float(azimuth)] = probe_vertical_plane(float(azimuth), "uniform_azimuth")
    coarse_complete = len(plane_summaries) == s.azimuth_count

    # First round probes every interval midpoint. Later rounds concentrate on feasible
    # intervals and legal/illegal boundaries while keeping the original uniform bins fixed.
    for round_index in range(s.azimuth_refinement_rounds):
        ordered = sorted(plane_summaries)
        intervals = [(ordered[i], ordered[(i + 1) % len(ordered)]) for i in range(len(ordered))]
        additions = []
        for lo, raw_hi in intervals:
            hi = raw_hi + (2 * math.pi if raw_hi <= lo else 0)
            left, right = plane_summaries[lo], plane_summaries[raw_hi]
            if round_index > 0 and not (left["legal"] or right["legal"]
                                        or left["legal"] != right["legal"]):
                continue
            additions.append(((lo + hi) / 2) % (2 * math.pi))
        for azimuth in additions:
            if (len(direction_cache) >= s.max_direction_evaluations
                    or len(position_cache) >= s.max_position_evaluations):
                break
            plane_summaries[azimuth] = probe_vertical_plane(
                azimuth, f"azimuth_refine_{round_index}")

    if not records:
        raise ValueError("no quality-qualified target view found; inspect height, distance, occlusion and framing limits")

    # Improve a few azimuth-distributed seeds in real coordinates. Every trial must still
    # pass all hard quality constraints and shares the global position budget.
    seeds, _ = azimuth_bin_indices(records, azimuth_centers, s.local_refine_seeds, 1)
    for index in seeds:
        best = records[index]
        step = evaluator.scale * 0.06
        for _ in range(s.local_refine_rounds):
            trials = []
            for axis in range(3):
                for sign in (-1, 1):
                    position = best["camera"].center.copy()
                    position[axis] += sign * step
                    candidate, _ = evaluate_position(position, local=True)
                    if candidate is not None:
                        annotate_angles(candidate)
                        trials.append(candidate)
            if trials:
                candidate = max(trials, key=lambda r: r["composition"])
                if candidate["composition"] > best["composition"] + 1e-8:
                    best = candidate
                    records.append(best)
            step /= 2

    unique = {}
    for record in records:
        key = tuple(np.round(record["camera"].center, 7))
        if key not in unique or record["composition"] > unique[key]["composition"]:
            unique[key] = record
    records = list(unique.values())
    before_cap = len(records)
    chosen, bins = azimuth_bin_indices(records, azimuth_centers, s.max_candidates,
                                       s.max_candidates_per_azimuth_bin)
    before_counts = Counter(int(v) for v in bins)
    after_counts = Counter(int(bins[i]) for i in chosen)
    records = [records[i] for i in chosen]
    for i, record in enumerate(records):
        record["camera"].camera_id = f"view_{i:04d}"
        record["azimuth_bin"] = int(bins[chosen[i]])
    stop = ("position_budget" if len(position_cache) >= s.max_position_evaluations else
            "direction_budget" if len(direction_cache) >= s.max_direction_evaluations else
            "configured_refinement_complete")
    return records, {"search_policy": "azimuth_first_vertical_plane_v3",
                     "parameterization": "uniform/refined azimuth; elevation and radius inside vertical half-plane",
                     "coarse_azimuth_count": s.azimuth_count,
                     "coarse_azimuth_sweep_complete": coarse_complete,
                     "direction_queries": len(direction_cache), "radial_queries": radial_queries,
                     "local_position_queries": local_queries, "position_queries": len(position_cache),
                     "max_direction_evaluations": s.max_direction_evaluations,
                     "max_position_evaluations": s.max_position_evaluations,
                     "stop_reason": stop, "before_candidate_cap": before_cap,
                     "retained_candidates": len(records), "requested_candidate_cap": s.max_candidates,
                     "max_candidates_per_azimuth_bin": s.max_candidates_per_azimuth_bin,
                     "azimuth_bins": [{"bin": i, "center_degrees": math.degrees(angle),
                                        "before": before_counts[i], "retained": after_counts[i]}
                                       for i, angle in enumerate(azimuth_centers)],
                     "vertical_planes": [{"azimuth_degrees": math.degrees(angle), **summary}
                                         for angle, summary in sorted(plane_summaries.items())],
                     "direction_trace": trace, "rejections": dict(evaluator.rejections),
                     "complete_continuous_domain_certified": False}


def diverse_indices(records, count):
    if count == 0:
        return []
    vectors = np.array([r["direction"] for r in records])
    quality = np.array([r["composition"] for r in records])
    chosen = [int(np.argmax(quality))]
    while len(chosen) < count:
        gain = 1 - (vectors @ vectors[chosen].T).max(axis=1) + 0.03 * quality
        gain[chosen] = -np.inf
        chosen.append(int(np.argmax(gain)))
    return chosen


class TargetPathGrid(FreeGrid):
    """Camera clearance plus sampled target framing/visibility on every searched edge."""
    def __init__(self, evaluator):
        req = evaluator.request
        super().__init__(req.allowed_camera_region, req.path.voxel_size_m, evaluator.geometry,
                         req.path.max_grid_cells, req.candidates.camera_clearance_m)
        self.validation_step = req.path.validation_step_m
        self.evaluator = evaluator
        self.view_cache = {}

    def segment_clear(self, a, b):
        if not super().segment_clear(a, b):
            return False
        n = max(1, math.ceil(np.linalg.norm(np.asarray(b) - a) / self.evaluator.s.transition_check_step_m))
        for position in np.linspace(a, b, n + 1):
            key = tuple(np.round(position, 7))
            if key not in self.view_cache:
                if len(self.view_cache) >= self.evaluator.s.max_transition_view_checks:
                    return False
                self.view_cache[key] = self.evaluator.evaluate(position)[0] is not None
            if not self.view_cache[key]:
                return False
        return True


def select_tour(evaluator, records):
    req, s = evaluator.request, evaluator.s
    vectors = np.array([r["direction"] for r in records])
    quality = np.array([r["composition"] for r in records])
    azimuths = np.array([r["azimuth_rad"] for r in records])
    elevations = np.array([r["elevation_rad"] for r in records])
    refs = np.arange(s.azimuth_count) * 2 * math.pi / s.azimuth_count
    cos_limit = math.cos(math.radians(s.coverage_angle_degrees))
    separation = np.abs((azimuths[:, None] - refs[None, :] + math.pi) % (2 * math.pi) - math.pi)
    kernel = np.clip((np.cos(separation) - cos_limit) / (1 - cos_limit), 0, 1)
    reachable = kernel.max(axis=0) > 0
    kernel = kernel[:, reachable]
    if kernel.shape[1] == 0:
        raise ValueError("no angular quadrature nodes covered")

    def score(ids, length):
        coverage = float(kernel[ids].max(axis=0).mean())
        pairs = [max(0.0, 1 - float(vectors[a] @ vectors[b])**2) for a, b in itertools.combinations(ids, 2)]
        baseline = float(np.mean(pairs)) if pairs else 0.0
        composition = float(quality[ids].mean())
        worst_composition = float(quality[ids].min())
        elevation_span = math.radians(s.elevation_max_degrees - s.elevation_min_degrees)
        elevation_pairs = [abs(float(elevations[a] - elevations[b])) / elevation_span
                           for a, b in itertools.combinations(ids, 2)]
        elevation_diversity = float(np.mean(np.clip(elevation_pairs, 0, 1))) if elevation_pairs else 0.0
        total = (s.direction_weight * coverage + s.baseline_weight * baseline
                 + s.composition_weight * composition
                 + s.worst_composition_weight * worst_composition
                 + s.elevation_diversity_weight * elevation_diversity
                 - s.path_weight * length / evaluator.scale)
        return total, {"direction_coverage": coverage, "anchor_baseline": baseline,
                       "composition": composition, "worst_composition": worst_composition,
                       "elevation_diversity": elevation_diversity,
                       "path_length_m": length, "total": total}

    grid, links = TargetPathGrid(evaluator), {}
    queries = 0

    def link(a, b):
        nonlocal queries
        if (a, b) not in links:
            if queries >= s.max_path_queries:
                return None
            queries += 1
            path = connect(grid, records[a]["camera"].center, records[b]["camera"].center,
                           req.path.max_expansions)
            value = (float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), path) if path else None
            links[a, b] = value
            links[b, a] = (value[0], value[1][::-1]) if value else None
        return links[a, b]

    best, partial = None, []
    starts, _ = azimuth_bin_indices(records, refs, min(s.selection_starts, len(records)), 1)
    for start in starts:
        order, length = [start], 0.0
        while len(order) < req.budget:
            available = [i for i in range(len(records)) if i not in order]
            # Cheap optimistic score (zero extra travel); true costs used before acceptance.
            available.sort(key=lambda i: score(order + [i], length)[0], reverse=True)
            winner = None
            for candidate in available[:s.insertion_shortlist]:
                for slot in range(len(order) + 1):
                    before = order[slot - 1] if slot else None
                    after = order[slot] if slot < len(order) else None
                    left = link(before, candidate) if before is not None else (0.0, [])
                    right = link(candidate, after) if after is not None else (0.0, [])
                    if left is None or right is None:
                        continue
                    removed = links[before, after][0] if before is not None and after is not None else 0
                    new_length = length + left[0] + right[0] - removed
                    if req.path.max_length_m is not None and new_length > req.path.max_length_m:
                        continue
                    trial = order[:slot] + [candidate] + order[slot:]
                    utility, metrics = score(trial, new_length)
                    if winner is None or utility > winner[0]:
                        winner = utility, trial, new_length, metrics
            if winner is None:
                break
            _, order, length, _ = winner
        partial.append({"start": start, "selected_count": len(order), "length_m": length})
        if len(order) == req.budget:
            utility, metrics = score(order, length)
            if best is None or utility > best[0]:
                best = utility, order, metrics
    diagnostics = {"path_queries": queries, "path_query_cap": s.max_path_queries,
                   "transition_view_checks": len(grid.view_cache),
                   "transition_view_check_cap": s.max_transition_view_checks,
                   "starts": partial, "angular_reference_count": int(reachable.sum()),
                   "angular_reference_semantics": "uniform horizontal azimuth bins near feasible candidates"}
    if best is None:
        return None, None, diagnostics
    _, order, metrics = best
    path = {"format_version": "camera_open_path_v1", "status": "planned", "closed": False,
            "length_unit": "meter", "length_m": metrics["path_length_m"],
            "geometry_backend": evaluator.geometry.name,
            "selection_coupling": "bounded multi-start greedy insertion using obstacle-aware edge costs",
            "safety_model": "geometry clearance; target composition/visibility sampled along edges",
            "transition_check_step_m": s.transition_check_step_m,
            "focus_world_m": evaluator.focus.tolist(), "world_up": evaluator.up.tolist(),
            "orientation_policy": "look_at_fixed_target_aim",
            "ordered_camera_ids": [records[i]["camera"].camera_id for i in order],
            "legs": [{"from": records[a]["camera"].camera_id, "to": records[b]["camera"].camera_id,
                      "points": links[a, b][1], "length_m": links[a, b][0]}
                     for a, b in itertools.pairwise(order)]}
    diagnostics["metrics"] = metrics
    return order, path, diagnostics


def plan_target_views(request, output, seed=0):
    from .planner import rig

    root = new_output(output)
    evaluator = ViewEvaluator(request, load_geometry(request))
    records, search = generate_views(evaluator)
    write_json(root / "request.json", request.model_dump(mode="json"))
    write_json(root / "viewspace_search.json", search)
    write_json(root / "candidate_cameras.json", rig(request.scene_id, [r["camera"] for r in records]))
    write_json(root / "target_view_evidence.json", {
        "target_corners_world_m": evaluator.corners.tolist(), "focus_world_m": evaluator.focus.tolist(),
        "visibility_semantics": "front-facing target surface samples; not rendered pixels or human visibility",
        "candidates": [{"camera_id": r["camera"].camera_id, "bbox_normalized": r["bbox"],
                        "visibility_fraction": r["visibility"], "composition": r["composition"],
                        "fill": r["fill"], "distance_m": r["distance_m"],
                        "height_above_floor_m": r["height_above_floor_m"],
                        "azimuth_degrees": math.degrees(r["azimuth_rad"]),
                        "elevation_degrees": math.degrees(r["elevation_rad"]),
                        "azimuth_bin": r["azimuth_bin"]} for r in records]})
    if len(records) < request.budget:
        write_json(root / "selection_diagnostics.json", {
            "status": "blocked", "reason": "insufficient_candidates_after_azimuth_bin_cap",
            "candidate_count": len(records), "requested_cameras": request.budget,
            "max_candidates_per_azimuth_bin": request.target_view.max_candidates_per_azimuth_bin})
        raise ValueError("too few candidates after azimuth-bin retention; inspect viewspace_search.json "
                         "and adjust search budget or explicit bin cap; no cap was silently relaxed")
    order, path, selection = select_tour(evaluator, records)
    write_json(root / "selection_diagnostics.json", selection)
    if order is None:
        write_json(root / "camera_path.json", {"status": "blocked", "closed": False,
                   "reason": "no_complete_tour_within_search_budget", **selection})
        raise ValueError(f"could not select {request.budget} connected target views; see {root / 'selection_diagnostics.json'}")
    cameras = [records[i]["camera"] for i in order]
    camera_rig = rig(request.scene_id, cameras)
    path["rig_hash"] = json_hash(camera_rig)
    write_json(root / "cameras.json", camera_rig)
    write_json(root / "camera_path.json", path)
    write_json(root / "camera_placements.json", {"length_unit": "meter", "cameras": [
        {"camera_id": c.camera_id, "position_world_m": c.center.tolist(),
         "view_direction_world": np.asarray(c.R)[2].tolist()} for c in cameras]})
    # Keep preview compatible while removing hypothetical human/free-space point clouds entirely.
    np.savez_compressed(root / "evidence.npz", points=evaluator.corners, kinds=np.full(8, 3, np.int8))
    write_json(root / "observation_domain.json", {"mode": "target_viewspace", "human_scale_used": False,
               "human_position_samples": 0, "bbox_source": request.target_obb.method if request.target_obb else "aabb_fallback"})
    write_json(root / "post_generation_contract.json", {
        "scene_id": request.scene_id, "expected_subject": "one human static in world coordinates",
        "camera_source": "tour/camera_trajectory.json (actual per-frame K/R/T)",
        "requires": ["verified background/camera correspondence after video generation",
                     "person bbox and common-layout 2D keypoints with confidence and track_id"],
        "next_command": "camera_planning select-human-frames",
        "detector_or_video_model_executed": False})
    result = {"schema_version": "camera_plan_result_v2", "planning_mode": "target_viewspace",
              "scene_id": request.scene_id, "budget": request.budget, "seed": seed,
              "feasible_count": len(records), "selected_ids": [c.camera_id for c in cameras],
              "metrics": selection["metrics"], "trajectory_status": "planned",
              "geometry_backend": evaluator.geometry.name, "reconstruction_status": "not_run",
              "limitations": ["bounded heuristic; no global optimum or exhaustive feasible-domain guarantee",
                              "visibility and transitions are finite samples; not exact pixel coverage",
                              "AABB backend can overestimate occlusion and closes concave free regions",
                              "OBB geometry axes do not identify semantic furniture front",
                              "image margins do not guarantee a complete future person or visible floor",
                              "azimuth/elevation/radius search is bounded and may miss a narrow feasible region",
                              "generated video must be checked before inheriting calibrated cameras"]}
    write_json(root / "camera_plan_result.json", result)
    return result
