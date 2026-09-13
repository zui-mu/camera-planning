"""Bounded coarse-to-fine candidate proposals and incremental evidence rows."""

import numpy as np

from .geometry import look_at, project
from .observability import point_quality
from .scoring import Evidence, build_evidence, evaluate_set
from .selection import select


def evidence_bytes(cameras, points):
    # float64 information[3,3], direction[3], resolution + visibility bool.
    return int(cameras * points * (13 * 8 + 1))


def check_evidence_budget(request, count, points):
    size = evidence_bytes(count, points)
    if size > request.adaptive.max_evidence_mb * 1024**2:
        raise ValueError(
            f"evidence arrays need {size / 1024**2:.1f} MiB; reduce candidates/samples"
        )


def pose_rejection(request, geometry, camera):
    position = camera.center
    region, margin = request.allowed_camera_region, request.candidates.camera_clearance_m
    if np.any(position < np.asarray(region.minimum) + margin) or np.any(
        position > np.asarray(region.maximum) - margin
    ):
        return "outside_allowed_camera_region"
    if geometry.occupied(position[None], margin)[0]:
        return "camera_in_geometry_or_clearance"
    distance = geometry.distance_to_object(position[None], request.target.object_id)[0]
    if (
        not request.candidates.min_target_distance_m
        <= distance
        <= request.candidates.max_target_distance_m
    ):
        return "outside_declared_standoff_range"
    low, high = request.target.minimum, request.target.maximum
    corners = np.array(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
    )
    uv, depth = project(camera, corners)
    border = request.observation.image_margin_ratio
    framed = (
        (depth > 0.05)
        & (uv[:, 0] >= border * camera.width)
        & (uv[:, 0] < (1 - border) * camera.width)
        & (uv[:, 1] >= border * camera.height)
        & (uv[:, 1] < (1 - border) * camera.height)
    )
    if framed.mean() < request.candidates.min_target_in_frame_fraction:
        return "target_bounds_out_of_frame"
    return None


def filter_evidence(request, evidence, rejected):
    surface = evidence.kinds == 0
    space = evidence.kinds == (2 if request.observation.mode == "furniture_free_space" else 1)
    a = evidence.visible[:, surface].mean(axis=1)
    b = evidence.visible[:, space].mean(axis=1)
    minimum = (
        request.candidates.min_space_visible_fraction
        if request.observation.mode == "furniture_free_space"
        else request.candidates.min_envelope_visible_fraction
    )
    keep = (a >= request.candidates.min_surface_visible_fraction) & (b >= minimum)
    for i in np.flatnonzero(~keep):
        rejected.append(
            {
                "camera_id": evidence.cameras[i].camera_id,
                "reason": "insufficient_surface_or_observation_visibility",
                "surface_visible_fraction": float(a[i]),
                "observation_visible_fraction": float(b[i]),
            }
        )
    return evidence.subset(np.flatnonzero(keep))


def concatenate_evidence(left, right):
    if not np.array_equal(left.points, right.points) or not np.array_equal(
        left.weights, right.weights
    ):
        raise ValueError("incremental evidence must use the identical weighted observation domain")
    return Evidence(
        left.cameras + right.cameras,
        left.points,
        left.zones,
        *(
            np.concatenate((getattr(left, key), getattr(right, key)))
            for key in ("visible", "information", "resolution", "directions")
        ),
        left.kinds,
        left.weights,
    )


def propose(request, evidence, indices, iteration):
    settings = request.adaptive
    points = evidence.points[evidence.kinds == 2]
    quality = point_quality(evidence, indices, request.scoring)
    poor = []
    for i in np.argsort(quality, kind="stable"):
        if all(np.linalg.norm(points[i] - x) >= request.observation.voxel_size_m * 2 for x in poor):
            poor.append(points[i])
        if len(poor) >= settings.target_count:
            break
    center = (np.asarray(request.target.minimum) + request.target.maximum) / 2
    up = np.eye(3)[request.up_index]
    horizontal = [i for i in range(3) if i != request.up_index]
    proposed = []
    # Exploit selected poses, but keep it bounded and leave half the budget to exploration.
    step = settings.position_step_m / (iteration + 1)
    for delta in (np.eye(3) * step).tolist() + (-np.eye(3) * step).tolist():
        for index in indices:
            proposed.append(
                (evidence.cameras[index].center + delta, center, "selected_neighborhood")
            )
            if len(proposed) >= settings.proposals_per_round // 2:
                break
        if len(proposed) >= settings.proposals_per_round // 2:
            break
    radius = float(np.median([np.linalg.norm(c.center - center) for c in evidence.cameras]))
    slots = settings.proposals_per_round - len(proposed)
    for k in range(slots):
        target = poor[k % len(poor)]
        # Deterministic golden-angle directions, alternating elevations and standoffs.
        angle = (k + iteration * slots) * 2.399963229728653
        direction = np.zeros(3)
        elevation = (0.0, 0.25, 0.5)[(k // len(poor)) % 3]
        direction[horizontal] = [np.cos(angle), np.sin(angle)]
        direction[request.up_index] = elevation
        direction /= np.linalg.norm(direction)
        position = target + direction * radius * (0.85 if k % 2 else 1.05)
        focus = 0.7 * center + 0.3 * target
        proposed.append((position, focus, "poor_space_direction"))
    for serial, (position, focus, source) in enumerate(proposed):
        yield (
            look_at(position, focus, up, request.intrinsics, f"ref_{iteration:02d}_{serial:04d}"),
            source,
        )


def refine(request, geometry, evidence, rejected, method, seed):
    if request.adaptive.enabled and len(evidence.cameras) < request.budget:
        indices = list(range(len(evidence.cameras)))
        trace = [{"event": "bootstrap_incomplete_pool", "initial_candidates": len(indices)}]
    else:
        indices, trace = select(
            evidence, request.budget, request.scoring, method, seed, request.up_index
        )
    if not request.adaptive.enabled:
        return evidence, indices, trace, []
    if method != "greedy":
        raise ValueError(
            "adaptive generation uses greedy robust selection; compare baselines on exported pool"
        )
    history = []
    seen = {tuple(np.round(np.r_[c.center, np.asarray(c.R).ravel()], 7)) for c in evidence.cameras}
    for iteration in range(request.adaptive.rounds):
        before = evaluate_set(evidence, indices, request.scoring)["total"]
        complete_before = len(indices) == request.budget
        remaining = request.adaptive.max_candidates - len(evidence.cameras)
        if remaining <= 0:
            history.append({"round": iteration, "stop": "candidate_cap"})
            break
        new, sources = [], {}
        attempted = 0
        for camera, source in propose(request, evidence, indices, iteration):
            attempted += 1
            key = tuple(np.round(np.r_[camera.center, np.asarray(camera.R).ravel()], 7))
            reason = "duplicate_pose" if key in seen else pose_rejection(request, geometry, camera)
            seen.add(key)
            if reason:
                rejected.append(
                    {
                        "camera_id": camera.camera_id,
                        "reason": reason,
                        "position_world_m": camera.center.tolist(),
                        "source": source,
                    }
                )
                continue
            new.append(camera)
            sources[camera.camera_id] = source
            if len(new) >= remaining:
                break
        if not new:
            history.append(
                {"round": iteration, "attempted": attempted, "stop": "no_legal_proposals"}
            )
            break
        check_evidence_budget(request, len(evidence.cameras) + len(new), len(evidence.points))
        # Crucially, previous rows are reused unchanged. Only NEW cameras ray-cast.
        added = build_evidence(
            new,
            evidence.points,
            evidence.zones,
            geometry,
            request.scoring,
            evidence.kinds,
            request.observation.image_margin_ratio,
            evidence.weights,
        )
        added = filter_evidence(request, added, rejected)
        evidence = concatenate_evidence(evidence, added)
        if len(evidence.cameras) < request.budget:
            candidate, extra_trace = list(range(len(evidence.cameras))), []
        else:
            candidate, extra_trace = select(
                evidence, request.budget, request.scoring, method, seed, request.up_index
            )
        after = evaluate_set(evidence, candidate, request.scoring)["total"]
        if not complete_before or after >= before:
            indices = candidate
        else:
            after = before  # Growing the pool must never discard the incumbent.
        trace.extend(extra_trace)
        history.append(
            {
                "round": iteration,
                "attempted": attempted,
                "raycast_camera_rows": len(new),
                "accepted": len(added.cameras),
                "score_before": before,
                "score_after": after,
                "candidate_sources": sources,
                "old_evidence_rows_recomputed": 0,
                "complete_set_before": complete_before,
                "complete_set_after": len(indices) == request.budget,
            }
        )
        if complete_before and after - before < request.adaptive.min_gain:
            history[-1]["stop"] = "gain_below_threshold"
            break
    if len(indices) != request.budget:
        raise ValueError(
            "adaptive search exhausted before enough legal cameras; inspect region, framing or increase proposal budget"
        )
    return evidence, indices, trace, history
