"""Auditable support relations and pose-agnostic interaction evidence.

This module intentionally uses only scene geometry.  It does not infer a
furniture class, a human pose, or a language prompt.  A supported object is an
object whose lower face is close to the target's upper support plane and whose
horizontal footprint substantially overlaps the target.  The free interaction
region is then sampled above support-surface cells not occupied by any scene box.
"""

from __future__ import annotations

import math

import numpy as np

from .contracts import Box, InteractionRegion, TargetViewSettings


def _horizontal_indices(up_index: int) -> list[int]:
    return [axis for axis in range(3) if axis != up_index]


def _footprint_overlap(target: Box, child: Box, up_index: int) -> tuple[float, float]:
    horizontal = _horizontal_indices(up_index)
    target_low, target_high = np.asarray(target.minimum), np.asarray(target.maximum)
    child_low, child_high = np.asarray(child.minimum), np.asarray(child.maximum)
    overlap = np.maximum(
        0.0,
        np.minimum(target_high[horizontal], child_high[horizontal])
        - np.maximum(target_low[horizontal], child_low[horizontal]),
    )
    child_extent = child_high[horizontal] - child_low[horizontal]
    target_extent = target_high[horizontal] - target_low[horizontal]
    overlap_area = float(np.prod(overlap))
    child_area = float(np.prod(child_extent))
    target_area = float(np.prod(target_extent))
    return overlap_area / max(child_area, 1e-12), child_area / max(target_area, 1e-12)


def detect_supported_objects(
    target: Box,
    obstacles: list[Box],
    settings: TargetViewSettings,
    up_index: int,
) -> tuple[list[Box], list[Box], dict]:
    """Split ordinary obstacles into target-supported context and other scene geometry."""
    supported, remaining, relations = [], [], []
    target_top = float(target.maximum[up_index])
    for box in obstacles:
        child_low = float(box.minimum[up_index])
        child_high = float(box.maximum[up_index])
        gap = child_low - target_top
        overlap_fraction, area_ratio = _footprint_overlap(target, box, up_index)
        vertical_match = (
            -settings.support_max_vertical_penetration_m
            <= gap
            <= settings.support_max_vertical_gap_m
        )
        accepted = bool(
            settings.support_detection_enabled
            and vertical_match
            and child_high > target_top + 1e-4
            and overlap_fraction >= settings.support_min_footprint_overlap_fraction
            and area_ratio <= settings.support_max_footprint_area_ratio
        )
        if accepted:
            supported.append(box)
            gap_quality = 1.0 - max(0.0, gap) / max(
                settings.support_max_vertical_gap_m, 1e-9
            )
            confidence = float(
                np.clip(
                    min(
                        overlap_fraction,
                        gap_quality,
                        settings.support_max_footprint_area_ratio / max(area_ratio, 1e-9),
                    ),
                    0.0,
                    1.0,
                )
            )
            relations.append(
                {
                    "supporter_id": target.object_id,
                    "supported_id": box.object_id,
                    "relation": "supported_by_target_top",
                    "vertical_gap_m": gap,
                    "child_footprint_overlap_fraction": overlap_fraction,
                    "child_to_target_footprint_area_ratio": area_ratio,
                    "confidence": confidence,
                }
            )
        else:
            remaining.append(box)
    graph = {
        "format_version": "camera_support_graph_v1",
        "method": "aabb_top_contact_and_horizontal_footprint_overlap",
        "target_id": target.object_id,
        "supported_object_ids": [box.object_id for box in supported],
        "relations": relations,
        "thresholds": {
            "max_vertical_gap_m": settings.support_max_vertical_gap_m,
            "max_vertical_penetration_m": settings.support_max_vertical_penetration_m,
            "min_child_footprint_overlap_fraction": (
                settings.support_min_footprint_overlap_fraction
            ),
            "max_child_to_target_footprint_area_ratio": (
                settings.support_max_footprint_area_ratio
            ),
        },
        "limitations": [
            "AABB contact is a conservative support proxy, not a semantic classifier",
            "rotated or concave support surfaces may require mesh-level downward-ray confirmation",
        ],
    }
    return supported, remaining, graph


def _target_frame(target: Box, target_obb) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_obb is not None:
        data = target_obb.model_dump() if hasattr(target_obb, "model_dump") else target_obb
        return (
            np.asarray(data["center"], float),
            np.asarray(data["axes"], float),
            np.asarray(data["half_extents"], float),
        )
    low, high = np.asarray(target.minimum, float), np.asarray(target.maximum, float)
    return (low + high) / 2, np.eye(3), (high - low) / 2


def _inside_any_box(points: np.ndarray, boxes: list[Box], clearance: float) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    for box in boxes:
        low = np.asarray(box.minimum, float) - clearance
        high = np.asarray(box.maximum, float) + clearance
        result |= np.all((points >= low) & (points <= high), axis=1)
    return result


def build_interaction_region(
    target: Box,
    target_obb,
    supported: list[Box],
    other_obstacles: list[Box],
    settings: TargetViewSettings,
    up_index: int,
) -> tuple[InteractionRegion | None, dict]:
    """Sample free support cells and vertical unknown-human proxy points.

    The support-aware domain is activated only when supported context exists.
    This avoids pretending that the highest point of every arbitrary furniture
    mesh is a usable support surface (for example, a sofa backrest).
    """
    if not supported:
        return None, {
            "format_version": "camera_interaction_region_v1",
            "status": "not_applicable",
            "reason": "no_supported_objects_detected",
        }
    center, axes, half = _target_frame(target, target_obb)
    world_up = np.eye(3)[up_index]
    local_up = int(np.argmax(np.abs(axes.T @ world_up)))
    if float(axes[:, local_up] @ world_up) < 0:
        axes[:, local_up] *= -1
    horizontal = [axis for axis in range(3) if axis != local_up]
    widths = 2 * half[horizontal]
    ratio = float(widths[0] / max(widths[1], 1e-9))
    count0 = max(2, int(round(math.sqrt(settings.interaction_anchor_count * ratio))))
    count1 = max(2, int(math.ceil(settings.interaction_anchor_count / count0)))
    coordinates = [
        np.linspace(-half[axis], half[axis], count, endpoint=False)
        + half[axis] / count
        for axis, count in zip(horizontal, (count0, count1), strict=True)
    ]
    local = np.zeros((count0 * count1, 3), dtype=float)
    local[:, local_up] = half[local_up] + 1e-4
    local[:, horizontal[0]] = np.repeat(coordinates[0], count1)
    local[:, horizontal[1]] = np.tile(coordinates[1], count0)
    anchors = center + local @ axes.T

    scene_boxes = [*supported, *other_obstacles]
    # A support cell is occupied if any point in its representative body column
    # intersects scene geometry after a configurable human-clearance expansion.
    offsets = np.asarray(settings.interaction_height_offsets_m, float)
    proxy = (
        anchors[:, None, :]
        + offsets[None, :, None] * world_up[None, None, :]
    )
    proxy_occupied = _inside_any_box(
        proxy.reshape(-1, 3), scene_boxes, settings.interaction_clearance_m
    ).reshape(len(anchors), len(offsets))
    occupied = proxy_occupied.any(axis=1)
    free_anchors = anchors[~occupied]
    occupied_anchors = anchors[occupied]
    samples = (
        free_anchors[:, None, :]
        + offsets[None, :, None] * world_up[None, None, :]
    ).reshape(-1, 3)
    if not len(samples):
        return None, {
            "format_version": "camera_interaction_region_v1",
            "status": "blocked",
            "reason": "supported_surface_has_no_collision_free_proxy_column",
            "support_object_ids": [box.object_id for box in supported],
            "anchor_count": len(anchors),
            "occupied_anchor_count": len(occupied_anchors),
        }
    region = InteractionRegion(
        free_anchor_points_world_m=[tuple(point) for point in free_anchors.tolist()],
        occupied_anchor_points_world_m=[tuple(point) for point in occupied_anchors.tolist()],
        sample_points_world_m=[tuple(point) for point in samples.tolist()],
        support_object_ids=[box.object_id for box in supported],
    )
    manifest = {
        "format_version": "camera_interaction_region_v1",
        "status": "ready",
        "mode": region.mode,
        "target_id": target.object_id,
        "support_object_ids": region.support_object_ids,
        "anchor_count": len(anchors),
        "free_anchor_count": len(free_anchors),
        "occupied_anchor_count": len(occupied_anchors),
        "sample_count": len(samples),
        "height_offsets_m": settings.interaction_height_offsets_m,
        "clearance_m": settings.interaction_clearance_m,
        "free_anchor_points_world_m": region.free_anchor_points_world_m,
        "occupied_anchor_points_world_m": region.occupied_anchor_points_world_m,
        "sample_points_world_m": region.sample_points_world_m,
        "semantics": region.semantics,
    }
    return region, manifest
