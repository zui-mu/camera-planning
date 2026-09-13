import numpy as np

from .contracts import PlanningRequest
from .geometry import GeometryBackend, look_at


def _horizontal_obb(vertices, up_index):
    """Deterministic PCA-oriented box used only to organize the candidate shell."""
    horizontal = [axis for axis in range(3) if axis != up_index]
    planar = np.asarray(vertices)[:, horizontal]
    covariance = np.cov(planar - planar.mean(axis=0), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis0 = vectors[:, np.argmax(values)]
    if axis0[np.argmax(abs(axis0))] < 0:
        axis0 = -axis0
    axis1 = np.array([-axis0[1], axis0[0]])
    basis2 = np.column_stack((axis0, axis1))
    projected = planar @ basis2
    low, high = projected.min(axis=0), projected.max(axis=0)
    center2 = (low + high) / 2
    center = np.zeros(3)
    center[horizontal] = basis2 @ center2
    center[up_index] = (vertices[:, up_index].min() + vertices[:, up_index].max()) / 2
    return center, basis2, (high - low) / 2


def _distance_to_target_obb(position, center, basis2, half2, target_low, target_high, up_index):
    horizontal = [axis for axis in range(3) if axis != up_index]
    local2 = (position[horizontal] - center[horizontal]) @ basis2
    horizontal_outside = np.maximum(abs(local2) - half2, 0)
    up_outside = max(
        target_low[up_index] - position[up_index],
        position[up_index] - target_high[up_index],
        0,
    )
    return float(np.linalg.norm(np.append(horizontal_outside, up_outside)))


def generate_candidates(request: PlanningRequest, geometry: GeometryBackend):
    vertices = geometry.object_vertices(request.target.object_id)
    center, basis2, half2 = _horizontal_obb(vertices, request.up_index)
    target_low = np.asarray(request.target.minimum)
    target_high = np.asarray(request.target.maximum)
    horizontal = [i for i in range(3) if i != request.up_index]
    up = np.eye(3)[request.up_index]
    cameras, rejected = [], []
    seen_poses = set()
    serial = 0
    for offset in request.candidates.shell_offsets_m:
        expanded_half = half2 + offset
        for height in request.candidates.heights_above_floor_m:
            for focus_height in request.candidates.focus_heights_above_floor_m:
                for angle in np.linspace(
                    0, 2 * np.pi, request.candidates.azimuth_count, endpoint=False
                ):
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    distance = np.min(expanded_half / np.maximum(abs(direction), 1e-12))
                    position = center.copy()
                    position[horizontal] += basis2 @ (direction * distance)
                    position[request.up_index] = request.floor_height_m + height
                    focus = center.copy()
                    focus[request.up_index] = request.floor_height_m + focus_height
                    camera_id = f"cam_{serial:04d}"
                    serial += 1
                    margin = request.candidates.camera_clearance_m
                    region = request.allowed_camera_region
                    fits = np.all(position >= np.array(region.minimum) + margin) and np.all(
                        position <= np.array(region.maximum) - margin
                    )
                    target_distance = _distance_to_target_obb(
                        position,
                        center,
                        basis2,
                        half2,
                        target_low,
                        target_high,
                        request.up_index,
                    )
                    reason = None
                    if not fits:
                        reason = "outside_allowed_region_or_clearance"
                    elif target_distance < request.candidates.min_target_distance_m:
                        reason = "too_close_for_declared_full_body_framing"
                    elif target_distance > request.candidates.max_target_distance_m:
                        reason = "beyond_declared_camera_distance"
                    elif geometry.occupied(position[None, :], margin)[0]:
                        reason = "camera_in_obstacle_or_clearance"
                    if reason:
                        rejected.append(
                            {
                                "camera_id": camera_id,
                                "reason": reason,
                                "target_distance_m": target_distance,
                            }
                        )
                        continue
                    pose_key = tuple(np.round(np.concatenate([position, focus]), 8))
                    if pose_key in seen_poses:
                        rejected.append({"camera_id": camera_id, "reason": "duplicate_pose"})
                        continue
                    seen_poses.add(pose_key)
                    cameras.append(look_at(position, focus, up, request.intrinsics, camera_id))
    return cameras, rejected
