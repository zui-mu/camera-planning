from pathlib import Path

import numpy as np
import yaml

from . import __version__
from .artifacts import file_hash, json_hash, new_output, write_json
from .candidates import generate_candidates
from .contracts import PlanningRequest
from .geometry import load_geometry, look_at
from .observation import sample_observation_data
from .refinement import check_evidence_budget, filter_evidence, pose_rejection, refine
from .scoring import build_evidence, evaluate_set


def load_request(path):
    path = Path(path).resolve()
    request = PlanningRequest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if request.geometry.mesh_file:
        mesh = Path(request.geometry.mesh_file)
        if not mesh.is_absolute():
            request.geometry.mesh_file = str((path.parent / mesh).resolve())
    return request


def _prepare(request):
    geometry = load_geometry(request)
    points, zones, kinds, weights, metadata = sample_observation_data(request, geometry)
    cameras, rejected = generate_candidates(request, geometry)
    if request.observation.mode == "furniture_free_space":
        center = (np.asarray(request.target.minimum) + request.target.maximum) / 2
        up = np.eye(3)[request.up_index]
        legal = []
        for old in cameras:
            camera = look_at(old.center, center, up, request.intrinsics, old.camera_id)
            reason = pose_rejection(request, geometry, camera)
            # Step back along this view direction only when framing is too tight.
            for _ in range(3):
                if reason != "target_bounds_out_of_frame":
                    break
                camera = look_at(
                    center + (camera.center - center) * 1.15,
                    center,
                    up,
                    request.intrinsics,
                    old.camera_id,
                )
                reason = pose_rejection(request, geometry, camera)
            if reason:
                rejected.append(
                    {
                        "camera_id": camera.camera_id,
                        "reason": reason,
                        "position_world_m": camera.center.tolist(),
                    }
                )
            elif not any(np.allclose(camera.center, c.center, atol=1e-7) for c in legal):
                legal.append(camera)
        cameras = legal
    if request.adaptive.enabled and len(cameras) > request.adaptive.max_candidates:
        keep = set(np.linspace(0, len(cameras) - 1, request.adaptive.max_candidates, dtype=int))
        for i, camera in enumerate(cameras):
            if i not in keep:
                rejected.append(
                    {"camera_id": camera.camera_id, "reason": "initial_candidate_budget"}
                )
        cameras = [camera for i, camera in enumerate(cameras) if i in keep]
    if not cameras:
        raise ValueError("no camera is geometrically feasible; check bounds/shells/units")
    check_evidence_budget(request, len(cameras), len(points))
    evidence = build_evidence(
        cameras,
        points,
        zones,
        geometry,
        request.scoring,
        kinds,
        request.observation.image_margin_ratio,
        weights,
    )
    evidence = filter_evidence(request, evidence, rejected)
    if not evidence.cameras:
        raise ValueError("no camera passes surface and observation-domain visibility thresholds")
    metadata["initial_raycast_camera_rows"] = len(cameras)
    return evidence, rejected, geometry, metadata


def prepare(request):
    evidence, rejected, _, _ = _prepare(request)
    return evidence, rejected


def rig(scene_id, cameras):
    # Do not add units/up_axis/score here: YSynthetic CameraRig forbids extra fields.
    return {"scene_id": scene_id, "cameras": [c.model_dump(mode="json") for c in cameras]}


def plan(request: PlanningRequest, output, method="greedy", seed=0):
    if request.planning_mode == "target_viewspace":
        from .target_viewspace import plan_target_views

        if method != "greedy":
            raise ValueError("target_viewspace currently implements joint greedy insertion only")
        return plan_target_views(request, output, seed)
    if Path(output).exists():
        raise FileExistsError(output)
    evidence, rejected, geometry, domain_metadata = _prepare(request)
    evidence, indices, trace, adaptive_trace = refine(
        request,
        geometry,
        evidence,
        rejected,
        method,
        seed,
    )
    output = new_output(output)
    request_dict = request.model_dump(mode="json")
    selected_cameras = [evidence.cameras[i] for i in indices]
    write_json(output / "request.json", request_dict)
    write_json(output / "cameras.json", rig(request.scene_id, selected_cameras))
    write_json(
        output / "camera_placements.json",
        {
            "schema_version": "camera_placements_v1",
            "scene_id": request.scene_id,
            "length_unit": "meter",
            "coordinate_system": "source_scene_world",
            "cameras": [
                {
                    "camera_id": camera.camera_id,
                    "position_world_m": camera.center.tolist(),
                    "view_direction_world": np.asarray(camera.R).T[:, 2].tolist(),
                }
                for camera in selected_cameras
            ],
        },
    )
    write_json(output / "candidate_cameras.json", rig(request.scene_id, evidence.cameras))
    np.savez_compressed(
        output / "evidence.npz",
        points=evidence.points,
        zones=evidence.zones,
        visible=evidence.visible,
        information=evidence.information,
        resolution=evidence.resolution,
        directions=evidence.directions,
        kinds=evidence.kinds,
        weights=evidence.weights,
    )
    write_json(output / "observation_domain.json", domain_metadata)
    trajectory_status = "not_planned"
    if request.path.enabled:
        from .trajectory import plan_open_path

        trajectory = plan_open_path(request, geometry, selected_cameras)
        trajectory["rig_hash"] = json_hash(rig(request.scene_id, selected_cameras))
        write_json(output / "camera_path.json", trajectory)
        trajectory_status = trajectory["status"]
    pool_diagnostic = None
    if request.scoring.objective == "robust_space":
        from .observability import point_quality, spatial_metrics

        selected_quality = point_quality(evidence, indices, request.scoring)
        pool_quality = point_quality(evidence, range(len(evidence.cameras)), request.scoring)
        np.savez_compressed(
            output / "observability.npz",
            points=evidence.points[evidence.kinds == 2],
            weights=evidence.weights[evidence.kinds == 2],
            selected_quality=selected_quality,
            candidate_pool_quality=pool_quality,
        )
        pool_diagnostic = {
            "semantics": "all-candidate reference bound; ignores camera budget and path constraints; not a physical ceiling",
            "metrics": spatial_metrics(evidence, range(len(evidence.cameras)), request.scoring),
            "candidate_pool_zero_quality_points": int((pool_quality <= 1e-8).sum()),
            "hard_regions_removed_from_objective": False,
        }
    result = {
        "schema_version": "camera_plan_result_v2",
        "package_version": __version__,
        "implementation_hash": json_hash(
            {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
        ),
        "scene_id": request.scene_id,
        "request_hash": json_hash(request_dict),
        "candidate_rig_hash": json_hash(rig(request.scene_id, evidence.cameras)),
        "evidence_sha256": file_hash(output / "evidence.npz"),
        "length_unit": "meter",
        "up_axis": request.up_axis,
        "geometry_backend": (
            "solid_aabb_proxy_v1" if request.geometry.backend == "aabb" else "triangle_mesh_bvh_v1"
        ),
        "method": method,
        "seed": seed,
        "budget": request.budget,
        "feasible_count": len(evidence.cameras),
        "selected_ids": [evidence.cameras[i].camera_id for i in indices],
        "metrics": evaluate_set(evidence, indices, request.scoring),
        "trace": trace,
        "rejected": rejected,
        "trajectory_status": trajectory_status,
        "reconstruction_status": "not_run",
        "observation_domain": (
            "scene_constrained_furniture_free_space_v1"
            if request.observation.mode == "furniture_free_space"
            else "pose_agnostic_furniture_surface_and_framing_envelope_v1"
        ),
        "adaptive_trace": adaptive_trace,
        "candidate_pool_reference": pool_diagnostic,
        "limitations": (
            ["AABB proxy, not exact mesh collision"]
            if request.geometry.backend == "aabb"
            else [
                "triangle mesh uses static line-of-sight and surface-clearance queries; "
                "it does not model lens housing or a continuous camera path"
            ]
        )
        + [
            "no person location or pose is predicted during planning",
            (
                "no body scale used by the free-space domain; grid resolution and neighborhood radius are explicit"
                if request.observation.mode == "furniture_free_space"
                else "legacy envelope includes a declared full-body height; not a zero-body-scale baseline"
            ),
            "human self-occlusion is measured only by the downstream reconstruction experiment",
            "geometry utility is not measured HMR accuracy",
            "any path guarantee is conditional on declared static geometry and numerical tolerances",
        ],
    }
    write_json(output / "camera_plan_result.json", result)
    return result
