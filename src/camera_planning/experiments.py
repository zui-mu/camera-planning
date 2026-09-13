"""RQ1 experiment packaging. This module may see H_ref; planner.py must not."""

import json
from pathlib import Path

import numpy as np

from .artifacts import file_hash, json_hash, new_output, write_json
from .contracts import Camera, PlanningRequest
from .integration import write_scene_package
from .scoring import Evidence, evaluate_set
from .selection import select


def prepare_rq1(plan_dir, output, reference_npz, random_sets=12):
    if random_sets < 1:
        raise ValueError("at least one random set required")
    source = Path(plan_dir).resolve()
    reference = Path(reference_npz).resolve()
    with np.load(reference, allow_pickle=False) as ref:
        if str(ref["format_version"].item()) != "mhr_fused_body_parameters_v1":
            raise ValueError("reference must be a fused MHR v1 artifact")
        if str(ref["length_unit"].item()) != "meter":
            raise ValueError("reference must use meters")
    request_dict = json.loads((source / "request.json").read_text(encoding="utf-8"))
    request = PlanningRequest.model_validate(request_dict)
    if request.planning_mode == "target_viewspace":
        raise ValueError("target_viewspace has no human-space evidence; use select-human-frames after generation")
    pool = json.loads((source / "candidate_cameras.json").read_text(encoding="utf-8"))
    planned = json.loads((source / "camera_plan_result.json").read_text(encoding="utf-8"))
    if (
        planned["request_hash"] != json_hash(request_dict)
        or planned["candidate_rig_hash"] != json_hash(pool)
        or planned["evidence_sha256"] != file_hash(source / "evidence.npz")
    ):
        raise ValueError(
            "source plan changed; regenerate evidence before selecting experiment sets"
        )
    cameras = [Camera.model_validate(c) for c in pool["cameras"]]
    with np.load(source / "evidence.npz", allow_pickle=False) as a:
        evidence = Evidence(
            cameras,
            **{
                key: a[key].copy()
                for key in (
                    "points",
                    "zones",
                    "visible",
                    "information",
                    "resolution",
                    "directions",
                    "kinds",
                )
            },
            weights=a["weights"].copy() if "weights" in a.files else None,
        )
    if evidence.visible.shape != (len(cameras), len(evidence.points)):
        raise ValueError("camera/evidence shape mismatch")
    selections = [(method, 0) for method in ("uniform", "farthest", "coverage", "greedy")]
    if request.scoring.objective == "robust_space":
        selections += [("surface_coverage", 0), ("lazy_coverage", 0), ("planned_incumbent", 0)]
    selections += [("random", seed) for seed in range(random_sets)]
    sets = []
    seen = {}
    used_indices = set()
    for method, seed in selections:
        if method == "planned_incumbent":
            by_id = {camera.camera_id: i for i, camera in enumerate(cameras)}
            indices = [by_id[camera_id] for camera_id in planned["selected_ids"]]
        else:
            indices, _ = select(
                evidence, request.budget, request.scoring, method, seed, request.up_index
            )
        selected = [cameras[i] for i in sorted(indices)]
        used_indices.update(int(i) for i in indices)
        set_id = f"{method}_{seed:03d}"
        signature = tuple(c.camera_id for c in selected)
        sets.append(
            {
                "set_id": set_id,
                "method": method,
                "seed": seed,
                "camera_ids": list(signature),
                "duplicate_of": seen.get(signature),
                "proxy_metrics": evaluate_set(evidence, indices, request.scoring),
                "reconstruction_status": "not_run",
            }
        )
        seen.setdefault(signature, set_id)
    # Body fitting is expensive. Render and fit only the union of cameras that
    # actually appears in a tested set, rather than the entire feasible pool.
    used_pool = {
        "scene_id": request.scene_id,
        "cameras": [cameras[i].model_dump(mode="json") for i in sorted(used_indices)],
    }
    output = new_output(output)
    write_json(output / "pool_cameras.json", used_pool)
    # Set files are written after the root is reserved, so a failed experiment
    # cannot silently reuse an old directory.
    by_id = {camera.camera_id: camera for camera in cameras}
    for entry in sets:
        selected = [by_id[camera_id] for camera_id in entry["camera_ids"]]
        set_dir = output / "sets" / entry["set_id"]
        write_scene_package(
            set_dir,
            request.scene_id,
            selected,
            output / "renders" / "empty",
            output / "renders" / "human",
            set_dir / "outputs",
        )
    manifest = {
        "schema_version": "camera_rq1_experiment_v1",
        "scene_id": request.scene_id,
        "request_hash": json_hash(request_dict),
        "source_plan": str(source),
        "reference": {
            "path": str(reference),
            "sha256": file_hash(reference),
            "role": "evaluation_and_render_only",
            "not_real_world_ground_truth": True,
        },
        "source_candidate_rig_hash": json_hash(pool),
        "pool_rig_hash": json_hash(used_pool),
        "source_feasible_camera_count": len(cameras),
        "used_camera_count": len(used_pool["cameras"]),
        "budget": request.budget,
        "render_status": "not_run",
        "inference_status": "not_run",
        "scene_consistency_protocol": "UNSET_requires_explicit_choice",
        "sets": sets,
        "required_cache_keys": [
            "scene_snapshot_hash",
            "reference_hash",
            "camera_KRT_and_raster_hash",
            "image_hash",
            "mask_bbox_and_prompts_hash",
            "model_checkpoint_hash",
            "effective_config_hash",
            "code_revision",
            "random_seed",
        ],
        "warning": "No automatic inference cache reuse is implemented. Verify every key before reuse.",
    }
    write_json(output / "experiment.json", manifest)
    return manifest
