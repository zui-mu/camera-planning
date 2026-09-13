"""Re-select a frozen camera pool without importing geometry or rendering anything."""

import json
from pathlib import Path

import numpy as np

from .artifacts import file_hash, json_hash, new_output, write_json
from .contracts import Camera, PlanningRequest
from .scoring import Evidence, evaluate_set
from .selection import select


def load_cached_evidence(plan_dir):
    source = Path(plan_dir).resolve()
    request_data = json.loads((source / "request.json").read_text(encoding="utf-8"))
    if request_data.get("planning_mode") == "target_viewspace":
        raise ValueError("target_viewspace selection requires path geometry; use replan-geometry instead")
    pool = json.loads((source / "candidate_cameras.json").read_text(encoding="utf-8"))
    result = json.loads((source / "camera_plan_result.json").read_text(encoding="utf-8"))
    for actual, expected in (
        (json_hash(request_data), result["request_hash"]),
        (json_hash(pool), result["candidate_rig_hash"]),
        (file_hash(source / "evidence.npz"), result["evidence_sha256"]),
    ):
        if actual != expected:
            raise ValueError("frozen pool/request/evidence hash mismatch; regenerate the plan")
    cameras = [Camera.model_validate(c) for c in pool["cameras"]]
    with np.load(source / "evidence.npz", allow_pickle=False) as archive:
        evidence = Evidence(
            cameras,
            **{
                key: archive[key].copy()
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
            weights=archive["weights"].copy() if "weights" in archive.files else None,
        )
    if evidence.visible.shape != (len(cameras), len(evidence.points)):
        raise ValueError("invalid cached evidence dimensions")
    return PlanningRequest.model_validate(request_data), evidence


def select_cached(plan_dir, output, method="greedy", budget=None, seed=0):
    if Path(output).exists():
        raise FileExistsError(output)
    request, evidence = load_cached_evidence(plan_dir)
    budget = request.budget if budget is None else budget
    indices, trace = select(evidence, budget, request.scoring, method, seed, request.up_index)
    root = new_output(output)
    write_json(
        root / "cameras.json",
        {
            "scene_id": request.scene_id,
            "cameras": [evidence.cameras[i].model_dump(mode="json") for i in indices],
        },
    )
    result = {
        "source_plan": str(Path(plan_dir).resolve()),
        "method": method,
        "budget": budget,
        "selected_ids": [evidence.cameras[i].camera_id for i in indices],
        "metrics": evaluate_set(evidence, indices, request.scoring),
        "trace": trace,
        "geometry_queries": 0,
        "render_calls": 0,
        "trajectory_status": "not_replanned",
    }
    write_json(root / "selection_result.json", result)
    return result
