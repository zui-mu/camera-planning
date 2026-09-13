"""Strict paired-reference metrics; never interpret optimizer residuals as accuracy."""

from pathlib import Path

import numpy as np

from .artifacts import file_hash


def joint_metrics(reference, prediction, root_index=1, joint_indices=None):
    ref, pred = np.asarray(reference, float), np.asarray(prediction, float)
    if ref.shape != pred.shape or ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError("joint arrays must have identical Nx3 shape")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(pred)):
        raise ValueError("non-finite joints")
    if not 0 <= root_index < len(ref):
        raise ValueError("invalid named root index")
    indices = np.arange(len(ref)) if joint_indices is None else np.asarray(joint_indices, int)
    if not len(indices) or np.any(indices < 0) or np.any(indices >= len(ref)):
        raise ValueError("invalid joint evaluation indices")
    absolute = np.linalg.norm(pred[indices] - ref[indices], axis=1)
    aligned = (pred - pred[root_index]) - (ref - ref[root_index])
    return {
        "joint_count": int(len(indices)),
        "root_index": root_index,
        "world_mpjpe_mm": float(absolute.mean() * 1000),
        "root_aligned_mpjpe_mm": float(np.linalg.norm(aligned[indices], axis=1).mean() * 1000),
        "root_translation_error_mm": float(
            np.linalg.norm(pred[root_index] - ref[root_index]) * 1000
        ),
    }


def evaluate_fused(reference, prediction):
    with np.load(reference, allow_pickle=False) as r, np.load(prediction, allow_pickle=False) as p:
        for field in (
            "format_version",
            "profile",
            "length_unit",
            "mhr_asset_hash",
            "checkpoint_hash",
            "topology_hash",
        ):
            if str(r[field].item()) != str(p[field].item()):
                raise ValueError(f"incompatible {field}; separate experiment stratum required")
        if str(r["length_unit"].item()) != "meter":
            raise ValueError("expected meter")
        if str(r["format_version"].item()) != "mhr_fused_body_parameters_v1":
            raise ValueError("unsupported fused format")
        if str(r["profile"].item()) != "mhr_127_v1":
            raise ValueError("unsupported MHR joint profile")
        if r["joints_world"].shape != (127, 3) or p["joints_world"].shape != (127, 3):
            raise ValueError("expected current MHR-127 joint layout")
        # Joint 0 is body_world, not an anatomical pelvis. Joint 1 is MHR root.
        result = joint_metrics(r["joints_world"], p["joints_world"], 1, range(1, 127))
        rotations = []
        for a in (r["root_rotation_world"], p["root_rotation_world"]):
            if (
                not np.all(np.isfinite(a))
                or not np.allclose(a @ a.T, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(a), 1, atol=1e-4)
            ):
                raise ValueError("invalid root rotation")
            rotations.append(a)
        angle = np.arccos(np.clip((np.trace(rotations[0].T @ rotations[1]) - 1) / 2, -1, 1))
        result["root_rotation_error_deg"] = float(np.rad2deg(angle))
    result.update(
        {
            "metric_semantics": "round_trip_deviation_against_frozen_estimate",
            "reference_path": str(Path(reference).resolve()),
            "reference_sha256": file_hash(reference),
            "prediction_path": str(Path(prediction).resolve()),
            "prediction_sha256": file_hash(prediction),
            "mpvpe_status": "not_computed_requires_corresponding_world_vertices_and_verified_faces",
            "joint_subset_note": "MHR indices 1..126 includes hand/helper joints; add a preregistered body-only subset for paper reporting",
        }
    )
    return result
