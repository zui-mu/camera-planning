import json

import numpy as np

from camera_planning.artifacts import file_hash, write_json
from camera_planning.reporting import summarize_experiment


def fused_fixture(path, translation=0.0):
    np.savez(
        path,
        format_version="mhr_fused_body_parameters_v1",
        profile="mhr_127_v1",
        length_unit="meter",
        mhr_asset_hash="a" * 64,
        checkpoint_hash="b" * 64,
        topology_hash="c" * 64,
        joints_world=np.ones((127, 3)) * translation,
        root_rotation_world=np.eye(3),
    )


def test_summary_evaluates_and_reuses_duplicate(tmp_path):
    root = tmp_path / "experiment"
    reference = tmp_path / "reference.npz"
    prediction = root / "sets" / "greedy_000" / "outputs" / "body_fuse" / "prediction.npz"
    prediction.parent.mkdir(parents=True)
    fused_fixture(reference)
    fused_fixture(prediction, 0.1)
    proxy = {
        "total": 1.0,
        "coverage": 0.8,
        "multiview": 0.7,
        "information": 0.6,
        "resolution": 0.5,
        "worst_zone": 0.4,
    }
    manifest = {
        "reference": {"path": str(reference), "sha256": file_hash(reference)},
        "sets": [
            {
                "set_id": "greedy_000",
                "method": "greedy",
                "seed": 0,
                "duplicate_of": None,
                "camera_ids": ["a", "b"],
                "proxy_metrics": proxy,
            },
            {
                "set_id": "random_000",
                "method": "random",
                "seed": 0,
                "duplicate_of": "greedy_000",
                "camera_ids": ["a", "b"],
                "proxy_metrics": proxy,
            },
        ],
    }
    write_json(root / "experiment.json", manifest)
    write_json(
        prediction.parent / "body_fuse_result.json",
        {
            "status": "ok",
            "valid": True,
            "active_view_ids": ["a", "b"],
            "used_view_ids": ["a", "b"],
            "outputs": {"fused_body_parameters": str(prediction)},
        },
    )
    summary = summarize_experiment(root)
    assert summary["rows"][0]["status"] == "evaluated"
    assert summary["rows"][1]["status"] == "duplicate_reused"
    assert (root / "summary" / "rq1_summary.csv").is_file()
    loaded = json.loads((root / "summary" / "rq1_summary.json").read_text(encoding="utf-8"))
    assert loaded["evaluated_count"] == 2
