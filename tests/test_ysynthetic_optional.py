"""Set YSYNTHETIC_ROOT to verify the real checkout without changing its files."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from camera_planning.contracts import Intrinsics
from camera_planning.geometry import look_at
from camera_planning.integration import write_scene_package
from camera_planning.evaluation import evaluate_fused
from camera_planning.controls import prepare_constant_scene_control


@pytest.fixture
def ys_root(monkeypatch):
    value = os.environ.get("YSYNTHETIC_ROOT")
    if not value:
        pytest.skip("optional real YSynthetic checkout probe")
    root = Path(value)
    monkeypatch.syspath_prepend(str(root / "src"))
    return root


def test_real_scene_loader(ys_root, tmp_path):
    from ysynthetic.config import load_scene_bundle

    camera = look_at(np.array([0.0, 1.0, 3.0]), [0, 1, 0], [0, 1, 0], Intrinsics(), "cam_0")
    scene = write_scene_package(
        tmp_path / "scene",
        "test",
        [camera],
        tmp_path / "empty",
        tmp_path / "human",
        tmp_path / "outputs",
    )
    bundle = load_scene_bundle(scene, ys_root)
    assert bundle.scene.scene_id == "test"
    assert bundle.output_dir == tmp_path / "outputs"


def test_real_control_schema(ys_root, tmp_path):
    from ysynthetic.schemas import SceneConsistencyResult
    from test_controls import fixture_renders

    fixture_renders(tmp_path)
    result = prepare_constant_scene_control(tmp_path, ["RefHuman"], True)
    for path in result["records"]:
        record = SceneConsistencyResult.model_validate(json.loads(Path(path).read_text()))
        assert not record.metadata["is_measured_scene_score"]
        assert record.r_scene == 1


def test_actual_reference_self_comparison(ys_root):
    reference = (
        ys_root
        / ".verification/body_fit_real/hierarchical_pipeline/body_fuse/fused_body_parameters.npz"
    )
    if not reference.exists():
        pytest.skip("local reference asset absent")
    metrics = evaluate_fused(reference, reference)
    assert metrics["world_mpjpe_mm"] == 0
    assert metrics["root_aligned_mpjpe_mm"] == 0
    assert metrics["root_rotation_error_deg"] < 0.1
