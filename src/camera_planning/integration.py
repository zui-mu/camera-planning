"""Artifact-only integration; never imports or edits YSynthetic during planning."""

from pathlib import Path

from .artifacts import write_json
from .planner import rig


def write_scene_package(directory, scene_id, cameras, empty_dir, human_dir, outputs_dir):
    directory = Path(directory).resolve()
    write_json(directory / "cameras.json", rig(scene_id, cameras))
    scene = {
        "scene_id": scene_id,
        "description": "External camera experiment; reference/render provenance lives in experiment manifest.",
        "paths": {
            "camera_file": str(directory / "cameras.json"),
            "empty_views_dir": str(Path(empty_dir).resolve()),
            "output_dir": str(Path(outputs_dir).resolve()),
        },
        "artifacts": {
            "empty_views": {
                "type": "view_images",
                "path": str(Path(empty_dir).resolve()),
                "view_mapping": "default",
            },
            "inpaint": {
                "type": "view_images",
                "path": str(Path(human_dir).resolve()),
                "view_mapping": "default",
            },
        },
        "views": [{"view_id": c.camera_id, "camera_id": c.camera_id} for c in cameras],
    }
    # The loader dispatches by .yaml/.yml suffix; valid JSON content alone is insufficient.
    import yaml

    (directory / "scene.yaml").write_text(
        yaml.safe_dump(scene, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return directory / "scene.yaml"


def command_plan(experiment_dir, ysynthetic_root, python_executable, body_config, fuse_config):
    """Print-only commands. Caller must inspect files/config/model and run explicitly."""
    import json

    root = Path(experiment_dir).resolve()
    ys = Path(ysynthetic_root).resolve()
    if root == ys or ys in root.parents:
        raise ValueError("experiment outputs must be outside YSynthetic")
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    prefix = [str(Path(python_executable).resolve()), "-B", "-m", "ysynthetic"]
    common = ["--project-root", str(ys)]
    commands = [
        {
            "stage": "body_fit_pool",
            "requires": ["human renders for every candidate"],
            "argv": prefix
            + [
                "body-fit-batch",
                "--scene-id",
                manifest["scene_id"],
                "--image-dir",
                str(root / "renders" / "human"),
                "--camera-file",
                str(root / "pool_cameras.json"),
                "--output-dir",
                str(root / "body_fit_pool"),
                "--backend",
                "sam3d_body",
                "--config",
                str(Path(body_config).resolve()),
            ]
            + common,
        }
    ]
    skipped_duplicates = []
    for entry in manifest["sets"]:
        if entry.get("duplicate_of"):
            skipped_duplicates.append(
                {"set_id": entry["set_id"], "duplicate_of": entry["duplicate_of"]}
            )
            continue
        set_dir = root / "sets" / entry["set_id"]
        scene_arg = ["--scene", str(set_dir / "scene.yaml")]
        commands += [
            {
                "stage": f"pose_consensus:{entry['set_id']}",
                "requires": [
                    "body_fit_pool",
                    "explicitly prepared/measured scene_consistency inputs",
                ],
                "argv": prefix
                + ["pose-consensus"]
                + scene_arg
                + [
                    "--body-fit-dir",
                    str(root / "body_fit_pool"),
                    "--scene-consistency-dir",
                    str(root / "scene_consistency_pool"),
                    "--output-dir",
                    str(set_dir / "outputs"),
                ]
                + common,
            },
            {
                "stage": f"body_fuse:{entry['set_id']}",
                "requires": ["pose_consensus successful"],
                "argv": prefix
                + ["body-fuse"]
                + scene_arg
                + [
                    "--body-fit-dir",
                    str(root / "body_fit_pool"),
                    "--pose-consensus-dir",
                    str(set_dir / "outputs" / "pose_consensus"),
                    "--output-dir",
                    str(set_dir / "outputs"),
                    "--config",
                    str(Path(fuse_config).resolve()),
                ]
                + common,
            },
        ]
    return {
        "execution": "NOT_EXECUTED",
        "cwd": str(root),
        "environment": {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ys / "src")},
        "scene_gate": "Explicitly run prepare-controls for verified controlled renders, OR use real measured scene_consistency outputs. No automatic stub substitution.",
        "skipped_duplicate_sets": skipped_duplicates,
        "commands": commands,
    }
