"""Read-only contract probe. Run with YSynthetic's lightweight Python and -B."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ysynthetic-root", required=True)
    parser.add_argument("--rig", required=True)
    parser.add_argument("--scene")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(args.ysynthetic_root) / "src"))
    from ysynthetic.schemas import CameraRig, Scene
    from ysynthetic.config import load_scene_bundle

    rig = CameraRig.from_json(args.rig)
    result = {"rig_valid": True, "scene_id": rig.scene_id, "camera_count": len(rig.cameras)}
    if args.scene:
        scene = Scene.from_yaml(args.scene)
        assert scene.scene_id == rig.scene_id
        assert {v.camera_id for v in scene.views} <= {c.camera_id for c in rig.cameras}
        result["scene_schema_valid"] = True
        bundle = load_scene_bundle(Path(args.scene).resolve(), args.ysynthetic_root)
        result["scene_bundle_valid"] = bundle.scene.scene_id == rig.scene_id
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
