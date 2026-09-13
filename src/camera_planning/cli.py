import argparse
import json
from pathlib import Path

from .artifacts import new_output, write_json
from .cached_selection import select_cached
from .contracts import Camera
from .controls import prepare_constant_scene_control
from .evaluation import evaluate_fused
from .execution import execute_command_plan
from .experiments import prepare_rq1
from .integration import command_plan, write_scene_package
from .planner import load_request, plan
from .reporting import summarize_experiment
from .workflow import run_pilot


def main():
    parser = argparse.ArgumentParser(description="Geometry-only camera planning prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--method",
        default="greedy",
        choices=[
            "greedy",
            "random",
            "uniform",
            "farthest",
            "coverage",
            "exact",
            "lazy_coverage",
            "surface_coverage",
            "milp_coverage",
        ],
    )
    p.add_argument("--seed", default=0, type=int)
    p = commands.add_parser("select-cached")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--method",
        default="greedy",
        choices=[
            "greedy",
            "coverage",
            "lazy_coverage",
            "surface_coverage",
            "milp_coverage",
            "random",
            "uniform",
            "farthest",
            "exact",
        ],
    )
    p = commands.add_parser("prepare-rq1")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--random-sets", type=int, default=12)
    p = commands.add_parser("integration-plan")
    for name in ("experiment", "ysynthetic-root", "python", "body-config", "fuse-config", "output"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("evaluate")
    for name in ("reference", "prediction", "output"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("export-scene")
    for name in ("rig", "output", "empty-dir", "human-dir"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("prepare-controls")
    p.add_argument("--experiment", required=True)
    p.add_argument("--human-object", action="append", required=True)
    p.add_argument("--acknowledge-assumption", action="store_true")
    p = commands.add_parser("run-pilot")
    p.add_argument("--config", required=True)
    p = commands.add_parser("select-human-frames")
    p.add_argument("--trajectory", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=int, default=8)
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--min-shared", type=int, default=6)
    p.add_argument("--max-error-px", type=float, default=4.0)
    p.add_argument("--min-angle-degrees", type=float, default=3.0)
    p.add_argument("--max-frames", type=int, default=96)
    p = commands.add_parser("replan-geometry")
    p.add_argument("--geometry-request", required=True)
    p.add_argument("--pilot-config", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("execute-integration")
    p.add_argument("--commands", required=True)
    p.add_argument("--acknowledge-heavy", action="store_true")
    p.add_argument("--resume", action="store_true")
    p = commands.add_parser("summarize-rq1")
    p.add_argument("--experiment", required=True)
    p.add_argument("--output")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(load_request(args.request), args.output, args.method, args.seed)
    elif args.command == "select-cached":
        result = select_cached(args.plan_dir, args.output, args.method, args.budget, args.seed)
    elif args.command == "prepare-rq1":
        result = prepare_rq1(args.plan_dir, args.output, args.reference, args.random_sets)
    elif args.command == "integration-plan":
        result = command_plan(
            args.experiment, args.ysynthetic_root, args.python, args.body_config, args.fuse_config
        )
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
    elif args.command == "evaluate":
        result = evaluate_fused(args.reference, args.prediction)
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
    elif args.command == "prepare-controls":
        result = prepare_constant_scene_control(
            args.experiment, args.human_object, args.acknowledge_assumption
        )
    elif args.command == "run-pilot":
        result = run_pilot(args.config)
    elif args.command == "select-human-frames":
        from .human_frames import select_human_frames

        result = select_human_frames(args.trajectory, args.detections, args.output, args.budget,
                                     args.confidence, args.min_shared, args.max_error_px,
                                     args.min_angle_degrees, args.max_frames)
    elif args.command == "replan-geometry":
        from .workflow import replan_geometry

        result = replan_geometry(args.geometry_request, args.pilot_config, args.output)
    elif args.command == "execute-integration":
        result = execute_command_plan(
            args.commands, acknowledge_heavy=args.acknowledge_heavy, resume=args.resume
        )
    elif args.command == "summarize-rq1":
        result = summarize_experiment(args.experiment, args.output)
    else:
        source_rig = json.loads(Path(args.rig).read_text(encoding="utf-8"))
        cameras = [Camera.model_validate(c) for c in source_rig["cameras"]]
        output = new_output(args.output)
        scene = write_scene_package(
            output,
            source_rig["scene_id"],
            cameras,
            args.empty_dir,
            args.human_dir,
            output / "outputs",
        )
        result = {"scene": str(scene), "images_exist": "not_checked"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
