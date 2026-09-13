"""One-command Blender acquisition pilot without modifying YSynthetic."""

import json
import subprocess
from pathlib import Path
from typing import Literal

import imageio_ffmpeg
import yaml
from pydantic import Field, model_validator

from .artifacts import file_hash, new_output, write_json
from .contracts import (
    AdaptiveSettings,
    CandidateSettings,
    Intrinsics,
    ObservationSettings,
    PathSettings,
    PlanningRequest,
    ScoreSettings,
    StrictModel,
    TargetViewSettings,
)
from .controls import prepare_constant_scene_control
from .experiments import prepare_rq1
from .integration import command_plan
from .planner import load_request, plan


class IntegrationSettings(StrictModel):
    ysynthetic_root: str
    python: str
    body_config: str
    fuse_config: str


class TourSettings(StrictModel):
    enabled: bool = False
    fps: int = Field(default=24, ge=1, le=120)
    seconds_per_leg: float = Field(default=1.0, gt=0, le=30)
    close_loop: bool = False
    resolution_percentage: int = Field(default=50, ge=10, le=100)


class PilotConfig(StrictModel):
    schema_version: Literal["camera_pilot_config_v2"] = "camera_pilot_config_v2"
    scene_id: str
    planning_mode: Literal["legacy", "target_viewspace"] = "legacy"
    target_view: TargetViewSettings = Field(default_factory=TargetViewSettings)
    source_scene: str
    reference_mesh: str | None = None
    reference_object_name: str = "ReferenceHuman"
    reference_body_fuse_result: str | None = None
    blender_executable: str
    output: str
    # Preferred contract: source_scene is the complete room and target_scene is
    # a separately exported copy of the target furniture in the same world frame.
    # target_object remains as a backwards-compatible explicit full-scene selector.
    target_scene: str | None = None
    target_asset_object: str | None = None
    target_object: str | None = None
    human_objects: list[str] = Field(default_factory=list)
    exclude_objects: list[str] = Field(default_factory=list)
    solid_objects: list[str] = Field(default_factory=list)
    planning_geometry: Literal["aabb_boxes", "triangle_mesh_proxy"] = "aabb_boxes"
    box_ignore_name_tokens: list[str] = Field(default_factory=list)
    box_structure_policy: Literal["auto_geometry", "name_tokens", "none"] = "auto_geometry"
    box_minimum_thickness_m: float = Field(default=0.002, gt=0)
    force_aabb_proxy_objects: list[str] = Field(default_factory=list)
    large_object_proxy_threshold_triangles: int = Field(default=250000, ge=1000)
    target_max_triangles: int = Field(default=60000, ge=1000)
    allowed_region_object: str | None = None
    allowed_min_m: tuple[float, float, float] | None = None
    allowed_max_m: tuple[float, float, float] | None = None
    floor_height_m: float | None = None
    meters_per_blender_unit: float | None = Field(default=None, gt=0)
    budget: int = Field(default=4, ge=2, le=64)
    intrinsics: Intrinsics = Field(default_factory=Intrinsics)
    candidates: CandidateSettings = Field(default_factory=CandidateSettings)
    observation: ObservationSettings = Field(default_factory=ObservationSettings)
    scoring: ScoreSettings = Field(default_factory=ScoreSettings)
    adaptive: AdaptiveSettings = Field(default_factory=AdaptiveSettings)
    path: PathSettings = Field(default_factory=PathSettings)
    method: Literal[
        "greedy",
        "random",
        "uniform",
        "farthest",
        "coverage",
        "exact",
        "lazy_coverage",
        "surface_coverage",
        "milp_coverage",
    ] = "greedy"
    seed: int = Field(default=0, ge=0)
    frame: int = Field(default=1, ge=0)
    reference_npz: str | None = None
    random_sets: int = Field(default=12, ge=1)
    acknowledge_reference_mesh_correspondence: bool = False
    assume_constant_scene_quality: bool = False
    integration: IntegrationSettings | None = None
    tour: TourSettings = Field(default_factory=TourSettings)

    @model_validator(mode="after")
    def consistent(self):
        if self.planning_mode == "target_viewspace":
            if not self.path.enabled or (self.tour.enabled and self.tour.close_loop):
                raise ValueError("target_viewspace requires an enabled open path")
            if self.reference_npz:
                raise ValueError("target_viewspace uses post-generation frame selection; use legacy mode for existing RQ1 evidence experiments")
        if not self.target_scene and not self.target_object:
            raise ValueError("declare target_scene or legacy target_object")
        if (self.allowed_min_m is None) != (self.allowed_max_m is None):
            raise ValueError("allowed_min_m and allowed_max_m must be provided together")
        if not self.allowed_region_object and self.allowed_min_m is None:
            raise ValueError("declare allowed_region_object or explicit allowed_min_m/max_m")
        if self.reference_mesh and self.reference_object_name not in self.human_objects:
            self.human_objects.append(self.reference_object_name)
        if self.reference_npz and not self.human_objects:
            raise ValueError("RQ1 requires the reference-human object names in the Blender scene")
        if self.reference_npz and not self.acknowledge_reference_mesh_correspondence:
            raise ValueError(
                "confirm that reference_npz corresponds to human_objects before preparing RQ1"
            )
        if self.assume_constant_scene_quality and not self.reference_npz:
            raise ValueError("constant-scene control is meaningful only for an RQ1 reference")
        if self.integration and not self.assume_constant_scene_quality:
            raise ValueError(
                "integration commands require controlled scene-consistency or a separately measured input"
            )
        return self


def _resolve(value, base):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_pilot_config(path):
    source = Path(path).resolve()
    config = PilotConfig.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))
    base = source.parent
    for name in (
        "source_scene",
        "target_scene",
        "reference_mesh",
        "reference_body_fuse_result",
        "blender_executable",
        "output",
        "reference_npz",
    ):
        value = getattr(config, name)
        if value:
            setattr(config, name, str(_resolve(value, base)))
    if config.integration:
        for name in ("ysynthetic_root", "python", "body_config", "fuse_config"):
            setattr(
                config.integration,
                name,
                str(_resolve(getattr(config.integration, name), base)),
            )
    return config


def _run(argv, stage, root, cwd=None):
    completed = subprocess.run(
        [str(value) for value in argv],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        cwd=str(cwd) if cwd else None,
    )
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = logs / f"{stage}.stdout.log"
    stderr = logs / f"{stage}.stderr.log"
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    record = {
        "stage": stage,
        "argv": [str(value) for value in argv],
        "returncode": completed.returncode,
        "stdout": str(stdout),
        "stderr": str(stderr),
    }
    if completed.returncode:
        raise RuntimeError(f"{stage} failed; inspect {stderr} and {stdout}")
    return record


def _blender(blender, snapshot, script, arguments):
    command = [blender, "--background"]
    if snapshot:
        command.append(snapshot)
    else:
        command.append("--factory-startup")
    # Without --python-exit-code Blender may print a Python traceback yet return
    # process code 0, which would make the workflow falsely mark a stage successful.
    return command + ["--python-exit-code", "2", "--python", script, "--", *arguments]


def _csv(values):
    return ",".join(str(value) for value in values)


def replan_geometry(geometry_request, pilot_config, output):
    """Explicit reuse of a frozen world-geometry export; no Blender or human loading."""
    request = load_request(geometry_request)
    config = load_pilot_config(pilot_config)
    expected = "aabb" if config.planning_geometry == "aabb_boxes" else "triangle_mesh"
    if request.geometry.backend != expected:
        raise ValueError("cached geometry backend differs from requested pilot geometry")
    data = request.model_dump(mode="json")
    for name in ("intrinsics", "candidates", "observation", "scoring", "adaptive", "path", "target_view"):
        data[name] = getattr(config, name).model_dump(mode="json")
    data["planning_mode"] = config.planning_mode
    data["budget"] = config.budget
    data["scene_id"] = config.scene_id
    result = plan(PlanningRequest.model_validate(data), output, config.method, config.seed)
    write_json(
        Path(output) / "reused_geometry.json",
        {
            "request": str(Path(geometry_request).resolve()),
            "sha256": file_hash(geometry_request),
            "pilot_config": str(Path(pilot_config).resolve()),
            "warning": "uses the original export's geometry and camera-region coordinates; source assets are NOT re-imported or re-verified",
        },
    )
    return result


def _triangle_export_arguments(config, geometry):
    if not config.target_object:
        raise ValueError("triangle_mesh_proxy currently requires target_object")
    arguments = [
        "--output",
        geometry,
        "--scene-id",
        config.scene_id,
        "--target-object",
        config.target_object,
        "--budget",
        config.budget,
        "--width",
        config.intrinsics.width,
        "--height",
        config.intrinsics.height,
        "--fx",
        config.intrinsics.fx,
        "--fy",
        config.intrinsics.fy,
        "--principal-x",
        config.intrinsics.cx,
        "--principal-y",
        config.intrinsics.cy,
        "--azimuth-count",
        config.candidates.azimuth_count,
        "--shell-offsets",
        _csv(config.candidates.shell_offsets_m),
        "--camera-heights",
        _csv(config.candidates.heights_above_floor_m),
        "--focus-heights",
        _csv(config.candidates.focus_heights_above_floor_m),
        "--camera-clearance",
        config.candidates.camera_clearance_m,
        "--min-target-distance",
        config.candidates.min_target_distance_m,
        "--max-target-distance",
        config.candidates.max_target_distance_m,
        "--min-surface-visible-fraction",
        config.candidates.min_surface_visible_fraction,
        "--min-envelope-visible-fraction",
        config.candidates.min_envelope_visible_fraction,
        "--surface-samples",
        config.observation.surface_samples,
        "--envelope-samples",
        config.observation.envelope_samples,
        "--envelope-horizontal-clearance",
        config.observation.horizontal_clearance_m,
        "--envelope-min-height",
        config.observation.min_height_above_floor_m,
        "--full-body-height",
        config.observation.full_body_height_m,
        "--envelope-top-clearance",
        config.observation.top_clearance_m,
        "--image-margin-ratio",
        config.observation.image_margin_ratio,
        "--underside-normal-threshold",
        config.observation.underside_normal_threshold,
        "--observation-seed",
        config.observation.seed,
        "--large-object-proxy-threshold",
        config.large_object_proxy_threshold_triangles,
        "--target-max-triangles",
        config.target_max_triangles,
        "--coverage-weight",
        config.scoring.coverage_weight,
        "--multiview-weight",
        config.scoring.multiview_weight,
        "--information-weight",
        config.scoring.information_weight,
        "--resolution-weight",
        config.scoring.resolution_weight,
        "--worst-zone-weight",
        config.scoring.worst_zone_weight,
        "--parallax-min-degrees",
        config.scoring.parallax_min_degrees,
        "--pixel-noise-sigma",
        config.scoring.pixel_noise_sigma,
        "--information-scale",
        config.scoring.information_scale_m,
        "--target-pixels-per-meter",
        config.scoring.target_pixels_per_meter,
        "--swap-passes",
        config.scoring.swap_passes,
    ]
    for name in sorted(set(config.exclude_objects + config.human_objects)):
        arguments += ["--exclude-object", name]
    for name in config.solid_objects:
        arguments += ["--solid-object", name]
    for name in config.force_aabb_proxy_objects:
        arguments += ["--force-aabb-proxy", name]
    if config.allowed_region_object:
        arguments += ["--allowed-region-object", config.allowed_region_object]
    else:
        arguments += [
            "--allowed-min=" + _csv(config.allowed_min_m),
            "--allowed-max=" + _csv(config.allowed_max_m),
        ]
    if config.floor_height_m is not None:
        arguments += ["--floor-height", config.floor_height_m]
    if config.meters_per_blender_unit is not None:
        arguments += ["--meters-per-unit", config.meters_per_blender_unit]
    return arguments


def _box_export_arguments(config, output, target_bounds=None):
    arguments = [
        "--output",
        output,
        "--minimum-thickness",
        config.box_minimum_thickness_m,
        "--structure-policy",
        config.box_structure_policy,
    ]
    if config.target_object:
        arguments += ["--target-object", config.target_object]
    if target_bounds:
        arguments += ["--target-bounds", target_bounds]
    if config.allowed_region_object:
        arguments += ["--allowed-region-object", config.allowed_region_object]
    else:
        arguments += [
            "--allowed-min=" + _csv(config.allowed_min_m),
            "--allowed-max=" + _csv(config.allowed_max_m),
        ]
    for name in sorted(set(config.exclude_objects + config.human_objects)):
        arguments += ["--exclude-object", name]
    for token in config.box_ignore_name_tokens:
        arguments += ["--ignore-name-token", token]
    if config.meters_per_blender_unit is not None:
        arguments += ["--meters-per-unit", config.meters_per_blender_unit]
    return arguments


def _write_box_request(config, boxes_path, request_path):
    manifest = json.loads(Path(boxes_path).read_text(encoding="utf-8"))
    records = manifest["boxes"]
    target = next(record for record in records if record["role"] == "target")
    allowed_records = [record for record in records if record["role"] == "allowed_region"]
    if config.allowed_region_object:
        if len(allowed_records) != 1:
            raise ValueError("box extraction did not produce exactly one allowed region")
        allowed_low = allowed_records[0]["minimum_m"]
        allowed_high = allowed_records[0]["maximum_m"]
        allowed_id = allowed_records[0]["object_id"]
    else:
        allowed_low, allowed_high = config.allowed_min_m, config.allowed_max_m
        allowed_id = "allowed_camera_region"
    floor = (
        config.floor_height_m
        if config.floor_height_m is not None
        else min(record["minimum_m"][2] for record in records if record["role"] != "allowed_region")
    )
    request = PlanningRequest.model_validate(
        {
            "schema_version": "camera_planning_request_v2",
            "scene_id": config.scene_id,
            "length_unit": "meter",
            "up_axis": "Z",
            "floor_height_m": floor,
            "budget": config.budget,
            "planning_mode": config.planning_mode,
            "target_view": config.target_view.model_dump(mode="json"),
            "target_obb": (manifest.get("target_asset") or {}).get("obb"),
            "target": {
                "object_id": target["object_id"],
                "minimum": target["minimum_m"],
                "maximum": target["maximum_m"],
            },
            "obstacles": [
                {
                    "object_id": record["object_id"],
                    "minimum": record["minimum_m"],
                    "maximum": record["maximum_m"],
                }
                for record in records
                if record["role"] == "obstacle"
            ],
            "allowed_camera_region": {
                "object_id": allowed_id,
                "minimum": allowed_low,
                "maximum": allowed_high,
            },
            "intrinsics": config.intrinsics.model_dump(mode="json"),
            "candidates": config.candidates.model_dump(mode="json"),
            "observation": config.observation.model_dump(mode="json"),
            "scoring": config.scoring.model_dump(mode="json"),
            "adaptive": config.adaptive.model_dump(mode="json"),
            "path": config.path.model_dump(mode="json"),
            "geometry": {"backend": "aabb"},
        }
    )
    write_json(request_path, request.model_dump(mode="json"))


def run_pilot(config_path):
    config = load_pilot_config(config_path)
    source = Path(config.source_scene)
    blender = Path(config.blender_executable)
    if not source.is_file():
        raise FileNotFoundError(source)
    if config.target_scene and not Path(config.target_scene).is_file():
        raise FileNotFoundError(config.target_scene)
    if not blender.is_file():
        raise FileNotFoundError(blender)
    if config.reference_npz and not Path(config.reference_npz).is_file():
        raise FileNotFoundError(config.reference_npz)
    if config.reference_mesh and not Path(config.reference_mesh).is_file():
        raise FileNotFoundError(config.reference_mesh)
    if config.reference_body_fuse_result:
        body_result_path = Path(config.reference_body_fuse_result)
        body_result = yaml.safe_load(body_result_path.read_text(encoding="utf-8"))
        if not body_result.get("valid"):
            raise ValueError("reference body_fuse_result is not valid")
        expected = {
            "fused_body_parameters": config.reference_npz,
            "fused_mesh_world": config.reference_mesh,
        }
        for key, value in expected.items():
            if value and Path(body_result["outputs"][key]).resolve() != Path(value).resolve():
                raise ValueError(f"reference {key} does not match body_fuse_result")
        if config.reference_npz:
            import numpy as np

            with np.load(config.reference_npz, allow_pickle=False) as archive:
                if str(archive["topology_hash"].item()) != body_result.get("topology_hash"):
                    raise ValueError("reference topology hash does not match body_fuse_result")
    root = new_output(config.output)
    write_json(root / "pilot_config.resolved.json", config.model_dump(mode="json"))
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    records = []
    try:
        snapshot = root / "snapshot" / "scene_snapshot.blend"
        prepare_args = ["--source", source, "--output", snapshot]
        if config.planning_mode == "target_viewspace":
            for name in config.exclude_objects:
                prepare_args += ["--exclude-object", name]
        if config.reference_mesh:
            prepare_args += [
                "--reference-mesh",
                config.reference_mesh,
                "--reference-object-name",
                config.reference_object_name,
            ]
        records.append(
            _run(
                _blender(
                    blender,
                    None,
                    scripts / "blender_prepare_scene.py",
                    prepare_args,
                ),
                "01_prepare_scene",
                root,
                cwd=source.parent,
            )
        )
        geometry = root / "geometry"
        target_bounds = None
        if config.target_scene:
            target_bounds = geometry / "target_asset_bounds.json"
            target_args = [
                "--source",
                config.target_scene,
                "--output",
                target_bounds,
            ]
            if config.target_asset_object:
                target_args += ["--target-object", config.target_asset_object]
            for name in config.exclude_objects:
                target_args += ["--exclude-object", name]
            if config.meters_per_blender_unit is not None:
                target_args += ["--meters-per-unit", config.meters_per_blender_unit]
            records.append(
                _run(
                    _blender(
                        blender,
                        None,
                        scripts / "blender_extract_target_bounds.py",
                        target_args,
                ),
                "02_extract_target_asset",
                root,
                cwd=Path(config.target_scene).parent if config.target_scene else source.parent,
            )
        )
        if config.planning_geometry == "aabb_boxes":
            boxes_path = geometry / "scene_boxes.json"
            export_script = scripts / "blender_extract_scene_boxes.py"
            export_args = _box_export_arguments(config, boxes_path, target_bounds)
        else:
            boxes_path = None
            export_script = scripts / "blender_export_scene_geometry.py"
            export_args = _triangle_export_arguments(config, geometry)
        records.append(
            _run(
                _blender(blender, snapshot, export_script, export_args),
                "03_export_geometry",
                root,
            )
        )
        if boxes_path is not None:
            _write_box_request(config, boxes_path, geometry / "request.json")
        else:
            # Blender exports geometry; Python applies the full modern planner contract.
            exported_request = json.loads((geometry / "request.json").read_text(encoding="utf-8"))
            for name in ("observation", "scoring", "candidates", "adaptive", "path", "target_view"):
                exported_request[name] = getattr(config, name).model_dump(mode="json")
            exported_request["planning_mode"] = config.planning_mode
            if target_bounds:
                exported_request["target_obb"] = json.loads(target_bounds.read_text(encoding="utf-8")).get("obb")
            write_json(
                geometry / "request.json",
                PlanningRequest.model_validate(exported_request).model_dump(mode="json"),
            )
        plan_dir = root / "plan"
        plan_result = plan(
            load_request(geometry / "request.json"), plan_dir, config.method, config.seed
        )
        records.append({"stage": "04_plan", "returncode": 0, "result": plan_result})
        preview = root / "preview" / "camera_plan_preview.blend"
        records.append(
            _run(
                _blender(
                    blender,
                    snapshot,
                    scripts / "blender_preview_plan.py",
                    [
                        "--selected-rig",
                        plan_dir / "cameras.json",
                        "--candidate-rig",
                        plan_dir / "candidate_cameras.json",
                        "--request",
                        plan_dir / "request.json",
                        "--evidence",
                        plan_dir / "evidence.npz",
                        "--output",
                        preview,
                    ],
                ),
                "05_preview",
                root,
            )
        )
        if config.reference_npz:
            experiment_dir = root / "rq1"
            prepare_rq1(plan_dir, experiment_dir, config.reference_npz, config.random_sets)
            rig_path = experiment_dir / "pool_cameras.json"
            render_root = experiment_dir / "renders"
        else:
            experiment_dir = None
            rig_path = plan_dir / "cameras.json"
            render_root = root / "captures"
        if config.planning_mode == "target_viewspace":
            # Camera renderer writes positions in Blender world units; this implementation
            # explicitly supports metre-native snapshots rather than silently rescaling K/R/T.
            meters = config.meters_per_blender_unit
            if meters is not None and abs(meters - 1.0) > 1e-8:
                raise ValueError("target_viewspace rendering currently requires metre-native Blender coordinates")
        primary_render_dir = (
            render_root / "human" if config.human_objects else render_root / "selected"
        )
        records.append(
            _run(
                _blender(
                    blender,
                    snapshot,
                    scripts / "blender_render_rig.py",
                    [
                        "--rig",
                        rig_path,
                        "--request",
                        plan_dir / "request.json",
                        "--output",
                        primary_render_dir,
                        "--frame",
                        config.frame,
                    ],
                ),
                "06_render_human" if config.human_objects else "06_render_selected_views",
                root,
            )
        )
        if config.human_objects:
            empty_args = [
                "--rig",
                rig_path,
                "--output",
                render_root / "empty",
                "--frame",
                config.frame,
            ]
            for name in config.human_objects:
                empty_args += ["--hide-object", name]
            records.append(
                _run(
                    _blender(
                        blender,
                        snapshot,
                        scripts / "blender_render_rig.py",
                        empty_args,
                    ),
                    "07_render_empty",
                    root,
                )
            )
        if experiment_dir and config.assume_constant_scene_quality:
            prepare_constant_scene_control(
                experiment_dir,
                config.human_objects,
                acknowledge_assumption=True,
            )
            records.append({"stage": "08_prepare_controls", "returncode": 0})
        command_path = None
        if experiment_dir and config.integration:
            command_path = experiment_dir / "integration_commands.json"
            commands = command_plan(
                experiment_dir,
                config.integration.ysynthetic_root,
                config.integration.python,
                config.integration.body_config,
                config.integration.fuse_config,
            )
            write_json(command_path, commands)
            records.append({"stage": "09_integration_plan", "returncode": 0})
        tour_video = None
        tour_blend = None
        if config.tour.enabled:
            tour_root = root / "tour"
            tour_video = tour_root / "camera_tour.mp4"
            tour_blend = tour_root / "camera_tour.blend"
            tour_args = [
                "--rig",
                plan_dir / "cameras.json",
                "--request",
                plan_dir / "request.json",
                "--output-video",
                tour_video,
                "--output-blend",
                tour_blend,
                "--frame",
                config.frame,
                "--fps",
                config.tour.fps,
                "--seconds-per-leg",
                config.tour.seconds_per_leg,
                "--resolution-percentage",
                config.tour.resolution_percentage,
                "--ffmpeg-executable",
                imageio_ffmpeg.get_ffmpeg_exe(),
            ]
            if config.tour.close_loop:
                tour_args += ["--close-loop"]
            if config.path.enabled:
                if plan_result["trajectory_status"] != "planned":
                    raise ValueError(
                        "camera path is blocked; inspect plan/camera_path.json; tour not rendered"
                    )
                if config.tour.close_loop:
                    raise ValueError("planned camera_path is open; close_loop must be false")
                tour_args += ["--path-plan", plan_dir / "camera_path.json"]
            records.append(
                _run(
                    _blender(
                        blender,
                        snapshot,
                        scripts / "blender_render_camera_tour.py",
                        tour_args,
                    ),
                    "10_render_camera_tour",
                    root,
                )
            )
        status = {
            "schema_version": "camera_pilot_run_v2",
            "status": "prepared",
            "root": str(root),
            "snapshot": str(snapshot),
            "preview": str(preview),
            "selected_rig": str(plan_dir / "cameras.json"),
            "capture_rig": str(rig_path),
            "capture_root": str(render_root),
            "tour_video": str(tour_video) if tour_video else None,
            "tour_blend": str(tour_blend) if tour_blend else None,
            "experiment": str(experiment_dir) if experiment_dir else None,
            "integration_commands": str(command_path) if command_path else None,
            "stages": records,
            "downstream_execution": "not_run_requires_explicit_command",
        }
        write_json(root / "pilot_run.json", status)
        return status
    except Exception as error:
        write_json(
            root / "pilot_run.json",
            {
                "schema_version": "camera_pilot_run_v2",
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "stages": records,
            },
        )
        raise
