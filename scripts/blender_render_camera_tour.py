"""Create an animated camera tour through the selected planned viewpoints."""

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_world(spec):
    rotation = Matrix(spec["R"])
    center = -(rotation.transposed() @ Vector(spec["T"]))
    world = (rotation.transposed() @ Matrix.Diagonal((1.0, -1.0, -1.0))).to_4x4()
    world.translation = center
    return world


def inside(point, minimum, maximum, margin=0.0, epsilon=1e-6):
    return all(
        float(minimum[i]) - margin - epsilon <= point[i] <= float(maximum[i]) + margin + epsilon
        for i in range(3)
    )


def configure_lens(scene, camera, spec):
    k = spec["K"]
    width, height = spec["width"], spec["height"]
    if abs(k[0][0] - k[1][1]) > 1e-6:
        raise ValueError("tour renderer requires square pixels")
    if abs(k[0][1]) > 1e-8 or abs(k[0][2] - width / 2) > 1e-6 or abs(k[1][2] - height / 2) > 1e-6:
        raise ValueError("tour renderer requires zero skew and a centered principal point")
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.lens = k[0][0] * 36.0 / width
    camera.data.shift_x = camera.data.shift_y = 0.0
    camera.data.clip_start, camera.data.clip_end = 0.01, 1000.0
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0


def add_path_curve(points):
    curve = bpy.data.curves.new("CameraPlan_TourPath", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.012
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for control, point in zip(spline.points, points):
        control.co = (*point, 1.0)
    obj = bpy.data.objects.new("CameraPlan_TourPath", curve)
    obj.color = (0.05, 0.65, 1.0, 1.0)
    obj.hide_render = True
    bpy.context.scene.collection.objects.link(obj)


def _hermite_point(a, b, tangent_a, tangent_b, u):
    """Interpolating cubic used for one control-polyline segment."""
    u2, u3 = u * u, u * u * u
    return (
        (2 * u3 - 3 * u2 + 1) * a
        + (u3 - 2 * u2 + u) * tangent_a
        + (-2 * u3 + 3 * u2) * b
        + (u3 - u2) * tangent_b
    )


def _curve_segments(points, smoothing_strength, dense_count=33):
    """Return dense C1 Hermite segments through every certified polyline vertex.

    ``smoothing_strength=None`` is the exact piecewise-linear safety fallback.
    Non-zero strengths share one tangent at each interior vertex, so the camera
    passes through waypoints without stopping or changing direction abruptly.
    """
    if len(points) < 2:
        raise ValueError("tour curve requires at least two control points")
    if smoothing_strength is None:
        return [[a, b] for a, b in itertools.pairwise(points)]
    tangents = []
    for index, point in enumerate(points):
        if index == 0:
            tangent = points[1] - point
        elif index == len(points) - 1:
            tangent = point - points[index - 1]
        else:
            tangent = (points[index + 1] - points[index - 1]) * 0.5
        tangents.append(tangent * smoothing_strength)
    parameters = [index / (dense_count - 1) for index in range(dense_count)]
    return [
        [
            _hermite_point(points[index], points[index + 1], tangents[index],
                           tangents[index + 1], u)
            for u in parameters
        ]
        for index in range(len(points) - 1)
    ]


def _polyline_length(points):
    return sum((b - a).length for a, b in itertools.pairwise(points))


def _allocate_intervals(lengths, total_intervals):
    """Allocate integer frame intervals by arc length, at least one per segment."""
    if total_intervals < len(lengths):
        total_intervals = len(lengths)
    total = max(sum(lengths), 1e-12)
    raw = [length / total * (total_intervals - len(lengths)) for length in lengths]
    counts = [1 + int(math.floor(value)) for value in raw]
    remaining = total_intervals - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def _arc_samples(points, intervals):
    """Sample one dense polyline at nearly equal arc-length spacing."""
    if intervals <= 0:
        raise ValueError("arc sampling needs a positive interval count")
    lengths = [(b - a).length for a, b in itertools.pairwise(points)]
    total = sum(lengths)
    if total <= 1e-12:
        return [points[0].copy() for _ in range(intervals + 1)]
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    result, edge = [], 0
    for distance in (total * index / intervals for index in range(intervals + 1)):
        while edge + 1 < len(lengths) and distance > cumulative[edge + 1]:
            edge += 1
        length = lengths[edge]
        u = 0.0 if length <= 1e-12 else (distance - cumulative[edge]) / length
        result.append(points[edge].lerp(points[edge + 1], min(1.0, max(0.0, u))))
    return result


def planned_samples(path_plan, rig, fps, seconds_per_leg, smoothing_strength=1.0):
    if path_plan.get("status") != "planned" or path_plan.get("closed"):
        raise ValueError("path-plan must contain a successful open path")
    rig_hash = hashlib.sha256(
        json.dumps(rig, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if path_plan.get("rig_hash") != rig_hash:
        raise ValueError("path-plan does not match this camera rig")
    specs = {item["camera_id"]: item for item in rig["cameras"]}
    ids = path_plan["ordered_camera_ids"]
    if len(ids) != len(specs) or set(ids) != set(specs):
        raise ValueError("path-plan must visit every selected camera exactly once")
    if [(leg["from"], leg["to"]) for leg in path_plan["legs"]] != list(itertools.pairwise(ids)):
        raise ValueError("path-plan legs do not match open visit order")
    controls = []
    segment_labels = []
    arrival_controls = {0: ids[0]}
    for leg in path_plan["legs"]:
        left, right = camera_world(specs[leg["from"]]), camera_world(specs[leg["to"]])
        points = [Vector(point) for point in leg["points"]]
        if (
            len(points) < 2
            or (points[0] - left.translation).length > 1e-6
            or (points[-1] - right.translation).length > 1e-6
        ):
            raise ValueError("path endpoints differ from selected cameras")
        if not controls:
            controls.append(points[0])
        elif (controls[-1] - points[0]).length > 1e-6:
            raise ValueError("adjacent path legs are disconnected")
        for point in points[1:]:
            controls.append(point)
            segment_labels.append((leg["from"], leg["to"]))
        arrival_controls[len(controls) - 1] = leg["to"]

    dense_segments = _curve_segments(controls, smoothing_strength)
    lengths = [_polyline_length(points) for points in dense_segments]
    # Preserve the old duration contract, but spend it continuously over the
    # complete path rather than restarting an ease curve at every waypoint.
    total_intervals = max(1, round(fps * seconds_per_leg * (len(ids) - 1)))
    interval_counts = _allocate_intervals(lengths, total_intervals)
    result, arrivals = [], []
    for control_index, (dense, count, label) in enumerate(
        zip(dense_segments, interval_counts, segment_labels, strict=True)
    ):
        positions = _arc_samples(dense, count)
        for position in positions[0 if not result else 1 :]:
            if path_plan.get("orientation_policy") == "look_at_fixed_target_aim":
                forward = (Vector(path_plan["focus_world_m"]) - position).normalized()
                right_axis = forward.cross(Vector(path_plan["world_up"])).normalized()
                up_axis = right_axis.cross(forward).normalized()
                rotation = Matrix((right_axis, up_axis, -forward)).transposed().to_quaternion()
            else:
                progress = control_index / max(len(controls) - 1, 1)
                rotation = camera_world(specs[ids[0]]).to_quaternion().slerp(
                    camera_world(specs[ids[-1]]).to_quaternion(), progress
                )
            result.append({
                "frame": len(result) + 1,
                "position": list(position),
                "rotation": list(rotation),
                "from": label[0],
                "to": label[1],
            })
        endpoint_control = control_index + 1
        if endpoint_control in arrival_controls:
            arrivals.append({
                "camera_id": arrival_controls[endpoint_control],
                "frame": len(result),
            })
    arrivals.insert(0, {"camera_id": ids[0], "frame": 1})
    return [specs[i] for i in ids], result, arrivals


def collision_failures(samples, request):
    allowed = request["allowed_camera_region"]
    clearance = float(request["candidates"]["camera_clearance_m"])
    failures = []
    for sample in samples:
        point = sample["position"]
        if not inside(point, allowed["minimum"], allowed["maximum"], -clearance):
            failures.append({"frame": sample["frame"], "object_id": "outside mark"})
        for obstacle in [request["target"], *request["obstacles"]]:
            if inside(point, obstacle["minimum"], obstacle["maximum"], clearance):
                failures.append({"frame": sample["frame"], "object_id": obstacle["object_id"]})
    return failures


def mesh_view_failures(samples, specs, evaluator):
    from mathutils import Quaternion

    failures = []
    for sample in samples:
        rotation = (
            Matrix.Diagonal((1.0, -1.0, -1.0))
            @ Quaternion(sample["rotation"]).to_matrix().transposed()
        )
        translation = -(rotation @ Vector(sample["position"]))
        check = evaluator.check(
            {**specs[0], "R": [list(row) for row in rotation], "T": list(translation)}
        )
        if not check["valid"]:
            failures.append({"frame": sample["frame"], **check})
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds-per-leg", type=float, default=1.0)
    parser.add_argument("--resolution-percentage", type=int, default=50)
    parser.add_argument("--ffmpeg-executable", required=True)
    parser.add_argument("--close-loop", action="store_true")
    parser.add_argument("--path-plan")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    rig_path, request_path = Path(args.rig).resolve(), Path(args.request).resolve()
    output_video = Path(args.output_video).resolve()
    output_blend = Path(args.output_blend).resolve()
    if output_video.exists() or output_blend.exists():
        raise FileExistsError("tour output already exists")
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    specs = rig["cameras"]
    if len(specs) < 2:
        raise ValueError("tour requires at least two cameras")
    target_min = Vector(request["target"]["minimum"])
    target_max = Vector(request["target"]["maximum"])
    center = (target_min + target_max) / 2
    worlds = {spec["camera_id"]: camera_world(spec) for spec in specs}
    specs = sorted(
        specs,
        key=lambda spec: math.atan2(
            worlds[spec["camera_id"]].translation.y - center.y,
            worlds[spec["camera_id"]].translation.x - center.x,
        ),
    )
    ids = [spec["camera_id"] for spec in specs]
    legs = list(itertools.pairwise(specs))
    if args.close_loop:
        legs.append((specs[-1], specs[0]))
    frames_per_leg = max(2, round(args.fps * args.seconds_per_leg))
    samples = []
    arrival_frames = []
    frame_number = 1
    for leg_index, (left, right) in enumerate(legs):
        left_world, right_world = worlds[left["camera_id"]], worlds[right["camera_id"]]
        p0, p1 = left_world.translation.copy(), right_world.translation.copy()
        a0 = math.atan2(p0.y - center.y, p0.x - center.x)
        a1 = math.atan2(p1.y - center.y, p1.x - center.x)
        while a1 <= a0:
            a1 += 2 * math.pi
        r0 = math.hypot(p0.x - center.x, p0.y - center.y)
        r1 = math.hypot(p1.x - center.x, p1.y - center.y)
        q0, q1 = left_world.to_quaternion(), right_world.to_quaternion()
        start = 0 if leg_index == 0 else 1
        for step in range(start, frames_per_leg + 1):
            t = step / frames_per_leg
            angle = (1 - t) * a0 + t * a1
            radius = (1 - t) * r0 + t * r1
            position = Vector(
                (
                    center.x + radius * math.cos(angle),
                    center.y + radius * math.sin(angle),
                    (1 - t) * p0.z + t * p1.z,
                )
            )
            rotation = q0.slerp(q1, t)
            samples.append(
                {
                    "frame": frame_number,
                    "position": list(position),
                    "rotation": list(rotation),
                    "from": left["camera_id"],
                    "to": right["camera_id"],
                }
            )
            frame_number += 1
        arrival_frames.append({"camera_id": right["camera_id"], "frame": frame_number - 1})
    samples[0]["position"] = list(worlds[specs[0]["camera_id"]].translation)
    samples[0]["rotation"] = list(worlds[specs[0]["camera_id"]].to_quaternion())
    arrival_frames.insert(0, {"camera_id": specs[0]["camera_id"], "frame": 1})
    path_plan = None
    smoothing_strength = None
    smoothing_attempts = []
    if args.path_plan:
        if args.close_loop:
            raise ValueError("path-plan is open; --close-loop is not supported")
        path_plan = json.loads(Path(args.path_plan).read_text(encoding="utf-8"))
        evaluator = None
        if request.get("planning_mode") == "target_viewspace":
            # Recheck every proposed output frame.  Curve fitting may cut the
            # corner of an otherwise certified polyline, so progressively reduce
            # tangent magnitude until both geometry and target view remain valid.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from blender_target_visibility import MeshViewValidator

            bpy.context.scene.frame_set(args.frame)
            evaluator = MeshViewValidator(request)
        accepted = None
        for strength in (1.0, 0.75, 0.5, 0.25, None):
            trial_specs, trial_samples, trial_arrivals = planned_samples(
                path_plan, rig, args.fps, args.seconds_per_leg, strength
            )
            trial_collisions = collision_failures(trial_samples, request)
            trial_bad_views = (
                mesh_view_failures(trial_samples, trial_specs, evaluator)
                if evaluator is not None and not trial_collisions
                else []
            )
            smoothing_attempts.append({
                "strength": strength,
                "collision_failures": len(trial_collisions),
                "view_failures": len(trial_bad_views),
            })
            if not trial_collisions and not trial_bad_views:
                accepted = (
                    trial_specs,
                    trial_samples,
                    trial_arrivals,
                    trial_collisions,
                    trial_bad_views,
                )
                smoothing_strength = strength
                break
        if accepted is None:
            report_path = output_video.parent / "trajectory_view_validation.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({
                "status": "failed",
                "smoothing_attempts": smoothing_attempts,
                "reason": "no collision-and-view-valid smooth or linear trajectory",
            }, indent=2), encoding="utf-8")
            raise ValueError(f"tour curve fitting failed; see {report_path}")
        specs, samples, arrival_frames, collision_samples, bad_frames = accepted
        ids = [spec["camera_id"] for spec in specs]
    else:
        collision_samples = collision_failures(samples, request)
        bad_frames = []

    if request.get("planning_mode") == "target_viewspace":
        if path_plan is None:
            # Legacy cylindrical interpolation still receives an authoritative
            # per-frame mesh check when target-view mode is requested.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from blender_target_visibility import MeshViewValidator

            bpy.context.scene.frame_set(args.frame)
            evaluator = MeshViewValidator(request)
            bad_frames = mesh_view_failures(samples, specs, evaluator)
        report_path = output_video.parent / "trajectory_view_validation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"checked_frames": len(samples), "failures": bad_frames,
            "status": "failed" if bad_frames or collision_samples else "passed",
            "collision_failures": collision_samples,
            "collision_backend": "AABB proxy and declared allowed region",
            "geometry_backend": "Blender evaluated scene mesh ray casts",
            "smoothing_strength": smoothing_strength,
            "smoothing_attempts": smoothing_attempts}, indent=2), encoding="utf-8")
        if bad_frames or collision_samples:
            raise ValueError(f"tour frames failed geometry/composition validation; see {report_path}")

    scene = bpy.context.scene
    scene.frame_set(args.frame)
    # Imported production scenes may contain timeline markers bound to their
    # original cameras. During animation renders Blender uses those bindings
    # instead of scene.camera, so remove them in this derived output copy.
    for marker in list(scene.timeline_markers):
        scene.timeline_markers.remove(marker)
    camera = bpy.data.objects.new(
        "CameraPlanning_Tour", bpy.data.cameras.new("CameraPlanning_Tour")
    )
    scene.collection.objects.link(camera)
    scene.camera = camera
    configure_lens(scene, camera, specs[0])
    camera.rotation_mode = "QUATERNION"
    for sample in samples:
        camera.location = sample["position"]
        camera.rotation_quaternion = sample["rotation"]
        camera.keyframe_insert(data_path="location", frame=sample["frame"])
        camera.keyframe_insert(data_path="rotation_quaternion", frame=sample["frame"])
    # One key is baked at every rendered frame, so no interpolation-mode API is
    # needed. This also works with Blender 5's layered Action representation.
    for spec in specs:
        marker_data = bpy.data.cameras.new(f"TourStop_{spec['camera_id']}")
        marker_data.display_size = 0.20
        marker = bpy.data.objects.new(f"TourStop_{spec['camera_id']}", marker_data)
        marker.matrix_world = worlds[spec["camera_id"]]
        marker.show_name = True
        marker.color = (0.05, 0.8, 0.25, 1.0)
        scene.collection.objects.link(marker)
    add_path_curve([sample["position"] for sample in samples])
    scene.frame_start, scene.frame_end = 1, samples[-1]["frame"]
    scene.render.fps = args.fps
    scene.render.resolution_percentage = args.resolution_percentage
    actual_width = max(1, scene.render.resolution_x * args.resolution_percentage // 100)
    actual_height = max(1, scene.render.resolution_y * args.resolution_percentage // 100)
    if actual_width % 2 or actual_height % 2:
        raise ValueError("H.264 yuv420p requires even rendered width and height")
    frames_dir = output_video.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    scaled_k = [row[:] for row in specs[0]["K"]]
    for axis, scale in enumerate(
        (actual_width / specs[0]["width"], actual_height / specs[0]["height"])
    ):
        scaled_k[axis] = [v * scale for v in scaled_k[axis]]
    frames = []
    from mathutils import Quaternion

    for sample in samples:
        world_rotation = Quaternion(sample["rotation"]).to_matrix()
        rotation = Matrix.Diagonal((1.0, -1.0, -1.0)) @ world_rotation.transposed()
        translation = -(rotation @ Vector(sample["position"]))
        frames.append(
            {
                "frame": sample["frame"],
                "time_seconds": (sample["frame"] - 1) / args.fps,
                "R": [list(row) for row in rotation],
                "T": list(translation),
            }
        )
    (output_video.parent / "camera_trajectory.json").write_text(
        json.dumps(
            {
                "coordinate_convention": "world_to_camera",
                "width": actual_width,
                "height": actual_height,
                "K": scaled_k,
                "fps": args.fps,
                "frames": frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    scene.render.filepath = str(frames_dir / "camera_tour_")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend), check_existing=False)
    bpy.ops.render.render(animation=True)
    completed = subprocess.run(
        [
            str(Path(args.ffmpeg_executable).resolve()),
            "-y",
            "-framerate",
            str(args.fps),
            "-start_number",
            "1",
            "-i",
            str(frames_dir / "camera_tour_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"FFmpeg failed: {completed.stderr[-2000:]}")
    if not output_video.is_file():
        raise RuntimeError("Blender finished without creating the requested MP4")
    manifest = {
        "format_version": "camera_tour_v1",
        "status": "rendered",
        "video": str(output_video),
        "video_sha256": sha256(output_video),
        "blend": str(output_blend),
        "frames": str(frames_dir),
        "ffmpeg_executable": str(Path(args.ffmpeg_executable).resolve()),
        "rig": str(rig_path),
        "request": str(request_path),
        "ordered_camera_ids": ids,
        "arrival_frames": arrival_frames,
        "fps": args.fps,
        "frame_count": samples[-1]["frame"],
        "duration_seconds": samples[-1]["frame"] / args.fps,
        "interpolation": (
            ("arc-length sampled C1 Hermite curve; continuous pass-through; fixed target look-at"
             if path_plan.get("orientation_policy") == "look_at_fixed_target_aim"
             else "arc-length sampled C1 Hermite curve; continuous pass-through")
            if path_plan
            else "azimuth-ordered cylindrical arc; quaternion slerp"
        ),
        "motion_profile": "single continuous arc-length schedule; no intermediate dwell",
        "curve_smoothing_strength": smoothing_strength,
        "curve_smoothing_attempts": smoothing_attempts,
        "curve_safety_fallback": (
            "piecewise-linear constant-speed path"
            if path_plan and smoothing_strength is None
            else None
        ),
        "path_plan": str(Path(args.path_plan).resolve()) if args.path_plan else None,
        "path_safety_model": path_plan["safety_model"] if path_plan else None,
        "trajectory_parameters": str(output_video.parent / "camera_trajectory.json"),
        "continuous_collision_model": "sampled selected-camera-center vs expanded AABBs",
        "collision_free_under_aabb_proxy": not collision_samples,
        "collision_samples": collision_samples,
        "warning": (
            "This is a visualization tour through selected static viewpoints. A clean AABB "
            "check does not prove that a physical camera rig can execute the path."
        ),
    }
    (output_video.parent / "camera_tour_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
