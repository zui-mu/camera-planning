"""Select calibrated observations of one generated, world-static human.

Consumes external detections and an external camera/background correspondence
report. Does not run a detector, video generator, SAM 3D Body or MHR fusion.
"""

import itertools
import json
from pathlib import Path

import numpy as np

from .artifacts import file_hash, new_output, write_json
from .contracts import Camera
from .geometry import project


def skew(t):
    x, y, z = t
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def epipolar_error(a, b, pa, pb):
    r1, r2, t1, t2 = map(np.asarray, (a.R, b.R, a.T, b.T))
    relative_r = r2 @ r1.T
    relative_t = t2 - relative_r @ t1
    f = np.linalg.inv(b.K).T @ skew(relative_t) @ relative_r @ np.linalg.inv(a.K)
    x, y = np.column_stack((pa, np.ones(len(pa)))), np.column_stack((pb, np.ones(len(pb))))
    line_b, line_a = x @ f.T, y @ f
    numerator = np.abs(np.einsum("ij,ij->i", y, line_b))
    denominator = np.sqrt(np.sum(line_a[:, :2]**2, axis=1) + np.sum(line_b[:, :2]**2, axis=1))
    return numerator / np.maximum(denominator, 1e-12)


def triangulate(observations):
    rows = []
    for camera, keypoint in observations:
        # Calibrated coordinates improve conditioning when intrinsics/resolutions differ.
        xy = np.linalg.inv(camera.K) @ [keypoint[0], keypoint[1], 1.0]
        p = np.column_stack((camera.R, camera.T))
        weight = np.sqrt(keypoint[2])
        rows.extend((weight * (xy[0] * p[2] - xy[2] * p[0]),
                     weight * (xy[1] * p[2] - xy[2] * p[1])))
    _, _, vh = np.linalg.svd(np.asarray(rows))
    if abs(vh[-1, 3]) < 1e-10:
        return None
    return vh[-1, :3] / vh[-1, 3]


def static_joint_fit(frames, confidence, min_shared, max_error, min_angle):
    points, errors, geometry_scores = [], [], []
    for joint in range(len(frames[0]["keypoints"])):
        observations = [(f["camera"], f["keypoints"][joint]) for f in frames
                        if f["keypoints"][joint, 2] >= confidence]
        if len(observations) < 2:
            points.append(None)
            continue
        point = triangulate(observations)
        if point is None:
            return None
        rays, joint_errors = [], []
        for camera, keypoint in observations:
            uv, depth = project(camera, point[None])
            if depth[0] <= 0.01:
                return None
            joint_errors.append(float(np.linalg.norm(uv[0] - keypoint[:2])))
            delta = camera.center - point
            rays.append(delta / max(np.linalg.norm(delta), 1e-12))
        sin2 = [max(0.0, 1 - float(a @ b)**2) for a, b in itertools.combinations(rays, 2)]
        if max(sin2) < np.sin(np.deg2rad(min_angle))**2:
            points.append(None)
            continue
        if max(joint_errors) > max_error:
            return None
        points.append(point.tolist())
        errors.extend(joint_errors)
        geometry_scores.append(max(sin2))
    if len(geometry_scores) < min_shared:
        return None
    return {"joint_positions_world_m": points, "checked_joint_count": len(geometry_scores),
            "reprojection_error_max_px": max(errors), "geometry_score": float(np.mean(geometry_scores))}


def select_human_frames(trajectory_file, detections_file, output, budget=8,
                        confidence=0.5, min_shared=6, max_error_px=4.0,
                        min_angle_degrees=3.0, max_frames=96):
    if not 2 <= budget <= 64 or not budget <= max_frames <= 512:
        raise ValueError("require 2 <= budget <= 64 and budget <= max_frames <= 512")
    if min_shared < 1:
        raise ValueError("min_shared must be positive")
    if not np.isfinite([confidence, max_error_px, min_angle_degrees]).all() or not 0 < confidence <= 1 or max_error_px <= 0 or not 0 < min_angle_degrees < 90:
        raise ValueError("invalid confidence/error/angle threshold")
    source = Path(detections_file).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    trajectory_file = Path(trajectory_file).resolve()
    trajectory = json.loads(trajectory_file.read_text(encoding="utf-8"))
    if trajectory.get("coordinate_convention") != "world_to_camera":
        raise ValueError("trajectory must use world_to_camera")
    if data.get("schema_version") != "generated_human_detections_v1" or data.get("world_static_subject") is not True:
        raise ValueError("requires generated_human_detections_v1 and declared world_static_subject")
    report_path = (source.parent / data["camera_validation_report"]).resolve()
    camera_report = json.loads(report_path.read_text(encoding="utf-8"))
    if camera_report.get("status") != "passed" or camera_report.get("trajectory_sha256") != file_hash(trajectory_file):
        raise ValueError("camera/background validation must pass for this exact trajectory")
    # The external matcher must explicitly certify which frame correspondences it checked.
    verified = {int(f["frame"]): int(f["camera_frame"]) for f in camera_report["verified_frames"]}
    if len(verified) != len(camera_report["verified_frames"]):
        raise ValueError("duplicate verified frame correspondences")
    poses = {int(f["frame"]): f for f in trajectory["frames"]}
    if len(poses) != len(trajectory["frames"]):
        raise ValueError("duplicate camera trajectory frame ids")
    accepted, rejected, seen = [], [], set()
    layout = data["keypoint_names"]
    if len(layout) < min_shared or len(set(layout)) != len(layout):
        raise ValueError("keypoint names must be unique and sufficient")
    for entry in data["frames"]:
        frame, camera_frame = int(entry["frame"]), int(entry["camera_frame"])
        if frame in seen:
            raise ValueError("duplicate detection frame ids")
        seen.add(frame)
        reason = None
        if verified.get(frame) != camera_frame or camera_frame not in poses:
            reason = "unverified_camera_correspondence"
        keypoints = np.asarray(entry["keypoints"], float)
        bbox = np.asarray(entry["bbox_xyxy"], float)
        if keypoints.shape != (len(layout), 3) or bbox.shape != (4,) or not (
            np.isfinite(keypoints).all() and np.isfinite(bbox).all()
        ) or np.any((keypoints[:, 2] < 0) | (keypoints[:, 2] > 1)):
            raise ValueError(f"malformed keypoints/bbox at frame {frame}")
        if entry.get("track_id") != data["track_id"]:
            reason = "different_person_track"
        width, height = trajectory["width"], trajectory["height"]
        if entry.get("width") != width or entry.get("height") != height:
            reason = "detection_resolution_mismatch"
        if bbox[0] <= .01*width or bbox[1] <= .01*height or bbox[2] >= .99*width or bbox[3] >= .99*height:
            reason = "person_bbox_touches_frame"
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            reason = "empty_bbox"
        good = keypoints[:, 2] >= confidence
        if good.sum() < min_shared or np.any((keypoints[good, :2] < 0) | (keypoints[good, :2] >= [width, height])):
            reason = "insufficient_valid_keypoints"
        image = (source.parent / entry["image"]).resolve()
        if not image.is_file():
            reason = "missing_generated_frame_image"
        if reason:
            rejected.append({"frame": frame, "reason": reason})
            continue
        pose = poses[camera_frame]
        camera = Camera(camera_id=f"generated_{frame:06d}", width=width, height=height,
                        K=trajectory["K"], R=pose["R"], T=pose["T"])
        accepted.append({"frame": frame, "camera_frame": camera_frame, "image": str(image),
                         "camera": camera, "keypoints": keypoints,
                         "quality": float(keypoints[good, 2].mean())})
    # Bounded, time-stratified preselection avoids O(T^2) on every video frame.
    accepted.sort(key=lambda f: f["frame"])
    if len(accepted) > max_frames:
        accepted = [max([accepted[int(i)] for i in group], key=lambda f: f["quality"])
                    for group in np.array_split(np.arange(len(accepted)), max_frames)]
    root = new_output(output)
    compatible, pair_scores = {}, []
    for a, b in itertools.combinations(range(len(accepted)), 2):
        left, right = accepted[a], accepted[b]
        shared = (left["keypoints"][:, 2] >= confidence) & (right["keypoints"][:, 2] >= confidence)
        if shared.sum() < min_shared or np.linalg.norm(left["camera"].center - right["camera"].center) < 1e-5:
            continue
        residual = epipolar_error(left["camera"], right["camera"], left["keypoints"][shared, :2], right["keypoints"][shared, :2])
        if residual.max() > max_error_px:
            continue
        fit = static_joint_fit([left, right], confidence, min_shared, max_error_px, min_angle_degrees)
        if fit is None:
            continue
        compatible[a, b] = compatible[b, a] = True
        pair_scores.append((fit["geometry_score"], a, b))
    best = None
    for _, a, b in sorted(pair_scores, reverse=True)[:16]:
        selected = [a, b]
        fit = static_joint_fit([accepted[a], accepted[b]], confidence, min_shared, max_error_px, min_angle_degrees)
        while len(selected) < budget:
            winner = None
            for i in range(len(accepted)):
                if i in selected or not all(compatible.get((i, j), False) for j in selected):
                    continue
                trial = static_joint_fit([accepted[j] for j in selected + [i]], confidence, min_shared, max_error_px, min_angle_degrees)
                if trial is not None:
                    score = trial["geometry_score"] + 0.1 * accepted[i]["quality"]
                    if winner is None or score > winner[0]:
                        winner = score, i, trial
            if winner is None:
                break
            selected.append(winner[1])
            fit = winner[2]
        if len(selected) == budget and (best is None or fit["geometry_score"] > best[0]):
            best = fit["geometry_score"], selected, fit
    report = {"schema_version": "human_frame_selection_v1", "status": "selected" if best else "insufficient_consistent_views",
              "camera_validation_report": str(report_path), "trajectory_sha256": file_hash(trajectory_file),
              "detections_sha256": file_hash(source), "candidate_frames": len(accepted),
              "rejected": rejected, "compatible_pairs": len(pair_scores), "budget": budget,
              "thresholds": {"confidence": confidence, "min_shared_joints": min_shared,
                             "max_error_px": max_error_px, "min_angle_degrees": min_angle_degrees},
              "limitation": "sparse-keypoint consistency does not prove fixed shape, identity, or contact; detector and background validator are external"}
    if best:
        _, selected, fit = best
        report["static_fit"] = fit
        report["selected"] = [{k: accepted[i][k] for k in ("frame", "camera_frame", "image")} for i in selected]
        write_json(root / "cameras.json", {"scene_id": data["scene_id"], "cameras": [accepted[i]["camera"].model_dump(mode="json") for i in selected]})
    write_json(root / "human_frame_selection.json", report)
    return report
