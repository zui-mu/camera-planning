"""Actual target/room mesh ray checks shared by selected-view and tour rendering."""

import itertools

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


class MeshViewValidator:
    def __init__(self, request):
        self.request = request
        self.s = request["target_view"]
        for obj in bpy.context.scene.objects:
            if obj.hide_render:
                obj.hide_set(True)
        bpy.context.view_layer.update()
        self.deps = bpy.context.evaluated_depsgraph_get()
        unit = bpy.context.scene.unit_settings
        self.scale = float(unit.scale_length) if unit.system == "METRIC" else 1.0
        if abs(self.scale - 1.0) > 1e-8:
            raise ValueError("target_viewspace rendering requires metre-native scene coordinates")
        target = bpy.data.objects.get(request["target"]["object_id"])
        if target is None or target.type != "MESH":
            raise ValueError("target mesh unavailable for final visibility validation")
        self.name = target.name
        evaluated = target.evaluated_get(self.deps)
        mesh = evaluated.to_mesh()
        try:
            vertices = [evaluated.matrix_world @ v.co for v in mesh.vertices]
            self.bvh = BVHTree.FromPolygons(vertices, [list(p.vertices) for p in mesh.polygons])
        finally:
            evaluated.to_mesh_clear()
        obb = request.get("target_obb")
        if obb:
            target_center = np.asarray(obb["center"], float)
            target_axes = np.asarray(obb["axes"], float)
            target_half = np.asarray(obb["half_extents"], float)
        else:
            low = np.asarray(request["target"]["minimum"], float)
            high = np.asarray(request["target"]["maximum"], float)
            target_center, target_axes, target_half = (low + high) / 2, np.eye(3), (high - low) / 2
        self.up_index = {"X": 0, "Y": 1, "Z": 2}[request["up_axis"]]
        up = np.eye(3)[self.up_index]
        signs = np.asarray(list(itertools.product((-1, 1), repeat=3)), float)
        self.target_corners = target_center + (signs * target_half) @ target_axes.T
        self.center = self.target_corners.mean(axis=0)
        height = float(np.ptp(self.target_corners[:, self.up_index]))

        capture_axes, capture_half = target_axes.copy(), target_half.copy()
        local_up = int(np.argmax(np.abs(capture_axes.T @ up)))
        if float(capture_axes[:, local_up] @ up) < 0:
            capture_axes[:, local_up] *= -1
        horizontal = [i for i in range(3) if i != local_up]
        capture_half[horizontal] += self.s.get("capture_side_padding_m", 0.0)
        top = self.s.get("capture_top_padding_m", 0.0)
        bottom = self.s.get("capture_bottom_padding_m", 0.0)
        capture_center = target_center + up * (top - bottom) / 2
        capture_half[local_up] += (top + bottom) / 2
        aim_reference = self.s.get("aim_reference", "target_center")
        self.focus = (
            capture_center.copy() if aim_reference == "capture_center" else self.center.copy()
        )
        self.focus[self.up_index] += height * self.s.get("aim_height_ratio", 0.0)
        focus_local = (self.focus - capture_center) @ capture_axes
        if np.any(np.abs(focus_local) > capture_half + 1e-9):
            raise ValueError("aim point must remain inside the capture box")
        self.corners = capture_center + (signs * capture_half) @ capture_axes.T
        self.capture_surface = self._sample_box_surface(
            capture_center,
            capture_axes,
            capture_half,
            self.s.get("capture_visibility_samples", 128),
        )

    @staticmethod
    def _sample_box_surface(center, axes, half, count):
        faces = []
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            area = 4 * half[others[0]] * half[others[1]]
            for sign in (-1.0, 1.0):
                faces.append((axis, sign, area))
        weights = np.asarray([face[2] for face in faces], float)
        weights /= weights.sum()
        rng = np.random.default_rng(29)
        selected = rng.choice(len(faces), count, p=weights)
        local = rng.uniform(-half, half, size=(count, 3))
        for i, face_index in enumerate(selected):
            axis, sign, _ = faces[face_index]
            local[i, axis] = sign * half[axis]
        return center + local @ axes.T

    def _externally_blocked(self, camera_center, endpoint):
        """Trace to the capture boundary while stepping through the target mesh."""
        start = Vector(camera_center / self.scale)
        finish = Vector(endpoint / self.scale)
        epsilon = 1e-5 / self.scale
        for _ in range(16):
            delta = finish - start
            distance = delta.length
            if distance <= epsilon:
                return False
            direction = delta.normalized()
            result = bpy.context.scene.ray_cast(
                self.deps, start, direction, distance=max(0.0, distance - epsilon)
            )
            if not result[0]:
                return False
            if result[4].original.name != self.name:
                return True
            start = result[1] + direction * epsilon
        return True

    def check(self, spec):
        r, t, k = (np.asarray(spec[key], float) for key in ("R", "T", "K"))
        center = -r.T @ t
        height_above_floor = float(center[self.up_index] - self.request["floor_height_m"])
        camera_distance = float(np.linalg.norm(center - self.focus))
        if not self.s["min_camera_height_m"] <= height_above_floor <= self.s["max_camera_height_m"]:
            return {"valid": False, "reason": "actual_mesh_camera_height",
                    "height_above_floor_m": height_above_floor, "distance_m": camera_distance}
        if camera_distance > self.s["max_camera_distance_m"]:
            return {"valid": False, "reason": "actual_mesh_camera_distance",
                    "height_above_floor_m": height_above_floor, "distance_m": camera_distance}
        camera_points = self.corners @ r.T + t
        if (camera_points[:, 2] <= 0.01).any():
            return {"valid": False, "reason": "target_behind_camera"}
        pix = camera_points @ k.T
        uv = pix[:, :2] / pix[:, 2, None]
        low, high = uv.min(axis=0), uv.max(axis=0)
        ratios_low, ratios_high = low / [spec["width"], spec["height"]], high / [spec["width"], spec["height"]]
        wh = ratios_high - ratios_low
        s = self.s
        if (ratios_low < [s["side_margin_ratio"], s["top_margin_ratio"]]).any() or (
            ratios_high > [1-s["side_margin_ratio"], 1-s["bottom_margin_ratio"]]
        ).any() or (wh > [s["max_width_ratio"], s["max_height_ratio"]]).any() or max(wh) < s["min_extent_ratio"]:
            return {"valid": False, "reason": "mesh_bbox_composition"}
        size = max(4, int(np.ceil(np.sqrt(s["visibility_samples"]))))
        xs = low[0] + (np.arange(size) + 0.5) / size * (high[0] - low[0])
        ys = low[1] + (np.arange(size) + 0.5) / size * (high[1] - low[1])
        inverse_k = np.linalg.inv(k)
        hits = visible = 0
        origin = Vector(center / self.scale)
        for x, y in itertools.product(xs, ys):
            direction = Vector(r.T @ (inverse_k @ [x, y, 1.0])).normalized()
            location, _, _, hit_distance = self.bvh.ray_cast(origin, direction)
            if location is None:
                continue
            hits += 1
            result = bpy.context.scene.ray_cast(self.deps, origin, direction,
                                               distance=hit_distance + 1e-4 / self.scale)
            if result[0] and (result[4].original.name == self.name or
                              (result[1] - location).length < 1e-5 / self.scale):
                visible += 1
        fraction = visible / hits if hits else 0.0
        capture_camera = self.capture_surface @ r.T + t
        capture_pixels = capture_camera @ k.T
        capture_uv = capture_pixels[:, :2] / capture_pixels[:, 2, None]
        capture_in_frame = (capture_camera[:, 2] > 0.01) & np.all(
            (capture_uv >= 0) & (capture_uv <= [spec["width"], spec["height"]]), axis=1
        )
        capture_visible = capture_in_frame.copy()
        for index in np.flatnonzero(capture_in_frame):
            capture_visible[index] = not self._externally_blocked(center, self.capture_surface[index])
        capture_fraction = float(capture_visible.mean())
        fill = float(min(1.0, max(wh[0] / s["max_width_ratio"], wh[1] / s["max_height_ratio"])))
        proximity = max(0.0, 1.0 - camera_distance / s["max_camera_distance_m"])
        remaining = 1.0 - s["distance_preference_weight"]
        effective_fraction = min(fraction, capture_fraction)
        composition = float(remaining * (0.6 * effective_fraction + 0.4 * fill)
                            + s["distance_preference_weight"] * proximity)
        valid = bool(hits > 0 and fraction >= s["min_visibility_fraction"]
                     and capture_fraction >= s.get("min_capture_visibility_fraction", 0.90)
                     and composition >= s["min_composition_score"])
        reason = None
        if fraction < s["min_visibility_fraction"]:
            reason = "actual_mesh_target_occlusion"
        elif capture_fraction < s.get("min_capture_visibility_fraction", 0.90):
            reason = "actual_mesh_capture_corridor_occlusion"
        elif composition < s["min_composition_score"]:
            reason = "actual_mesh_composition_score"
        return {"valid": valid, "reason": reason,
                "target_hit_rays": hits, "visible_target_rays": visible,
                "visible_fraction": fraction,
                "capture_visibility_fraction": capture_fraction,
                "fill": fill, "composition": composition,
                "distance_m": camera_distance, "height_above_floor_m": height_above_floor,
                "semantics": "uniform image-grid mesh rays; finite silhouette-area estimate"}
