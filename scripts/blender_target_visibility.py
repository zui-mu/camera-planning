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
            self.corners = np.array(obb["center"]) + (
                np.array(list(itertools.product((-1, 1), repeat=3))) * obb["half_extents"]
            ) @ np.asarray(obb["axes"]).T
        else:
            self.corners = np.array(list(itertools.product(*zip(
                request["target"]["minimum"], request["target"]["maximum"]))))
        self.up_index = {"X": 0, "Y": 1, "Z": 2}[request["up_axis"]]
        self.center = self.corners.mean(axis=0)
        height = float(np.ptp(self.corners[:, self.up_index]))
        self.focus = self.center.copy()
        self.focus[self.up_index] += height * self.s["aim_height_ratio"]

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
        fill = float(min(1.0, max(wh[0] / s["max_width_ratio"], wh[1] / s["max_height_ratio"])))
        proximity = max(0.0, 1.0 - camera_distance / s["max_camera_distance_m"])
        remaining = 1.0 - s["distance_preference_weight"]
        composition = float(remaining * (0.6 * fraction + 0.4 * fill)
                            + s["distance_preference_weight"] * proximity)
        valid = bool(hits > 0 and fraction >= s["min_visibility_fraction"]
                     and composition >= s["min_composition_score"])
        reason = None
        if fraction < s["min_visibility_fraction"]:
            reason = "actual_mesh_target_occlusion"
        elif composition < s["min_composition_score"]:
            reason = "actual_mesh_composition_score"
        return {"valid": valid, "reason": reason,
                "target_hit_rays": hits, "visible_target_rays": visible,
                "visible_fraction": fraction, "fill": fill, "composition": composition,
                "distance_m": camera_distance, "height_above_floor_m": height_above_floor,
                "semantics": "uniform image-grid mesh rays; finite silhouette-area estimate"}
