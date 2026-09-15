"""Target-centred view generation and bounded, path-aware open-tour selection.

This module never predicts a human pose.  When target-supported context is
detected, it samples a geometry-only free interaction volume; otherwise it uses
the original expanded target-box evidence.  Visibility remains a geometric ray
proxy supplied by the backend, not rendered segmentation area.
"""

import itertools
import math
from collections import Counter, deque

import numpy as np

from .artifacts import json_hash, new_output, write_json
from .free_space import FreeGrid
from .geometry import inside_box, load_geometry, look_at, project, unit
from .trajectory import connect, open_order


def box_frame_for(request):
    """Return target center, orthonormal axes and half extents."""
    if request.target_obb is not None:
        box = request.target_obb
        return (
            np.asarray(box.center, dtype=float),
            np.asarray(box.axes, dtype=float),
            np.asarray(box.half_extents, dtype=float),
        )
    lo, hi = np.asarray(request.target.minimum), np.asarray(request.target.maximum)
    return (lo + hi) / 2, np.eye(3), (hi - lo) / 2


def capture_box_for(request):
    """Expand the target box into the finite region that every view must preserve."""
    center, axes, half = box_frame_for(request)
    center, axes, half = center.copy(), axes.copy(), half.copy()
    up = np.eye(3)[request.up_index]
    local_up = int(np.argmax(np.abs(axes.T @ up)))
    if float(axes[:, local_up] @ up) < 0:
        axes[:, local_up] *= -1
    horizontal = [i for i in range(3) if i != local_up]
    half[horizontal] += request.target_view.capture_side_padding_m
    top = request.target_view.capture_top_padding_m
    bottom = request.target_view.capture_bottom_padding_m
    center += up * (top - bottom) / 2
    half[local_up] += (top + bottom) / 2
    return center, axes, half


def frame_corners(center, axes, half):
    signs = np.asarray(list(itertools.product((-1, 1), repeat=3)), dtype=float)
    return np.asarray(center) + (signs * np.asarray(half)) @ np.asarray(axes).T


def sample_box_surface(center, axes, half, count, seed):
    """Deterministic area-weighted samples on a capture box boundary."""
    center, axes, half = np.asarray(center), np.asarray(axes), np.asarray(half)
    faces = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        area = 4 * half[others[0]] * half[others[1]]
        for sign in (-1.0, 1.0):
            faces.append((axis, sign, others, area))
    weights = np.asarray([face[3] for face in faces], dtype=float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(faces), count, p=weights)
    local = rng.uniform(-half, half, size=(count, 3))
    for i, face_index in enumerate(selected):
        axis, sign, _, _ = faces[face_index]
        local[i, axis] = sign * half[axis]
    return center + local @ axes.T


def corners_for(request):
    return frame_corners(*box_frame_for(request))


def ray_interval(origin, direction, lo, hi):
    near, far = -np.inf, np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < lo[axis] or origin[axis] > hi[axis]:
                return None
        else:
            ends = sorted(((lo[axis] - origin[axis]) / direction[axis],
                           (hi[axis] - origin[axis]) / direction[axis]))
            near, far = max(near, ends[0]), min(far, ends[1])
    return (near, far) if far > near else None


class ViewEvaluator:
    def __init__(self, request, geometry):
        self.request, self.geometry, self.s = request, geometry, request.target_view
        self.target_corners = corners_for(request)
        capture_center, capture_axes, capture_half = capture_box_for(request)
        self.corners = frame_corners(capture_center, capture_axes, capture_half)
        self.capture_surface = sample_box_surface(
            capture_center,
            capture_axes,
            capture_half,
            self.s.capture_visibility_samples,
            29,
        )
        self.support_ids = frozenset(
            box.object_id for box in request.supported_objects
        )
        self.support_aware = request.interaction_region is not None
        self.interaction_points = (
            np.asarray(request.interaction_region.sample_points_world_m, dtype=float)
            if self.support_aware
            else np.empty((0, 3), dtype=float)
        )
        self.coverage_points = (
            self.interaction_points if self.support_aware else self.capture_surface
        )
        self.coverage_semantics = (
            "collision-free unknown-human support-volume samples"
            if self.support_aware
            else "expanded target-box boundary samples"
        )
        self.framing_corners = (
            self.target_corners if self.s.framing_region == "target" else self.corners
        )
        # A separate, much smaller deterministic set defines a lazy 3-D shadow
        # prefilter.  It follows the active evidence domain (free interaction
        # volume or capture boundary).  Full visibility checks remain authoritative.
        if self.support_aware:
            shadow_count = min(self.s.shadow_prefilter_samples, len(self.coverage_points))
            shadow_indices = np.linspace(
                0, len(self.coverage_points) - 1, shadow_count, dtype=int
            )
            self.shadow_surface = self.coverage_points[shadow_indices]
        else:
            self.shadow_surface = sample_box_surface(
                capture_center,
                capture_axes,
                capture_half,
                self.s.shadow_prefilter_samples,
                43,
            )
        self.center = self.target_corners.mean(axis=0)
        self.up = np.eye(3)[request.up_index]
        self.scale = max(float(np.linalg.norm(np.ptp(self.corners, axis=0))), 1e-3)
        height = float(np.ptp(self.target_corners[:, request.up_index]))
        base_focus = capture_center if self.s.aim_reference == "capture_center" else self.center
        self.focus = base_focus + self.up * height * self.s.aim_height_ratio
        focus_local = (self.focus - capture_center) @ capture_axes
        if np.any(np.abs(focus_local) > capture_half + 1e-9):
            raise ValueError("aim point must remain inside the capture box")
        self.surface, self.normals = geometry.sample_surface(
            request.target.object_id, self.s.visibility_samples, 17, request.up_index, -1.0
        )
        self.rejections = Counter()

    def basic_position_reason(self, position):
        """Reject a camera centre without doing any target visibility work."""
        position = np.asarray(position, float)
        req, s = self.request, self.s
        clearance = req.candidates.camera_clearance_m
        height = float(position[req.up_index] - req.floor_height_m)
        if height < s.min_camera_height_m or height > s.max_camera_height_m:
            return "camera_height_out_of_range"
        distance = float(np.linalg.norm(position - self.focus))
        if distance > s.max_camera_distance_m:
            return "camera_too_far"
        if not inside_box(position, req.allowed_camera_region, -clearance):
            return "outside_camera_region"
        if self.geometry.occupied(position[None], clearance)[0]:
            return "camera_collision"
        if abs(np.dot(unit(self.focus - position), self.up)) > 0.995:
            return "vertical_look_at_singularity"
        return None

    def shadow_prefilter(self, position):
        """Cheaply exclude a clearly occluded point in the bounded 3-D search domain.

        This is intentionally a point test, not an azimuth veto.  A low table can
        therefore reject low/radially-behind positions while a higher camera in
        the same vertical half-plane survives.  The target itself is excluded:
        only other scene geometry may cast this coarse shadow.
        """
        reason = self.basic_position_reason(position)
        if reason is not None:
            return None, reason
        if not self.s.shadow_prefilter_enabled:
            return 1.0, None
        position = np.asarray(position, float)
        endpoints = self.shadow_surface + (position - self.shadow_surface) * 1e-6
        blocked = self.geometry.blocked(
            position,
            endpoints,
            exclude_object_id=(
                None if self.support_aware else self.request.target.object_id
            ),
        )
        visible_fraction = float((~blocked).mean())
        if visible_fraction < self.s.shadow_prefilter_min_visible_fraction:
            return visible_fraction, "coarse_capture_shadow"
        return visible_fraction, None

    def evaluate(self, position):
        position = np.asarray(position, float)
        req, s = self.request, self.s
        reason = self.basic_position_reason(position)
        if reason is not None:
            return None, reason
        height = float(position[req.up_index] - req.floor_height_m)
        distance = float(np.linalg.norm(position - self.focus))
        camera = look_at(position, self.focus, self.up, req.intrinsics, "pending")
        # Strict image bounds apply to the declared framing region.  By default
        # this is the furniture itself; the expanded capture box is allowed to
        # be partially visible and is evaluated independently below.
        uv, depth = project(camera, self.framing_corners)
        if (depth <= 0.01).any():
            return None, "target_behind_near_plane"
        xy = uv / [camera.width, camera.height]
        low, high = xy.min(axis=0), xy.max(axis=0)
        wh = high - low
        if (low < [s.side_margin_ratio, s.top_margin_ratio]).any() or (
            high > [1 - s.side_margin_ratio, 1 - s.bottom_margin_ratio]
        ).any() or (wh > [s.max_width_ratio, s.max_height_ratio]).any():
            return None, "composition_too_tight"
        if max(wh) < s.min_extent_ratio:
            return None, "target_too_small"
        facing = np.einsum("ij,ij->i", self.normals, position - self.surface) > 1e-8
        if not facing.any():
            return None, "no_front_facing_samples"
        points = self.surface[facing]
        # Offset toward the camera to avoid an intersection with the hit surface itself.
        endpoints = points + (position - points) * 1e-6
        pix, z = project(camera, points)
        in_frame = (z > 0.01) & np.all((pix >= 0) & (pix <= [camera.width, camera.height]), axis=1)
        blocked = self.geometry.blocked(position, endpoints)
        strictly_visible = in_frame & ~blocked
        if self.support_ids:
            blocked_without_support = self.geometry.blocked(
                position,
                endpoints,
                exclude_object_ids=self.support_ids,
            )
            support_only = in_frame & blocked & ~blocked_without_support
        else:
            support_only = np.zeros(len(points), dtype=bool)
        visibility_credit = strictly_visible.astype(float)
        visibility_credit[support_only] = 1.0 - s.supported_target_occlusion_penalty
        visibility = float(visibility_credit.mean())
        strict_visibility = float(strictly_visible.mean())
        if visibility < s.min_visibility_fraction:
            return None, "target_occluded"
        # The active evidence is either the support-aware free interaction volume
        # or the legacy expanded capture boundary.  Interaction rays keep target
        # and supported objects solid; legacy capture rays exclude target self-occlusion.
        evidence_points = self.coverage_points
        capture_pix, capture_depth = project(camera, evidence_points)
        capture_in_frame = (capture_depth > 0.01) & np.all(
            (capture_pix >= 0) & (capture_pix <= [camera.width, camera.height]), axis=1
        )
        capture_endpoints = evidence_points + (position - evidence_points) * 1e-6
        capture_visible = capture_in_frame & ~self.geometry.blocked(
            position,
            capture_endpoints,
            exclude_object_id=(None if self.support_aware else req.target.object_id),
        )
        capture_visibility = float(capture_visible.mean())
        minimum_evidence_visibility = (
            s.min_interaction_visibility_fraction
            if self.support_aware
            else s.min_capture_visibility_fraction
        )
        if capture_visibility < minimum_evidence_visibility:
            return None, (
                "interaction_region_occluded"
                if self.support_aware
                else "capture_corridor_occluded"
            )
        fill = min(1.0, max(wh[0] / s.max_width_ratio, wh[1] / s.max_height_ratio))
        proximity = max(0.0, 1.0 - distance / s.max_camera_distance_m)
        remaining = 1.0 - s.distance_preference_weight
        # Furniture visibility is a strict per-camera requirement.  The expanded
        # capture region is only a weak per-camera preference because different
        # cameras are expected to cover complementary parts of that region.
        # Do not let capture visibility cap otherwise excellent target views.
        if self.support_aware:
            composition = remaining * (
                0.25 * visibility + 0.30 * fill + 0.45 * capture_visibility
            ) + s.distance_preference_weight * proximity
        else:
            composition = remaining * (
                0.5 * visibility + 0.3 * fill + 0.2 * capture_visibility
            ) + s.distance_preference_weight * proximity
        if composition < s.min_candidate_composition_score:
            return None, "composition_score_too_low"
        return {
            "camera": camera,
            "direction": unit(position - self.center),
            "bbox": [*low.tolist(), *high.tolist()],
            "visibility": visibility,
            "strict_target_visibility": strict_visibility,
            "support_discounted_target_fraction": float(support_only.mean()),
            "capture_visibility": capture_visibility,
            "capture_visible_mask": capture_visible,
            "interaction_visibility": (
                capture_visibility if self.support_aware else None
            ),
            "coverage_semantics": self.coverage_semantics,
            "fill": float(fill),
            "distance_m": distance,
            "height_above_floor_m": height,
            "composition": float(composition),
        }, None

    def transition_view(self, position):
        """Apply the deliberately weaker view contract used between acquisition stops.

        Final cameras still go through :meth:`evaluate`.  During motion we only
        require collision-safe placement (also checked by ``FreeGrid``), a
        non-trivial projected target size, and enough visible target surface to
        avoid an obviously blind/fully occluded shot.  The expanded interaction
        region and publication-quality margins are intentionally not hard path
        constraints.
        """
        position = np.asarray(position, float)
        req, s = self.request, self.s
        clearance = req.candidates.camera_clearance_m
        if not inside_box(position, req.allowed_camera_region, -clearance):
            return None, "transition_outside_camera_region"
        if self.geometry.occupied(position[None], clearance)[0]:
            return None, "transition_camera_collision"
        if abs(np.dot(unit(self.focus - position), self.up)) > 0.995:
            return None, "transition_vertical_look_at_singularity"
        if s.transition_view_policy == "clearance_only":
            return {"policy": "clearance_only"}, None
        if s.transition_view_policy == "strict":
            return self.evaluate(position)

        camera = look_at(position, self.focus, self.up, req.intrinsics, "transition")
        uv, depth = project(camera, self.framing_corners)
        if (depth <= 0.01).any():
            return None, "transition_target_behind_near_plane"
        xy = uv / [camera.width, camera.height]
        extent = float(np.ptp(xy, axis=0).max())
        if extent < s.transition_min_extent_ratio:
            return None, "transition_target_too_small"

        facing = np.einsum("ij,ij->i", self.normals, position - self.surface) > 1e-8
        if not facing.any():
            return None, "transition_no_front_facing_samples"
        points = self.surface[facing]
        endpoints = points + (position - points) * 1e-6
        pix, z = project(camera, points)
        in_frame = (z > 0.01) & np.all(
            (pix >= 0) & (pix <= [camera.width, camera.height]), axis=1
        )
        blocked = self.geometry.blocked(position, endpoints)
        strictly_visible = in_frame & ~blocked
        if self.support_ids:
            blocked_without_support = self.geometry.blocked(
                position,
                endpoints,
                exclude_object_ids=self.support_ids,
            )
            support_only = in_frame & blocked & ~blocked_without_support
        else:
            support_only = np.zeros(len(points), dtype=bool)
        credit = strictly_visible.astype(float)
        credit[support_only] = 1.0 - s.supported_target_occlusion_penalty
        visibility = float(credit.mean())
        if visibility < s.transition_min_visibility_fraction:
            return None, "transition_target_heavily_occluded"
        return {
            "policy": "relaxed",
            "visibility": visibility,
            "strict_target_visibility": float(strictly_visible.mean()),
            "projected_extent_ratio": extent,
        }, None

    def projected_extent_ratio(self, direction, radius):
        """Largest normalized hard-framing extent at one orbit radius."""
        direction = unit(direction)
        probe = look_at(
            self.focus + np.asarray(direction, float),
            self.focus,
            self.up,
            self.request.intrinsics,
            "extent_probe",
        )
        local = (self.framing_corners - self.focus) @ np.asarray(probe.R).T
        return self._projected_extent_from_local(local, radius)

    def _projected_extent_from_local(self, local, radius):
        """Project fixed camera-basis coordinates without rebuilding the pose."""
        depth = local[:, 2] + float(radius)
        if (depth <= 0.01).any():
            return math.inf
        k = self.request.intrinsics
        u = k.fx * local[:, 0] / depth + k.cx
        v = k.fy * local[:, 1] / depth + k.cy
        return float(max(np.ptp(u) / k.width, np.ptp(v) / k.height))

    def fit_interval(self, direction):
        """Direction-dependent near/far radii of the pinhole framing shell.

        Orbit origin is the fixed aim point; orientation stays constant along a ray.
        The near boundary follows directly from linear corner/frustum inequalities.
        Beyond it projected box extent decreases, so a bounded 1-D bisection finds
        the far boundary imposed by ``min_extent_ratio``. Collision and visibility
        stay outside this calculation because they are non-monotone.
        """
        req, s = self.request, self.s
        direction = unit(direction)
        if abs(direction @ self.up) > 0.995:
            return None
        probe = look_at(self.focus + direction, self.focus, self.up, req.intrinsics, "probe")
        local = (self.framing_corners - self.focus) @ np.asarray(probe.R).T
        k = req.intrinsics
        left = max(s.side_margin_ratio * k.width, k.cx - s.max_width_ratio * k.width / 2)
        right = min((1 - s.side_margin_ratio) * k.width, k.cx + s.max_width_ratio * k.width / 2)
        top = max(s.top_margin_ratio * k.height, k.cy - s.max_height_ratio * k.height / 2)
        bottom = min((1 - s.bottom_margin_ratio) * k.height, k.cy + s.max_height_ratio * k.height / 2)
        if not (left < k.cx < right and top < k.cy < bottom):
            raise ValueError("principal point must lie within composition margins")
        rho = max(0.01, float(np.max(0.011 - local[:, 2])))
        for axis, focal, principal, lower, upper in (
            (0, k.fx, k.cx, left, right), (1, k.fy, k.cy, top, bottom)
        ):
            bound = np.maximum(focal * local[:, axis] / (upper - principal),
                               -focal * local[:, axis] / (principal - lower)) - local[:, 2]
            rho = max(rho, float(bound.max()))
        margin = req.candidates.camera_clearance_m
        interval = ray_interval(self.focus, direction,
                                np.array(req.allowed_camera_region.minimum) + margin,
                                np.array(req.allowed_camera_region.maximum) - margin)
        if interval is None:
            return None
        lo = max(rho * (1 + 1e-6), interval[0], 0.01)
        hi_limit = min(interval[1], s.max_camera_distance_m)
        if hi_limit <= lo:
            return None
        def extent_at(radius):
            return self._projected_extent_from_local(local, radius)

        if extent_at(lo) < s.min_extent_ratio:
            return None
        if extent_at(hi_limit) >= s.min_extent_ratio:
            return lo, hi_limit

        feasible, too_far = lo, hi_limit
        while too_far - feasible > s.framing_interval_tolerance_m:
            middle = (feasible + too_far) / 2
            if extent_at(middle) >= s.min_extent_ratio:
                feasible = middle
            else:
                too_far = middle
        return (lo, feasible) if feasible > lo else None

    def radial_intervals(self, direction, interval=None):
        interval = self.fit_interval(direction) if interval is None else interval
        if interval is None:
            return []
        intervals = [interval]
        # Exact interval subtraction for the selected AABB proxy backend.
        # Mesh backends retain adaptive occupied/visibility tests below.
        if hasattr(self.geometry, "boxes"):
            margin = self.request.candidates.camera_clearance_m + 1e-6
            for box in self.geometry.boxes:
                cut = ray_interval(self.focus, direction, np.array(box.minimum) - margin,
                                   np.array(box.maximum) + margin)
                if cut is None:
                    continue
                pieces = []
                for lo, hi in intervals:
                    if cut[1] <= lo or cut[0] >= hi:
                        pieces.append((lo, hi))
                    else:
                        if lo < cut[0]:
                            pieces.append((lo, cut[0]))
                        if hi > cut[1]:
                            pieces.append((cut[1], hi))
                intervals = pieces
        return intervals


def circular_distance(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def feasible_azimuth_intervals(plane_summaries):
    """Approximate connected feasible arcs using sampled-plane Voronoi cells."""
    if not plane_summaries:
        return []
    ordered = sorted(
        (float(angle) % (2 * math.pi), bool(value["legal"]))
        for angle, value in plane_summaries.items()
    )
    angles = np.asarray([item[0] for item in ordered])
    legal = np.asarray([item[1] for item in ordered], dtype=bool)
    if not legal.any():
        return []
    if legal.all():
        return [
            {
                "start_rad": 0.0,
                "end_rad": 2 * math.pi,
                "width_rad": 2 * math.pi,
                "legal_samples": len(ordered),
            }
        ]

    anchor = int(np.flatnonzero(~legal)[0])
    indices = [(anchor + step) % len(ordered) for step in range(len(ordered) + 1)]
    unwrapped = []
    previous = None
    for index in indices:
        angle = float(angles[index])
        if previous is not None:
            while angle <= previous:
                angle += 2 * math.pi
        unwrapped.append((angle, bool(legal[index])))
        previous = angle

    result, start, count = [], None, 0
    for (left_angle, left_legal), (right_angle, right_legal) in itertools.pairwise(unwrapped):
        boundary = (left_angle + right_angle) / 2
        if not left_legal and right_legal:
            start, count = boundary, 0
        if right_legal:
            count += 1
        if left_legal and not right_legal and start is not None:
            width = boundary - start
            normalized_start = start % (2 * math.pi)
            result.append(
                {
                    "start_rad": normalized_start,
                    "end_rad": normalized_start + width,
                    "width_rad": width,
                    "legal_samples": count,
                }
            )
            start, count = None, 0
    return result


def stratified_azimuth_centers(intervals, count):
    """Allocate angular strata by feasible-arc length, with one per arc first."""
    if count <= 0 or not intervals:
        return np.empty(0)
    intervals = sorted(intervals, key=lambda item: item["width_rad"], reverse=True)[:count]
    widths = np.asarray([item["width_rad"] for item in intervals], dtype=float)
    quotas = np.ones(len(intervals), dtype=int)
    remaining = count - len(intervals)
    if remaining:
        raw = remaining * widths / widths.sum()
        extra = np.floor(raw).astype(int)
        quotas += extra
        left = remaining - int(extra.sum())
        order = np.argsort(-(raw - extra), kind="stable")
        quotas[order[:left]] += 1
    centers = []
    for interval, quota in zip(intervals, quotas, strict=True):
        start, width = interval["start_rad"], interval["width_rad"]
        centers.extend(
            (start + (index + 0.5) * width / quota) % (2 * math.pi)
            for index in range(quota)
        )
    return np.asarray(sorted(centers))


def azimuth_bin_indices(records, centers, count, per_bin):
    """Round-robin azimuth retention; quality breaks ties inside equal-occupancy bins."""
    if not records or count <= 0:
        return [], np.empty(0, dtype=int)
    angles = np.asarray([r["azimuth_rad"] for r in records])
    quality = np.asarray([r["composition"] for r in records])
    bins = np.asarray([int(np.argmin([circular_distance(a, c) for c in centers])) for a in angles])
    selected, occupancy = [], np.zeros(len(centers), dtype=int)
    available = np.ones(len(records), dtype=bool)
    nearest = np.full(len(records), math.pi)
    while len(selected) < min(count, len(records)):
        eligible = available & (occupancy[bins] < per_bin)
        if not eligible.any():
            break
        level = occupancy[bins[eligible]].min()
        eligible &= occupancy[bins] == level
        gain = nearest / math.pi + 0.15 * quality if selected else quality.copy()
        gain[~eligible] = -np.inf
        index = int(np.argmax(gain))
        selected.append(index)
        occupancy[bins[index]] += 1
        available[index] = False
        nearest = np.minimum(nearest, np.asarray([circular_distance(a, angles[index]) for a in angles]))
    return selected, bins


def generate_views(evaluator):
    """Uniform azimuths outside, elevation/radius search inside each vertical half-plane."""
    s, req = evaluator.s, evaluator.request
    records, direction_cache, position_cache, trace = [], {}, {}, []
    radial_queries = local_queries = exact_position_queries = 0
    shadow_queries, shadow_rejections = 0, 0
    horizontal_axes = [i for i in range(3) if i != req.up_index]
    azimuth_centers = np.arange(s.azimuth_count) * 2 * math.pi / s.azimuth_count
    initial_elevations = np.linspace(math.radians(s.elevation_min_degrees),
                                     math.radians(s.elevation_max_degrees), s.elevation_count)

    def vector(azimuth, elevation):
        horizontal = np.zeros(3)
        horizontal[horizontal_axes[0]] = math.cos(azimuth)
        horizontal[horizontal_axes[1]] = math.sin(azimuth)
        return math.cos(elevation) * horizontal + math.sin(elevation) * evaluator.up

    def annotate_angles(record):
        delta = record["camera"].center - evaluator.center
        horizontal = delta[horizontal_axes]
        record["azimuth_rad"] = math.atan2(horizontal[1], horizontal[0]) % (2 * math.pi)
        record["elevation_rad"] = math.atan2(delta[req.up_index], np.linalg.norm(horizontal))

    def evaluate_position(position, local=False):
        nonlocal radial_queries, local_queries, exact_position_queries
        nonlocal shadow_queries, shadow_rejections
        key = tuple(np.round(position, 10))
        if key in position_cache:
            return position_cache[key], True
        if len(position_cache) >= s.max_position_evaluations:
            return None, False
        coarse_visibility, reason = evaluator.shadow_prefilter(position)
        if s.shadow_prefilter_enabled and coarse_visibility is not None:
            shadow_queries += 1
        if reason is None:
            value, reason = evaluator.evaluate(position)
            exact_position_queries += 1
        else:
            value = None
            if reason == "coarse_capture_shadow":
                shadow_rejections += 1
        position_cache[key] = value
        if local:
            local_queries += 1
        else:
            radial_queries += 1
        if reason:
            evaluator.rejections[reason] += 1
        return value, True

    def sample_direction(azimuth, elevation, stage):
        azimuth %= 2 * math.pi
        key = (round(azimuth, 10), round(elevation, 10))
        if key in direction_cache:
            return direction_cache[key]
        if len(direction_cache) >= s.max_direction_evaluations:
            return None
        direction = vector(azimuth, elevation)
        legal, radial_cache = [], {}

        def at(radius):
            if radius not in radial_cache:
                if len(radial_cache) >= s.max_radial_evaluations_per_direction:
                    return None, False
                value, tested = evaluate_position(evaluator.focus + radius * direction)
                if not tested:
                    return None, False
                radial_cache[radius] = value
                if value is not None:
                    annotate_angles(value)
                    legal.append(value)
            return radial_cache[radius], True

        shell_interval = evaluator.fit_interval(direction)
        intervals = (
            evaluator.radial_intervals(direction, shell_interval)
            if shell_interval is not None
            else []
        )
        queue = deque((lo, hi, 0) for lo, hi in intervals)
        truncated = False
        while queue:
            lo, hi, depth = queue.popleft()
            mid = (lo + hi) / 2
            samples = [at(r) for r in (lo, mid, hi)]
            if not all(tested for _, tested in samples):
                truncated = True
                break
            valid = [value for value, _ in samples if value is not None]
            changing = len(valid) not in (0, 3)
            if len(valid) > 1:
                changing |= np.ptp([v["visibility"] for v in valid]) > s.max_visibility_change
                changing |= np.max(np.ptp([v["bbox"] for v in valid], axis=0)) > s.max_bbox_change
            if (depth < s.radial_max_depth
                    and hi - lo > s.min_radial_step_target_ratio * evaluator.scale
                    and (depth < s.radial_min_depth or changing)):
                queue.extend(((lo, mid, depth + 1), (mid, hi, depth + 1)))

        kept = []
        if legal:
            nearest = min(legal, key=lambda r: r["distance_m"])
            best = max(legal, key=lambda r: r["composition"])
            for record in (best, nearest):
                if not any(np.allclose(record["camera"].center, other["camera"].center) for other in kept):
                    kept.append(record)
            records.extend(kept)
        best = max(kept, key=lambda r: r["composition"]) if kept else None
        summary = {"legal": best is not None,
                   "composition": best["composition"] if best else 0.0,
                   "visibility": best["visibility"] if best else 0.0,
                   "capture_visibility": best["capture_visibility"] if best else 0.0,
                   "distance_m": best["distance_m"] if best else None}
        direction_cache[key] = summary
        trace.append({"stage": stage, "azimuth_degrees": math.degrees(azimuth),
                      "elevation_degrees": math.degrees(elevation),
                      "framing_shell_interval_m": shell_interval,
                      "collision_free_intervals_m": intervals,
                      "framing_free_intervals_m": intervals, "tested_radii": len(radial_cache),
                      "radial_budget_truncated": truncated, "retained": len(kept), **summary})
        return summary

    def probe_vertical_plane(azimuth, stage):
        summaries = {}
        for elevation in initial_elevations:
            if len(direction_cache) >= s.max_direction_evaluations:
                break
            summaries[float(elevation)] = sample_direction(azimuth, float(elevation), stage)
        for round_index in range(s.elevation_refinement_rounds):
            options = []
            ordered = sorted(summaries)
            for lo, hi in itertools.pairwise(ordered):
                left, right = summaries[lo], summaries[hi]
                if left is None or right is None:
                    continue
                mixed = left["legal"] != right["legal"]
                quality = max(left["composition"], right["composition"])
                change = abs(left["visibility"] - right["visibility"])
                priority = 2.0 * float(mixed) + quality + change + (hi - lo) / math.pi
                options.append((priority, lo, hi))
            if not options or len(direction_cache) >= s.max_direction_evaluations:
                break
            _, lo, hi = max(options)
            mid = (lo + hi) / 2
            summaries[mid] = sample_direction(azimuth, mid, f"{stage}_elevation_refine_{round_index}")
        valid = [v for v in summaries.values() if v and v["legal"]]
        return {"legal": bool(valid),
                "best_composition": max((v["composition"] for v in valid), default=0.0)}

    # Complete uniform horizontal sweep before any azimuth refinement.
    plane_summaries = {}
    for azimuth in azimuth_centers:
        plane_summaries[float(azimuth)] = probe_vertical_plane(float(azimuth), "uniform_azimuth")
    coarse_complete = len(plane_summaries) == s.azimuth_count

    # First round probes every interval midpoint. Later rounds concentrate on feasible
    # intervals and legal/illegal boundaries while keeping the original uniform bins fixed.
    for round_index in range(s.azimuth_refinement_rounds):
        ordered = sorted(plane_summaries)
        intervals = [(ordered[i], ordered[(i + 1) % len(ordered)]) for i in range(len(ordered))]
        additions = []
        for lo, raw_hi in intervals:
            hi = raw_hi + (2 * math.pi if raw_hi <= lo else 0)
            left, right = plane_summaries[lo], plane_summaries[raw_hi]
            if round_index > 0 and not (left["legal"] or right["legal"]
                                        or left["legal"] != right["legal"]):
                continue
            additions.append(((lo + hi) / 2) % (2 * math.pi))
        for azimuth in additions:
            if (len(direction_cache) >= s.max_direction_evaluations
                    or len(position_cache) >= s.max_position_evaluations):
                break
            plane_summaries[azimuth] = probe_vertical_plane(
                azimuth, f"azimuth_refine_{round_index}")

    if not records:
        raise ValueError("no quality-qualified target view found; inspect height, distance, occlusion and framing limits")

    # Improve a few azimuth-distributed seeds in real coordinates. Every trial must still
    # pass all hard quality constraints and shares the global position budget.
    seeds, _ = azimuth_bin_indices(records, azimuth_centers, s.local_refine_seeds, 1)
    for index in seeds:
        best = records[index]
        step = evaluator.scale * 0.06
        for _ in range(s.local_refine_rounds):
            trials = []
            for axis in range(3):
                for sign in (-1, 1):
                    position = best["camera"].center.copy()
                    position[axis] += sign * step
                    candidate, _ = evaluate_position(position, local=True)
                    if candidate is not None:
                        annotate_angles(candidate)
                        trials.append(candidate)
            if trials:
                candidate = max(trials, key=lambda r: r["composition"])
                if candidate["composition"] > best["composition"] + 1e-8:
                    best = candidate
                    records.append(best)
            step /= 2

    unique = {}
    for record in records:
        key = tuple(np.round(record["camera"].center, 7))
        if key not in unique or record["composition"] > unique[key]["composition"]:
            unique[key] = record
    records = list(unique.values())
    before_cap = len(records)
    feasible_intervals = feasible_azimuth_intervals(plane_summaries)
    retention_centers = stratified_azimuth_centers(feasible_intervals, s.azimuth_count)
    if not len(retention_centers):
        retention_centers = azimuth_centers
    chosen, bins = azimuth_bin_indices(records, retention_centers, s.max_candidates,
                                       s.max_candidates_per_azimuth_bin)
    before_counts = Counter(int(v) for v in bins)
    after_counts = Counter(int(bins[i]) for i in chosen)
    records = [records[i] for i in chosen]
    for i, record in enumerate(records):
        record["camera"].camera_id = f"view_{i:04d}"
        record["azimuth_bin"] = int(bins[chosen[i]])
    stop = ("position_budget" if len(position_cache) >= s.max_position_evaluations else
            "direction_budget" if len(direction_cache) >= s.max_direction_evaluations else
            "configured_refinement_complete")
    return records, {"search_policy": "target_framing_shell_then_capture_shadow_then_stratified_azimuth_v6",
                     "parameterization": "direction-dependent pinhole target-framing shell; partial 3-D capture-corridor coverage; uniformly refined feasible azimuth arcs; elevation and radius inside vertical half-plane",
                     "framing_shell": {
                         "region": s.framing_region,
                         "aim_reference": s.aim_reference,
                         "aim_world_m": evaluator.focus.tolist(),
                         "near_boundary": "analytic framing-region corner/frustum inequalities",
                         "far_boundary": "bounded 1-D solve of projected extent >= min_extent_ratio",
                         "minimum_extent_ratio": s.min_extent_ratio,
                         "distance_tolerance_m": s.framing_interval_tolerance_m,
                     },
                     "shadow_prefilter": {
                         "enabled": s.shadow_prefilter_enabled,
                         "semantics": "sampled camera-to-capture-box corridor; excludes positions, never whole azimuths",
                         "samples_per_position": s.shadow_prefilter_samples,
                         "minimum_visible_fraction": s.shadow_prefilter_min_visible_fraction,
                         "queries": shadow_queries,
                         "rejected_positions": shadow_rejections,
                         "exact_visibility_queries_avoided": shadow_rejections,
                         "exact_position_queries": exact_position_queries,
                     },
                     "coarse_azimuth_count": s.azimuth_count,
                     "coarse_azimuth_sweep_complete": coarse_complete,
                     "direction_queries": len(direction_cache), "radial_queries": radial_queries,
                     "local_position_queries": local_queries, "position_queries": len(position_cache),
                     "max_direction_evaluations": s.max_direction_evaluations,
                     "max_position_evaluations": s.max_position_evaluations,
                     "stop_reason": stop, "before_candidate_cap": before_cap,
                     "retained_candidates": len(records), "requested_candidate_cap": s.max_candidates,
                     "max_candidates_per_azimuth_bin": s.max_candidates_per_azimuth_bin,
                     "feasible_azimuth_intervals": [
                         {"start_degrees": math.degrees(item["start_rad"]),
                          "end_degrees_unwrapped": math.degrees(item["end_rad"]),
                          "width_degrees": math.degrees(item["width_rad"]),
                          "legal_samples": item["legal_samples"]}
                         for item in feasible_intervals
                     ],
                     "azimuth_strata_centers_degrees": [
                         math.degrees(angle) for angle in retention_centers
                     ],
                     "azimuth_bins": [{"bin": i, "center_degrees": math.degrees(angle),
                                        "before": before_counts[i], "retained": after_counts[i]}
                                       for i, angle in enumerate(retention_centers)],
                     "vertical_planes": [{"azimuth_degrees": math.degrees(angle), **summary}
                                         for angle, summary in sorted(plane_summaries.items())],
                     "direction_trace": trace, "rejections": dict(evaluator.rejections),
                     "complete_continuous_domain_certified": False}


def diverse_indices(records, count):
    if count == 0:
        return []
    vectors = np.array([r["direction"] for r in records])
    quality = np.array([r["composition"] for r in records])
    chosen = [int(np.argmax(quality))]
    while len(chosen) < count:
        gain = 1 - (vectors @ vectors[chosen].T).max(axis=1) + 0.03 * quality
        gain[chosen] = -np.inf
        chosen.append(int(np.argmax(gain)))
    return chosen


class TargetPathGrid(FreeGrid):
    """Camera clearance plus the configured moving-shot view contract."""
    def __init__(self, evaluator):
        req = evaluator.request
        super().__init__(req.allowed_camera_region, req.path.voxel_size_m, evaluator.geometry,
                         req.path.max_grid_cells, req.candidates.camera_clearance_m)
        self.validation_step = req.path.validation_step_m
        self.evaluator = evaluator
        self.view_cache = {}
        self.view_rejections = Counter()

    def segment_clear(self, a, b):
        if not super().segment_clear(a, b):
            return False
        n = max(1, math.ceil(np.linalg.norm(np.asarray(b) - a) / self.evaluator.s.transition_check_step_m))
        for position in np.linspace(a, b, n + 1):
            key = tuple(np.round(position, 7))
            if key not in self.view_cache:
                if len(self.view_cache) >= self.evaluator.s.max_transition_view_checks:
                    self.view_rejections["transition_view_check_budget_exhausted"] += 1
                    return False
                _, reason = self.evaluator.transition_view(position)
                self.view_cache[key] = reason is None
                if reason is not None:
                    self.view_rejections[reason] += 1
            if not self.view_cache[key]:
                return False
        return True


def view_pair_geometry(record_a, record_b):
    """Return geodesic viewing-angle and camera-center separation for two records."""
    cosine = float(np.clip(record_a["direction"] @ record_b["direction"], -1, 1))
    angle = math.degrees(math.acos(cosine))
    distance = float(
        np.linalg.norm(record_a["camera"].center - record_b["camera"].center)
    )
    return angle, distance


def views_are_distinct(record_a, record_b, settings):
    angle, distance = view_pair_geometry(record_a, record_b)
    return (
        angle >= settings.min_view_direction_separation_degrees
        and distance >= settings.min_camera_position_separation_m
    )


def select_tour(evaluator, records):
    req, s = evaluator.request, evaluator.s
    original_candidate_count = len(records)
    source_indices = [
        i for i, record in enumerate(records)
        if record["composition"] >= s.min_composition_score
    ]
    if len(source_indices) < req.budget:
        return None, None, {
            "status": "blocked",
            "reason": "insufficient_final_quality_candidates",
            "candidate_count": len(records),
            "final_quality_candidate_count": len(source_indices),
            "requested_cameras": req.budget,
            "min_candidate_composition_score": s.min_candidate_composition_score,
            "min_final_composition_score": s.min_composition_score,
        }
    # Work on the final-quality subset, then map the selected local indices back
    # to the complete candidate list returned by candidate generation.
    records = [records[i] for i in source_indices]
    vectors = np.array([r["direction"] for r in records])
    quality = np.array([r["composition"] for r in records])
    distances = np.array([r["distance_m"] for r in records])
    capture_masks = np.asarray([r["capture_visible_mask"] for r in records], dtype=bool)
    azimuths = np.array([r["azimuth_rad"] for r in records])
    elevations = np.array([r["elevation_rad"] for r in records])
    refs = np.arange(s.azimuth_count) * 2 * math.pi / s.azimuth_count
    cos_limit = math.cos(math.radians(s.coverage_angle_degrees))
    separation = np.abs((azimuths[:, None] - refs[None, :] + math.pi) % (2 * math.pi) - math.pi)
    kernel = np.clip((np.cos(separation) - cos_limit) / (1 - cos_limit), 0, 1)
    reachable = kernel.max(axis=0) > 0
    kernel = kernel[:, reachable]
    if kernel.shape[1] == 0:
        raise ValueError("no angular quadrature nodes covered")

    def pair_geometry(a, b):
        return view_pair_geometry(records[a], records[b])

    def distinct(ids, candidate):
        for i in ids:
            if not views_are_distinct(records[i], records[candidate], s):
                return False
            azimuth_gap = abs(
                (float(azimuths[i] - azimuths[candidate]) + math.pi)
                % (2 * math.pi) - math.pi
            )
            if math.degrees(azimuth_gap) < s.min_azimuth_separation_degrees:
                return False
        return True

    def score(ids):
        coverage = float(kernel[ids].max(axis=0).mean())
        pairs = [max(0.0, 1 - float(vectors[a] @ vectors[b])**2) for a, b in itertools.combinations(ids, 2)]
        baseline = float(np.mean(pairs)) if pairs else 0.0
        geometry_pairs = [pair_geometry(a, b) for a, b in itertools.combinations(ids, 2)]
        minimum_angle = min((pair[0] for pair in geometry_pairs), default=180.0)
        minimum_position = min((pair[1] for pair in geometry_pairs), default=float("inf"))
        composition = float(quality[ids].mean())
        worst_composition = float(quality[ids].min())
        capture_counts = capture_masks[ids].sum(axis=0)
        capture_set_coverage = float(np.mean(capture_counts >= 1))
        capture_multiview = float(np.mean(capture_counts >= 2))
        proximity = float(np.mean(np.clip(
            1.0 - distances[ids] / s.max_camera_distance_m, 0.0, 1.0
        )))
        elevation_span = math.radians(s.elevation_max_degrees - s.elevation_min_degrees)
        elevation_pairs = [abs(float(elevations[a] - elevations[b])) / elevation_span
                           for a, b in itertools.combinations(ids, 2)]
        elevation_diversity = float(np.mean(np.clip(elevation_pairs, 0, 1))) if elevation_pairs else 0.0
        total = (s.direction_weight * coverage + s.baseline_weight * baseline
                 + s.composition_weight * composition
                 + s.worst_composition_weight * worst_composition
                 + s.elevation_diversity_weight * elevation_diversity
                 + s.capture_set_coverage_weight * capture_set_coverage
                 + s.capture_multiview_weight * capture_multiview
                 + s.set_proximity_weight * proximity)
        evidence_metrics = (
            {
                "interaction_set_coverage": capture_set_coverage,
                "interaction_multiview_coverage": capture_multiview,
            }
            if evaluator.support_aware
            else {
                "capture_set_coverage": capture_set_coverage,
                "capture_multiview_coverage": capture_multiview,
            }
        )
        return total, {"direction_coverage": coverage, "anchor_baseline": baseline,
                       "composition": composition, "worst_composition": worst_composition,
                       **evidence_metrics,
                       "mean_proximity": proximity,
                       "elevation_diversity": elevation_diversity,
                       "min_pairwise_view_angle_degrees": minimum_angle,
                       "min_pairwise_camera_distance_m": minimum_position,
                       "total": total}

    grid, links = TargetPathGrid(evaluator), {}
    queries = 0

    def link(a, b):
        nonlocal queries
        if (a, b) not in links:
            if queries >= s.max_path_queries:
                return None
            queries += 1
            path = connect(grid, records[a]["camera"].center, records[b]["camera"].center,
                           req.path.max_expansions)
            value = (float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), path) if path else None
            links[a, b] = value
            links[b, a] = (value[0], value[1][::-1]) if value else None
        return links[a, b]

    best, partial, complete_sets = None, [], []
    starts, _ = azimuth_bin_indices(records, refs, min(s.selection_starts, len(records)), 1)
    for start in starts:
        selected = [start]
        while len(selected) < req.budget:
            available = [
                i for i in range(len(records))
                if i not in selected and distinct(selected, i)
            ]
            if not available:
                break
            selected.append(max(available, key=lambda i: score(selected + [i])[0]))
        entry = {"start": start, "selected_count": len(selected)}
        if len(selected) == req.budget:
            utility, metrics = score(selected)
            entry["selection_total"] = utility
            complete_sets.append((utility, selected, metrics, entry))
        partial.append(entry)

    # Selection is complete before path planning.  Build an exact shortest open
    # visit order over each bounded greedy set; route length breaks only true
    # selection-score ties and can no longer buy a duplicate camera.
    for utility, selected, metrics, entry in complete_sets:
        costs = np.full((len(selected), len(selected)), np.inf)
        np.fill_diagonal(costs, 0.0)
        for a, b in itertools.combinations(range(len(selected)), 2):
            edge = link(selected[a], selected[b])
            if edge is not None:
                costs[a, b] = costs[b, a] = edge[0]
        local_order = open_order(costs)
        if local_order is None:
            entry["path_status"] = "blocked"
            continue
        length = float(sum(costs[a, b] for a, b in itertools.pairwise(local_order)))
        if req.path.max_length_m is not None and length > req.path.max_length_m:
            entry["path_status"] = "too_long"
            entry["length_m"] = length
            continue
        order = [selected[i] for i in local_order]
        entry["path_status"] = "planned"
        entry["length_m"] = length
        metrics = {**metrics, "path_length_m": length}
        if best is None or utility > best[0] + 1e-12 or (
            abs(utility - best[0]) <= 1e-12 and length < best[3]
        ):
            best = utility, order, metrics, length
    diagnostics = {"path_queries": queries, "path_query_cap": s.max_path_queries,
                   "transition_view_checks": len(grid.view_cache),
                   "transition_view_check_cap": s.max_transition_view_checks,
                   "transition_view_policy": s.transition_view_policy,
                   "transition_min_visibility_fraction": s.transition_min_visibility_fraction,
                   "transition_min_extent_ratio": s.transition_min_extent_ratio,
                   "transition_view_rejections": dict(grid.view_rejections),
                   "starts": partial, "angular_reference_count": int(reachable.sum()),
                   "angular_reference_semantics": "uniform horizontal azimuth bins near feasible candidates",
                   "selection_policy": "quality/diversity set first; exact open route second",
                   "candidate_count": original_candidate_count,
                   "final_quality_candidate_count": len(records),
                   "min_candidate_composition_score": s.min_candidate_composition_score,
                   "min_final_composition_score": s.min_composition_score,
                   "hard_min_view_direction_separation_degrees": s.min_view_direction_separation_degrees,
                   "hard_min_azimuth_separation_degrees": s.min_azimuth_separation_degrees,
                   "hard_min_camera_position_separation_m": s.min_camera_position_separation_m,
                   "coverage_evidence_semantics": evaluator.coverage_semantics,
                   "support_aware": evaluator.support_aware,
                   "path_role": "feasibility and tie-break only"}
    if best is None:
        return None, None, diagnostics
    _, order, metrics, _ = best
    path = {"format_version": "camera_open_path_v1", "status": "planned", "closed": False,
            "length_unit": "meter", "length_m": metrics["path_length_m"],
            "geometry_backend": evaluator.geometry.name,
            "selection_coupling": "bounded multi-start greedy camera set, then exact open ordering",
            "safety_model": (
                "geometry clearance plus configured moving-shot target visibility; "
                "strict acquisition composition applies only at selected stops"
            ),
            "transition_view_policy": s.transition_view_policy,
            "transition_min_visibility_fraction": s.transition_min_visibility_fraction,
            "transition_min_extent_ratio": s.transition_min_extent_ratio,
            "transition_check_step_m": s.transition_check_step_m,
            "focus_world_m": evaluator.focus.tolist(), "world_up": evaluator.up.tolist(),
            "orientation_policy": "look_at_fixed_target_aim",
            "ordered_camera_ids": [records[i]["camera"].camera_id for i in order],
            "legs": [{"from": records[a]["camera"].camera_id, "to": records[b]["camera"].camera_id,
                      "points": links[a, b][1], "length_m": links[a, b][0]}
                     for a, b in itertools.pairwise(order)]}
    diagnostics["metrics"] = metrics
    return [source_indices[i] for i in order], path, diagnostics


def plan_target_views(request, output, seed=0):
    from .planner import rig

    root = new_output(output)
    evaluator = ViewEvaluator(request, load_geometry(request))
    records, search = generate_views(evaluator)
    write_json(root / "request.json", request.model_dump(mode="json"))
    write_json(root / "viewspace_search.json", search)
    write_json(root / "candidate_cameras.json", rig(request.scene_id, [r["camera"] for r in records]))
    write_json(root / "target_view_evidence.json", {
        "target_corners_world_m": evaluator.target_corners.tolist(),
        "capture_corners_world_m": evaluator.corners.tolist(),
        "focus_world_m": evaluator.focus.tolist(),
        "visibility_semantics": (
            "target visibility discounts occlusion caused only by target-supported context objects; "
            "unknown-human interaction samples still treat every object as solid"
            if evaluator.support_aware
            else "target visibility uses front-facing target-surface samples; capture visibility "
            "uses the finite expanded target-box corridor and excludes target self-occlusion"
        ),
        "coverage_evidence_semantics": evaluator.coverage_semantics,
        "support_aware": evaluator.support_aware,
        "supported_object_ids": sorted(evaluator.support_ids),
        "candidates": [{"camera_id": r["camera"].camera_id, "bbox_normalized": r["bbox"],
                        "visibility_fraction": r["visibility"], "composition": r["composition"],
                        "strict_target_visibility_fraction": r["strict_target_visibility"],
                        "support_discounted_target_fraction": r["support_discounted_target_fraction"],
                        "capture_visibility_fraction": r["capture_visibility"],
                        "interaction_visibility_fraction": r["interaction_visibility"],
                        "fill": r["fill"], "distance_m": r["distance_m"],
                        "height_above_floor_m": r["height_above_floor_m"],
                        "azimuth_degrees": math.degrees(r["azimuth_rad"]),
                        "elevation_degrees": math.degrees(r["elevation_rad"]),
                        "azimuth_bin": r["azimuth_bin"]} for r in records]})
    if len(records) < request.budget:
        write_json(root / "selection_diagnostics.json", {
            "status": "blocked", "reason": "insufficient_candidates_after_azimuth_bin_cap",
            "candidate_count": len(records), "requested_cameras": request.budget,
            "max_candidates_per_azimuth_bin": request.target_view.max_candidates_per_azimuth_bin})
        raise ValueError("too few candidates after azimuth-bin retention; inspect viewspace_search.json "
                         "and adjust search budget or explicit bin cap; no cap was silently relaxed")
    order, path, selection = select_tour(evaluator, records)
    write_json(root / "selection_diagnostics.json", selection)
    if order is None:
        write_json(root / "camera_path.json", {"status": "blocked", "closed": False,
                   "reason": "no_complete_tour_within_search_budget", **selection})
        raise ValueError(f"could not select {request.budget} connected target views; see {root / 'selection_diagnostics.json'}")
    cameras = [records[i]["camera"] for i in order]
    camera_rig = rig(request.scene_id, cameras)
    path["rig_hash"] = json_hash(camera_rig)
    write_json(root / "cameras.json", camera_rig)
    write_json(root / "camera_path.json", path)
    write_json(root / "camera_placements.json", {"length_unit": "meter", "cameras": [
        {"camera_id": c.camera_id, "position_world_m": c.center.tolist(),
         "view_direction_world": np.asarray(c.R)[2].tolist()} for c in cameras]})
    evidence_points = [evaluator.corners]
    evidence_kinds = [np.full(len(evaluator.corners), 3, np.int8)]
    if evaluator.support_aware:
        region = request.interaction_region
        free_anchors = np.asarray(region.free_anchor_points_world_m, dtype=float)
        occupied_anchors = np.asarray(region.occupied_anchor_points_world_m, dtype=float)
        interaction_points = np.asarray(region.sample_points_world_m, dtype=float)
        for points, kind in (
            (free_anchors, 4),
            (occupied_anchors, 5),
            (interaction_points, 6),
        ):
            if len(points):
                evidence_points.append(points)
                evidence_kinds.append(np.full(len(points), kind, np.int8))
    np.savez_compressed(
        root / "evidence.npz",
        points=np.concatenate(evidence_points),
        kinds=np.concatenate(evidence_kinds),
    )
    write_json(root / "observation_domain.json", {"mode": "target_viewspace", "human_scale_used": False,
                "human_position_samples": len(evaluator.interaction_points),
                "human_pose_predicted": False,
                "capture_region": (
                    "support_surface_free_volume_v1"
                    if evaluator.support_aware
                    else "expanded_target_box_view_corridor"
                ),
                "supported_object_ids": sorted(evaluator.support_ids),
                "bbox_source": request.target_obb.method if request.target_obb else "aabb_fallback"})
    write_json(root / "post_generation_contract.json", {
        "scene_id": request.scene_id, "expected_subject": "one human static in world coordinates",
        "camera_source": "tour/camera_trajectory.json (actual per-frame K/R/T)",
        "requires": ["verified background/camera correspondence after video generation",
                     "person bbox and common-layout 2D keypoints with confidence and track_id"],
        "next_command": "camera_planning select-human-frames",
        "detector_or_video_model_executed": False})
    result = {"schema_version": "camera_plan_result_v2", "planning_mode": "target_viewspace",
              "scene_id": request.scene_id, "budget": request.budget, "seed": seed,
              "feasible_count": len(records), "selected_ids": [c.camera_id for c in cameras],
              "metrics": selection["metrics"], "trajectory_status": "planned",
              "geometry_backend": evaluator.geometry.name, "reconstruction_status": "not_run",
               "limitations": ["bounded heuristic; no global optimum or exhaustive feasible-domain guarantee",
                            "visibility and transitions are finite samples; not exact pixel coverage",
                            "AABB backend can overestimate occlusion and closes concave free regions",
                            "OBB geometry axes do not identify semantic furniture front",
                            "interaction samples are geometry-only proxies and do not predict a future pose",
                              "azimuth/elevation/radius search is bounded and may miss a narrow feasible region",
                              "generated video must be checked before inheriting calibrated cameras"]}
    write_json(root / "camera_plan_result.json", result)
    return result
