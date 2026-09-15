"""Standalone contracts. Export YSynthetic's smaller schema in integration.py."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Box(StrictModel):
    object_id: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @model_validator(mode="after")
    def ordered(self):
        if np.any(np.array(self.maximum) <= np.array(self.minimum)):
            raise ValueError("box extent must be positive on all axes")
        return self


class Intrinsics(StrictModel):
    width: int = Field(default=1024, gt=0)
    height: int = Field(default=1024, gt=0)
    fx: float = Field(default=900, gt=0)
    fy: float = Field(default=900, gt=0)
    cx: float = 512
    cy: float = 512

    def matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


class CandidateSettings(StrictModel):
    azimuth_count: int = Field(default=24, ge=4, le=720)
    shell_offsets_m: list[float] = Field(default_factory=lambda: [2.0, 3.0], min_length=1)
    heights_above_floor_m: list[float] = Field(default_factory=lambda: [1.2, 1.8], min_length=1)
    focus_heights_above_floor_m: list[float] = Field(default_factory=lambda: [1.0], min_length=1)
    camera_clearance_m: float = Field(default=0.12, ge=0)
    min_target_distance_m: float = Field(default=1.5, gt=0)
    max_target_distance_m: float = Field(default=6.0, gt=0)
    min_surface_visible_fraction: float = Field(default=0.10, ge=0, le=1)
    min_envelope_visible_fraction: float = Field(default=0.25, ge=0, le=1)
    min_space_visible_fraction: float = Field(default=0.02, ge=0, le=1)
    min_target_in_frame_fraction: float = Field(default=1.0, gt=0, le=1)

    @model_validator(mode="after")
    def positive_lists(self):
        for values in (
            self.shell_offsets_m,
            self.heights_above_floor_m,
            self.focus_heights_above_floor_m,
        ):
            if not np.all(np.isfinite(values)) or min(values) <= 0:
                raise ValueError("shell offsets and heights must be finite and positive")
        if self.max_target_distance_m <= self.min_target_distance_m:
            raise ValueError("max_target_distance_m must exceed min_target_distance_m")
        return self


class ObservationSettings(StrictModel):
    """Pose-agnostic evidence domain, derived only from target furniture geometry."""

    mode: Literal["furniture_surface_and_envelope", "furniture_free_space"] = (
        "furniture_surface_and_envelope"
    )
    # Explicit observation bounds are NOT the camera-admissible bounds.
    region: Box | None = None
    voxel_size_m: float = Field(default=0.20, gt=0)
    neighborhood_radius_m: float = Field(default=1.0, gt=0)
    # Conservative radius around a sampled body/reference point.  Obstacles
    # are treated as expanded by this amount before the free-space grid is
    # built, so points near a coffee table or wall are not presented as valid
    # human locations.  This is a geometric proxy, not a pose/body model.
    occupancy_clearance_m: float = Field(default=0.0, ge=0)
    max_grid_cells: int = Field(default=120000, ge=100, le=2000000)
    spatial_samples: int = Field(default=2048, ge=24, le=50000)
    surface_samples: int = Field(default=360, ge=24, le=20000)
    envelope_samples: int = Field(default=360, ge=24, le=20000)
    horizontal_clearance_m: float = Field(default=0.80, gt=0)
    min_height_above_floor_m: float = Field(default=0.05, ge=0)
    full_body_height_m: float = Field(default=2.10, gt=0)
    top_clearance_m: float = Field(default=0.25, ge=0)
    image_margin_ratio: float = Field(default=0.08, ge=0, lt=0.45)
    underside_normal_threshold: float = Field(default=-0.15, ge=-1, le=1)
    seed: int = Field(default=17, ge=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.mode == "furniture_free_space" and self.region is None:
            raise ValueError("furniture_free_space requires an explicit indoor observation.region")
        if self.full_body_height_m <= self.min_height_above_floor_m:
            raise ValueError("full_body_height_m must exceed min_height_above_floor_m")
        return self


class ScoreSettings(StrictModel):
    objective: Literal["legacy", "robust_space"] = "legacy"
    tail_fraction: float = Field(default=0.20, gt=0, le=1)
    tail_weight: float = Field(default=0.50, ge=0, lt=1)
    observability_length_scale_m: float = Field(default=0.01, gt=0)
    multistarts: int = Field(default=3, ge=1, le=16)
    max_score_evaluations: int = Field(default=12000, ge=100)
    exact_max_subsets: int = Field(default=100000, ge=1, le=1000000)
    solver_time_limit_s: float = Field(default=30, gt=0)
    coverage_weight: float = Field(default=1.0, ge=0)
    multiview_weight: float = Field(default=1.0, ge=0)
    information_weight: float = Field(default=0.3, ge=0)
    resolution_weight: float = Field(default=0.3, ge=0)
    worst_zone_weight: float = Field(default=0.5, ge=0)
    parallax_min_degrees: float = Field(default=15, gt=0, lt=90)
    pixel_noise_sigma: float = Field(default=3, gt=0)
    information_scale_m: float = Field(default=0.20, gt=0)
    target_pixels_per_meter: float = Field(default=300, gt=0)
    swap_passes: int = Field(default=2, ge=0, le=20)


class AdaptiveSettings(StrictModel):
    enabled: bool = False
    rounds: int = Field(default=2, ge=0, le=10)
    max_candidates: int = Field(default=256, ge=8, le=5000)
    proposals_per_round: int = Field(default=48, ge=4, le=1000)
    target_count: int = Field(default=4, ge=1, le=32)
    position_step_m: float = Field(default=0.35, gt=0)
    min_gain: float = Field(default=0.0001, ge=0)
    max_evidence_mb: float = Field(default=512, gt=0)


class PathSettings(StrictModel):
    enabled: bool = False
    voxel_size_m: float = Field(default=0.30, gt=0)
    max_grid_cells: int = Field(default=100000, ge=100)
    max_expansions: int = Field(default=50000, ge=100)
    validation_step_m: float = Field(default=0.05, gt=0)
    max_length_m: float | None = Field(default=None, gt=0)


class GeometrySettings(StrictModel):
    backend: Literal["aabb", "triangle_mesh"] = "aabb"
    mesh_file: str | None = None
    sha256: str | None = None

    @model_validator(mode="after")
    def mesh_required(self):
        if self.backend == "triangle_mesh" and not self.mesh_file:
            raise ValueError("triangle_mesh geometry requires mesh_file")
        if self.sha256 is not None and (
            len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError("geometry sha256 must be a lowercase SHA-256 digest")
        return self


class TargetOBB(StrictModel):
    center: tuple[float, float, float]
    axes: list[list[float]]  # columns are local axes in world coordinates
    half_extents: tuple[float, float, float]
    method: str = "upright_mesh_pca"

    @model_validator(mode="after")
    def valid(self):
        q = np.asarray(self.axes)
        if q.shape != (3, 3) or not np.allclose(q.T @ q, np.eye(3), atol=1e-6):
            raise ValueError("OBB axes must be orthonormal")
        if np.linalg.det(q) < 0 or min(self.half_extents) <= 0:
            raise ValueError("OBB needs right-handed axes and positive extents")
        return self


class TargetViewSettings(StrictModel):
    # Only used by planning_mode=target_viewspace; no body-space sampling.
    # The capture box is the target OBB/AABB expanded into a conservative region.
    # It is sampled for partial/set-level coverage, separately from strict target
    # framing and target-surface visibility.
    capture_side_padding_m: float = Field(default=0.0, ge=0)
    capture_top_padding_m: float = Field(default=0.0, ge=0)
    capture_bottom_padding_m: float = Field(default=0.0, ge=0)
    capture_visibility_samples: int = Field(default=128, ge=24, le=4096)
    min_capture_visibility_fraction: float = Field(default=0.90, ge=0, le=1)
    # Objects resting on the target (monitor, lamp, books, etc.) are a third
    # semantic/geometric role.  They remain solid for camera/path collision and
    # for future-human visibility, but their occlusion of furniture appearance
    # is discounted.  Detection is deliberately geometric and auditable.
    support_detection_enabled: bool = True
    support_max_vertical_gap_m: float = Field(default=0.035, ge=0, le=0.25)
    support_max_vertical_penetration_m: float = Field(default=0.015, ge=0, le=0.25)
    support_min_footprint_overlap_fraction: float = Field(default=0.50, gt=0, le=1)
    support_max_footprint_area_ratio: float = Field(default=0.80, gt=0, le=2)
    supported_target_occlusion_penalty: float = Field(default=0.15, ge=0, le=1)
    # If support objects are detected, a deterministic free-space proxy is
    # sampled above the unoccupied part of the support surface.  This is an
    # unknown-human interaction domain, not a predicted pose.
    interaction_anchor_count: int = Field(default=64, ge=4, le=1024)
    interaction_height_offsets_m: list[float] = Field(
        default_factory=lambda: [0.15, 0.50, 0.90, 1.30], min_length=1, max_length=16
    )
    interaction_clearance_m: float = Field(default=0.08, ge=0, le=0.50)
    min_interaction_visibility_fraction: float = Field(default=0.55, ge=0, le=1)
    # Cheap first-stage test of the finite camera-to-capture-box corridor.  It
    # removes only 3-D positions whose corridor is already clearly occluded;
    # it never removes a whole azimuth merely because an obstacle is on the floor.
    shadow_prefilter_enabled: bool = True
    shadow_prefilter_samples: int = Field(default=32, ge=8, le=512)
    shadow_prefilter_min_visible_fraction: float = Field(default=0.55, ge=0, le=1)
    # Hard framing normally applies to the furniture itself. ``capture`` is
    # retained for ablations which require each camera to contain the entire
    # expanded region and consequently tend to produce distant views.
    framing_region: Literal["target", "capture"] = "target"
    # The aim point may remain at the capture centre even when the target alone
    # defines the hard framing shell; this reserves image space above furniture.
    aim_reference: Literal["capture_center", "target_center"] = "capture_center"
    framing_interval_tolerance_m: float = Field(default=0.005, gt=0, le=0.10)
    max_width_ratio: float = Field(default=0.70, gt=0, lt=1)
    max_height_ratio: float = Field(default=0.65, gt=0, lt=1)
    top_margin_ratio: float = Field(default=0.20, ge=0, lt=0.5)
    side_margin_ratio: float = Field(default=0.10, ge=0, lt=0.5)
    bottom_margin_ratio: float = Field(default=0.10, ge=0, lt=0.5)
    min_extent_ratio: float = Field(default=0.18, gt=0, lt=1)
    # World-up offset from the chosen target/capture reference, scaled by furniture height.
    aim_height_ratio: float = Field(default=0.0, ge=-0.5, le=0.5)
    min_visibility_fraction: float = Field(default=0.85, ge=0, le=1)
    visibility_samples: int = Field(default=256, ge=24, le=2048)
    # Candidate generation uses a deliberately loose composition floor so that
    # useful azimuths survive to set selection.  The stricter floor below is
    # enforced only for cameras which may appear in the final rig.
    min_candidate_composition_score: float = Field(default=0.60, ge=0, le=1)
    min_composition_score: float = Field(default=0.68, ge=0, le=1)
    min_camera_height_m: float = Field(default=1.0, ge=0)
    max_camera_height_m: float = Field(default=2.4, gt=0)
    max_camera_distance_m: float = Field(default=4.20, gt=0)
    distance_preference_weight: float = Field(default=0.15, ge=0, lt=1)
    # Candidate coordinates are (azimuth, elevation, radius).  Azimuth is the
    # outer variable; elevations and radii live in its vertical half-plane.
    azimuth_count: int = Field(default=24, ge=8, le=360)
    azimuth_refinement_rounds: int = Field(default=1, ge=0, le=4)
    elevation_min_degrees: float = Field(default=0.0, ge=-30, le=80)
    elevation_max_degrees: float = Field(default=40.0, ge=-30, le=85)
    elevation_count: int = Field(default=5, ge=2, le=33)
    elevation_refinement_rounds: int = Field(default=2, ge=0, le=6)
    max_candidates_per_azimuth_bin: int = Field(default=6, ge=1, le=32)
    max_direction_evaluations: int = Field(default=600, ge=42, le=10000)
    # Candidate generation only; transition/path checks have their own budget.
    max_position_evaluations: int = Field(default=4096, ge=240, le=1000000)
    max_radial_evaluations_per_direction: int = Field(default=17, ge=3, le=4097)
    max_visibility_change: float = Field(default=0.15, gt=0, le=1)
    max_bbox_change: float = Field(default=0.10, gt=0, le=1)
    radial_max_depth: int = Field(default=5, ge=1, le=12)
    radial_min_depth: int = Field(default=2, ge=0, le=6)
    min_radial_step_target_ratio: float = Field(default=0.025, gt=0, le=1)
    max_candidates: int = Field(default=128, ge=8, le=512)
    local_refine_seeds: int = Field(default=8, ge=0, le=32)
    local_refine_rounds: int = Field(default=2, ge=0, le=6)
    coverage_angle_degrees: float = Field(default=35, gt=0, lt=90)
    direction_weight: float = Field(default=1.0, ge=0)
    baseline_weight: float = Field(default=0.25, ge=0)
    composition_weight: float = Field(default=0.20, ge=0)
    worst_composition_weight: float = Field(default=0.35, ge=0)
    elevation_diversity_weight: float = Field(default=0.15, ge=0)
    # Set-level terms.  The capture region is not required to be fully visible
    # from every camera; instead the selected rig should cover it collectively
    # and, where possible, from at least two views.  Proximity is only a soft
    # preference and can never override hard framing/visibility constraints.
    capture_set_coverage_weight: float = Field(default=0.30, ge=0)
    capture_multiview_weight: float = Field(default=0.15, ge=0)
    set_proximity_weight: float = Field(default=0.10, ge=0)
    # A pair must clear both floors.  This is a hard set constraint, not a soft
    # average which can hide one duplicate pair among many diverse pairs.
    min_view_direction_separation_degrees: float = Field(default=6.0, ge=0, lt=90)
    min_azimuth_separation_degrees: float = Field(default=5.0, ge=0, lt=90)
    min_camera_position_separation_m: float = Field(default=0.35, ge=0)
    # Kept for old configuration files.  Path length is now only a tie-breaker
    # after a quality/diversity-valid camera set has been fixed.
    path_weight: float = Field(default=0.10, ge=0)
    insertion_shortlist: int = Field(default=12, ge=2, le=64)
    selection_starts: int = Field(default=3, ge=1, le=8)
    max_path_queries: int = Field(default=300, ge=10, le=5000)
    transition_check_step_m: float = Field(default=0.15, gt=0)
    max_transition_view_checks: int = Field(default=60000, ge=100, le=1000000)

    @model_validator(mode="after")
    def consistent(self):
        if self.radial_min_depth > self.radial_max_depth:
            raise ValueError("radial_min_depth exceeds radial_max_depth")
        if self.elevation_min_degrees >= self.elevation_max_degrees:
            raise ValueError("elevation_min_degrees must be below elevation_max_degrees")
        if self.min_camera_height_m >= self.max_camera_height_m:
            raise ValueError("min_camera_height_m must be below max_camera_height_m")
        coarse_directions = self.azimuth_count * self.elevation_count
        if self.max_direction_evaluations < coarse_directions:
            raise ValueError("direction budget must cover every initial azimuth/elevation probe")
        if self.max_position_evaluations < 3 * coarse_directions:
            raise ValueError("position budget must reserve three radii per initial direction")
        if self.min_extent_ratio >= max(self.max_width_ratio, self.max_height_ratio):
            raise ValueError("minimum image extent cannot exceed both maximum extents")
        if self.min_candidate_composition_score > self.min_composition_score:
            raise ValueError(
                "candidate composition floor cannot exceed final-camera composition floor"
            )
        if (
            self.shadow_prefilter_enabled
            and self.shadow_prefilter_min_visible_fraction
            > min(
                self.min_capture_visibility_fraction,
                self.min_interaction_visibility_fraction,
            )
        ):
            raise ValueError(
                "shadow prefilter must be looser than the authoritative capture/interaction visibility test"
            )
        if not np.all(np.isfinite(self.interaction_height_offsets_m)) or min(
            self.interaction_height_offsets_m
        ) <= 0:
            raise ValueError("interaction height offsets must be finite and positive")
        return self


class InteractionRegion(StrictModel):
    """Finite, geometry-only proxy for unknown human interaction around a support surface."""

    mode: Literal["support_surface_free_volume_v1"] = "support_surface_free_volume_v1"
    free_anchor_points_world_m: list[tuple[float, float, float]] = Field(default_factory=list)
    occupied_anchor_points_world_m: list[tuple[float, float, float]] = Field(default_factory=list)
    sample_points_world_m: list[tuple[float, float, float]] = Field(default_factory=list)
    support_object_ids: list[str] = Field(default_factory=list)
    semantics: str = (
        "pose-agnostic vertical proxy samples above free support-surface anchors; "
        "points colliding with target, supported objects, or other obstacles are removed"
    )


class PlanningRequest(StrictModel):
    schema_version: Literal["camera_planning_request_v2"] = "camera_planning_request_v2"
    scene_id: str
    length_unit: Literal["meter"] = "meter"
    up_axis: Literal["Y", "Z"]
    floor_height_m: float
    target: Box
    planning_mode: Literal["legacy", "target_viewspace"] = "legacy"
    target_obb: TargetOBB | None = None
    target_view: TargetViewSettings = Field(default_factory=TargetViewSettings)
    supported_objects: list[Box] = Field(default_factory=list)
    interaction_region: InteractionRegion | None = None
    obstacles: list[Box] = Field(default_factory=list)
    allowed_camera_region: Box
    budget: int = Field(default=4, ge=2, le=64)
    intrinsics: Intrinsics = Field(default_factory=Intrinsics)
    candidates: CandidateSettings = Field(default_factory=CandidateSettings)
    observation: ObservationSettings = Field(default_factory=ObservationSettings)
    scoring: ScoreSettings = Field(default_factory=ScoreSettings)
    adaptive: AdaptiveSettings = Field(default_factory=AdaptiveSettings)
    path: PathSettings = Field(default_factory=PathSettings)

    @model_validator(mode="after")
    def research_mode(self):
        if self.planning_mode == "target_viewspace":
            if not self.path.enabled:
                raise ValueError("target_viewspace requires path.enabled for joint tour selection")
            if self.budget > self.target_view.max_candidates:
                raise ValueError("target_view candidate cap below camera budget")
            return self
        if (
            self.scoring.objective == "robust_space"
            and self.observation.mode != "furniture_free_space"
        ):
            raise ValueError(
                "robust_space requires furniture_free_space, not the legacy body envelope"
            )
        if self.adaptive.enabled and self.scoring.objective != "robust_space":
            raise ValueError("adaptive refinement currently requires robust_space")
        if self.adaptive.enabled and self.budget > self.adaptive.max_candidates:
            raise ValueError("adaptive candidate cap is smaller than camera budget")
        return self

    geometry: GeometrySettings = Field(default_factory=GeometrySettings)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = (
            [self.target.object_id]
            + [box.object_id for box in self.supported_objects]
            + [box.object_id for box in self.obstacles]
        )
        if len(ids) != len(set(ids)):
            raise ValueError("target/support/obstacle roles must have unique object_id values")
        if self.interaction_region is not None:
            declared = {box.object_id for box in self.supported_objects}
            if set(self.interaction_region.support_object_ids) != declared:
                raise ValueError("interaction support ids must match supported_objects")
            if not self.interaction_region.sample_points_world_m:
                raise ValueError("interaction region must contain at least one free sample")
        return self

    @property
    def up_index(self):
        return 1 if self.up_axis == "Y" else 2


class Camera(StrictModel):
    camera_id: str
    width: int
    height: int
    K: list[list[float]]
    R: list[list[float]]
    T: list[float]
    coordinate_convention: Literal["world_to_camera"] = "world_to_camera"

    @property
    def center(self):
        return -np.asarray(self.R).T @ np.asarray(self.T)

    @model_validator(mode="after")
    def calibrated(self):
        k, r, t = np.asarray(self.K), np.asarray(self.R), np.asarray(self.T)
        if k.shape != (3, 3) or r.shape != (3, 3) or t.shape != (3,):
            raise ValueError("K/R must be 3x3, T must be length 3")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("positive image dimensions required")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("invalid K")
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r), 1):
            raise ValueError("R must be a proper SO(3) rotation")
        return self
