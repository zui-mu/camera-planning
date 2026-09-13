"""Deterministic, pose-agnostic observation domain around target furniture."""

import numpy as np

from .contracts import PlanningRequest
from .geometry import GeometryBackend

SURFACE_KIND = 0
ENVELOPE_KIND = 1
SPACE_KIND = 2


def sample_observation_data(request, geometry):
    if request.observation.mode != "furniture_free_space":
        points, zones, kinds = sample_observation_domain(request, geometry)
        return points, zones, kinds, np.ones(len(points)), {"mode": "legacy_framing_envelope"}
    from .free_space import furniture_domain

    settings = request.observation
    surface, _ = geometry.sample_surface(
        request.target.object_id,
        settings.surface_samples,
        settings.seed,
        request.up_index,
        settings.underside_normal_threshold,
    )
    space, weights, metadata = furniture_domain(request, geometry)
    center = (np.asarray(request.target.minimum) + request.target.maximum) / 2
    return (
        np.concatenate((surface, space)),
        np.concatenate(
            (
                _spatial_zones(surface, center, request.up_index),
                _spatial_zones(space, center, request.up_index) + 8,
            )
        ),
        np.concatenate(
            (np.zeros(len(surface), dtype=np.int8), np.full(len(space), SPACE_KIND, dtype=np.int8))
        ),
        np.concatenate((np.ones(len(surface)), weights)),
        metadata,
    )


def _spatial_zones(points, center, up_index):
    horizontal = [axis for axis in range(3) if axis != up_index]
    delta = points[:, horizontal] - center[horizontal]
    azimuth = np.arctan2(delta[:, 1], delta[:, 0])
    sectors = np.floor((azimuth + np.pi) / (np.pi / 2)).astype(int) % 4
    height = (points[:, up_index] >= center[up_index]).astype(int)
    return sectors + 4 * height


def sample_observation_domain(request: PlanningRequest, geometry: GeometryBackend):
    """Return points, spatial zones and semantic kind labels.

    The envelope is a deterministic framing volume derived from target bounds and
    a declared full-body height.  It deliberately contains no pose, joint or
    learned human-location prior.
    """
    if request.observation.mode == "furniture_free_space":
        return sample_observation_data(request, geometry)[:3]
    settings = request.observation
    object_id = request.target.object_id
    surface, _normals = geometry.sample_surface(
        object_id,
        settings.surface_samples,
        settings.seed,
        request.up_index,
        settings.underside_normal_threshold,
    )

    target_low = np.asarray(request.target.minimum, dtype=float)
    target_high = np.asarray(request.target.maximum, dtype=float)
    low, high = target_low.copy(), target_high.copy()
    horizontal = [axis for axis in range(3) if axis != request.up_index]
    low[horizontal] -= settings.horizontal_clearance_m
    high[horizontal] += settings.horizontal_clearance_m
    low[request.up_index] = request.floor_height_m + settings.min_height_above_floor_m
    high[request.up_index] = max(
        request.floor_height_m + settings.full_body_height_m,
        target_high[request.up_index] + settings.top_clearance_m,
    )
    if high[request.up_index] <= low[request.up_index]:
        raise ValueError("invalid furniture framing envelope height")

    rng = np.random.default_rng(settings.seed + 1)
    accepted = []
    count = 0
    for _ in range(20):
        batch = rng.uniform(low, high, (settings.envelope_samples * 3, 3))
        batch = batch[~geometry.occupied(batch, 0.0)]
        accepted.append(batch)
        count += len(batch)
        if count >= settings.envelope_samples:
            break
    if count < settings.envelope_samples:
        raise ValueError(
            "furniture framing envelope has too little free volume; inspect target, floor and solids"
        )
    envelope = np.concatenate(accepted)[: settings.envelope_samples]

    center = (low + high) / 2
    surface_zones = _spatial_zones(surface, center, request.up_index)
    envelope_zones = _spatial_zones(envelope, center, request.up_index) + 8
    return (
        np.concatenate((surface, envelope)),
        np.concatenate((surface_zones, envelope_zones)),
        np.concatenate(
            (
                np.full(len(surface), SURFACE_KIND, dtype=np.int8),
                np.full(len(envelope), ENVELOPE_KIND, dtype=np.int8),
            )
        ),
    )
