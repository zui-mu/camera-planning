"""Interpretable set utility. Scores are hypotheses, not reconstruction accuracy."""

from dataclasses import dataclass

import numpy as np

from .contracts import Camera, ScoreSettings
from .geometry import GeometryBackend, project, projection_jacobian


@dataclass
class Evidence:
    cameras: list[Camera]
    points: np.ndarray
    zones: np.ndarray
    visible: np.ndarray
    information: np.ndarray
    resolution: np.ndarray
    directions: np.ndarray
    kinds: np.ndarray
    weights: np.ndarray | None = None

    def subset(self, indices):
        return Evidence(
            [self.cameras[i] for i in indices],
            self.points,
            self.zones,
            self.visible[indices],
            self.information[indices],
            self.resolution[indices],
            self.directions[indices],
            self.kinds,
            self.weights,
        )


def build_evidence(
    cameras,
    points,
    zones,
    geometry: GeometryBackend,
    settings: ScoreSettings,
    kinds=None,
    image_margin_ratio=0.0,
    weights=None,
):
    n, m = len(cameras), len(points)
    kinds = np.zeros(m, dtype=np.int8) if kinds is None else np.asarray(kinds, dtype=np.int8)
    if kinds.shape != (m,):
        raise ValueError("kinds must have one label per evidence point")
    visible = np.zeros((n, m), dtype=bool)
    information = np.zeros((n, m, 3, 3))
    resolution = np.zeros((n, m))
    directions = np.zeros((n, m, 3))
    for i, camera in enumerate(cameras):
        uv, depth = project(camera, points)
        margin_x = camera.width * image_margin_ratio
        margin_y = camera.height * image_margin_ratio
        visible[i] = (
            (depth > 0.05)
            & (uv[:, 0] >= margin_x)
            & (uv[:, 0] < camera.width - margin_x)
            & (uv[:, 1] >= margin_y)
            & (uv[:, 1] < camera.height - margin_y)
            & ~geometry.blocked(camera.center, points)
        )
        jac = projection_jacobian(camera, points)
        information[i] = np.einsum("nki,nkj->nij", jac, jac) / settings.pixel_noise_sigma**2
        information[i] *= visible[i, :, None, None]
        # Frontoparallel area-density surrogate; no surface normals or human image used.
        k = np.array(camera.K)
        density = k[0, 0] * k[1, 1] / np.maximum(depth, 0.05) ** 2
        resolution[i] = visible[i] * np.minimum(density / settings.target_pixels_per_meter**2, 1.0)
        delta = camera.center - points
        directions[i] = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
    return Evidence(
        cameras, points, zones, visible, information, resolution, directions, kinds, weights
    )


def evaluate_set(evidence: Evidence, indices, settings: ScoreSettings):
    indices = list(indices)
    robust = {}
    if settings.objective == "robust_space":
        from .observability import spatial_metrics

        robust = spatial_metrics(evidence, indices, settings)
    if not indices:
        return dict(
            total=0.0,
            coverage=0.0,
            multiview=0.0,
            information=0.0,
            resolution=0.0,
            worst_zone=0.0,
            furniture_surface_coverage=0.0,
            framing_envelope_coverage=0.0,
            furniture_surface_multiview=0.0,
            framing_envelope_multiview=0.0,
            **robust,
        )
    vis = evidence.visible[indices]
    coverage = vis.any(axis=0).astype(float)
    multi = np.zeros(len(evidence.points), dtype=bool)
    cosine_limit = np.cos(np.deg2rad(settings.parallax_min_degrees))
    for a in range(len(indices)):
        for b in range(a):
            cosine = np.sum(
                evidence.directions[indices[a]] * evidence.directions[indices[b]], axis=1
            )
            # abs: both nearly parallel AND nearly opposite collinear rays are degenerate.
            multi |= vis[a] & vis[b] & (abs(cosine) <= cosine_limit)
    info = evidence.information[indices].sum(axis=0) * settings.information_scale_m**2
    _, logdet = np.linalg.slogdet(np.eye(3)[None] + info)
    # Smooth bounded transform stabilizes scale, not a calibrated success probability.
    information = 1 - np.exp(-logdet / 12.0)
    resolution = evidence.resolution[indices].max(axis=0)
    zones = np.unique(evidence.zones)

    def balanced(values):
        return float(np.mean([np.mean(values[evidence.zones == z]) for z in zones]))

    components = dict(
        coverage=balanced(coverage),
        multiview=balanced(multi),
        information=balanced(information),
        resolution=balanced(resolution),
        worst_zone=float(min(np.mean(multi[evidence.zones == z]) for z in zones)),
    )
    total = sum(components[name] * getattr(settings, f"{name}_weight") for name in components)
    surface = evidence.kinds == 0
    envelope = evidence.kinds == 1

    def kind_mean(values, mask):
        return float(np.mean(values[mask])) if np.any(mask) else 0.0

    return dict(
        total=float(robust.get("robust_utility", total)),
        **components,
        furniture_surface_coverage=kind_mean(coverage, surface),
        framing_envelope_coverage=kind_mean(coverage, envelope),
        furniture_surface_multiview=kind_mean(multi, surface),
        framing_envelope_multiview=kind_mean(multi, envelope),
        **robust,
    )
