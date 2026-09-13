"""Task-independent geometry proxy, not a learned HMR success probability."""

import numpy as np


def spatial_mask(evidence):
    mask = evidence.kinds == 2
    if not mask.any():
        raise ValueError("robust_space needs kind=2 free-space evidence")
    return mask


def normalized_weights(evidence, mask):
    weights = np.ones(len(evidence.points)) if evidence.weights is None else evidence.weights
    weights = np.asarray(weights, float)[mask]
    if not len(weights) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sample weights must be finite and positive")
    return weights / weights.sum()


def lower_tail(values, weights, fraction):
    """Exact weighted lower-tail mean, with partial mass of the boundary sample."""
    values, weights = np.asarray(values), np.asarray(weights, float)
    if not 0 < fraction <= 1 or values.shape != weights.shape or not len(values):
        raise ValueError("invalid weighted tail input")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("tail values/weights must be finite; weights positive")
    order = np.argsort(values, kind="stable")
    weights = weights[order] / weights.sum()
    mass = np.clip(fraction - (np.cumsum(weights) - weights), 0, weights)
    return float(np.dot(values[order], mass) / fraction)


def quality_from_information(information, counts, scale):
    # A single view and opposed collinear rays have a rank-deficient constraint.
    eigen = np.maximum(np.linalg.eigvalsh(information * scale**2)[:, 0], 0)
    eigen[eigen < 1e-9] = 0
    return np.where(counts >= 2, eigen / (1 + eigen), 0.0)


def point_quality(evidence, indices, settings):
    mask = spatial_mask(evidence)
    ids = list(indices)
    if not ids:
        return np.zeros(int(mask.sum()))
    return quality_from_information(
        evidence.information[ids][:, mask].sum(axis=0),
        evidence.visible[ids][:, mask].sum(axis=0),
        settings.observability_length_scale_m,
    )


def spatial_metrics(evidence, indices, settings):
    mask = spatial_mask(evidence)
    weights = normalized_weights(evidence, mask)
    q = point_quality(evidence, indices, settings)
    counts = evidence.visible[list(indices)][:, mask].sum(axis=0)
    mean = float(np.dot(q, weights))
    tail = lower_tail(q, weights, settings.tail_fraction)
    return {
        "robust_utility": (1 - settings.tail_weight) * mean + settings.tail_weight * tail,
        "spatial_mean_quality": mean,
        "spatial_lower_tail_quality": tail,
        "spatial_visible_fraction": float(np.dot(counts >= 1, weights)),
        "spatial_multiview_fraction": float(np.dot(counts >= 2, weights)),
        "spatial_informative_fraction": float(np.dot(q > 1e-8, weights)),
    }


class SetObjective:
    """Bounded memoization over precomputed arrays. Never performs ray casting."""

    def __init__(self, evidence, settings):
        self.evidence, self.settings = evidence, settings
        mask = spatial_mask(evidence)
        self.information = evidence.information[:, mask]
        self.visible = evidence.visible[:, mask]
        self.weights = normalized_weights(evidence, mask)
        self.cache = {}
        self.evaluations = 0

    def __call__(self, indices):
        key = tuple(sorted(indices))
        if key in self.cache:
            return self.cache[key]
        if self.evaluations >= self.settings.max_score_evaluations:
            raise EvaluationBudgetExceeded
        self.evaluations += 1
        q = quality_from_information(
            self.information[list(key)].sum(axis=0),
            self.visible[list(key)].sum(axis=0),
            self.settings.observability_length_scale_m,
        )
        result = (1 - self.settings.tail_weight) * float(q @ self.weights)
        result += self.settings.tail_weight * lower_tail(
            q,
            self.weights,
            self.settings.tail_fraction,
        )
        self.cache[key] = result
        return result


class EvaluationBudgetExceeded(RuntimeError):
    pass


def coverage_weights(evidence, surface_only=False):
    mask = evidence.kinds == 0 if surface_only else evidence.kinds == 2
    weights = np.zeros(len(evidence.points))
    if mask.any():
        weights[mask] = normalized_weights(evidence, mask)
    elif surface_only:
        raise ValueError("no surface samples for surface coverage baseline")
    else:
        # Match legacy region-balanced coverage exactly.
        zones = np.unique(evidence.zones)
        for zone in zones:
            region = evidence.zones == zone
            weights[region] = 1 / (len(zones) * region.sum())
    return weights
