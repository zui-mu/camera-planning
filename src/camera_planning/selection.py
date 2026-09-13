"""Fixed-budget baselines, greedy marginal utility, and deterministic one-swap search."""

import heapq
import itertools
import math

import numpy as np

from .scoring import Evidence, evaluate_set


def _coverage_select(evidence, budget, method, settings):
    from .observability import coverage_weights

    weights = coverage_weights(evidence, surface_only=method == "surface_coverage")
    visible = evidence.visible
    if method == "milp_coverage":
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import csc_matrix, eye, hstack, vstack
        except ImportError as error:
            raise RuntimeError(
                "milp_coverage needs optional solver extra: uv sync --extra solver"
            ) from error
        n, m = visible.shape
        matrix = vstack(
            [
                hstack([-csc_matrix(visible.T, dtype=float), eye(m)]),
                csc_matrix(np.r_[np.ones(n), np.zeros(m)][None]),
            ],
            format="csc",
        )
        result = milp(
            c=np.r_[np.zeros(n), -weights],
            integrality=np.r_[np.ones(n), np.zeros(m)],
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(
                matrix,
                np.r_[np.full(m, -np.inf), budget],
                np.r_[np.zeros(m), budget],
            ),
            options={"time_limit": settings.solver_time_limit_s},
        )
        if result.x is None:
            raise RuntimeError(f"coverage MILP has no incumbent: {result.message}")
        ids = np.flatnonzero(result.x[:n] > 0.5).tolist()
        if len(ids) != budget:
            raise RuntimeError("coverage MILP did not return the required integer budget")
        return ids, [
            {
                "event": "milp_coverage",
                "objective": "weighted_single_view_coverage",
                "status": int(result.status),
                "proven_optimal": result.status == 0,
                "gap": float(result.mip_gap),
                "not_robust_observability_oracle": True,
            }
        ]
    selected, covered, trace = [], np.zeros(visible.shape[1], bool), []
    heap = [(-float(row @ weights), i, 0) for i, row in enumerate(visible)]
    heapq.heapify(heap)
    evaluations = len(heap)
    for iteration in range(budget):
        if method == "lazy_coverage":
            while True:
                gain, winner, stamp = heapq.heappop(heap)
                if stamp == iteration:
                    break
                gain = -float((visible[winner] & ~covered) @ weights)
                evaluations += 1
                heapq.heappush(heap, (gain, winner, iteration))
        else:
            remaining = [i for i in range(len(visible)) if i not in selected]
            winner = max(remaining, key=lambda i: float((visible[i] & ~covered) @ weights))
            evaluations += len(remaining)
        covered |= visible[winner]
        selected.append(winner)
        trace.append(
            {
                "event": "coverage_add",
                "camera_id": evidence.cameras[winner].camera_id,
                "coverage": float(covered @ weights),
            }
        )
    trace.append(
        {
            "event": "selection_stats",
            "score_evaluations": evaluations,
            "objective": "weighted_single_view_coverage",
            "lazy": method == "lazy_coverage",
        }
    )
    return selected, trace


def _robust_select(evidence, budget, settings):
    from .observability import EvaluationBudgetExceeded, SetObjective

    objective = SetObjective(evidence, settings)
    # Multiple deterministic seeds avoid all-zero one-view information scores.
    seed_scores = evidence.visible[:, evidence.kinds == 2] @ objective.weights
    seeds = np.argsort(-seed_scores, kind="stable")[: settings.multistarts]
    best, best_value, trace = None, -np.inf, []
    try:
        for seed in seeds:
            ids = [int(seed)]
            while len(ids) < budget:
                choices = [i for i in range(len(evidence.cameras)) if i not in ids]
                winner = max(choices, key=lambda i: objective(ids + [i]))
                ids.append(winner)
            value = objective(ids)
            trace.append({"event": "multistart", "seed_camera": int(seed), "score": value})
            if value > best_value:
                best, best_value = ids, value
        for _ in range(settings.swap_passes):
            replacement, value = None, best_value
            for slot in range(budget):
                for candidate in range(len(evidence.cameras)):
                    if candidate in best:
                        continue
                    proposal = best.copy()
                    proposal[slot] = candidate
                    score = objective(proposal)
                    if score > value + 1e-10:
                        replacement, value = proposal, score
            if replacement is None:
                break
            best, best_value = replacement, value
            trace.append({"event": "swap", "score": value})
    except EvaluationBudgetExceeded:
        trace.append({"event": "evaluation_budget_reached"})
    if best is None:
        raise ValueError("score evaluation budget exhausted before a complete set; increase cap")
    trace.append(
        {
            "event": "selection_stats",
            "score_evaluations": objective.evaluations,
            "objective": "robust_space",
            "optimality": "heuristic_no_submodular_guarantee",
        }
    )
    return best, trace


def select(evidence: Evidence, budget, settings, method="greedy", seed=0, up_index=2):
    n = len(evidence.cameras)
    if budget > n or budget < 2:
        raise ValueError(f"need 2 <= budget <= feasible candidates ({n}), got {budget}")
    if method in ("coverage", "lazy_coverage", "surface_coverage", "milp_coverage"):
        return _coverage_select(evidence, budget, method, settings)
    if method == "greedy" and settings.objective == "robust_space":
        return _robust_select(evidence, budget, settings)

    def score(indices):
        return evaluate_set(evidence, indices, settings)["total"]

    trace = []
    if method == "random":
        indices = np.random.default_rng(seed).choice(n, budget, replace=False).tolist()
    elif method == "farthest":
        directions = np.array([np.array(c.R).T[:, 2] for c in evidence.cameras])
        indices = [int(np.argmax(evidence.visible.mean(axis=1)))]
        while len(indices) < budget:
            distance = np.min(1 - directions @ directions[indices].T, axis=1)
            distance[indices] = -np.inf
            indices.append(int(np.argmax(distance)))
    elif method == "uniform":
        # Snap evenly spaced azimuth targets to feasible candidates; prefer median height/range.
        centers = np.array([c.center for c in evidence.cameras])
        focus = evidence.points.mean(axis=0)
        axis = up_index
        horizontal = [i for i in range(3) if i != axis]
        delta = centers[:, horizontal] - focus[horizontal]
        azimuth = np.arctan2(delta[:, 1], delta[:, 0])
        radius = np.linalg.norm(delta, axis=1)
        tie = abs(centers[:, axis] - np.median(centers[:, axis])) + abs(radius - np.median(radius))
        indices = []
        for target in np.linspace(0, 2 * np.pi, budget, endpoint=False):
            difference = abs(np.arctan2(np.sin(azimuth - target), np.cos(azimuth - target)))
            rank = sorted(range(n), key=lambda i: (round(float(difference[i]), 6), tie[i], i))
            indices.append(next(i for i in rank if i not in indices))
    elif method == "exact":
        if math.comb(n, budget) > settings.exact_max_subsets:
            raise ValueError(f"exact enumeration capped at {settings.exact_max_subsets} subsets")
        indices = list(max(itertools.combinations(range(n), budget), key=score))
    elif method in ("greedy", "coverage"):
        indices = []
        for _ in range(budget):
            remaining = [i for i in range(n) if i not in indices]
            objective = (
                score
                if method == "greedy"
                else lambda ids: evaluate_set(evidence, ids, settings)["coverage"]
            )
            winner = max(remaining, key=lambda i: objective(indices + [i]))
            indices.append(winner)
            trace.append(
                {
                    "event": "add",
                    "camera_id": evidence.cameras[winner].camera_id,
                    "score": score(indices),
                }
            )
        if method == "greedy":
            for _ in range(settings.swap_passes):
                current, best = score(indices), None
                for slot in range(budget):
                    for candidate in range(n):
                        if candidate in indices:
                            continue
                        proposal = indices.copy()
                        proposal[slot] = candidate
                        value = score(proposal)
                        if value > current + 1e-10:
                            current, best = value, proposal
                if best is None:
                    break
                indices = best
                trace.append({"event": "swap", "score": current})
    else:
        raise ValueError(f"unknown method {method}")
    return indices, trace
