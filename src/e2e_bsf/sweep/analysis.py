"""Global: conditional image uncertainty, never fictitious seed replication; local: curve math.

Source: https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html.
"""

import numpy as np


def normalized_auc(values, coordinates):
    """Integrate the observed operating points over log2 coordinates without extrapolation."""
    x = np.log2(np.asarray(coordinates, dtype=float))
    values = np.asarray(values, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all() or not np.all(np.diff(x) > 0) or values.shape[-1] != len(x):
        raise ValueError("A complete curve needs strictly increasing positive coordinate budgets")
    return np.trapezoid(values, x=x, axis=-1) / (x[-1] - x[0])


def bootstrap_columns(values, groups, replicates=2000, seed=42):
    """Use identical duplicate-group draws for every column and condition."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    unique, inverse = np.unique(groups, return_inverse=True)
    if not len(unique) or len(inverse) != len(values) or not np.isfinite(values).all():
        raise ValueError("Finite observations and matching image groups are required")
    totals = np.zeros((len(unique), values.shape[1]))
    np.add.at(totals, inverse, values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, values.shape[1]))
    for index in range(replicates):
        draw = rng.integers(0, len(unique), len(unique))
        samples[index] = totals[draw].sum(0) / counts[draw].sum()
    return np.quantile(samples, [.025, .975], axis=0).T


def paired_curves(local, proposed, coordinates, groups, replicates=2000):
    """Pair both operating points and images; the resulting intervals condition on one seed."""
    difference = np.asarray(local) - np.asarray(proposed)
    auc = normalized_auc(difference, coordinates)
    intervals = bootstrap_columns(np.column_stack([auc, difference]), groups, replicates)
    return {"mean_improvement": float(auc.mean()), "ci95": intervals[0].tolist(),
            "per_k_improvement": difference.mean(0).tolist(), "per_k_ci95": intervals[1:].tolist(),
            "uncertainty": "paired image-group bootstrap conditional on seed 200; no seed uncertainty"}


def threshold_budgets(values, coordinates, thresholds=(.01, .05, .1)):
    """Use measured points only; missing fidelity targets are null rather than extrapolated."""
    return {f"{threshold:g}": next((int(k) for k, value in zip(coordinates, values, strict=True)
                                   if value <= threshold), None) for threshold in thresholds}


def pareto_indices(values):
    """Return measured points not strictly dominated when minimizing every listed quantity."""
    values = np.asarray(values, dtype=float)
    # A tie is retained: dominance requires no worse coordinates and at least one strict gain.
    return [i for i, point in enumerate(values)
            if not np.any(np.all(values <= point, axis=1) & np.any(values < point, axis=1))]
