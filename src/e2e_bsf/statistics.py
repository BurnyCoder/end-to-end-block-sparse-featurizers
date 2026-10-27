"""Global: honest uncertainty. Local: paired seeds and image-group bootstrap.

Sources: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html;
https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.html.
"""

import numpy as np  # Seeded resampling is reproducible.
from scipy import stats  # Student intervals describe seed-level mean differences.
from statsmodels.stats.power import TTestPower  # Standard paired-difference power calculation.


def paired_summary(local, kl):
    """Local: paired local-minus-KL effects; global: do not pseudo-replicate patches."""
    seeds = sorted(set(local) & set(kl))  # Incomplete conditions are explicitly excluded and listed.
    differences = np.array([local[seed] - kl[seed] for seed in seeds], dtype=float)
    result = {"n_pairs": len(seeds), "seeds": seeds,
              "unpaired_local": sorted(set(local) - set(kl)), "unpaired_kl": sorted(set(kl) - set(local))}
    if not len(seeds):
        return result
    mean = float(differences.mean())
    half = float(stats.t.ppf(0.975, len(seeds) - 1) * stats.sem(differences)) if len(seeds) > 1 else None
    result.update(mean_improvement=mean, differences=differences.tolist(),
                  ci95=[mean - half, mean + half] if half is not None else None)
    return result


def bootstrap_images(values, groups, seed=42, replicates=2000):
    """Local: resample duplicate groups; global: image-related patches are not independent."""
    values, groups = np.asarray(values), np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    totals = np.bincount(inverse, weights=values)  # Preserve image-weighted endpoint.
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for i in range(replicates):
        draw = rng.integers(0, len(unique), len(unique))
        means[i] = totals[draw].sum() / counts[draw].sum()
    return np.quantile(means, [0.025, 0.975]).tolist()


def required_seeds(local, kl):
    """Local: pilot-only variance; global: freeze replication before seeing test results."""
    seeds = sorted(set(local) & set(kl))
    if len(seeds) < 3:
        return {"required": None, "reason": "fewer than three independent pilot pairs"}
    differences = np.array([local[s] - kl[s] for s in seeds])
    minimum_effect = 0.1 * np.mean([local[s] for s in seeds])  # Prespecified practical improvement.
    standard_deviation = differences.std(ddof=1)
    if minimum_effect <= 0 or standard_deviation <= 0:
        return {"required": None, "reason": "degenerate pilot variance/effect; use conservative cap"}
    effect_size = minimum_effect / standard_deviation
    analysis = TTestPower()  # Evaluate the allowed minimum before numerical root finding.
    power_at_five = float(analysis.power(effect_size=effect_size, nobs=5, alpha=.05, alternative="two-sided"))
    required = 5.0 if power_at_five >= .8 else float(analysis.solve_power(
        effect_size=effect_size, alpha=.05, power=.8, alternative="two-sided"))
    return {"required": int(np.ceil(required)) if np.isfinite(required) else None,
            "effect_size": float(effect_size), "power_at_five": power_at_five if np.isfinite(power_at_five) else None,
            "minimum_absolute_effect": float(minimum_effect), "pilot_pairs": len(seeds),
            "reason": "minimum five meets target" if power_at_five >= .8 else "numerical paired-t solution"}
