"""Global: uncertainty is explicit for every displayed metric; local: verify seed aggregation.

Student intervals follow https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html.
"""

import numpy as np  # Compare means and interval widths independently.
from scipy import stats  # Reference t quantile for three independent training seeds.
from e2e_bsf.reporting import aggregate_results


def test_every_comparison_metric_has_a_seed_interval():
    """Local: retain between-seed variability; global: reconstruction and accuracy need uncertainty too."""
    rows = [{"condition": "block_mse", "seed": seed, "images": 20, "parameters": 100,
             "kl": value, "nmse": 2 * value, "accuracy": 1 - value}
            for seed, value in enumerate([.1, .2, .3])]
    summaries, _ = aggregate_results(rows)
    row = summaries[0]
    half = stats.t.ppf(.975, 2) * .1 / np.sqrt(3)
    np.testing.assert_allclose(row["nmse_seed_ci95"], [.4 - 2 * half, .4 + 2 * half])
    np.testing.assert_allclose(row["accuracy_seed_ci95"], [.8 - half, .8 + half])
    assert row["stable_rank_seed_ci95"] is None


def test_single_seed_has_no_seed_interval():
    """Local: a missing interval stays missing; global: the smoke study cannot estimate seed variance."""
    summaries, _ = aggregate_results([{"condition": "block_kl", "seed": 100, "images": 20,
                                       "parameters": 100, "kl": .1, "nmse": .5}])
    assert summaries[0]["nmse_seed_ci95"] is None
