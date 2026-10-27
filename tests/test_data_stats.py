"""Global: evidence integrity; local: independent image groups and paired analysis."""

import numpy as np  # Synthetic inputs have known ground truth.

from e2e_bsf.data import assign_splits
from e2e_bsf.statistics import paired_summary


def test_duplicate_groups_do_not_cross_splits():
    """Exact/perceptual group identities must survive split assignment."""
    rows = [{"id": str(i), "label": i % 2, "official_split": "train", "group": i // 2}
            for i in range(40)]
    rows.append({"id": "test", "label": 0, "official_split": "val", "group": 0})
    result = assign_splits(rows, seed=42)
    assert result[0]["split"] == result[1]["split"] == "excluded"
    for group in {row["group"] for row in result} - {0}:
        assert len({row["split"] for row in result if row["group"] == group}) == 1
    assert result[-1]["split"] == "test"
    assert {row["split"] for row in result} >= {"train", "dev", "test"}


def test_paired_summary_reports_all_seeds():
    """Pairing uses seed IDs; incomplete pairs cannot silently change the claim."""
    summary = paired_summary({1: 2.0, 2: 3.0, 3: 4.0}, {1: 1.0, 2: 2.0, 3: 3.0})
    assert summary["n_pairs"] == 3
    assert np.isclose(summary["mean_improvement"], 1.0)
    assert summary["ci95"] == [1.0, 1.0]
    assert paired_summary({1: 2.0}, {2: 1.0})["n_pairs"] == 0
