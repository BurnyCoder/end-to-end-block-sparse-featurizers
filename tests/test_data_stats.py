"""Global: evidence integrity; local: independent image groups and paired analysis."""

import numpy as np  # Synthetic inputs have known ground truth.

from e2e_bsf.data import assign_splits, exclude_previewed_test_groups
from e2e_bsf.statistics import paired_summary, required_seeds


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


def test_engineering_preview_excludes_entire_test_group():
    """Local: preview exclusion uses identities; global: no inspected test group is reused."""
    rows = [{"id": "seen", "group": 1, "split": "test"},
            {"id": "near_duplicate", "group": 1, "split": "test"},
            {"id": "unseen", "group": 2, "split": "test"},
            {"id": "fitting", "group": 3, "split": "train"}]
    excluded = exclude_previewed_test_groups(rows, {"seen"})
    assert excluded == {1}
    assert [row["split"] for row in rows] == ["excluded", "excluded", "test", "train"]


def test_high_pilot_effect_uses_minimum_replication_without_solver_failure():
    """A near-zero paired variance must not crash the bounded design's power calculation."""
    result = required_seeds({0: .65, 1: .67, 2: .69}, {0: .59, 1: .609, 2: .628})
    assert result["required"] == 5
    assert result["power_at_five"] >= .8
