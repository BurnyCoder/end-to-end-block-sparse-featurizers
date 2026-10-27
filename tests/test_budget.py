"""Global: use the local budget faithfully; local: account for each architecture's measured cost."""

import pytest  # Exercise both feasible fallback and mandatory budget failure.

from e2e_bsf.config import Config, CONDITIONS
from e2e_bsf.io import write_json
from e2e_bsf.study import freeze_schedule


@pytest.fixture
def pilot_records(tmp_path):
    """Local: fixed pilot costs; global: test allocation without consulting held-out outcomes."""
    write_json(tmp_path / "manifest.json", {"counts": {"train": 7575, "dev": 1894, "test": 3875}})
    records = []
    for seed in range(3):
        for lr in (1e-4, 3e-4):
            for condition in CONDITIONS:
                kl = .65 + .02 * seed - (.06 + .001 * seed if condition == "block_kl" else 0)
                records.append({"condition": condition, "seed": seed, "lr": lr, "dev_kl": kl,
                                "seconds": 285 if condition.startswith("scalar") else 90, "microbatch": 8})
    return records


def test_allocation_uses_each_condition_cost_without_changing_exposure(tmp_path, pilot_records):
    """Different QR costs still permit one shared epoch across five independently paired seeds."""
    schedule = freeze_schedule(Config(root=str(tmp_path)), pilot_records, 20 * 3600)
    assert len(schedule["seeds"]) == 5
    assert schedule["epochs"] == 1
    assert 17 * 3600 < schedule["predicted_optimization_seconds"] <= 20 * 3600
    assert schedule["forecast_seconds_per_epoch"]["scalar_mse"] > schedule["forecast_seconds_per_epoch"]["block_mse"]


def test_budget_shortfall_uses_declared_exploratory_fallback(tmp_path, pilot_records, monkeypatch):
    """Local: six required pairs exceed budget; global: keep five complete, explicitly exploratory pairs."""
    monkeypatch.setattr("e2e_bsf.study.required_seeds", lambda *args: {"required": 6, "power_at_five": .775})
    schedule = freeze_schedule(Config(root=str(tmp_path)), pilot_records, 20 * 3600)
    assert schedule["seeds"] == list(range(100, 105))
    assert schedule["epochs"] == 1
    assert schedule["requested_seed_count"] == 6
    assert schedule["power"]["required"] == 6
    assert schedule["exploratory"] is True
    assert schedule["allocation_reason"] == "replication_reduced_to_fit_original_budget"
    assert schedule["predicted_optimization_seconds"] <= 20 * 3600


def test_budget_never_shrinks_below_five_pairs(tmp_path, pilot_records):
    """Local: fail before schedule creation; global: do not silently sacrifice minimum replication."""
    with pytest.raises(TimeoutError):
        freeze_schedule(Config(root=str(tmp_path)), pilot_records, 3600)
    assert not (tmp_path / "study_schedule.json").exists()
