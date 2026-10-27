"""Global: use the local budget faithfully; local: account for each architecture's measured cost."""

from e2e_bsf.config import Config, CONDITIONS
from e2e_bsf.io import write_json
from e2e_bsf.study import freeze_schedule


def test_allocation_uses_each_condition_cost_without_changing_exposure(tmp_path):
    """Different QR costs still permit one shared epoch across five independently paired seeds."""
    write_json(tmp_path / "manifest.json", {"counts": {"train": 7575, "dev": 1894, "test": 3875}})
    records = []
    for seed in range(3):
        for lr in (1e-4, 3e-4):
            for condition in CONDITIONS:
                kl = .65 + .02 * seed - (.06 + .001 * seed if condition == "block_kl" else 0)
                records.append({"condition": condition, "seed": seed, "lr": lr, "dev_kl": kl,
                                "seconds": 285 if condition.startswith("scalar") else 90, "microbatch": 8})
    schedule = freeze_schedule(Config(root=str(tmp_path)), records, 20 * 3600)
    assert len(schedule["seeds"]) == 5
    assert schedule["epochs"] == 1
    assert 17 * 3600 < schedule["predicted_optimization_seconds"] <= 20 * 3600
    assert schedule["forecast_seconds_per_epoch"]["scalar_mse"] > schedule["forecast_seconds_per_epoch"]["block_mse"]
