"""Global: independent 2026-11-22 claim recheck; local: recompute documented numbers from saved artifacts.

Run from the checkout with ``uv run python scripts/recheck_claims.py``. It only reads committed reports,
committed sweep logs and, when present, the ignored local ``outputs/`` predictions and logs. It never
loads DINOv3, never instantiates a featurizer and never trains or runs inference.
Sources: SciPy ttest_rel/wilcoxon (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html,
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html), statsmodels paired-t power
(https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.html), and the exact
sign-flip permutation test implemented in e2e_bsf.statistics.exact_sign_flip.
"""

from datetime import UTC, datetime  # Logs use UTC ISO timestamps; the study clock is Unix time.
from pathlib import Path  # Stay inside this checkout and its ignored output link.
import json  # Read JSON artifacts and JSONL-free logs as plain text.
import re  # Parse the timestamp prefix of sanitized log lines.
import numpy as np  # Double-precision recomputation of per-image arrays.
from scipy import stats  # Student paired interval and exact Wilcoxon signed-rank test.
from statsmodels.stats.power import TTestPower  # Same paired-t power family as the preregistration.

from e2e_bsf.io import read_json, write_json  # Reuse the repository's atomic JSON helpers.
from e2e_bsf.statistics import bootstrap_images, design_sensitivity  # Registered bootstrap; shared post-hoc sensitivity.

STUDY, SWEEP, OUTPUTS = Path("reports/study"), Path("reports/downstream-sweep"), Path("outputs/study")
STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z")  # Logger prefix written by logging_utils.


def utc(seconds):
    """Local: format Unix seconds; global: compare artifact clocks with log clocks in one zone."""
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_times(path, event):
    """Return every timestamp of one logged event, so first/last ordering is checked from evidence."""
    return [STAMP.match(line).group(1) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if f"event={event} " in line and STAMP.match(line)]


def primary(evidence, pilot_power):
    """Recheck the registered contrast and add clearly labelled post-hoc sensitivity diagnostics."""
    seeds = evidence["primary"]["per_seed"]
    local = np.array([row["local_kl"] for row in seeds])
    kl = np.array([row["proposed_kl"] for row in seeds])
    difference = local - kl  # Positive values favor output-KL training, as registered.
    test = stats.ttest_rel(local, kl)  # Same Student paired interval as statistics.paired_summary.
    interval = test.confidence_interval()
    sensitivity = design_sensitivity(local, kl)  # Shared with the paper generator; 10% registered minimum effect.
    return {"mean_difference": float(difference.mean()), "ci95": [float(interval.low), float(interval.high)],
            "t": float(test.statistic), "p_two_sided": float(test.pvalue), "seeds_favoring_kl": int((difference > 0).sum()),
            "relative_kl_reduction_percent": 100 * sensitivity["relative_effect"],
            "ci95_percent_of_local_kl": [100 * value for value in sensitivity["ci95_relative"]],
            "registered_minimum_practical_effect_percent": 100 * sensitivity["minimum_relative_effect"],
            "point_estimate_below_minimum_practical_effect": sensitivity["relative_effect"] < sensitivity["minimum_relative_effect"],
            "exact_sign_flip_p": sensitivity["exact_sign_flip_p"],
            "smallest_attainable_exact_p_with_five_pairs": sensitivity["smallest_exact_p"],
            "wilcoxon_exact_p": float(stats.wilcoxon(difference, method="exact").pvalue),
            "post_hoc_sensitivity": {
                "label": "post hoc; uses the main-study seed SD for the preregistered 10% effect, not the observed effect",
                "pilot_relative_difference_sd": pilot_power["relative_sd"],
                "main_relative_difference_sd": sensitivity["relative_difference_sd"],
                "standardized_10_percent_effect": sensitivity["standardized_minimum_effect"],
                "power_at_five_seeds": sensitivity["power_at_n"],
                "seeds_for_80_percent_power": sensitivity["n_for_80_percent"]}}


def pilot_power(pilots, schedule):
    """Recompute LR choices and the pilot power inputs from all 30 retained pilot records."""
    mean = lambda condition, lr: float(np.mean([r["dev_kl"] for r in pilots if r["condition"] == condition and r["lr"] == lr]))
    selected = {c: min((mean(c, lr), lr) for lr in (1e-4, 3e-4))[1] for c in schedule["conditions"]}
    by_seed = {c: {r["seed"]: r["dev_kl"] for r in pilots if r["condition"] == c and r["lr"] == selected[c]}
               for c in ("block_mse", "block_kl")}
    difference = np.array([by_seed["block_mse"][s] - by_seed["block_kl"][s] for s in sorted(by_seed["block_mse"])])
    local = float(np.mean(list(by_seed["block_mse"].values())))
    effect = 0.1 * local / difference.std(ddof=1)
    power = TTestPower()
    return {"selected_learning_rates": selected, "matches_frozen_schedule": selected == schedule["learning_rates"],
            "standardized_effect": float(effect), "relative_sd": float(difference.std(ddof=1) / local),
            "power_at_five": float(power.power(effect_size=effect, nobs=5, alpha=.05, alternative="two-sided")),
            "required_seeds": int(np.ceil(power.solve_power(effect_size=effect, alpha=.05, power=.8, alternative="two-sided"))),
            "pilot_mean_relative_kl_reduction_percent": 100 * float(difference.mean()) / local}


def local_predictions(evidence):
    """Recompute condition means and within-seed paired image-bootstrap intervals from complete test arrays."""
    if not (OUTPUTS / "runs").exists():
        return {"available": False, "reason": "ignored outputs/ predictions are not present in this checkout"}
    means, paired = {}, []
    for summary in evidence["summaries"]:
        arrays = [np.load(OUTPUTS / "runs" / f"{summary['condition']}_seed{seed}" / "test_predictions.npz") for seed in summary["seeds"]]
        means[summary["condition"]] = {key: float(np.mean([a[key].astype(np.float64).mean() for a in arrays]))
                                       for key in ("kl", "accuracy", "nmse")}
    for seed in evidence["primary"]["seeds"]:
        local = np.load(OUTPUTS / "runs" / f"block_mse_seed{seed}" / "test_predictions.npz")
        kl = np.load(OUTPUTS / "runs" / f"block_kl_seed{seed}" / "test_predictions.npz")
        assert np.array_equal(local["indices"], kl["indices"])  # The same reserved images enter both conditions.
        difference = local["kl"].astype(np.float64) - kl["kl"].astype(np.float64)
        paired.append({"seed": seed, "mean_difference": float(difference.mean()),
                       "paired_image_group_ci95": bootstrap_images(difference, local["groups"])})
    return {"available": True, "condition_means_from_arrays": means, "within_seed_paired_bootstrap": paired,
            "label": "post hoc; conditional on each trained pair, so it does not measure training-seed uncertainty"}


def interventions(evidence):
    """Quantify how approximate the random energy-matched removal control was, per seed-100 dictionary."""
    rows = {}
    for diagnostic in evidence["diagnostics"]:
        records = diagnostic["interventions"]
        ratios = [r["random_energy"] / r["selected_energy"] for r in records if r["selected_energy"] > 0]
        rows[diagnostic["condition"]] = {"targets": len(records), "inactive_selected": sum(r["selected_energy"] == 0 for r in records),
                                         "random_to_selected_energy_ratio": [min(ratios), max(ratios)] if ratios else None,
                                         "mean_kl_change": {name: float(np.mean([r["kl"][name] - r["kl"]["baseline"] for r in records]))
                                                            for name in ("remove_selected", "remove_random_matched",
                                                                         "swap_same_class", "swap_other_class")}}
    return rows


def study_timing(evidence):
    """Recheck documented clock times from the status file and, when present, the original run logs."""
    status, schedule = evidence["status"], evidence["schedule"]
    result = {"started": utc(status["started_unix"]), "schedule_frozen_minutes": (schedule["frozen_at_unix"] - status["started_unix"]) / 60,
              "evaluation_completed": utc(status["completed_unix"]), "deadline": utc(status["started_unix"] + 24 * 3600),
              "training_hours": evidence["study_details"]["total_training_seconds"] / 3600,
              "peak_allocated_mib": evidence["study_details"]["peak_vram_bytes"] / 2 ** 20,
              "five_seed_forecast_hours": 5 * sum(schedule["forecast_seconds_per_epoch"].values()) / 3600,
              "six_seed_forecast_hours": 6 * sum(schedule["forecast_seconds_per_epoch"].values()) / 3600}
    logs = sorted((OUTPUTS / "logs").glob("*-run.log")) if (OUTPUTS / "logs").exists() else []
    if logs:
        development = log_times(logs[-2], "development")  # The pilot log precedes the recovered main-run log.
        start = datetime.fromtimestamp(status["started_unix"], UTC).replace(tzinfo=None)
        result.update(last_pilot_minutes=(datetime.fromisoformat(development[-1]) - start).total_seconds() / 60,
                      last_training=max(log_times(logs[-1], "training_complete")),
                      first_test_evaluation=min(log_times(logs[-1], "evaluation")),
                      first_complete_report=log_times(logs[-1], "report_generated")[0])
    return result


def sweep():
    """Recheck the follow-up's pilot inventory, stage durations and an explicitly labelled timing counterfactual."""
    evidence = read_json(SWEEP / "results.json")
    trials = evidence["progress"]["pilot_trials"]
    complete = {(t["cell"], t["lr"]): t["development"] for t in trials if t["development"]}
    matched = [(k, lr) for (cell, lr) in complete if cell.endswith("_kl") for k in [cell[:-3]] if (k + "_mse", lr) in complete]
    run_log = sorted((SWEEP / "logs").glob("*-sweep-run.txt"))
    prep = {event: log_times(run_log[0], event) for event in ("sweep_command", "sweep_cache", "sweep_train_variance", "sweep_prepared")}
    clock = lambda text: datetime.fromisoformat(text)
    minutes = lambda a, b: (clock(b) - clock(a)).total_seconds() / 60
    pilots = [clock(t) for t in log_times(run_log[1], "sweep_development")]
    durations = np.diff([clock(log_times(run_log[1], "sweep_command")[0])] + pilots).astype("timedelta64[s]").astype(float)
    status = read_json(SWEEP / "study_status.json")
    remaining = (status["preparation_deadline_unix"] - clock(prep["sweep_prepared"][0]).replace(tzinfo=UTC).timestamp()) / 60
    verify = log_times(next((SWEEP / "logs").glob("*-sweep-verify.txt")), "sweep_verified")[0]
    return {"pilots": {status_name: sum(t["state"]["status"] == status_name for t in trials)
                       for status_name in ("complete", "trained_epoch", "not_started")},
            "kl_below_mse_matched": [sum(complete[(k + "_kl", lr)]["kl"] < complete[(k + "_mse", lr)]["kl"] for k, lr in matched), len(matched)],
            "completed_dev_accuracy_range": [min(d["accuracy"] for d in complete.values()), max(d["accuracy"] for d in complete.values())],
            "stage_minutes": {"extraction": minutes(prep["sweep_command"][0], prep["sweep_cache"][-1]),
                              "training_statistics": minutes(prep["sweep_cache"][-1], prep["sweep_train_variance"][-1]),
                              "hashing": minutes(prep["sweep_train_variance"][-1], prep["sweep_prepared"][0])},
            "interruption_minutes": minutes(prep["sweep_prepared"][0], log_times(run_log[1], "sweep_command")[0]),
            "verified_at": verify,
            "counterfactual": {"label": "post hoc estimate from logged wall-clock gaps; assumes k>=16 pilots cost like k<=8 pilots",
                               "minutes_left_after_preparation": remaining,
                               "first_pilot_seconds": float(durations[0]), "later_pilot_seconds": [float(durations[1:].min()), float(durations[1:].max())],
                               "estimated_minutes_for_50": [(durations[0] + 49 * durations[1:].min()) / 60,
                                                            (durations[0] + 49 * np.median(durations[1:])) / 60]}}


def main():
    """Write one machine-readable audit record; every number cited in the review comes from it."""
    evidence = read_json(STUDY / "results.json")
    pilots = read_json(STUDY / "pilot.json")
    power = pilot_power(pilots, evidence["schedule"])
    record = {"audited_at_utc": datetime.now(UTC).isoformat(), "scope": "read-only; no model inference or training",
              "pilot": power, "primary": primary(evidence, power), "scalar_secondary": {
                  "mean_difference": evidence["secondary_scalar"]["mean_improvement"], "ci95": evidence["secondary_scalar"]["ci95"],
                  "learning_rates": {c: evidence["schedule"]["learning_rates"][c] for c in ("scalar_mse", "scalar_kl")}},
              "local_predictions": local_predictions(evidence), "interventions": interventions(evidence),
              "study_timing": study_timing(evidence), "sweep": sweep()}
    destination = Path("reports/reviews/2026-11-22-claim-recheck.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()  # Read-only recomputation; no optimizer, backbone or featurizer is constructed.
