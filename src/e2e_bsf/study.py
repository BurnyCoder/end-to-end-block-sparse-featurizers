"""Global: prespecified bounded study. Local: pilot, freeze schedule, train, then test.

Power analysis follows docs/preregistration.md and statsmodels' paired-difference convention.
The durable schedule is chosen without consulting any test metric.
"""

from dataclasses import replace
from pathlib import Path
import time
import numpy as np

from .config import CONDITIONS
from .io import read_json, write_json, load_checkpoint, digest, source_revision
from .logging_utils import log_event, redact_text
from .statistics import required_seeds
from .training import train_one


def run_pilot(config, logger, deadline):
    """Local: equal short runs; global: estimate feasibility without touching test outcomes."""
    root = Path(config.root)
    records = []
    for seed in (0, 1, 2):
        for learning_rate in (1e-4, 3e-4):
            for condition in CONDITIONS:
                recipe = replace(config, epochs=1, lr=learning_rate, train_limit=510, dev_limit=250)
                name = f"pilot_{condition}_seed{seed}_lr{learning_rate:g}"
                path = train_one(recipe, condition, seed, logger, run_name=name, deadline=deadline)
                state = load_checkpoint(path)
                if state["status"] != "complete":
                    raise TimeoutError("Equal pilot grid could not complete within preparation budget")
                development = read_json(path.parent / "development.json")
                records.append({"condition": condition, "seed": seed, "lr": learning_rate,
                                "dev_kl": development["kl"], "seconds": state["elapsed_seconds"],
                                "microbatch": state["microbatch"]})
                write_json(root / "pilot.json", records)
    return records


def freeze_schedule(config, records, optimization_seconds):
    """Local: deterministic allocation rule; global: no adaptive significance chasing."""
    root = Path(config.root)
    selected = {}
    for condition in CONDITIONS:
        candidates = [(float(np.mean([row["dev_kl"] for row in records if row["condition"] == condition and row["lr"] == lr])), lr)
                      for lr in (1e-4, 3e-4)]
        selected[condition] = min(candidates)[1]  # Equal trials and an explicit lower-LR tie break.
    primary = {}
    for condition in ("block_mse", "block_kl"):
        primary[condition] = {row["seed"]: row["dev_kl"] for row in records
                              if row["condition"] == condition and row["lr"] == selected[condition]}
    power = required_seeds(primary["block_mse"], primary["block_kl"])
    required = power.get("required")
    requested_seeds = min(10, max(5, required)) if required is not None else 10
    counts = read_json(root / "manifest.json")["counts"]
    seconds_per_epoch = {condition: 1.25 * max(row["seconds"] for row in records if row["condition"] == condition)
                        / 760 * (counts["train"] + counts["dev"]) for condition in CONDITIONS}
    seed_capacity = int(optimization_seconds // sum(seconds_per_epoch.values()))  # Complete all conditions at least once per seed.
    seeds = min(requested_seeds, seed_capacity)  # The preregistered exploratory fallback retains the five-seed minimum.
    if seeds < 5:
        raise TimeoutError("One common epoch across all conditions and at least five seeds cannot fit")
    seconds_per_paired_epoch = seeds * sum(seconds_per_epoch.values())  # Sum measured condition costs; exposure stays equal.
    epochs = min(20, int(optimization_seconds // seconds_per_paired_epoch))
    schedule = {"seeds": list(range(100, 100 + seeds)), "epochs": epochs, "learning_rates": selected,
                "microbatch": min(row["microbatch"] for row in records), "conditions": list(CONDITIONS),
                "power": power, "exploratory": required is None or required > seeds,
                "requested_seed_count": requested_seeds,
                "allocation_reason": "replication_reduced_to_fit_original_budget" if seeds < requested_seeds else "requested_replication_within_budget",
                "available_optimization_seconds": optimization_seconds,
                "predicted_optimization_seconds": epochs * seconds_per_paired_epoch,
                "forecast_seconds_per_epoch": seconds_per_epoch,
                "manifest_sha256": digest(root / "manifest.json"), "source_revision": source_revision(),
                "frozen_at_unix": time.time()}
    write_json(root / "study_schedule.json", schedule)
    return schedule


def run_study(config, logger):
    """Local: durable deadlines; global: finish the paired schedule before final evaluation."""
    from .pipeline import prepare, evaluate
    from .reporting import build_report

    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "study_status.json"
    if status_path.exists():
        status = read_json(status_path)
        if status["status"] == "complete":
            return build_report(config, logger)
        started = status["started_unix"]  # Restart does not silently reset the wall-time budget.
    else:
        started = time.time()
        status = {"status": "preparing", "started_unix": started, "max_hours": config.max_hours}
    deadline = started + config.max_hours * 3600
    preparation_deadline = min(started + 2 * 3600, deadline)
    optimization_deadline = min(started + 22 * 3600, deadline - min(2 * 3600, config.max_hours * 3600 / 12))
    write_json(status_path, status)
    try:
        schedule_path = root / "study_schedule.json"
        if not schedule_path.exists():
            prepare(config, logger, preparation_deadline)
            records = run_pilot(config, logger, preparation_deadline)
            optimization_deadline = min(optimization_deadline, time.time() + 20 * 3600)
            schedule = freeze_schedule(config, records, optimization_deadline - time.time())
            schedule["optimization_deadline_unix"] = optimization_deadline
            write_json(schedule_path, schedule)
        else:
            schedule = read_json(schedule_path)
            optimization_deadline = schedule["optimization_deadline_unix"]
        if "error" in status:
            status.setdefault("previous_failures", []).append({"error": status.pop("error"), "at_unix": status.get("updated_unix")})
        status.update(status="training", schedule=schedule, updated_unix=time.time())
        write_json(status_path, status)
        log_event(logger, "frozen_study_schedule", schedule)
        for seed in schedule["seeds"]:
            for condition in schedule["conditions"]:
                recipe = replace(config, epochs=schedule["epochs"], lr=schedule["learning_rates"][condition],
                                 microbatch=schedule["microbatch"])
                path = train_one(recipe, condition, seed, logger, deadline=optimization_deadline)
                if load_checkpoint(path)["status"] != "complete":
                    raise TimeoutError("Optimization deadline reached; incomplete paired study is exploratory")
                status.update(last_completed_run=path.parent.name, updated_unix=time.time())
                write_json(status_path, status)
        status.update(status="evaluating")
        write_json(status_path, status)
        evaluate(replace(config, microbatch=schedule["microbatch"]), logger,
                 deadline=min(deadline, time.time() + 2 * 3600))
        status.update(status="complete", completed_unix=time.time())
        write_json(status_path, status)
        build_report(config, logger)
        return status
    except Exception as error:
        status.update(status="incomplete", error=redact_text(str(error)), updated_unix=time.time())
        write_json(status_path, status)
        log_event(logger, "study_incomplete", status)
        build_report(config, logger)  # Missing measurements remain missing, never synthetic.
        raise
