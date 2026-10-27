"""Global: frozen allocation without outcome-dependent extensions; local: durable clock and grid.

Sources: docs/downstream-preregistration.md; Python time and atomic JSON helpers.
"""

from pathlib import Path
import time
from ..io import read_json, write_json, source_revision


def clock_state(spec, now=None):
    """Start once at preparation; every subsequent command inherits the original deadlines."""
    path = Path(spec.root) / "study_status.json"
    if path.exists():
        state = read_json(path)
        if state["spec"] != spec.to_dict():
            raise ValueError("Sweep configuration changed after the clock started")
        return state
    now = time.time() if now is None else now
    state = {"status": "preparing", "spec": spec.to_dict(), "started_unix": now,
             "preparation_deadline_unix": now + spec.preparation_hours * 3600,
             "optimization_deadline_unix": now + (spec.max_hours - spec.evaluation_hours) * 3600,
             "deadline_unix": now + spec.max_hours * 3600, "source_revision": source_revision()}
    write_json(path, state)
    return state


def allocate(spec, pilots, train_images, dev_images, available_seconds):
    """Freeze the entire grid using equal LR trials and conservative measured per-epoch cost."""
    expected = {(cell.key, lr) for cell in spec.cells() for lr in spec.learning_rates}
    actual = [(row["cell"], row["lr"]) for row in pilots]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("The equal pilot grid is incomplete or duplicated")
    selected, costs = {}, {}
    for cell in spec.cells():
        candidates = [row for row in pilots if row["cell"] == cell.key]
        selected[cell.key] = min(candidates, key=lambda row: (row["dev_kl"], row["lr"]))["lr"]
        costs[cell.key] = 1.25 * max(row["train_seconds_per_image"] * train_images +
            row["dev_seconds_per_image"] * dev_images + row["checkpoint_seconds"] for row in candidates)
    epochs = min(spec.max_epochs, int(available_seconds // sum(costs.values())))
    if epochs < 1:
        raise TimeoutError("The full 25-cell grid cannot fit one common epoch within the original deadline")
    return {"seed": spec.seed, "epochs": epochs, "learning_rates": selected,
            "cells": [cell.key for cell in spec.cells()], "microbatch": min(row["microbatch"] for row in pilots),
            "forecast_seconds_per_epoch": costs, "predicted_training_seconds": epochs * sum(costs.values()),
            "available_training_seconds": available_seconds, "exploratory": True,
            "allocation": "one seed; complete crossed grid; uniform exposure; no power claim"}
