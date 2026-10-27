"""Global: the registered phase order; local: short orchestration of independent modules.

The clock and configuration freeze precede main training; test results never change allocation.
"""

from pathlib import Path
import time
from ..io import read_json, write_json, load_checkpoint, source_revision, digest
from ..logging_utils import log_event, redact_text
from .cache import prepare, SweepRuntime, check_deadline
from .verification import verify
from .training import train_cell
from .evaluation import evaluate
from .schedule import clock_state, allocate


def pilot(spec, ctx, logger, deadline):
    """Complete the same two LR trials for all cells; retain every development endpoint."""
    records = []
    for cell in spec.cells():
        for lr in spec.learning_rates:
            path = train_cell(spec, ctx, cell, 1, lr, spec.pilot_seed, logger, pilot=True, deadline=deadline)
            state = load_checkpoint(path)
            if state["status"] != "complete":
                raise TimeoutError("Equal pilot grid exceeded the original preparation allocation")
            endpoint = read_json(path.parent / "development.json")
            records.append({"cell": cell.key, "lr": lr, "dev_kl": endpoint["kl"],
                            "train_seconds_per_image": state["train_seconds"] / state["images_seen"],
                            "dev_seconds_per_image": state["dev_seconds"] / endpoint["images"],
                            "checkpoint_seconds": state["checkpoint_seconds"], "microbatch": state["microbatch"]})
            write_json(Path(spec.root) / "pilot.json", records)
            log_event(logger, "sweep_pilot_cell", records[-1])
    return records


def freeze(spec, ctx, logger, status):
    """Freeze once, only after the complete equal pilot search and real-image verification."""
    path = Path(spec.root) / "study_schedule.json"
    if path.exists():
        schedule = read_json(path)
        if schedule["spec"] != spec.to_dict() or schedule["identity"] != ctx.identity:
            raise ValueError("Frozen study inputs changed")
        return schedule
    if not read_json(Path(spec.root) / "verification.json")["passed"]:
        raise RuntimeError("Real-image verification must pass first")
    schedule = allocate(spec, read_json(Path(spec.root) / "pilot.json"), len(ctx.indices("train")),
                        len(ctx.indices("dev")), status["optimization_deadline_unix"] - time.time())
    schedule.update(spec=spec.to_dict(), identity=ctx.identity, source_revision=source_revision(),
                    frozen_at_unix=time.time(), optimization_deadline_unix=status["optimization_deadline_unix"],
                    pilot_sha256=digest(Path(spec.root) / "pilot.json"))
    write_json(path, schedule)
    log_event(logger, "sweep_schedule_frozen", schedule)
    return schedule


def train_registered(spec, ctx, logger, status, only_cell=None):
    """Advance every cell by one epoch per round, preserving equal planned exposure on resume."""
    schedule = freeze(spec, ctx, logger, status)
    for epoch in range(1, schedule["epochs"] + 1):
        for cell in spec.cells():
            if only_cell and cell.key != only_cell:
                continue
            path = train_cell(spec, ctx, cell, schedule["epochs"], schedule["learning_rates"][cell.key],
                              spec.seed, logger, stop_epoch=epoch, deadline=schedule["optimization_deadline_unix"],
                              microbatch=schedule["microbatch"])
            state = load_checkpoint(path)
            if state["epoch"] < epoch:
                raise TimeoutError("Training stopped at the original deadline; no grid or exposure extension")
            status.update(status="training", last_completed_cell=cell.key, round=epoch, updated_unix=time.time())
            write_json(Path(spec.root) / "study_status.json", status)
    return schedule


def run(spec, logger, phase="run", only_cell=None):
    """Invoke readable phases; final publication remains explicit after visual and scientific review."""
    if phase == "report":
        from .reporting import report
        return report(spec, logger)
    if phase == "publish":
        from .publication import publish
        return publish(spec, logger)
    status = clock_state(spec)
    root = Path(spec.root)
    if status["status"] == "complete":
        from .reporting import report
        return report(spec, logger)
    try:
        if phase in ("run", "prepare"):
            prepare(spec, logger, status["preparation_deadline_unix"])
            if phase == "prepare":
                return
        ctx = SweepRuntime(spec)
        if phase in ("run", "verify") and not (root / "study_schedule.json").exists():
            verify(spec, ctx, logger, status["preparation_deadline_unix"])
            if phase == "verify":
                return
        if phase in ("run", "pilot") and not (root / "study_schedule.json").exists():
            pilot(spec, ctx, logger, status["preparation_deadline_unix"])
            if phase == "pilot":
                return
        if phase in ("run", "train"):
            train_registered(spec, ctx, logger, status, only_cell)
            if phase == "train":
                return
        if phase in ("run", "evaluate"):
            status.update(status="evaluating")
            status.setdefault("evaluation_started_unix", time.time())
            write_json(root / "study_status.json", status)
            evaluate(spec, ctx, logger, status["deadline_unix"])
            from .diagnostics import inspect_cells
            inspect_cells(spec, ctx, logger, status["deadline_unix"])
            check_deadline(status["deadline_unix"])
            status.update(status="complete", completed_unix=time.time(), evaluation_completed_unix=time.time())
            status.pop("error", None)
            write_json(root / "study_status.json", status)
        from .reporting import report
        return report(spec, logger)
    except Exception as error:
        status.update(status="incomplete", error=redact_text(str(error)), updated_unix=time.time())
        write_json(root / "study_status.json", status)
        log_event(logger, "sweep_incomplete", status)
        raise
