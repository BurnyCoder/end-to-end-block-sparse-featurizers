"""Global: readable execution phases. Local: compose independently testable components."""

from pathlib import Path  # Every phase shares the same explicit experiment directory.

from .config import CONDITIONS
from .data import prepare_manifest
from .extraction import extract_cache
from .teacher import fit_teacher
from .training import train_one
from .evaluation import evaluate_checkpoint, evaluate_controls
from .io import read_json, write_json


def prepare(config, logger, deadline=None):
    """Local: data → activations → frozen teacher; global: prepare the immutable experiment."""
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists() and read_json(root / "config.json") != config.to_dict():
        raise RuntimeError("Preparation configuration changed; choose another root")
    write_json(root / "config.json", config.to_dict())
    rows = prepare_manifest(config, logger)
    features = extract_cache(config, rows, logger, deadline)
    if not (root / "teacher.pt").exists():
        fit_teacher(features, rows, root, logger)
    return rows


def evaluate(config, logger, deadline=None):
    """Local: completed runs only; global: never promote partial training to final evidence."""
    root = Path(config.root)
    results = []
    for checkpoint in sorted((root / "runs").glob("*/checkpoint.pt")):
        if checkpoint.parent.name.startswith("pilot_"):
            continue  # Pilot seeds never enter the final test comparison.
        from .io import load_checkpoint  # Read status without loading a live model.
        if load_checkpoint(checkpoint)["status"] == "complete":
            results.append(evaluate_checkpoint(config, checkpoint, logger, deadline=deadline))
    evaluate_controls(config, logger, deadline=deadline)
    return results


def run(config, logger):
    """Local: one end-to-end command; global: smoke evidence is separate from confirmation."""
    if config.profile == "full":
        from .study import run_study  # Budgeted scheduling remains an independent subsystem.
        return run_study(config, logger)
    prepare(config, logger)
    from .verification import verify_real_pipeline
    verify_real_pipeline(config, logger)  # Exercise image-to-output and gradient parity before training.
    for condition in CONDITIONS:
        train_one(config, condition, 100, logger)
    evaluate(config, logger)
    from .reporting import build_report
    return build_report(config, logger)
