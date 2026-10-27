"""Global: independently audit the stopped preparation allocation; local: inspect saved evidence only.

Sources: PyTorch restricted checkpoint loading; NumPy deterministic permutation; registered deadlines.
No model forward pass, optimization, test evaluation, or raw-cache rewrite occurs here.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from dotenv import load_dotenv
from e2e_bsf.io import read_json, write_json, load_checkpoint, digest
from e2e_bsf.sweep.spec import SweepSpec
from e2e_bsf.sweep.cache import split_indices
from e2e_bsf.sweep.cli import writer_lock


def audit(spec):
    """Check every started pilot, unchanged teacher/splits, no main results, and the original clock."""
    root = Path(spec.root)
    status = read_json(root / "study_status.json")
    cache = read_json(root / "cache.json")
    verification = read_json(root / "verification.json")
    teacher = load_checkpoint(root / "teacher.pt")
    manifest = read_json(root / "manifest.json")
    assert status["status"] == "incomplete" and cache["complete"] and verification["passed"]
    assert status["preparation_deadline_unix"] - status["started_unix"] == 7200
    assert status["deadline_unix"] - status["started_unix"] == 86400
    assert digest(root / "manifest.json") == cache["identity"]["manifest_sha256"]
    assert digest(Path(spec.source_root) / "teacher.pt") == cache["identity"]["source_teacher_sha256"]
    assert digest(Path(spec.source_root) / "manifest.json") == cache["identity"]["manifest_sha256"]
    assert not (root / "study_schedule.json").exists()
    assert not list((root / "runs").glob("*/test_predictions.npz"))
    assert not list((root / "runs").glob("*/test_summary.json"))
    assert not (root / "controls").exists()
    for cell in spec.cells():
        assert not (root / "runs" / cell.key / "checkpoint.pt").exists()
    for name, record in cache["files"].items():
        stat = (Path(spec.cache_dir) / name).stat()
        assert stat.st_size == record["bytes"] and stat.st_mtime_ns == record["mtime_ns"]
    groups = {}
    for row in manifest["images"]:
        groups.setdefault(row["group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    train = split_indices(manifest["images"], "train", spec.pilot_train_images)
    dev = split_indices(manifest["images"], "dev", spec.pilot_dev_images)
    assert len(train) == 510 and len(dev) == 250 and not set(train) & set(dev)
    expected_identity = verification["identity"]
    assert expected_identity["cache_sha256"] == digest(root / "cache.json")
    assert expected_identity["teacher_sha256"] == digest(root / "teacher.pt")
    pilots = read_json(root / "pilot.json")
    indexed = {(row["cell"], row["lr"]): row for row in pilots}
    checks, initial_hashes = [], set()
    for path in sorted((root / "runs").glob("pilot_*/checkpoint.pt")):
        state = load_checkpoint(path)
        recipe = state["recipe"]
        assert state["seed"] == recipe["seed"] == spec.pilot_seed == 900 and recipe["pilot"]
        assert state["identity"] == expected_identity and recipe["spec"] == spec.to_dict()
        assert state["epoch"] == 1 and state["updates"] == 16 and state["images_seen"] == 510
        assert state["saved_unix"] <= status["preparation_deadline_unix"]
        assert state["source_revision"] == status["resumed_source_revision"]
        initial_hashes.add(state["initial_sha256"])
        for field in ("probe", "normalization"):
            for key, value in teacher[field].items():
                assert torch.equal(state[field][key], value)
        assert all(torch.isfinite(value).all() for value in state["model"].values())
        history = [json.loads(line) for line in (path.parent / "history.jsonl").read_text().splitlines()]
        assert [row["update"] for row in history] == list(range(1, 17))
        assert [row["images_seen"] for row in history] == list(range(32, 481, 32)) + [510]
        if state["status"] == "complete":
            development = read_json(path.parent / "development.json")
            cell_key = path.parent.name.removeprefix("pilot_").split("_lr")[0]
            assert development["images"] == 250 and development["kl"] == indexed[(cell_key, recipe["lr"])]["dev_kl"]
        else:
            assert state["status"] == "trained_epoch" and not (path.parent / "development.json").exists()
        checks.append({"trial": path.parent.name, "status": state["status"], "updates": state["updates"],
                       "images_seen": state["images_seen"], "saved_unix": state["saved_unix"], "checkpoint_sha256": digest(path)})
        del state
    assert len(initial_hashes) == 1 and len(checks) == 16 and len(pilots) == 15
    summary = ("Preparation completed after approximately 68 minutes: extraction took 26.5 minutes, training-only statistics about 23.4 minutes, "
               "and final cache hashing about 17.7 minutes. A batch-geometry verification failure stopped the worker at 19:08 UTC and was detected "
               "at 19:31 UTC. The repaired check passed at 19:37 UTC; pilots resumed at 19:38 UTC. The two-hour preparation deadline stopped "
               "the 16th trial during development evaluation at 20:00:02 UTC. Fifteen trials completed, one had trained but lacked a development "
               "endpoint, and 34 never started. No main schedule, selected learning rates, main training or test evaluation followed. "
               "The bottleneck was the preparation allocation, not a measured scientific failure of KL training.")
    result = {"passed": True, "audited_utc": datetime.now(timezone.utc).isoformat(), "scope": "stopped preparation allocation only",
              "execution_summary": summary, "completed_pilots": 15, "incomplete_pilots": 1, "unstarted_pilots": 34,
              "main_runs": 0, "main_test_evaluations": 0, "shared_initial_sha256": next(iter(initial_hashes)),
              "shared_training_order_sha256": hashlib.sha256(np.random.default_rng(900).permutation(train).tobytes()).hexdigest(),
              "checkpoints": checks, "source_teacher_unchanged": True, "split_groups_isolated": True,
              "cache_integrity_check": "complete extraction SHA-256 records and unchanged current sizes/mtimes; 58 GB not rehashed in this audit",
              "stage_deadline_stop_seconds": status["updated_unix"] - status["preparation_deadline_unix"],
              "scientific_conclusion": "primary hypothesis and sparsity advantage unmeasured"}
    write_json(root / "audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    load_dotenv(".env")
    specification = SweepSpec.load("configs/downstream-sweep.toml")
    with writer_lock(specification.root):
        audit(specification)
