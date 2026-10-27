"""Global: independently audit the completed study; local: inspect saved artifacts without model runs.

Run from the checkout with ``uv run python scripts/audit_study.py``. Numerical recomputation uses
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html and
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html.
"""

from datetime import datetime, timezone  # Record the audit independently of training timestamps.
from pathlib import Path  # Stay within the selected study and public report folders.
import json  # Inspect all history records rather than a sampled display.
import math  # Preserve the final incomplete effective batch.
import numpy as np  # Independent double-precision recomputation of per-image statistics.
from scipy import special, stats  # Established numerical definitions of log-softmax and paired intervals.
import torch  # Compare frozen checkpoint buffers exactly, without invoking a model.

from e2e_bsf.config import MODEL_REVISION, BSF_REVISION
from e2e_bsf.io import read_json, load_checkpoint, digest, write_json


def audit(root=Path("outputs/study")):
    """Verify every scheduled run, independent predictions, split isolation and immutable provenance."""
    schedule = read_json(root / "study_schedule.json")
    status = read_json(root / "study_status.json")
    manifest = read_json(root / "manifest.json")
    teacher = load_checkpoint(root / "teacher.pt")
    evidence = read_json(Path("reports/study/results.json"))
    rows = manifest["images"]
    test_ids = np.asarray([i for i, row in enumerate(rows) if row["split"] == "test"])
    labels = np.asarray([rows[i]["label"] for i in test_ids])
    groups = {split: {row["group"] for row in rows if row["split"] == split} for split in ("train", "dev", "test")}
    assert all(not groups[a] & groups[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")))
    assert status["status"] == "complete" and status["completed_unix"] <= status["started_unix"] + status["max_hours"] * 3600
    assert digest(root / "manifest.json") == schedule["manifest_sha256"]
    expected = {f"{condition}_seed{seed}" for seed in schedule["seeds"] for condition in schedule["conditions"]}
    measured = {path.parent.name for path in (root / "runs").glob("*/test_summary.json")}
    assert measured == expected  # No missing run, additional seed, or pilot can enter the endpoint.
    records, teacher_logits, primary = [], None, {"block_mse": [], "block_kl": []}
    for seed in schedule["seeds"]:
        for condition in schedule["conditions"]:
            run = root / "runs" / f"{condition}_seed{seed}"
            state = load_checkpoint(run / "checkpoint.pt")
            summary = read_json(run / "test_summary.json")
            history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
            assert state["status"] == "complete" and state["epoch"] == schedule["epochs"]
            assert state["config"]["lr"] == schedule["learning_rates"][condition]
            assert state["source_revision"] == schedule["source_revision"]
            assert state["model_revision"] == MODEL_REVISION and state["bsf_revision"] == BSF_REVISION
            assert state["manifest_sha256"] == schedule["manifest_sha256"]
            assert state["teacher_sha256"] == digest(root / "teacher.pt")
            assert summary["checkpoint_sha256"] == digest(run / "checkpoint.pt")
            assert summary["parameters"] == sum(value.numel() for value in state["model"].values())
            for field in ("probe", "normalization"):
                assert all(torch.equal(value, teacher[field][key]) for key, value in state[field].items())
            updates = math.ceil(manifest["counts"]["train"] / state["config"]["effective_batch"]) * schedule["epochs"]
            assert [row["update"] for row in history] == list(range(1, updates + 1))
            assert history[-1]["images_seen"] == manifest["counts"]["train"] * schedule["epochs"]
            assert all(np.isfinite(list(row.values())).all() for row in history)
            with np.load(run / "test_predictions.npz") as predictions:
                np.testing.assert_array_equal(predictions["indices"], test_ids)
                np.testing.assert_array_equal(predictions["groups"], [rows[i]["group"] for i in test_ids])
                logits = predictions["teacher_logits"].astype(np.float64)
                if teacher_logits is None:
                    teacher_logits = logits.copy()
                np.testing.assert_array_equal(logits, teacher_logits)
                inserted = predictions["modified_logits"].astype(np.float64)
                log_p = logits - special.logsumexp(logits, axis=-1, keepdims=True)
                log_q = inserted - special.logsumexp(inserted, axis=-1, keepdims=True)
                kl = (np.exp(log_p) * (log_p - log_q)).sum(-1)
                np.testing.assert_allclose(predictions["kl"], kl, rtol=3e-5, atol=3e-6)
                np.testing.assert_allclose(summary["kl"], kl.mean(), rtol=3e-5, atol=3e-6)
                np.testing.assert_allclose(summary["accuracy"], (inserted.argmax(-1) == labels).mean(), atol=1e-7)
            if condition in primary:
                primary[condition].append(summary["kl"])
            records.append({"run": run.name, "updates": updates, "images_seen": history[-1]["images_seen"],
                            "training_seconds": state["elapsed_seconds"], "peak_vram_bytes": state["peak_vram_bytes"],
                            "checkpoint_sha256": summary["checkpoint_sha256"]})
    paired = stats.ttest_rel(primary["block_mse"], primary["block_kl"]).confidence_interval()
    np.testing.assert_allclose([paired.low, paired.high], evidence["primary"]["ci95"], rtol=1e-12, atol=1e-12)
    log = sorted((root / "logs").glob("*run.log"))[-1].read_text(encoding="utf-8")
    training_times = [line.split(" INFO")[0] for line in log.splitlines() if "event=training_complete " in line]
    evaluation_times = [line.split(" INFO")[0] for line in log.splitlines() if "event=evaluation " in line]
    assert len(training_times) == len(expected) and len(evaluation_times) == len(expected)
    assert max(training_times) < min(evaluation_times)  # The reserved endpoint was opened only after all training.
    result = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
              "runs": records, "completed_pairs": len(schedule["seeds"]), "test_images": len(test_ids),
              "split_counts": manifest["counts"], "training_source_revision": schedule["source_revision"],
              "total_training_seconds": sum(row["training_seconds"] for row in records),
              "maximum_peak_vram_bytes": max(row["peak_vram_bytes"] for row in records),
              "last_training_completed_utc": max(training_times), "first_test_evaluation_completed_utc": min(evaluation_times),
              "primary_ci95_independent_recomputation": [paired.low, paired.high],
              "checks": ["complete frozen schedule", "equal image exposure", "disjoint duplicate groups",
                         "unchanged teacher and normalization", "pinned source/model revisions", "checkpoint hashes",
                         "identical test images and teacher logits", "independently recomputed forward KL and accuracy",
                         "independently recomputed paired interval", "all training precedes final evaluation"]}
    write_json(Path("reports/study/final_audit.json"), result)
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))
    return result


if __name__ == "__main__":
    audit()  # Read saved artifacts only; no optimizer or model inference is invoked.
