"""Global: reports must work before spending GPU time; local: isolated synthetic layout fixture.

These artificial arrays are test inputs only and never enter published study results.
Source: https://docs.pytest.org/en/stable/how-to/monkeypatch.html.
"""

from dataclasses import replace
import json
import logging
from pathlib import Path
import numpy as np
from e2e_bsf.io import write_json
from e2e_bsf.sweep.spec import SweepSpec
from e2e_bsf.sweep.reporting import report


def test_complete_report_uses_paired_predictions(tmp_path, monkeypatch):
    """Render all 25 cells and a PDF without model access, then independently check the primary effect."""
    monkeypatch.chdir(tmp_path)
    spec = replace(SweepSpec(), root="fixture", report_dir="report", bootstrap_replicates=20)
    root = Path(spec.root)
    write_json(root / "study_status.json", {"status": "synthetic_test_fixture"})
    write_json(root / "manifest.json", {"counts": {"train": 4, "dev": 4, "test": 4}})
    schedule = {"epochs": 2, "microbatch": 8, "learning_rates": {c.key: .0001 for c in spec.cells()}}
    write_json(root / "study_schedule.json", schedule)
    for cell in spec.cells():
        run = root / "runs" / cell.key
        kl = np.array([.1, .2, .3, .4]) / cell.k + (.01 if cell.objective == "mse" else 0)
        row = {"cell": cell.key, "label": cell.label, "k": cell.k, "kl": float(kl.mean()),
               "ce_delta": .01, "accuracy": .75, "agreement": .75, "active_coordinates": 3 * cell.k,
               "insertion_nmse": .5, "downstream_mse": 1., "alive_blocks": 4096,
               "layer_mse": np.ones((7, 3)).tolist(), "layer_nmse": np.full((7, 3), .5).tolist()}
        write_json(run / "test_summary.json", row)
        np.savez_compressed(run / "test_predictions.npz", kl=kl, indices=np.arange(4), groups=np.arange(4),
                            teacher_logits=np.zeros((4, 10)), ce_delta=np.full(4, .01),
                            accuracy=np.array([1., 1., 1., 0.]), agreement=np.array([1., 1., 1., 0.]),
                            layer_mse=np.ones((4, 7, 3)), layer_nmse=np.full((4, 7, 3), .5))
        (run / "history.jsonl").write_text(json.dumps({"images_seen": 32, "kl": row["kl"]}) + "\n")
        if cell.k == 16:
            example = {"id": "synthetic", "label": 0, "block_ids": [0], "heatmaps": [np.eye(14).tolist()]}
            write_json(run / "diagnostics.json", {"cell": cell.key, "label": cell.label,
                "examples": [example] * 3, "interventions": [], "selected_blocks": [0]})
    evidence = report(spec, logging.getLogger("report_fixture"))
    np.testing.assert_allclose(evidence["primary"]["mean_improvement"], .01)
    np.testing.assert_allclose(evidence["primary"]["ci95"], [.01, .01])
    assert len(evidence["runs"]) == 25 and len(evidence["secondary"]) == 3
    assert Path("report/paper.pdf").read_bytes().startswith(b"%PDF")
    assert Path("site/sweep/data.json").exists() and Path("report/figures/maps.png").exists()


def test_incomplete_pilot_is_reported_without_main_results(tmp_path, monkeypatch):
    """A preparation timeout produces a failure report, without pretending missing scores are zero."""
    monkeypatch.chdir(tmp_path)
    spec = replace(SweepSpec(), root="fixture", report_dir="report", bootstrap_replicates=20)
    root = Path(spec.root)
    write_json(root / "study_status.json", {"status": "incomplete", "error": "The original sweep stage deadline has elapsed"})
    write_json(root / "manifest.json", {"counts": {"train": 4, "dev": 4, "test": 4}})
    run = root / "runs" / "pilot_k004_mse_lr0.0001"
    write_json(run / "run.json", {"status": "trained_epoch", "epoch": 1, "images_seen": 510, "updates": 16})
    (run / "history.jsonl").write_text(json.dumps({"images_seen": 510, "kl": 2.}) + "\n")
    evidence = report(spec, logging.getLogger("failed_report_fixture"))
    assert evidence["progress"]["pilot_completed"] == 0
    assert evidence["progress"]["pilot_started"] == 1
    assert len(evidence["progress"]["pilot_trials"]) == 50
    assert evidence["primary"] is None and not evidence["runs"]
    assert "Allocation failure" in evidence["narrative"]
    assert "No main runs" in evidence["narrative"]
    assert Path("report/pilots/pilot_k004_mse_lr0.0001.md").exists()
    assert "figures/curves.png" not in Path("report/summary.md").read_text()


def test_publish_without_evaluated_main_models_has_no_remote_side_effect(tmp_path, monkeypatch):
    """An allocation failure cannot create an empty public collection or authenticate unnecessarily."""
    import pytest
    from e2e_bsf.sweep import publication

    def unexpected_access():
        raise AssertionError("Remote access must follow the local eligibility check")

    monkeypatch.setattr(publication, "check_access", unexpected_access)
    with pytest.raises(RuntimeError, match="No completed evaluated"):
        publication.publish(replace(SweepSpec(), root=str(tmp_path)), logging.getLogger("empty_publication"))
