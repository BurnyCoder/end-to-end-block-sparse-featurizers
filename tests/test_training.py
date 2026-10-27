"""Global: trustworthy execution. Local: real optimizer, checkpoints, and partial batches."""

from pathlib import Path
import logging
import numpy as np
import torch

from e2e_bsf.config import Config
from e2e_bsf.io import save_checkpoint, write_json, load_checkpoint, digest
from e2e_bsf.teacher import FrozenProbe, Normalization
from e2e_bsf.training import train_one


def fixture(root):
    """Build a small fixed activation dataset; no model or network access is required."""
    root.mkdir()
    features = np.random.default_rng(8).normal(size=(12, 4, 768)).astype("float32")
    np.save(root / "patches.npy", features)
    rows = [{"id": str(i), "label": i % 3, "group": i,
             "split": "train" if i < 7 else "dev" if i < 10 else "test"} for i in range(12)]
    write_json(root / "manifest.json", {"images": rows})
    torch.manual_seed(9)
    probe = FrozenProbe(torch.randn(3, 768) * 0.01, torch.zeros(3))
    normalization = Normalization.fit(features, range(7))
    save_checkpoint(root / "teacher.pt", {"probe": probe.state_dict(), "normalization": normalization.state_dict()})
    return Config(root=str(root), device="cpu", groups=4, block_size=2, active=2,
                  microbatch=3, effective_batch=4, epochs=2, lr=1e-3)


def test_resume_matches_uninterrupted_and_handles_remainder(tmp_path):
    """A 7-image epoch with effective batch 4 must use all images and resume exactly."""
    complete = fixture(tmp_path / "complete")
    resumed = fixture(tmp_path / "resumed")
    logger = logging.getLogger("test")
    teacher_hash = digest(Path(resumed.root) / "teacher.pt")
    complete_path = train_one(complete, "block_kl", 100, logger)
    partial_path = train_one(resumed, "block_kl", 100, logger, max_updates=1)
    partial = load_checkpoint(partial_path)
    assert partial["offset"] == 4 and partial["status"] == "paused"
    final_path = train_one(resumed, "block_kl", 100, logger)
    left, right = load_checkpoint(complete_path), load_checkpoint(final_path)
    assert left["updates"] == right["updates"] == 4
    assert right["epoch"] == 2 and right["offset"] == 0 and right["status"] == "complete"
    for name in left["model"]:
        torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)
    assert digest(Path(resumed.root) / "teacher.pt") == teacher_hash
