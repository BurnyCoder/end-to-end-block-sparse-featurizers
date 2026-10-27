"""Global: resumable equal exposure; local: tiny frozen-suffix optimizer integration.

Source: https://docs.pytorch.org/docs/stable/notes/randomness.html.
"""

from dataclasses import replace
import logging
import numpy as np
import torch
import pytest
from e2e_bsf.io import load_checkpoint
from e2e_bsf.teacher import Normalization
from e2e_bsf.sweep.spec import SweepSpec, Cell
from e2e_bsf.sweep.training import train_cell
from test_sweep import tiny_suffix


class TinyRuntime:
    """A read-only activation cache analogue with a frozen nonlinear downstream model."""

    def __init__(self):
        self.device = "cpu"
        self.suffix = tiny_suffix()
        clean = torch.randn(12, 7, 8)
        self.features = [clean.numpy()]
        self.targets = []
        with torch.no_grad():
            current = clean
            for layer in self.suffix.layers:
                current = layer(current)
                self.targets.append(current)
            self.teacher = self.suffix.probe(self.suffix.norm(current)[:, 2:])
        self.normalization = Normalization.fit(self.features[0], range(7))
        self.probe = self.suffix.probe
        self.rows = [{"label": i % 3, "group": i} for i in range(12)]
        self.identity = {"fixture": "unchanged"}
        self.variances = torch.ones(3, 3)

    def indices(self, split, limit=0):
        return np.arange(7) if split == "train" else np.arange(7, 10)

    def batch(self, indices):
        return torch.tensor(self.features[0][indices]), [value[indices] for value in self.targets], self.teacher[indices]


def test_sweep_resume_partial_batch_and_identity(tmp_path):
    """An interrupted two-epoch run equals uninterrupted training, including the last three images."""
    ctx = TinyRuntime()
    spec = replace(SweepSpec(), root=str(tmp_path / "full"), device="cpu", groups=6, block_size=2,
                   microbatch=2, effective_batch=4)
    cell = Cell(2, "downstream", .001)
    logger = logging.getLogger("sweep_test")
    frozen = {name: value.clone() for name, value in ctx.suffix.state_dict().items()}
    full = train_cell(spec, ctx, cell, 2, 1e-3, 200, logger)
    resumed_spec = replace(spec, root=str(tmp_path / "resumed"))
    interrupted = train_cell(resumed_spec, ctx, cell, 2, 1e-3, 200, logger, max_updates=1)
    assert load_checkpoint(interrupted)["updates"] == 1
    resumed = train_cell(resumed_spec, ctx, cell, 2, 1e-3, 200, logger)
    a, b = load_checkpoint(full), load_checkpoint(resumed)
    assert a["updates"] == b["updates"] == 4 and a["images_seen"] == b["images_seen"] == 14
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    for name, value in ctx.suffix.state_dict().items():
        torch.testing.assert_close(value, frozen[name], atol=0, rtol=0)
    with pytest.raises(ValueError):
        train_cell(resumed_spec, ctx, cell, 3, 1e-3, 200, logger)


def test_deadline_prevents_any_optimizer_update(tmp_path):
    """A restart cannot extend the clock, even when no update has yet completed."""
    spec = replace(SweepSpec(), root=str(tmp_path), device="cpu", groups=6, block_size=2)
    path = train_cell(spec, TinyRuntime(), Cell(2, "kl"), 1, 1e-3, 200,
                      logging.getLogger("deadline"), deadline=0)
    assert load_checkpoint(path)["updates"] == 0
