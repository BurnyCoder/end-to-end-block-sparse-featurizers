"""Global: protect the new study's causal and statistical contracts; local: CPU tests.

References: https://arxiv.org/html/2405.12241v2#S2.SS4 and
https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html.
"""

from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn
from e2e_bsf.featurizer import GrassmannianBSF, reconstruct
from e2e_bsf.teacher import FrozenProbe, Normalization
from e2e_bsf.sweep.spec import SweepSpec, Cell
from e2e_bsf.sweep.core import FrozenSuffix, loss_for
from e2e_bsf.sweep.analysis import normalized_auc, paired_curves, threshold_budgets, pareto_indices
from e2e_bsf.sweep.schedule import allocate, clock_state


class MixingLayer(nn.Module):
    """Mix all tokens so preserving clean special tokens would change the test's answer."""

    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, value, position_embeddings=None):
        return value + self.linear(value.mean(1, keepdim=True)).tanh()


def tiny_suffix():
    """Use a frozen two-block suffix with a patch-only readout and two special tokens."""
    torch.manual_seed(5)
    return FrozenSuffix([MixingLayer(8), MixingLayer(8)], nn.LayerNorm(8),
                        FrozenProbe(torch.randn(3, 8), torch.zeros(3)), (), special=2)


def test_full_replacement_and_connected_suffix():
    """Identity is exact; reconstructed special tokens and BSF parameters affect outputs."""
    suffix = tiny_suffix()
    clean = torch.randn(3, 7, 8)
    with torch.no_grad():
        targets, current = [], clean
        for layer in suffix.layers:
            current = layer(current)
            targets.append(current)
    expected = suffix.probe(suffix.norm(current)[:, 2:])
    logits, errors = suffix(clean, targets)
    torch.testing.assert_close(logits, expected, atol=0, rtol=0)
    assert errors.shape == (3, 2, 3) and torch.count_nonzero(errors) == 0
    changed = clean.clone()
    changed[:, :2] = 0
    assert not torch.allclose(suffix(changed, targets)[0], logits)
    normalization = Normalization(torch.randn(7, 8), 2.5)
    torch.testing.assert_close(normalization.inverse(normalization(clean)), clean)
    model = GrassmannianBSF(8, 6, 2, 2)
    modified, codes, local = reconstruct(model, normalization, clean)
    assert modified.shape == clean.shape
    assert torch.all((codes.norm(dim=-1) > 0).sum(-1) == 2)
    prediction, downstream = suffix(modified, targets)
    from e2e_bsf.objectives import forward_kl
    loss_for(Cell(2, "downstream", .01), forward_kl(prediction, logits).mean(), local,
             downstream[:, :, 0].mean()).backward()
    assert model.B_raw.grad.isfinite().all() and model.B_raw.grad.norm() > 0
    assert model.gamma.grad.isfinite().all() and model.gamma.grad.norm() > 0
    assert all(not p.requires_grad and p.grad is None for p in suffix.parameters())


def test_downstream_loss_has_no_hidden_local_term():
    """Raw downstream MSE averages layers; beta zero is exactly pure output KL."""
    kl = torch.tensor(2., requires_grad=True)
    local = torch.tensor(99., requires_grad=True)
    downstream = torch.tensor(4., requires_grad=True)
    value = loss_for(Cell(16, "downstream", .1), kl, local, downstream)
    assert value.item() == pytest.approx(2.4)
    value.backward()
    assert local.grad is None and downstream.grad.item() == pytest.approx(.1)
    assert loss_for(Cell(16, "downstream", 0), kl, local, downstream) == kl
    assert loss_for(Cell(16, "mse"), kl, local, downstream) is local


def test_raw_errors_and_exclusive_writer(tmp_path):
    """Raw squared errors retain image/layer/region reductions; competing workers cannot mutate state."""
    suffix = tiny_suffix()
    clean = torch.randn(2, 7, 8)
    current, targets, expected = clean, [], []
    for layer in suffix.layers:
        current = layer(current)
        target = torch.zeros_like(current)
        targets.append(target)
        expected.append(torch.stack([current.square().mean((1, 2)), current[:, 2:].square().mean((1, 2)),
                                     current[:, :2].square().mean((1, 2))], -1))
    torch.testing.assert_close(suffix(clean, targets)[1], torch.stack(expected, 1))
    from e2e_bsf.sweep.cli import writer_lock
    with writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match="writer"):
            with writer_lock(tmp_path):
                raise AssertionError("A second writer acquired the lock")


def test_curve_integration_and_paired_group_bootstrap():
    """Integrate log-spaced endpoints independently; paired offsets retain exact uncertainty."""
    coordinates = [12, 24, 48, 96, 192]
    y = np.array([[1., 2., 3., 4., 5.], [2., 4., 6., 8., 10.]])
    np.testing.assert_allclose(normalized_auc(y, coordinates), [3., 6.])
    result = paired_curves(y, y - .25, coordinates, [0, 0], replicates=40)
    assert result["mean_improvement"] == pytest.approx(.25)
    np.testing.assert_allclose(result["ci95"], [.25, .25])
    np.testing.assert_allclose(result["per_k_improvement"], [.25] * 5)
    with pytest.raises(ValueError):
        normalized_auc(y, coordinates[::-1])
    assert threshold_budgets([.2, .06, .08, .02, .015], coordinates, [.01, .05, .1]) == {
        "0.01": None, "0.05": 96, "0.1": 24}


def test_grid_allocation_and_original_deadline(tmp_path):
    """The complete grid and original clock survive restarts; no automatic cell dropping."""
    spec = replace(SweepSpec(), root=str(tmp_path / "sweep"))
    assert len(spec.cells()) == 25 and len({c.key for c in spec.cells()}) == 25
    first = clock_state(spec, now=100.)
    second = clock_state(spec, now=500.)
    assert first["started_unix"] == second["started_unix"] == 100.
    assert second["deadline_unix"] == 100. + 86400.
    pilots = [{"cell": c.key, "lr": lr, "dev_kl": lr, "train_seconds_per_image": .1,
               "dev_seconds_per_image": .01, "checkpoint_seconds": 1., "microbatch": 8}
              for c in spec.cells() for lr in spec.learning_rates]
    schedule = allocate(spec, pilots, 7575, 1894, 72000.)
    assert schedule["epochs"] == 2 and schedule["seed"] == 200
    assert set(schedule["learning_rates"]) == {c.key for c in spec.cells()}
    assert set(schedule["learning_rates"].values()) == {1e-4}
    with pytest.raises(TimeoutError):
        allocate(spec, pilots, 7575, 1894, 10.)
    with pytest.raises(ValueError):
        allocate(spec, pilots[:-1], 7575, 1894, 72000.)


def test_observed_pareto_and_shared_bootstrap():
    """Strict dominance retains ties; paired group draws agree with an independent calculation."""
    from e2e_bsf.sweep.analysis import bootstrap_columns
    assert pareto_indices([[12, .2], [24, .1], [24, .15], [48, .12], [24, .1]]) == [0, 1, 4]
    values = np.array([[1., 8.], [3., 6.], [7., 2.]])
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(200):
        draw = rng.integers(0, 2, 2)
        rows = np.concatenate([np.array([0, 1]) if group == 0 else np.array([2]) for group in draw])
        samples.append(values[rows].mean(0))
    expected = np.percentile(samples, [2.5, 97.5], axis=0).T
    np.testing.assert_allclose(bootstrap_columns(values, [0, 0, 1], 200), expected)
