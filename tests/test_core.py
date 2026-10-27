"""Scientific contracts written before implementation; see docs/methodology.md."""

import numpy as np  # Construct independent expected values.
import pytest  # Parameterize numerical edge cases.
import torch  # Exercise actual autograd rather than mocked mathematics.

from e2e_bsf.featurizer import GrassmannianBSF
from e2e_bsf.objectives import forward_kl, objective
from e2e_bsf.teacher import FrozenProbe, Normalization


def test_kl_orientation_and_reduction():
    """Local: asymmetric probabilities; global: prevent reversed distillation."""
    p = torch.tensor([[0.8, 0.2], [0.1, 0.9]])  # Teacher probabilities.
    q = torch.tensor([[0.3, 0.7], [0.4, 0.6]], requires_grad=True)  # Student.
    actual = forward_kl(q.log(), p.log())  # Input order follows PyTorch.
    expected = (p * (p.log() - q.log())).sum(-1)  # Definition, without API reuse.
    torch.testing.assert_close(actual, expected)  # Preserve one value per image.
    assert not torch.allclose(actual, (q * (q.log() - p.log())).sum(-1))
    actual.mean().backward()  # Ensure this is an optimization objective.
    assert torch.isfinite(q.grad).all()


def test_identical_extreme_logits():
    """Local: stable logs; global: identity insertion is a meaningful ceiling."""
    logits = torch.tensor([[1000.0, -1000.0, 0.0]])
    torch.testing.assert_close(forward_kl(logits, logits), torch.zeros(1))


@pytest.mark.parametrize("width,active", [(1, 6), (3, 2)])
def test_block_budget_and_orthogonality(width, active):
    """Local: QR and gating; global: enforce the intended sparse architecture."""
    torch.manual_seed(4)
    model = GrassmannianBSF(d=8, n_groups=6, group_size=width, l0=active)
    x = torch.randn(5, 8)
    reconstruction, codes = model(x)
    frames = model.frames()
    torch.testing.assert_close(frames.transpose(-1, -2) @ frames,
                               torch.eye(width).expand(6, width, width), atol=1e-5, rtol=1e-5)
    assert codes.shape == (5, 6, width)
    assert ((codes.norm(dim=-1) > 0).sum(-1) == active).all()
    reconstruction.square().mean().backward()
    assert torch.isfinite(model.B_raw.grad).all()
    assert model.B_raw.grad.norm() > 0


def test_normalization_roundtrip_and_training_only_fit():
    """Local: inverse transform; global: exclude validation distribution fitting."""
    rng = np.random.default_rng(2)
    training = rng.normal(size=(9, 4, 8)).astype("float32")
    normalization = Normalization.fit(training, range(9), batch_size=3)
    x = torch.from_numpy(training)
    torch.testing.assert_close(normalization.inverse(normalization(x)), x)
    assert abs(normalization(x).square().mean().item() - 1) < 1e-5
    mean_before = normalization.mean.clone()
    normalization(torch.randn_like(x) + 50)
    torch.testing.assert_close(mean_before, normalization.mean)


def test_teacher_frozen_and_intervention_connected():
    """Local: frozen probe; global: prevent a CLS shortcut or residual bypass."""
    torch.manual_seed(8)
    probe = FrozenProbe(torch.randn(3, 8), torch.randn(3))
    model = GrassmannianBSF(8, 12, 2, 3)
    patches = torch.randn(4, 4, 8)
    original = probe(patches).detach()
    reconstructed, _ = model(patches.flatten(0, 1))
    loss = forward_kl(probe(reconstructed.reshape_as(patches)), original).mean()
    loss.backward()
    assert loss.item() > 0
    assert model.B_raw.grad.norm() > 0
    assert all(p.grad is None and not p.requires_grad for p in probe.parameters())
    assert forward_kl(probe(torch.zeros_like(patches)), original).mean() > 0


def test_objective_has_no_hidden_mse_in_kl():
    """Local: objective dispatch; global: pure KL remains the primary method."""
    kl, mse = torch.tensor(2.0), torch.tensor(10.0)
    assert objective("kl", kl, mse).item() == 2
    assert objective("mse", kl, mse).item() == 10
    assert objective("kl_mse", kl, mse).item() == 3
    with pytest.raises(ValueError):
        objective("unknown", kl, mse)


def test_fixed_batch_kl_can_be_optimized():
    """Local: short optimization; global: exercise a real differentiable bottleneck."""
    torch.manual_seed(11)
    probe = FrozenProbe(torch.randn(3, 8) * 0.2, torch.zeros(3))
    model = GrassmannianBSF(8, 12, 2, 4)
    patches = torch.randn(8, 4, 8)
    target = probe(patches).detach()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for _ in range(35):
        optimizer.zero_grad()
        reconstruction, _ = model(patches.flatten(0, 1))
        loss = forward_kl(probe(reconstruction.reshape_as(patches)), target).mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert min(losses[-5:]) < losses[0] * 0.5
