"""Global: unchanged inference mathematics; local: reuse frozen QR frames only within one evaluation.

The inference-only scope follows https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html.
"""

import numpy as np  # Preserve full images and the final incomplete batch.
import pytest  # Check both registered block widths.
import torch  # Compare actual reconstructions and probe outputs.

from e2e_bsf.evaluation import measure
from e2e_bsf.featurizer import GrassmannianBSF, reconstruct
from e2e_bsf.teacher import FrozenProbe, Normalization


@pytest.mark.parametrize("width", [1, 3])
def test_frozen_frames_match_default_forward(width):
    """Local: explicit frames equal ordinary QR; global: the evaluation optimization changes no outputs."""
    torch.manual_seed(9)
    model = GrassmannianBSF(8, 6, width, 2)
    x = torch.randn(11, 8)
    with torch.no_grad():
        expected = model(x)
        actual = model(x, frames=model.frames())
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.parametrize("width", [1, 3])
def test_measure_reuses_frames_without_stale_state(width, monkeypatch):
    """Local: one QR per call including partial batches; global: never cache across parameter updates."""
    torch.manual_seed(10)
    model = GrassmannianBSF(8, 6, width, 2)
    normalization = Normalization(torch.randn(4, 8), 1.3)
    probe = FrozenProbe(torch.randn(3, 8), torch.zeros(3))
    features = torch.randn(7, 4, 8).numpy()
    rows = [{"label": index % 3} for index in range(7)]
    with torch.no_grad():
        expected = probe(reconstruct(model, normalization, torch.from_numpy(features))[0]).numpy()
    calls = []
    original_frames = model.frames

    def counted_frames():
        """Count expensive factorizations, retaining the real QR implementation."""
        calls.append(True)
        return original_frames()

    monkeypatch.setattr(model, "frames", counted_frames)
    summary, arrays = measure(model, normalization, probe, features, rows, np.arange(7), "cpu", batch_size=3)
    assert len(calls) == 1
    assert summary["images"] == 7
    np.testing.assert_allclose(arrays["modified_logits"], expected, rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        model.B_raw.add_(torch.randn_like(model.B_raw) * .1)
    _, changed = measure(model, normalization, probe, features, rows, np.arange(7), "cpu", batch_size=3)
    assert len(calls) == 2
    assert not np.allclose(changed["modified_logits"], expected)
    assert all(parameter.grad is None for parameter in model.parameters())
