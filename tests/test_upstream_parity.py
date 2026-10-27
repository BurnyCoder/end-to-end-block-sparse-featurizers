"""Global: match the cited implementation. Local: verify QR reuse preserves outputs/gradients."""

import importlib.util
from pathlib import Path
import sys
import types
import torch

from e2e_bsf.featurizer import GrassmannianBSF


def test_released_goodfire_output_and_gradient_parity():
    """Load the unmodified attributed source in an isolated module namespace."""
    location = Path(__file__).parents[1] / "vendor" / "goodfire"
    package = types.ModuleType("reference_bsf")
    package.__path__ = [str(location)]
    sys.modules["reference_bsf"] = package
    for name in ("base", "grassmannian"):
        spec = importlib.util.spec_from_file_location(f"reference_bsf.{name}", location / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    torch.manual_seed(10)
    reference = module.GrassmannianBSF(d=8, n_groups=9, group_size=3, l0=3)
    adapted = GrassmannianBSF(d=8, n_groups=9, group_size=3, l0=3)
    adapted.load_state_dict(reference.state_dict())
    x = torch.randn(6, 8)
    expected, expected_codes = reference(x)
    actual, actual_codes = adapted(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_codes, expected_codes)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (_, left), (_, right) in zip(reference.named_parameters(), adapted.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-6)
