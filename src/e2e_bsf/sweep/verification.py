"""Global: verify the actual causal insertion; local: real cached versus hooked online gradients.

Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook.
"""

from pathlib import Path
import time
import torch
from ..extraction import image_batch
from ..featurizer import GrassmannianBSF, reconstruct
from ..objectives import forward_kl
from ..io import read_json, write_json
from ..logging_utils import log_event
from .core import clean_states, region_mse
from .cache import check_deadline


def verify(spec, ctx, logger, deadline=None):
    """Use two training images, all 201 tokens, and the full dictionary; never inspect test scores."""
    destination = Path(spec.root) / "verification.json"
    if destination.exists():
        result = read_json(destination)
        if result["identity"] != ctx.identity:
            raise ValueError("Verification inputs changed")
        return result
    check_deadline(deadline)
    ids = ctx.indices("train")[:2]
    cached, targets, teacher = ctx.batch(ids)
    images = image_batch(ctx.rows, ids, Path(spec.source_root).parent / "data")
    pixels = ctx.processor(images=images, return_tensors="pt").to(ctx.device)
    online = clean_states(ctx.backbone, pixels["pixel_values"], spec.insertion_block)
    for a, b in zip(online, [cached, *targets], strict=True):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        original = ctx.probe(ctx.backbone(**pixels).last_hidden_state[:, 5:])
        identity, errors = ctx.suffix(cached, targets)
        torch.testing.assert_close(identity, original, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(teacher, original, atol=1e-4, rtol=1e-4)
        mean_logits, _ = ctx.suffix(ctx.normalization.mean.expand_as(cached), targets)
        mean_kl = float(forward_kl(mean_logits, teacher).mean())
    assert errors.abs().max() < 1e-6 and mean_kl > 1e-6
    torch.manual_seed(901)
    model = GrassmannianBSF(768, spec.groups, spec.block_size, 16).to(ctx.device)
    versions = {name: value._version for name, value in ctx.backbone.state_dict(keep_vars=True).items()}
    probe_before = {name: value.clone() for name, value in ctx.probe.state_dict().items()}
    gradients, losses = [], []
    for route in ("cached", "online_hook"):
        model.zero_grad(set_to_none=True)
        if route == "cached":
            modified, _, _ = reconstruct(model, ctx.normalization, cached)
            logits, downstream = ctx.suffix(modified, targets)
        else:
            captured, handles = [], []
            try:
                handles.append(ctx.backbone.model.layer[spec.insertion_block - 1].register_forward_hook(
                    lambda module, args, output: reconstruct(model, ctx.normalization, output)[0]))
                for layer in ctx.backbone.model.layer[spec.insertion_block:]:
                    handles.append(layer.register_forward_hook(lambda module, args, output: captured.append(output)))
                logits = ctx.probe(ctx.backbone(**pixels).last_hidden_state[:, 5:])
                downstream = torch.stack([region_mse(a, b) for a, b in zip(captured, targets, strict=True)], 1)
            finally:
                for handle in handles:
                    handle.remove()
        loss = forward_kl(logits, teacher).mean() + .001 * downstream[:, :, 0].mean()
        loss.backward()
        losses.append(float(loss.detach()))
        gradients.append({name: value.grad.detach().clone() for name, value in model.named_parameters()})
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=1e-4, rtol=1e-3)
        assert gradients[0][name].isfinite().all() and gradients[0][name].norm() > 0
    torch.testing.assert_close(torch.tensor(losses[0]), torch.tensor(losses[1]), atol=1e-4, rtol=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    curve = []
    for step in range(50):
        check_deadline(deadline)
        optimizer.zero_grad(set_to_none=True)
        modified, _, _ = reconstruct(model, ctx.normalization, cached)
        logits, _ = ctx.suffix(modified)
        loss = forward_kl(logits, teacher).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        curve.append(float(loss.detach()))
        if step % 10 == 0:
            log_event(logger, "sweep_verify_optimization", {"update": step, "kl": curve[-1]})
    assert min(curve[-10:]) < curve[0], "Fixed-batch optimization did not lower KL"
    assert all(not value.requires_grad and value.grad is None for value in ctx.backbone.parameters())
    assert versions == {name: value._version for name, value in ctx.backbone.state_dict(keep_vars=True).items()}
    for name, value in ctx.probe.state_dict().items():
        torch.testing.assert_close(value, probe_before[name], atol=0, rtol=0)
    result = {"passed": True, "identity": ctx.identity, "images": ids.tolist(), "cached_online_losses": losses,
              "maximum_gradient_difference": {name: float((gradients[0][name] - gradients[1][name]).abs().max()) for name in gradients[0]},
              "maximum_state_difference": max(float((a - b).abs().max()) for a, b in zip(online, [cached, *targets])),
              "identity_kl": float(forward_kl(identity, teacher).mean()), "mean_replacement_kl": mean_kl,
              "fixed_batch_kl": curve, "completed_unix": time.time()}
    write_json(destination, result)
    log_event(logger, "sweep_verified", result)
    return result
