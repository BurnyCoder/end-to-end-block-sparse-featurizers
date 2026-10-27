"""Global: prove the intervention is connected. Local: real image/cache/gradient parity.

The cache is an optimization only because the intervention follows the frozen backbone.
See https://docs.pytorch.org/docs/stable/notes/autograd.html#setting-requires-grad.
"""

from pathlib import Path
import numpy as np
import torch

from .extraction import load_backbone, extract_patches, image_batch
from .evaluation import load_teacher
from .featurizer import GrassmannianBSF, reconstruct
from .io import read_json, write_json
from .logging_utils import log_event
from .objectives import forward_kl


def verify_real_pipeline(config, logger):
    """Compare cached and online final patches, logits, KL and BSF gradients on training images."""
    root = Path(config.root)
    rows = read_json(root / "manifest.json")["images"]
    indices = [i for i, row in enumerate(rows) if row["split"] == "train"][:2]
    cache = np.load(root / "patches.npy", mmap_mode="r")
    cached = torch.from_numpy(np.array(cache[indices], copy=True)).to(config.device)
    processor, backbone = load_backbone(config)
    online = extract_patches(processor, backbone, image_batch(rows, indices, root.parent / "data"), config.device)
    torch.testing.assert_close(online, cached, atol=5e-5, rtol=5e-5)
    probe, normalization = load_teacher(root, config.device)
    torch.manual_seed(777)  # Separate verification initialization, never a reported study seed.
    model = GrassmannianBSF(768, 32, 3, 4).to(config.device)
    results = []
    gradients = []
    for patches in (online, cached):
        model.zero_grad(set_to_none=True)
        teacher = probe(patches).detach()
        modified, _, _ = reconstruct(model, normalization, patches)
        loss = forward_kl(probe(modified), teacher).mean()
        loss.backward()
        results.append(float(loss.detach()))
        gradients.append(model.B_raw.grad.detach().clone())
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-4, rtol=1e-3)
    identity = float(forward_kl(probe(cached), probe(cached)).mean())
    removed = float(forward_kl(probe(torch.zeros_like(cached)), probe(cached)).mean())
    assert abs(identity) < 1e-6 and removed > 1e-6 and gradients[0].norm() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in backbone.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    overfit = []
    target = probe(cached).detach()
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        modified, _, _ = reconstruct(model, normalization, cached)
        loss = forward_kl(probe(modified), target).mean()
        loss.backward()
        optimizer.step()
        overfit.append(float(loss.detach()))
    if min(overfit[-10:]) >= overfit[0]:
        raise RuntimeError("Real fixed-batch optimization failed to lower KL")
    result = {"images": indices, "online_kl": results[0], "cached_kl": results[1],
              "maximum_feature_difference": float((online - cached).abs().max()),
              "maximum_gradient_difference": float((gradients[0] - gradients[1]).abs().max()),
              "identity_kl": identity, "zero_patch_kl": removed,
              "overfit_curve": overfit, "passed": True}
    write_json(root / "verification.json", result)
    log_event(logger, "real_pipeline_verification", result)
    del backbone, model
    torch.cuda.empty_cache()
    return result
