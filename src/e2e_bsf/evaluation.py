"""Global: image-level evidence. Local: streamed predictions and block diagnostics.

KL follows Apollo's implementation; block-map diagnostics follow BSF Appendix G:
https://arxiv.org/html/2606.25234v1#A7. Statistics never treat patches as independent images.
"""

from pathlib import Path  # Keep each evaluation next to its immutable checkpoint.
import time  # Measure evaluation costs separately from training.
import numpy as np  # Persist complete numerical outputs without printing truncated tensors.
import torch  # Vectorized inference and streaming code moments.
import torch.nn.functional as F  # Cross-entropy is a secondary supervised diagnostic.

from .featurizer import reconstruct, GrassmannianBSF
from .io import load_checkpoint, read_json, write_json, digest
from .objectives import forward_kl
from .statistics import bootstrap_images
from .teacher import FrozenProbe, Normalization
from .logging_utils import log_event


def load_teacher(root, device):
    """Local: restore frozen preprocessing; global: every condition uses the identical teacher."""
    state = load_checkpoint(Path(root) / "teacher.pt")
    probe = FrozenProbe(state["probe"]["weight"], state["probe"]["bias"]).to(device)
    normalization = Normalization(state["normalization"]["mean"], state["normalization"]["scale"]).to(device)
    return probe, normalization


def load_model(checkpoint, device):
    """Local: strict architecture restore; global: loaded artifacts reproduce the trained model."""
    state = load_checkpoint(checkpoint)
    model = GrassmannianBSF(**state["architecture"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model, state


@torch.no_grad()
def measure(model, normalization, probe, features, rows, indices, device, batch_size=8,
            diagnostics=False, replacement=None, deadline=None):
    """Local: bounded batch inference; global: aggregate images and retain complete predictions."""
    values = {key: [] for key in ("kl", "ce_delta", "agreement", "accuracy", "teacher_accuracy",
                                   "nmse", "teacher_logits", "modified_logits")}
    counts = sums = second = None  # Code statistics are optional during frequent development checks.
    map_tv, map_dirichlet, map_count, patch_count, scalar_active = 0.0, 0.0, 0, 0, 0
    if diagnostics and model is not None:
        counts = torch.zeros(model.n_groups, device=device, dtype=torch.float64)
        sums = torch.zeros(model.n_groups, model.group_size, device=device, dtype=torch.float64)
        second = torch.zeros(model.n_groups, model.group_size, model.group_size, device=device, dtype=torch.float64)
    for start in range(0, len(indices), batch_size):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Evaluation budget reached; no partial summary will be treated as complete")
        ids = indices[start:start + batch_size]
        patches = torch.from_numpy(np.array(features[ids], copy=True)).to(device)
        teacher_logits = probe(patches)
        if replacement == "identity":
            modified, codes = patches, None
        elif replacement == "mean":
            modified, codes = normalization.mean.expand_as(patches), None
        else:
            modified, codes, _ = reconstruct(model, normalization, patches)
        logits = probe(modified)
        labels = torch.tensor([rows[int(i)]["label"] for i in ids], device=device)
        batch_values = {"kl": forward_kl(logits, teacher_logits),
                        "ce_delta": F.cross_entropy(logits, labels, reduction="none") - F.cross_entropy(teacher_logits, labels, reduction="none"),
                        "agreement": (logits.argmax(-1) == teacher_logits.argmax(-1)).float(),
                        "accuracy": (logits.argmax(-1) == labels).float(),
                        "teacher_accuracy": (teacher_logits.argmax(-1) == labels).float(),
                        "nmse": ((modified - patches) / normalization.scale).square().mean((1, 2)),
                        "teacher_logits": teacher_logits, "modified_logits": logits}
        for key, value in batch_values.items():
            values[key].append(value.cpu().numpy())  # Complete class vectors, no display truncation.
        if diagnostics and codes is not None:
            z = codes.double()  # Small streamed moments use double precision.
            active = z.norm(dim=-1) > 0
            counts += active.sum(0)
            sums += z.sum(0)
            second += torch.einsum("ngb,ngc->gbc", z, z)
            patch_count += len(z)
            scalar_active += int((z != 0).sum())
            heat = codes.norm(dim=-1).reshape(len(ids), patches.shape[1], -1)
            top = heat.square().sum(1).topk(min(8, model.n_groups), dim=-1).indices
            maps = heat.gather(2, top[:, None, :].expand(-1, heat.shape[1], -1))
            maps = maps / maps.square().sum(1, keepdim=True).sqrt().clamp_min(1e-12)
            side = int(np.sqrt(heat.shape[1]))
            maps = maps.reshape(len(ids), side, side, -1)
            dx, dy = maps[:, 1:] - maps[:, :-1], maps[:, :, 1:] - maps[:, :, :-1]
            map_tv += float(dx.abs().sum() + dy.abs().sum())
            map_dirichlet += float(dx.square().sum() + dy.square().sum())
            map_count += len(ids) * top.shape[-1]
    arrays = {key: np.concatenate(value) for key, value in values.items()}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError("Nonfinite evaluation output")
    summary = {key: float(value.mean()) for key, value in arrays.items() if not key.endswith("logits")}
    summary["images"] = len(indices)
    if counts is not None:
        means = sums / counts.clamp_min(1)[:, None]
        covariance = second / counts.clamp_min(1)[:, None, None] - means[..., None] * means[:, None, :]
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        valid = (counts > 1) & (eigenvalues.sum(-1) > 1e-12)
        stable_rank = eigenvalues.sum(-1) / eigenvalues[:, -1].clamp_min(1e-12)
        summary.update(active_blocks=float(counts.sum() / patch_count),
                       active_coordinates=scalar_active / patch_count,
                       dead_fraction=float((counts == 0).double().mean()),
                       stable_rank=float(stable_rank[valid].mean()) if valid.any() else None,
                       rank_eligible_blocks=int(valid.sum()), map_tv=map_tv / max(1, map_count),
                       map_dirichlet=map_dirichlet / max(1, map_count))
        arrays["block_counts"] = counts.cpu().numpy()
        arrays["block_stable_rank"] = stable_rank.cpu().numpy()
    return summary, arrays


def evaluate_checkpoint(config, checkpoint, logger, split="test", deadline=None):
    """Local: immutable evaluation record; global: reuse final test evidence rather than rerun selection."""
    root, checkpoint = Path(config.root), Path(checkpoint)
    destination = checkpoint.parent / f"{split}_summary.json"
    checkpoint_hash = digest(checkpoint)
    if destination.exists():
        existing = read_json(destination)
        if existing["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError("Checkpoint changed after evaluation; use a new run identity")
        return existing
    rows = read_json(root / "manifest.json")["images"]
    features = np.load(root / "patches.npy", mmap_mode="r")
    indices = np.array([i for i, row in enumerate(rows) if row["split"] == split])
    probe, normalization = load_teacher(root, config.device)
    model, state = load_model(checkpoint, config.device)
    if state["manifest_sha256"] != digest(root / "manifest.json") or state["teacher_sha256"] != digest(root / "teacher.pt"):
        raise RuntimeError("Evaluation data or teacher differ from the training artifact")
    started = time.monotonic()
    summary, arrays = measure(model, normalization, probe, features, rows, indices, config.device,
                              state["microbatch"], diagnostics=True, deadline=deadline)
    groups = [rows[int(i)]["group"] for i in indices]
    summary.update(condition=state["condition"], seed=state["seed"], split=split,
                   checkpoint_sha256=checkpoint_hash, completed_epochs=state["epoch"],
                   status=state["status"], parameters=sum(p.numel() for p in model.parameters()),
                   kl_image_cluster_ci95=bootstrap_images(arrays["kl"], groups),
                   evaluation_seconds=time.monotonic() - started)
    arrays.update(indices=indices, groups=np.asarray(groups))
    np.savez_compressed(checkpoint.parent / f"{split}_predictions.npz", **arrays)
    write_json(destination, summary)
    log_event(logger, "evaluation", summary)
    return summary


def evaluate_controls(config, logger, split="test", deadline=None):
    """Local: identity, mean and fresh random bottlenecks; global: quantify trivial ceilings/floors."""
    root = Path(config.root)
    destination = root / f"controls_{split}.json"
    if destination.exists():
        return read_json(destination)
    rows = read_json(root / "manifest.json")["images"]
    indices = np.array([i for i, row in enumerate(rows) if row["split"] == split])
    features = np.load(root / "patches.npy", mmap_mode="r")
    probe, normalization = load_teacher(root, config.device)
    results = {}
    for name in ("identity", "mean", "untrained"):
        torch.manual_seed(100)  # Match the first confirmatory initialization without fitting.
        model = GrassmannianBSF(768, config.groups, config.block_size, config.active).to(config.device) if name == "untrained" else None
        summary, arrays = measure(model, normalization, probe, features, rows, indices, config.device,
                                  config.microbatch, replacement=None if name == "untrained" else name, deadline=deadline)
        results[name] = summary
        np.savez_compressed(root / f"control_{name}_{split}.npz", indices=indices, **arrays)
        log_event(logger, "control", {"name": name, **summary})
    write_json(destination, results)
    return results
