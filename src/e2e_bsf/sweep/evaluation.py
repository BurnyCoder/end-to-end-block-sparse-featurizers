"""Global: all images and operating points remain visible; local: bounded suffix evaluation.

Sources: existing evaluation.py; https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigvalsh.html.
"""

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from ..featurizer import GrassmannianBSF, reconstruct
from ..objectives import forward_kl
from ..io import read_json, write_json, load_checkpoint, digest
from ..logging_utils import log_event
from .core import region_mse
from .cache import check_deadline


@torch.no_grad()
def measure(ctx, model, indices, microbatch=8, replacement=None, diagnostics=False, deadline=None):
    """Retain complete image-level predictions/errors, but only streamed latent moments."""
    records = {key: [] for key in ("kl", "ce_delta", "accuracy", "teacher_accuracy", "agreement",
                                  "layer_mse", "teacher_logits", "modified_logits")}
    frames = model.frames() if model is not None else None
    counts = sums = seconds = None
    tokens = active_coordinates = patch_active = special_active = 0
    map_tv = map_dirichlet = map_count = 0
    if diagnostics and model is not None:
        counts = torch.zeros(model.n_groups, dtype=torch.float64, device=ctx.device)
        sums = torch.zeros(model.n_groups, model.group_size, dtype=torch.float64, device=ctx.device)
        seconds = torch.zeros(model.n_groups, model.group_size, model.group_size, dtype=torch.float64, device=ctx.device)
    for offset in range(0, len(indices), microbatch):
        check_deadline(deadline)
        ids = indices[offset:offset + microbatch]
        clean, targets, teacher = ctx.batch(ids)
        codes = None
        if replacement == "identity":
            modified = clean
        elif replacement == "mean":
            modified = ctx.normalization.mean.expand_as(clean)
        else:
            modified, codes, _ = reconstruct(model, ctx.normalization, clean, frames)
        logits, downstream = ctx.suffix(modified, targets)
        insertion = region_mse(modified, clean, ctx.suffix.special)
        labels = torch.tensor([ctx.rows[int(i)]["label"] for i in ids], device=ctx.device)
        values = {"kl": forward_kl(logits, teacher),
                  "ce_delta": F.cross_entropy(logits, labels, reduction="none") - F.cross_entropy(teacher, labels, reduction="none"),
                  "accuracy": (logits.argmax(-1) == labels).float(),
                  "teacher_accuracy": (teacher.argmax(-1) == labels).float(),
                  "agreement": (logits.argmax(-1) == teacher.argmax(-1)).float(),
                  "layer_mse": torch.cat([insertion[:, None], downstream], 1),
                  "teacher_logits": teacher, "modified_logits": logits}
        for key, value in values.items():
            records[key].append(value.cpu().numpy())
        if counts is not None:
            z = codes.double()
            active = z.norm(dim=-1) > 0
            counts += active.sum(0)
            sums += z.sum(0)
            seconds += torch.einsum("ngb,ngc->gbc", z, z)
            tokens += len(z)
            active_coordinates += int((z != 0).sum())
            mask = active.reshape(len(ids), clean.shape[1], -1)
            patch_active += int(mask[:, ctx.suffix.special:].sum())
            special_active += int(mask[:, :ctx.suffix.special].sum())
            heat = codes.norm(dim=-1).reshape(len(ids), clean.shape[1], -1)[:, ctx.suffix.special:]
            side = int(np.sqrt(heat.shape[1]))
            if side * side == heat.shape[1]:
                chosen = heat.square().sum(1).topk(min(8, model.n_groups), dim=-1).indices
                maps = heat.gather(2, chosen[:, None].expand(-1, heat.shape[1], -1))
                maps /= maps.square().sum(1, keepdim=True).sqrt().clamp_min(1e-12)
                maps = maps.reshape(len(ids), side, side, -1)
                dx, dy = maps[:, 1:] - maps[:, :-1], maps[:, :, 1:] - maps[:, :, :-1]
                map_tv += float(dx.abs().sum() + dy.abs().sum())
                map_dirichlet += float(dx.square().sum() + dy.square().sum())
                map_count += len(ids) * chosen.shape[-1]
    arrays = {key: np.concatenate(value) for key, value in records.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("Nonfinite sweep evaluation")
    variances = ctx.variances.cpu().numpy()
    arrays["layer_nmse"] = arrays["layer_mse"] / np.maximum(variances, 1e-12)
    summary = {key: float(arrays[key].mean()) for key in ("kl", "ce_delta", "accuracy", "teacher_accuracy", "agreement")}
    summary.update(images=len(indices), layer_mse=arrays["layer_mse"].mean(0).tolist(),
                   layer_nmse=arrays["layer_nmse"].mean(0).tolist(),
                   insertion_nmse=float(arrays["layer_nmse"][:, 0, 0].mean()),
                   downstream_mse=float(arrays["layer_mse"][:, 1:, 0].mean()))
    if counts is not None:
        means = sums / counts.clamp_min(1)[:, None]
        covariance = seconds / counts.clamp_min(1)[:, None, None] - means[:, :, None] * means[:, None, :]
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        rank = eigenvalues.sum(-1) / eigenvalues[:, -1].clamp_min(1e-12)
        valid = (counts > 1) & (eigenvalues.sum(-1) > 1e-12)
        summary.update(active_blocks=float(counts.sum() / tokens), active_coordinates=active_coordinates / tokens,
                       patch_active_blocks=patch_active / (len(indices) * (clean.shape[1] - ctx.suffix.special)),
                       special_active_blocks=special_active / (len(indices) * ctx.suffix.special),
                       alive_blocks=int((counts > 0).sum()), dead_blocks=int((counts == 0).sum()),
                       parameters=sum(p.numel() for p in model.parameters()),
                       stable_rank=float(rank[valid].mean()) if valid.any() else None,
                       map_tv=map_tv / max(1, map_count), map_dirichlet=map_dirichlet / max(1, map_count))
        arrays.update(block_counts=counts.cpu().numpy(), block_stable_rank=rank.cpu().numpy())
    arrays.update(indices=np.asarray(indices), groups=np.asarray([ctx.rows[int(i)]["group"] for i in indices]))
    return summary, arrays


def load_cell(path, ctx):
    """Reject changed cache/teacher identities before using a checkpoint for inference."""
    state = load_checkpoint(path)
    if state["identity"] != ctx.identity:
        raise ValueError("Evaluation context differs from checkpoint identity")
    model = GrassmannianBSF(**state["architecture"]).to(ctx.device)
    model.load_state_dict(state["model"])
    return model, state


def evaluate(spec, ctx, logger, deadline=None):
    """Open reused test data only after every frozen main run has completed."""
    root = Path(spec.root)
    schedule = read_json(root / "study_schedule.json")
    for key in schedule["cells"]:
        state = load_checkpoint(root / "runs" / key / "checkpoint.pt")
        if state["status"] != "complete" or state["epoch"] != schedule["epochs"]:
            raise RuntimeError("All scheduled cells must finish equal exposure before test evaluation")
    indices = ctx.indices("test")
    for cell in spec.cells():
        run = root / "runs" / cell.key
        path, summary_path = run / "checkpoint.pt", run / "test_summary.json"
        checkpoint_hash = digest(path)
        if summary_path.exists():
            if read_json(summary_path)["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Checkpoint changed after evaluation")
            continue
        started = time.monotonic()
        model, state = load_cell(path, ctx)
        summary, arrays = measure(ctx, model, indices, state["microbatch"], diagnostics=True, deadline=deadline)
        summary.update(cell=cell.key, k=cell.k, objective=cell.objective, beta=cell.beta, label=cell.label,
                       seed=spec.seed, epochs=state["epoch"], checkpoint_sha256=checkpoint_hash,
                       evaluation_seconds=time.monotonic() - started, training_seconds=state["elapsed_seconds"],
                       peak_vram_bytes=state["peak_vram_bytes"], test_reuse=True)
        np.savez_compressed(run / "test_predictions.npz", **arrays)
        write_json(summary_path, summary)
        log_event(logger, "sweep_evaluation", summary)
        del model
    controls = root / "controls"
    controls.mkdir(exist_ok=True)
    for name, k in [("identity", None), ("mean", None), *[("untrained", k) for k in spec.active_blocks]]:
        key = name if k is None else f"untrained_k{k:03d}"
        if (controls / f"{key}.json").exists():
            continue
        torch.manual_seed(spec.seed)
        model = GrassmannianBSF(768, spec.groups, spec.block_size, k).to(ctx.device) if k else None
        summary, arrays = measure(ctx, model, indices, schedule["microbatch"],
                                  replacement=None if k else name, diagnostics=bool(k), deadline=deadline)
        summary.update(control=key, k=k, seed=spec.seed if k else None)
        np.savez_compressed(controls / f"{key}.npz", **arrays)
        write_json(controls / f"{key}.json", summary)
        log_event(logger, "sweep_control", summary)
