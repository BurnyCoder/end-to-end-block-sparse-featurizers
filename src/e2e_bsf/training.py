"""Global: matched from-scratch experiments. Local: resumable sample-weighted optimization.

Sources: https://docs.pytorch.org/docs/stable/optim.html;
https://docs.pytorch.org/docs/stable/notes/randomness.html;
https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html.
"""

from pathlib import Path  # Isolate every condition/seed/lr experiment.
import math  # Account for partial final accumulation groups.
import time  # Enforce a wall-time budget without looking at test outcomes.
import numpy as np  # Derive deterministic per-epoch image orders.
import torch  # Frozen readouts still propagate gradients to the BSF.

from .config import MODEL_ID, MODEL_REVISION, BSF_REVISION, CONDITIONS
from .evaluation import load_teacher, measure
from .featurizer import GrassmannianBSF, reconstruct
from .io import read_json, write_json, save_checkpoint, load_checkpoint, digest, source_revision
from .logging_utils import log_event
from .objectives import forward_kl, objective


def architecture(config, condition):
    """Local: shared capacity mapping; global: scalar controls match total/active coordinates."""
    scalar = condition.startswith("scalar_")
    return {"d": 768, "n_groups": config.groups * config.block_size if scalar else config.groups,
            "group_size": 1 if scalar else config.block_size,
            "l0": config.active * config.block_size if scalar else config.active}


def accumulate(model, normalization, probe, features, group, objective_name, device, microbatch):
    """Local: one complete effective batch; global: no dropped or overweighted partial batches."""
    totals = np.zeros(3)
    for start in range(0, len(group), microbatch):
        ids = group[start:start + microbatch]
        patches = torch.from_numpy(np.array(features[ids], copy=True)).to(device)
        target = probe(patches).detach()
        modified, _, mse = reconstruct(model, normalization, patches)
        kl = forward_kl(probe(modified), target).mean()
        loss = objective(objective_name, kl, mse)
        (loss * (len(ids) / len(group))).backward()  # Weight by real image count, including remainders.
        totals += np.array([loss.item(), kl.item(), mse.item()]) * len(ids)
    return totals / len(group)


def train_one(config, condition, seed, logger, run_name=None, deadline=None, max_updates=None):
    """Local: one deterministic run; global: compare only the declared objective/architecture."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    config.validate()
    root = Path(config.root)
    run = root / "runs" / (run_name or f"{condition}_seed{seed}")
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "checkpoint.pt"
    identity = {"manifest_sha256": digest(root / "manifest.json"), "teacher_sha256": digest(root / "teacher.pt")}
    if checkpoint.exists():
        previous = load_checkpoint(checkpoint)
        if previous["config"] != config.to_dict() or previous["condition"] != condition or previous["seed"] != seed:
            raise RuntimeError("Run identity/configuration changed; choose a different run name")
        if any(previous[key] != value for key, value in identity.items()):
            raise RuntimeError("Training inputs changed since the checkpoint")
        if previous["status"] == "complete":
            return checkpoint  # Idempotent study restarts never silently retrain a completed run.
    else:
        previous = None
    torch.manual_seed(seed)  # Same architecture/objective pairs receive identical initial parameters.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model = GrassmannianBSF(**architecture(config, condition)).to(config.device)
    probe, normalization = load_teacher(root, config.device)
    features = np.load(root / "patches.npy", mmap_mode="r")
    rows = read_json(root / "manifest.json")["images"]
    train = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    dev = np.array([i for i, row in enumerate(rows) if row["split"] == "dev"])
    for name, limit in (("train", config.train_limit), ("dev", config.dev_limit)):
        if limit:
            source = train if name == "train" else dev
            rng = np.random.default_rng(config.seed)  # Same slice for all pilot conditions/seeds.
            selected = np.concatenate([rng.permutation([i for i in source if rows[int(i)]["label"] == label])[:max(1, limit // 10)]
                                       for label in range(10)])
            if name == "train":
                train = selected
            else:
                dev = selected
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    updates_per_epoch = math.ceil(len(train) / config.effective_batch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: 0.1 + 0.9 * (1 + math.cos(math.pi * min(step / (updates_per_epoch * config.epochs), 1))) / 2)
    epoch, offset, updates, elapsed, microbatch = 0, 0, 0, 0.0, config.microbatch
    if previous is not None:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        epoch, offset, updates, elapsed = previous["epoch"], previous["offset"], previous["updates"], previous["elapsed_seconds"]
        microbatch = previous["microbatch"]
        torch.set_rng_state(previous["torch_rng"])
        if previous["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    started = time.monotonic()
    history_path = run / "history.jsonl"
    objective_name = condition.split("_", 1)[1]

    def checkpoint_state(status):
        """Local: complete restart cursor; global: preserve equal exposure across interruptions."""
        return {"format_version": 1, "config": config.to_dict(), "architecture": architecture(config, condition),
                "condition": condition, "seed": seed, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "probe": probe.state_dict(), "normalization": normalization.state_dict(),
                "epoch": epoch, "offset": offset, "updates": updates, "microbatch": microbatch,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "data_order": "numpy.default_rng(seed + epoch * 1000003).permutation(train_indices)",
                "elapsed_seconds": elapsed + time.monotonic() - started, "status": status,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "bsf_revision": BSF_REVISION,
                "source_revision": source_revision(), **identity}

    while epoch < config.epochs:
        order = np.random.default_rng(seed + epoch * 1000003).permutation(train)
        while offset < len(order):
            if (deadline is not None and time.time() >= deadline) or (max_updates is not None and updates >= max_updates):
                save_checkpoint(checkpoint, checkpoint_state("paused"))
                log_event(logger, "training_paused", {"run": run.name, "epoch": epoch, "offset": offset})
                return checkpoint
            group = order[offset:offset + config.effective_batch]
            optimizer.zero_grad(set_to_none=True)
            try:
                metrics = accumulate(model, normalization, probe, features, group, objective_name,
                                     config.device, microbatch)
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if microbatch == 1:
                    save_checkpoint(checkpoint, checkpoint_state("failed_oom"))
                    raise
                microbatch = max(1, microbatch // 2)
                torch.cuda.empty_cache()
                log_event(logger, "reduce_microbatch", {"run": run.name, "microbatch": microbatch})
                continue  # Retry the entire effective batch with zeroed gradients.
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            offset += len(group)
            updates += 1
            event = {"update": updates, "epoch": epoch, "images_seen": epoch * len(train) + offset,
                     "loss": float(metrics[0]), "kl": float(metrics[1]), "nmse": float(metrics[2]),
                     "gradient_norm": float(gradient_norm), "lr": optimizer.param_groups[0]["lr"],
                     "elapsed_seconds": elapsed + time.monotonic() - started}
            with history_path.open("a", encoding="utf-8") as stream:
                import json  # Standard JSONL records are append-only and independently readable.
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            if updates % 10 == 0 or offset == len(order):
                log_event(logger, "training_step", {"run": run.name, **event})
            if updates % 100 == 0:
                save_checkpoint(checkpoint, checkpoint_state("running"))
        epoch, offset = epoch + 1, 0
        summary, _ = measure(model, normalization, probe, features, rows, dev, config.device, microbatch)
        summary.update(epoch=epoch, update=updates)
        write_json(run / "development.json", summary)
        log_event(logger, "development", {"run": run.name, **summary})
        save_checkpoint(checkpoint, checkpoint_state("complete" if epoch == config.epochs else "running"))
    final = checkpoint_state("complete")
    final["peak_vram_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    save_checkpoint(checkpoint, final)
    write_json(run / "run.json", {key: final[key] for key in ("condition", "seed", "epoch", "updates",
               "microbatch", "elapsed_seconds", "status", "peak_vram_bytes", "source_revision")})
    log_event(logger, "training_complete", read_json(run / "run.json"))
    return checkpoint
