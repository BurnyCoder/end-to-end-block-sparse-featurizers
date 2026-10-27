"""Global: equal image exposure and durable round-robin execution; local: Adam on BSF only.

Reuses the repository's featurizer, KL, serialization and history recovery. Optimizer API:
https://docs.pytorch.org/docs/stable/optim.html.
"""

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import torch
from ..featurizer import GrassmannianBSF, reconstruct
from ..objectives import forward_kl
from ..io import load_checkpoint, save_checkpoint, read_json, write_json, source_revision, restore_history
from ..logging_utils import log_event
from .core import loss_for
from .evaluation import measure


def accumulate(ctx, model, cell, group, microbatch):
    """Backpropagate the actual image-weighted mean, including the last partial effective batch."""
    totals = np.zeros(4)
    for offset in range(0, len(group), microbatch):
        ids = group[offset:offset + microbatch]
        clean, targets, teacher = ctx.batch(ids)
        modified, codes, local = reconstruct(model, ctx.normalization, clean)
        # Local training logs suffix behavior without differentiating through an unused loss.
        with torch.set_grad_enabled(cell.objective != "mse"):
            logits, errors = ctx.suffix(modified if cell.objective != "mse" else modified.detach(), targets)
            kl = forward_kl(logits, teacher).mean()
            downstream = errors[:, :, 0].mean()
        loss = loss_for(cell, kl, local, downstream)
        (loss * len(ids) / len(group)).backward()
        totals += np.array([loss.item(), kl.item(), local.item(), downstream.item()]) * len(ids)
        del modified, codes, loss
    return totals / len(group)


def train_cell(spec, ctx, cell, epochs, lr, seed, logger, stop_epoch=None, max_updates=None,
               deadline=None, pilot=False, microbatch=None):
    """Advance one immutable run, optionally yielding at an epoch boundary for the frozen round-robin."""
    name = f"pilot_{cell.key}_lr{lr:g}" if pilot else cell.key
    run = Path(spec.root) / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "checkpoint.pt"
    train = ctx.indices("train", spec.pilot_train_images if pilot else 0)
    dev = ctx.indices("dev", spec.pilot_dev_images if pilot else 0)
    recipe = {"cell": asdict(cell), "epochs": epochs, "lr": lr, "seed": seed, "pilot": pilot,
              "spec": spec.to_dict(), "identity": ctx.identity}
    previous = load_checkpoint(checkpoint) if checkpoint.exists() else None
    if previous and previous["recipe"] != recipe:
        raise ValueError("Run configuration, cache, or teacher changed since the checkpoint")
    if previous and previous["status"] == "complete":
        return checkpoint
    torch.manual_seed(seed)
    architecture = {"d": ctx.features[0].shape[-1], "n_groups": spec.groups, "group_size": spec.block_size, "l0": cell.k}
    model = GrassmannianBSF(**architecture).to(ctx.device)
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps = math.ceil(len(train) / spec.effective_batch) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: .1 + .9 * (1 + math.cos(math.pi * min(step / steps, 1))) / 2)
    epoch = offset = updates = images_seen = 0
    elapsed = train_seconds = dev_seconds = checkpoint_seconds = 0.
    batch_size = microbatch or spec.microbatch
    if previous:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        epoch, offset, updates, images_seen = [previous[key] for key in ("epoch", "offset", "updates", "images_seen")]
        elapsed, train_seconds, dev_seconds, checkpoint_seconds = [previous[key] for key in
            ("elapsed_seconds", "train_seconds", "dev_seconds", "checkpoint_seconds")]
        batch_size = previous["microbatch"]
        torch.set_rng_state(previous["torch_rng"])
        if previous["cuda_rng"]:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    history = run / "history.jsonl"
    restore_history(history, updates)
    if ctx.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    revision = previous["source_revision"] if previous else source_revision()

    def save(status):
        """Bind the entire restart state and record active compute separately from wall-clock gaps."""
        nonlocal checkpoint_seconds
        state = {"format_version": 2, "recipe": recipe, "identity": ctx.identity, "cell": asdict(cell),
                 "architecture": architecture, "model": model.state_dict(), "probe": ctx.probe.state_dict(),
                 "normalization": ctx.normalization.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if ctx.device.startswith("cuda") else [],
                 "epoch": epoch, "offset": offset, "updates": updates, "images_seen": images_seen,
                 "microbatch": batch_size, "seed": seed, "source_revision": revision, "initial_sha256": initial_hash,
                 "status": status, "saved_unix": time.time(), "elapsed_seconds": elapsed + time.monotonic() - started,
                 "train_seconds": train_seconds, "dev_seconds": dev_seconds, "checkpoint_seconds": checkpoint_seconds,
                 "peak_vram_bytes": max(previous.get("peak_vram_bytes", 0) if previous else 0,
                    torch.cuda.max_memory_allocated() if ctx.device.startswith("cuda") else 0)}
        before = time.monotonic()
        save_checkpoint(checkpoint, state)
        checkpoint_seconds += time.monotonic() - before
        write_json(run / "run.json", {key: state[key] for key in ("cell", "seed", "epoch", "updates", "images_seen",
            "microbatch", "status", "elapsed_seconds", "train_seconds", "dev_seconds", "peak_vram_bytes", "source_revision", "initial_sha256")})

    while epoch < epochs and (stop_epoch is None or epoch < stop_epoch):
        order = np.random.default_rng(seed + epoch * 1000003).permutation(train)
        while offset < len(order):
            if (deadline is not None and time.time() >= deadline) or (max_updates is not None and updates >= max_updates):
                save("paused")
                return checkpoint
            group = order[offset:offset + spec.effective_batch]
            before = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            try:
                metrics = accumulate(ctx, model, cell, group, batch_size)
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                # Main runs use the shared pilot-selected microbatch; a new OOM is a recorded failure.
                if batch_size == 1 or (not pilot and microbatch is not None):
                    save("failed_oom")
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                log_event(logger, "sweep_reduce_microbatch", {"cell": name, "microbatch": batch_size})
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if ctx.device.startswith("cuda"):
                torch.cuda.synchronize()
            train_seconds += time.monotonic() - before
            offset += len(group)
            updates += 1
            images_seen += len(group)
            event = {"update": updates, "epoch": epoch, "images_seen": images_seen,
                     **dict(zip(("loss", "kl", "insertion_nmse", "downstream_mse"), map(float, metrics))),
                     "lr": optimizer.param_groups[0]["lr"], "gradient_norm": float(gradient_norm),
                     "elapsed_seconds": elapsed + time.monotonic() - started}
            with history.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            if updates % 10 == 0 or offset == len(order):
                log_event(logger, "sweep_training", {"cell": name, **event})
            if updates % 100 == 0:
                save("running")
        # Commit the epoch before optional development work; a failed evaluation never repeats training.
        epoch, offset = epoch + 1, 0
        save("trained_epoch")
        before = time.monotonic()
        summary, _ = measure(ctx, model, dev, batch_size, deadline=deadline)
        dev_seconds += time.monotonic() - before
        write_json(run / "development.json", dict(summary, epoch=epoch, images_seen=images_seen))
        log_event(logger, "sweep_development", {"cell": name, **summary})
        save("complete" if epoch == epochs else "running")
    # A resume after the final optimizer step still needs its development endpoint.
    if epoch > 0 and (not (run / "development.json").exists() or
                     read_json(run / "development.json")["epoch"] != epoch):
        before = time.monotonic()
        summary, _ = measure(ctx, model, dev, batch_size, deadline=deadline)
        dev_seconds += time.monotonic() - before
        write_json(run / "development.json", dict(summary, epoch=epoch, images_seen=images_seen))
        save("complete" if epoch == epochs else "running")
    return checkpoint
