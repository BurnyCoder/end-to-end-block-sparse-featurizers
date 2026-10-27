"""Global: reuse the frozen teacher without leaking test fits; local: resumable raw-state caches.

Sources: numpy.lib.format.open_memmap, existing extraction/normalization modules, and
https://huggingface.co/docs/transformers/model_doc/dinov3.
"""

from dataclasses import replace
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from ..config import Config, MODEL_ID, MODEL_REVISION, BSF_REVISION
from ..extraction import load_backbone, image_batch
from ..evaluation import load_teacher
from ..teacher import Normalization
from ..io import read_json, write_json, save_checkpoint, digest
from ..logging_utils import log_event
from .core import clean_states, FrozenSuffix


def split_indices(rows, split, limit=0):
    """Fixed class-balanced pilot subsets inherit the already isolated original split."""
    indices = np.array([i for i, row in enumerate(rows) if row["split"] == split], dtype=np.int64)
    if not limit:
        return indices
    rng = np.random.default_rng(42)
    return np.concatenate([rng.permutation([i for i in indices if rows[int(i)]["label"] == label])[:limit // 10]
                           for label in range(10)]).astype(np.int64)


def backbone_config(spec):
    """Model downloads and images remain in the original checkout's explicit source directory."""
    return replace(Config(), root=spec.source_root, device=spec.device, microbatch=spec.microbatch)


def check_deadline(deadline):
    """Never silently restart a stage after its original wall-time allocation."""
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("The original sweep stage deadline has elapsed")


def prepare(spec, logger, deadline):
    """Extract seven raw states per image; atomically advance only after all arrays are durable."""
    root, source, cache = Path(spec.root), Path(spec.source_root), Path(spec.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not (source / "teacher.pt").exists() or not (source / "manifest.json").exists():
        raise FileNotFoundError("Prepare the original Imagenette teacher first; source_root must contain teacher.pt and manifest.json")
    identity = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "bsf_revision": BSF_REVISION,
                "manifest_sha256": digest(source / "manifest.json"), "source_teacher_sha256": digest(source / "teacher.pt"),
                "layers": list(range(spec.insertion_block, 13)), "sequence_scope": "all_201", "dtype": "float32"}
    metadata_path = root / "cache.json"
    previous = read_json(metadata_path) if metadata_path.exists() else None
    if previous and previous["identity"] != identity:
        raise RuntimeError("Cache sources changed; use a new registered sweep directory")
    if previous and previous.get("complete") and (root / "teacher.pt").exists():
        return previous
    check_deadline(deadline)
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "duplicate_audit.json", "teacher.json"):
        if (source / name).exists():
            shutil.copy2(source / name, root / ("source_teacher.json" if name == "teacher.json" else name))
    rows = read_json(root / "manifest.json")["images"]
    shape = (len(rows), 201, 768)
    required = len(identity["layers"]) * int(np.prod(shape)) * 4
    if not previous and shutil.disk_usage(cache).free < required + 2 * 2**30:
        raise RuntimeError(f"Cache needs {required / 1e9:.1f} GB plus 2 GiB free space at {cache}")
    arrays = []
    for layer in identity["layers"]:
        path = cache / f"block{layer:02d}.npy"
        array = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32, shape=shape)
        if array.shape != shape or array.dtype != np.float32:
            raise RuntimeError("Existing cache array has a different shape or dtype")
        arrays.append(array)
    logits_path = cache / "teacher_logits.npy"
    logits = np.lib.format.open_memmap(logits_path, mode="r+" if logits_path.exists() else "w+",
                                      dtype=np.float32, shape=(len(rows), 10))
    processor, backbone = load_backbone(backbone_config(spec))
    probe, _ = load_teacher(source, spec.device)
    completed = previous["completed"] if previous else 0
    microbatch = previous.get("microbatch", spec.microbatch) if previous else spec.microbatch
    started = time.monotonic()
    while completed < len(rows):
        check_deadline(deadline)
        stop = min(completed + microbatch, len(rows))
        images = image_batch(rows, range(completed, stop), source.parent / "data")
        try:
            pixels = processor(images=images, return_tensors="pt").to(spec.device)["pixel_values"]
            states = clean_states(backbone, pixels, spec.insertion_block)
            if any(value.shape[1:] != (201, 768) or not torch.isfinite(value).all() for value in states):
                raise RuntimeError("Unexpected/nonfinite raw DINO state")
            for array, value in zip(arrays, states, strict=True):
                array[completed:stop] = value.cpu().numpy()
            with torch.no_grad():
                logits[completed:stop] = probe(backbone.norm(states[-1])[:, 5:]).cpu().numpy()
        except torch.OutOfMemoryError:
            if microbatch == 1:
                raise
            microbatch //= 2
            torch.cuda.empty_cache()
            log_event(logger, "sweep_cache_reduce_batch", {"microbatch": microbatch})
            continue
        for array in [*arrays, logits]:
            array.flush()
        completed = stop
        write_json(metadata_path, {"identity": identity, "shape": list(shape), "completed": completed,
                                   "microbatch": microbatch, "complete": False})
        if completed % (microbatch * 25) == 0 or completed == len(rows):
            log_event(logger, "sweep_cache", {"completed": completed, "images": len(rows),
                                            "seconds": time.monotonic() - started})
    del backbone, states, processor
    if spec.device.startswith("cuda"):
        torch.cuda.empty_cache()
    train = split_indices(rows, "train")
    variances, input_normalization = [], None
    for layer, array in zip(identity["layers"], arrays, strict=True):
        check_deadline(deadline)
        normalization = Normalization.fit(array, train)
        if layer == spec.insertion_block:
            input_normalization = normalization
        error_sum = np.zeros(3)
        for start in range(0, len(train), 32):
            check_deadline(deadline)
            squared = np.square(np.asarray(array[train[start:start + 32]], dtype=np.float64) - normalization.mean.numpy())
            error_sum += [squared.mean((1, 2)).sum(), squared[:, 5:].mean((1, 2)).sum(), squared[:, :5].mean((1, 2)).sum()]
        variances.append((error_sum / len(train)).tolist())
        log_event(logger, "sweep_train_variance", {"block": layer, "all_patch_special": variances[-1]})
    save_checkpoint(root / "teacher.pt", {"probe": {k: v.cpu() for k, v in probe.state_dict().items()},
                    "normalization": input_normalization.state_dict(), "layer_variances": variances})
    files = {path.name: {"sha256": digest(path), "bytes": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns}
             for path in [*[cache / f"block{layer:02d}.npy" for layer in identity["layers"]], logits_path]}
    metadata = {"identity": identity, "shape": list(shape), "completed": completed, "complete": True,
                "microbatch": microbatch, "files": files, "fit_images": len(train), "layer_variances": variances}
    write_json(metadata_path, metadata)
    log_event(logger, "sweep_prepared", {"images": len(rows), "cache_bytes": required, "fit_images": len(train)})
    return metadata


class SweepRuntime:
    """One shared frozen suffix and read-only cache handle per worker, reused across all cells."""

    def __init__(self, spec):
        self.spec, self.device = spec, spec.device
        root, cache = Path(spec.root), Path(spec.cache_dir)
        metadata = read_json(root / "cache.json")
        if not metadata["complete"]:
            raise RuntimeError("Finish extraction before opening the training cache")
        if digest(root / "manifest.json") != metadata["identity"]["manifest_sha256"]:
            raise ValueError("The cached split manifest changed")
        for name, record in metadata["files"].items():
            stat = (cache / name).stat()
            if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
                raise RuntimeError(f"Cache changed since hashing: {name}")
        self.features = [np.load(cache / f"block{layer:02d}.npy", mmap_mode="r") for layer in metadata["identity"]["layers"]]
        self.teacher_logits = np.load(cache / "teacher_logits.npy", mmap_mode="r")
        self.rows = read_json(root / "manifest.json")["images"]
        self.identity = {"cache_sha256": digest(root / "cache.json"), "manifest_sha256": digest(root / "manifest.json"),
                         "teacher_sha256": digest(root / "teacher.pt"), **metadata["identity"]}
        self.probe, self.normalization = load_teacher(root, spec.device)
        self.variances = torch.tensor(metadata["layer_variances"], device=spec.device)
        self.processor, self.backbone = load_backbone(backbone_config(spec))
        self.suffix = FrozenSuffix.from_backbone(self.backbone, self.probe, spec.insertion_block)

    def indices(self, split, limit=0):
        """Shared subset selection ensures every cell sees the same images."""
        return split_indices(self.rows, split, limit)

    def batch(self, indices):
        """Materialize only one image microbatch; latent datasets are never cached."""
        arrays = [torch.from_numpy(np.array(array[indices], copy=True)).to(self.device) for array in self.features]
        logits = torch.from_numpy(np.array(self.teacher_logits[indices], copy=True)).to(self.device)
        return arrays[0], arrays[1:], logits
