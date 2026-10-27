"""Global: fixed DINO representations. Local: resumable float32 patch cache.

Adapted extraction recipe from BurnyCoder/block-sparse-featurizer-experiments
@6af61910c3ad2ddc4320bff76fe607b6ede13c18. Transformers AutoModel/AutoImageProcessor:
https://huggingface.co/docs/transformers/model_doc/dinov3.
"""

from pathlib import Path  # Keep model/data caches inside the current workspace.
import time  # Measure extraction rather than guessing its cost.
import numpy as np  # Disk-backed .npy arrays retain complete image boundaries.
from PIL import Image, ImageOps  # Decode RGB images consistently.
import torch  # The backbone is always frozen and in evaluation mode.

from .config import MODEL_ID, MODEL_REVISION  # One revision for both processor and weights.
from .io import digest, read_json, write_json  # Cache validity follows manifest identity.
from .logging_utils import log_event  # Log every extraction batch without tensor truncation.


def load_backbone(config):
    """Local: pinned authorized model access; global: no silent substitute backbone."""
    from transformers import AutoImageProcessor, AutoModel  # Heavy imports are lazy.

    cache = Path(config.root).parent / "hf"  # Avoid searching another project's model cache.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=cache)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=cache,
                                     dtype=torch.float32).to(config.device).eval()
    model.requires_grad_(False)  # Frozen weights still allow suffix input gradients if needed.
    return processor, model


def image_batch(rows, indices, data_root):
    """Local: scoped image loading; global: exact same preprocessing online and cached."""
    images = []  # PIL copies detach from closed file handles.
    for i in indices:
        with Image.open(data_root / rows[int(i)]["id"]) as opened:
            images.append(ImageOps.exif_transpose(opened).convert("RGB"))
    return images


def extract_patches(processor, backbone, images, device):
    """Local: strip five special tokens; global: match the released Goodfire activation site."""
    batch = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():  # No backbone Jacobian is required for final-feature insertion.
        patches = backbone(**batch).last_hidden_state[:, 5:].float()
    if patches.shape[1:] != (196, 768) or not torch.isfinite(patches).all():
        raise RuntimeError(f"Unexpected/nonfinite DINO patch activations: {patches.shape}")
    return patches


def extract_cache(config, rows, logger, deadline=None):
    """Local: atomic progress and disk-backed arrays; global: reusable image-level evidence."""
    root = Path(config.root)
    path, metadata_path = root / "patches.npy", root / "cache.json"
    identity = {"model": MODEL_ID, "revision": MODEL_REVISION,
                "manifest_sha256": digest(root / "manifest.json"), "shape": [len(rows), 196, 768],
                "dtype": "float32"}
    completed = 0
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if any(metadata[key] != value for key, value in identity.items()):
            raise RuntimeError("Cache identity changed; choose a new output directory")
        completed = metadata["completed"]
    if completed == len(rows):
        return np.load(path, mmap_mode="r")  # Reuse only a complete, matching cache.
    processor, backbone = load_backbone(config)
    patches = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
                                       dtype=np.float32, shape=(len(rows), 196, 768))
    batch_size = config.microbatch
    started = time.monotonic()
    while completed < len(rows):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Preparation budget exhausted; extraction checkpoint is resumable")
        stop = min(completed + batch_size, len(rows))
        images = image_batch(rows, range(completed, stop), root.parent / "data")
        try:
            values = extract_patches(processor, backbone, images, config.device)
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)  # No precision or model changes on memory failure.
            torch.cuda.empty_cache()
            log_event(logger, "extraction_reduce_batch", {"microbatch": batch_size})
            continue
        patches[completed:stop] = values.cpu().numpy()
        patches.flush()  # Durability precedes advancing the committed row count.
        completed = stop
        write_json(metadata_path, dict(identity, completed=completed,
                                       seconds=time.monotonic() - started, microbatch=batch_size))
        log_event(logger, "extract", {"completed": completed, "total": len(rows),
                                      "elapsed_seconds": time.monotonic() - started})
    del backbone, patches  # Release the frozen backbone before BSF optimization.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.load(path, mmap_mode="r")
