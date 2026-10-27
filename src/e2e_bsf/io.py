"""Global: auditable artifacts. Local: atomic writes, hashes, and safe tensor loads.

Sources: https://docs.python.org/3/library/os.html#os.replace and
https://docs.pytorch.org/docs/stable/notes/serialization.html.
"""

import hashlib  # Bind files to their exact contents.
import json  # Human-readable metadata is the source of truth.
from pathlib import Path  # Use platform-neutral paths.
import subprocess  # Record the current source revision without shell interpolation.
import torch  # Persist tensor-only checkpoints.


def digest(path):
    """Local: streamed SHA-256; global: detect stale inputs without loading large files."""
    with Path(path).open("rb") as handle:  # Never inspect unrelated files.
        return hashlib.file_digest(handle, "sha256").hexdigest()  # Bounded memory.


def write_json(path, value):
    """Local: atomic JSON; global: interruption must not leave plausible partial results."""
    path = Path(path)  # Normalize callers' string/path inputs.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only requested output folders.
    temporary = path.with_suffix(path.suffix + ".tmp")  # Same filesystem for atomic replace.
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)  # Readers see either the previous or complete new document.


def read_json(path):
    """Local: UTF-8 decode; global: shared metadata interpretation."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_checkpoint(path, state):
    """Local: atomic tensor archive; global: permit safe reproducible resume."""
    path = Path(path)  # Checkpoints remain under the selected experiment.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")  # Incomplete writes are never selected as checkpoints.
    torch.save(state, temporary)  # State contains tensors and ordinary primitive containers.
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    """Local: restricted unpickler; global: avoid executable arbitrary checkpoint objects."""
    return torch.load(path, map_location=device, weights_only=True)


def source_revision():
    """Local: exact Git commit; global: tie every experiment to published source."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"
