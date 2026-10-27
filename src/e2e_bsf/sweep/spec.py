"""Global: explicit sweep identity; local: standard-library TOML and immutable cells.

Sources: https://docs.python.org/3/library/tomllib.html and docs/downstream-preregistration.md.
"""

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Cell:
    """One objective/sparsity operating point, independent of runtime allocation."""

    k: int
    objective: str
    beta: float = 0.

    @property
    def key(self):
        """Use unambiguous filesystem-safe names without rounding beta values."""
        return f"k{self.k:03d}_{self.objective}" + (f"_b{self.beta:g}" if self.objective == "downstream" else "")

    @property
    def label(self):
        """Keep raw-MSE coefficient units visible in every curve legend."""
        return {"mse": "Local MSE", "kl": "Output KL", "downstream": f"KL + {self.beta:g} downstream MSE"}[self.objective]


@dataclass(frozen=True)
class SweepSpec:
    """The human-selected grid; runtime paths may be configured through ignored .env."""

    root: str = "outputs/sweeps/downstream-v1"
    source_root: str = "outputs/study"
    cache_dir: str = "D:/AI/end-to-end-block-sparse-featurizers/downstream-v1"
    report_dir: str = "reports/downstream-sweep"
    device: str = "cuda"
    insertion_block: int = 6
    groups: int = 4096
    block_size: int = 3
    active_blocks: tuple = (4, 8, 16, 32, 64)
    downstream_weights: tuple = (1e-4, 1e-3, 1e-2)
    learning_rates: tuple = (1e-4, 3e-4)
    seed: int = 200
    pilot_seed: int = 900
    microbatch: int = 8
    effective_batch: int = 32
    max_epochs: int = 20
    max_hours: float = 24.
    preparation_hours: float = 2.
    evaluation_hours: float = 2.
    pilot_train_images: int = 510
    pilot_dev_images: int = 250
    bootstrap_replicates: int = 2000

    def cells(self):
        """Five objective curves crossed with every sparsity, in a fixed registered order."""
        return [cell for k in self.active_blocks for cell in
                [Cell(k, "mse"), Cell(k, "kl"), *[Cell(k, "downstream", b) for b in self.downstream_weights]]]

    def to_dict(self):
        """Plain JSON-compatible configuration, excluding credential environment variables."""
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    @classmethod
    def load(cls, path):
        """Reject unknown settings rather than silently running a misspelled experiment."""
        values = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        if set(values) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown sweep specification field")
        for name in ("active_blocks", "downstream_weights", "learning_rates"):
            if name in values:
                values[name] = tuple(values[name])
        for name in ("root", "source_root", "cache_dir", "device"):
            override = os.getenv(f"E2E_SWEEP_{name.upper()}")
            if override:
                values[name] = override
        result = cls(**values)
        if result.insertion_block != 6 or result.max_hours != 24:
            raise ValueError("This registration fixes block 6 and a 24-hour wall-time budget")
        if min(result.groups, result.block_size, result.microbatch, result.effective_batch, result.max_epochs) < 1:
            raise ValueError("Positive sizes required")
        if result.microbatch > result.effective_batch or max(result.active_blocks) > result.groups:
            raise ValueError("Invalid batch or sparsity budget")
        if sorted(set(result.active_blocks)) != list(result.active_blocks) or min(result.active_blocks) < 1:
            raise ValueError("Sparsity values must be positive and strictly increasing")
        if any(b <= 0 for b in result.downstream_weights) or any(lr <= 0 for lr in result.learning_rates):
            raise ValueError("Loss weights and learning rates must be positive")
        if Path(result.root).resolve() == Path(result.source_root).resolve():
            raise ValueError("The completed study cannot be the sweep output directory")
        return result
