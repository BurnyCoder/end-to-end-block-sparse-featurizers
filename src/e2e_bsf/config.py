"""Global: immutable experiment identity. Local: validated environment/CLI settings.

Sources: https://docs.python.org/3/library/dataclasses.html and
https://github.com/goodfire-ai/block-sparse-featurizer/tree/219f121ea82d2b19200d1dac918396e6058d7eb9.
"""

from dataclasses import asdict, dataclass  # Keep serialized settings explicit.
import os  # Read runtime settings without serializing credentials.
from pathlib import Path  # Resolve paths relative to this checkout.

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # Same backbone as Goodfire.
MODEL_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"  # Pin weights and processor.
BSF_REVISION = "219f121ea82d2b19200d1dac918396e6058d7eb9"  # Pin architecture provenance.
CONDITIONS = ("block_mse", "block_kl", "block_kl_mse", "scalar_mse", "scalar_kl")


@dataclass(frozen=True)
class Config:
    """Local: one reproducible run recipe; global: shared contracts across phases."""

    root: str = "outputs/study"  # Separate experiments from the source tree.
    profile: str = "full"  # Smoke is an explicitly separate feasibility experiment.
    device: str = "cuda"  # Fail clearly if requested CUDA is unavailable.
    microbatch: int = 8  # Initial memory probe from the preregistration.
    effective_batch: int = 32  # Accumulate complete image losses to this count.
    groups: int = 4096  # Three-dimensional blocks in the primary condition.
    block_size: int = 3  # Published successful DINO block width.
    active: int = 16  # Active blocks, not active individual coordinates.
    epochs: int = 20  # The pilot can reduce this uniformly before confirmation.
    lr: float = 1e-4  # Pilot candidates are 1e-4 and 3e-4.
    seed: int = 42  # Split seed; training seeds are explicit train arguments.
    max_hours: float = 24.0  # Hard supervisor wall-time budget.
    train_limit: int = 0  # Zero means all training images; pilots use a fixed stratified slice.
    dev_limit: int = 0  # Zero means all development images; pilots use a bounded slice.

    @classmethod
    def from_env(cls, profile="full", root=None):
        """Local: whitelist environment values; global: keep secrets out of configs."""
        base = Path(os.getenv("E2E_OUTPUT_DIR", "outputs"))  # Dedicated output root.
        values = dict(root=str(root or base / ("smoke" if profile == "smoke" else "study")),
                      profile=profile, device=os.getenv("E2E_DEVICE", "cuda"),
                      microbatch=int(os.getenv("E2E_MICROBATCH", "8")),
                      effective_batch=int(os.getenv("E2E_EFFECTIVE_BATCH", "32")),
                      seed=int(os.getenv("E2E_SEED", "42")),
                      max_hours=float(os.getenv("E2E_MAX_HOURS", "24")))
        if profile == "smoke":  # A separate, cheap real-image integration recipe.
            values.update(groups=32, active=4, epochs=3, microbatch=2, effective_batch=8)
        return cls(**values)  # Dataclass serialization never includes HF_TOKEN.

    def validate(self):
        """Local: bound resource settings; global: reject invalid scientific recipes."""
        if min(self.groups, self.active, self.microbatch, self.effective_batch, self.epochs) < 1:
            raise ValueError("Counts must be positive")  # No empty training loops.
        if self.active > self.groups or self.block_size > 768:
            raise ValueError("Invalid block budget")  # QR and TopK require these bounds.
        if self.effective_batch < self.microbatch:
            raise ValueError("microbatch cannot exceed effective_batch")
        return self  # Permit validation at construction boundaries.

    def to_dict(self):
        """Local: plain serializable settings; global: checkpoint/config parity."""
        return asdict(self)  # No ad-hoc hidden defaults.
