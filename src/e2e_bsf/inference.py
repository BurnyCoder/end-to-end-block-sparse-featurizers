"""Global: usable published models. Local: load a standalone safe tensor bundle.

Sources: https://huggingface.co/docs/safetensors/api/torch;
https://huggingface.co/docs/transformers/model_doc/dinov3.
"""

from pathlib import Path
import torch
from safetensors.torch import load_file

from .config import Config
from .extraction import load_backbone, extract_patches
from .featurizer import GrassmannianBSF, reconstruct
from .io import read_json
from .teacher import FrozenProbe, Normalization


class VisionFeaturizer:
    """Local: portable public artifact; global: run new images through the actual intervention."""

    def __init__(self, directory, device="cuda"):
        """Load immutable backbone identity, normalization, probe, and sparse dictionary."""
        directory = Path(directory)
        metadata = read_json(directory / "config.json")
        tensors = load_file(str(directory / "model.safetensors"), device=device)
        self.device = device
        self.featurizer = GrassmannianBSF(**metadata["architecture"]).to(device)
        self.featurizer.load_state_dict({key.removeprefix("bsf."): value for key, value in tensors.items() if key.startswith("bsf.")})
        self.probe = FrozenProbe(tensors["probe.weight"], tensors["probe.bias"]).to(device)
        self.normalization = Normalization(tensors["normalization.mean"], tensors["normalization.scale"]).to(device)
        self.processor, self.backbone = load_backbone(Config(device=device))
        from .config import MODEL_ID, MODEL_REVISION
        if metadata["model_id"] != MODEL_ID or metadata["model_revision"] != MODEL_REVISION:
            raise ValueError("This loader supports the project's exact pinned DINO backbone only")

    @torch.no_grad()
    def predict(self, images):
        """Return original and inserted class probabilities for a list of RGB PIL images."""
        patches = extract_patches(self.processor, self.backbone, images, self.device)
        modified, _, mse = reconstruct(self.featurizer, self.normalization, patches)
        return {"original": self.probe(patches).softmax(-1).cpu(),
                "inserted": self.probe(modified).softmax(-1).cpu(), "normalized_mse": float(mse)}


class IntermediateVisionFeaturizer(VisionFeaturizer):
    """Load a block-6 safe bundle with the shared loader, then execute its frozen suffix."""

    def __init__(self, directory, device="cuda"):
        """Require the complete-state intervention metadata; never infer a different insertion site."""
        metadata = read_json(Path(directory) / "config.json")
        if metadata.get("insertion_block") != 6 or metadata.get("sequence_scope") != "all_201":
            raise ValueError("Expected a registered block-6 all-token sweep bundle")
        super().__init__(directory, device)
        from .sweep.core import FrozenSuffix
        self.suffix = FrozenSuffix.from_backbone(self.backbone, self.probe, 6)

    @torch.no_grad()
    def predict(self, images):
        """Use reconstructed special tokens as well as patches; return original and inserted outputs."""
        from .sweep.core import clean_states
        pixels = self.processor(images=images, return_tensors="pt").to(self.device)["pixel_values"]
        states = clean_states(self.backbone, pixels, 6)
        modified, _, mse = reconstruct(self.featurizer, self.normalization, states[0])
        logits, _ = self.suffix(modified)
        original = self.probe(self.backbone.norm(states[-1])[:, 5:])
        return {"original": original.softmax(-1).cpu(), "inserted": logits.softmax(-1).cpu(),
                "normalized_mse": float(mse)}
