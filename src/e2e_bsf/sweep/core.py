"""Global: no bypass around the intermediate bottleneck; local: frozen differentiable suffix.

Sources: Transformers DINOv3ViTModel.forward and
https://raw.githubusercontent.com/ApolloResearch/e2e_sae/83b4add4d2652a4e5c6775a0e2b752897c871a87/e2e_sae/losses.py.
"""

import torch
from torch import nn


def region_mse(modified, original, special=5):
    """Per-image raw MSE for all, patch, and special tokens; images remain the sampling unit."""
    squared = (modified.float() - original.detach().float()).square()
    return torch.stack([squared.mean((1, 2)), squared[:, special:].mean((1, 2)),
                        squared[:, :special].mean((1, 2))], dim=-1)


class FrozenSuffix(nn.Module):
    """Execute remaining blocks without disabling input gradients or restoring clean tokens."""

    def __init__(self, layers, norm, probe, position, special=5):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm, self.probe, self.special = norm, probe, special
        for index, value in enumerate(position):
            self.register_buffer(f"position_{index}", value.detach())
        self.position_count = len(position)
        self.requires_grad_(False)
        self.eval()

    def forward(self, value, targets=None):
        """Compare each raw block output before applying DINO's terminal normalization once."""
        if targets is not None and len(targets) != len(self.layers):
            raise ValueError("Downstream targets must match every suffix block")
        position = tuple(getattr(self, f"position_{i}") for i in range(self.position_count))
        errors = []
        for index, layer in enumerate(self.layers):
            value = layer(value, position_embeddings=position or None)
            if targets is not None:
                errors.append(region_mse(value, targets[index], self.special))
        logits = self.probe(self.norm(value)[:, self.special:])
        return logits, torch.stack(errors, 1) if errors else None

    @classmethod
    def from_backbone(cls, backbone, probe, insertion=6):
        """RoPE depends on the fixed image geometry; retain the released evaluation behavior."""
        device = next(backbone.parameters()).device
        with torch.no_grad():
            position = backbone.rope_embeddings(torch.zeros(1, 3, 224, 224, device=device))
        return cls(list(backbone.model.layer[insertion:]), backbone.norm, probe, position,
                   special=1 + backbone.config.num_register_tokens)


@torch.no_grad()
def clean_states(backbone, pixels, insertion=6):
    """Extract the raw insertion state and all subsequent raw states, retaining all tokens."""
    value = backbone.embeddings(pixels)
    position = backbone.rope_embeddings(pixels)
    states = []
    for index, layer in enumerate(backbone.model.layer, start=1):
        value = layer(value, position_embeddings=position)
        if index >= insertion:
            states.append(value)
    return states


def loss_for(cell, kl, local_mse, downstream_mse):
    """Only the explicitly selected terms contribute gradients; no implicit local regularizer."""
    if cell.objective == "mse":
        return local_mse
    if cell.objective == "kl":
        return kl
    if cell.objective == "downstream":
        return kl + cell.beta * downstream_mse
    raise ValueError(f"Unknown objective: {cell.objective}")
