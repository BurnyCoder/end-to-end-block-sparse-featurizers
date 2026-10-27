"""Global: released BSF architecture. Local: an image-aware, differentiable adapter.

Adapted under MIT from Goodfire@219f121ea82d2b19200d1dac918396e6058d7eb9,
bsf/grassmannian.py and bsf/base.py (verbatim references in vendor/goodfire).
One QR result is reused within a forward pass; this preserves upstream mathematics.
"""

import torch  # Dense operations retain gradients through selected signed coefficients.
from torch import nn  # Register trainable charts and gains.


class GrassmannianBSF(nn.Module):
    """Local: orthonormal block charts; global: sparse across blocks, dense within them."""

    def __init__(self, d, n_groups, group_size=3, l0=16):
        """Initialize exactly the upstream random-QR charts and unit per-block gains."""
        super().__init__()  # Establish parameter registration.
        if not 1 <= l0 <= n_groups or not 1 <= group_size <= d:
            raise ValueError("Require 1 <= l0 <= groups and 1 <= block size <= d")
        self.d, self.n_groups, self.group_size, self.l0 = d, n_groups, group_size, l0
        frames, _ = torch.linalg.qr(torch.randn(n_groups, d, group_size))  # Upstream initialization.
        self.B_raw = nn.Parameter(frames)  # QR retraction is performed in each forward.
        self.gamma = nn.Parameter(torch.ones(n_groups))  # Upstream uses unconstrained gains.

    def frames(self):
        """Local: differentiable QR; global: constrain each block to an orthonormal chart."""
        return torch.linalg.qr(self.B_raw.float(), mode="reduced").Q  # QR stays float32.

    def forward(self, x):
        """Local: encode, block-TopK, decode; global: implement the reconstruction bottleneck."""
        frames = self.frames()  # Reuse the same charts for encoder and decoder.
        atoms = frames.transpose(1, 2).reshape(-1, self.d)  # Stack [group,coordinate] rows.
        codes = (x.float() @ atoms.T).reshape(-1, self.n_groups, self.group_size)
        codes = codes * self.gamma[None, :, None]  # One learned gain per block, as released.
        indices = codes.norm(dim=-1).topk(self.l0, dim=-1).indices  # Selection is discrete.
        mask = torch.zeros_like(codes[..., 0]).scatter_(1, indices, 1.0)  # No straight-through gate.
        codes = codes * mask[..., None]  # Selected coordinates retain ordinary gradients.
        return codes.flatten(1) @ atoms, codes  # No reconstruction-error bypass.

    def atoms(self):
        """Local: decoder geometry; global: support rank and intervention diagnostics."""
        return self.frames().transpose(1, 2)  # [groups, coordinates, activation dimension].


def reconstruct(model, normalization, patches):
    """Local: flatten only inside the BSF; global: preserve complete image readouts."""
    x = normalization(patches)  # Fixed training-derived transform.
    x_hat, codes = model(x.flatten(0, 1))  # Every patch uses the same dictionary.
    x_hat = x_hat.reshape_as(x)  # Restore image boundaries before pooling.
    return normalization.inverse(x_hat), codes, (x_hat - x).square().mean()
