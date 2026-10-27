# End-to-End Block-Sparse Featurizers for Vision

Train a **Grassmannian block-sparse featurizer from scratch** to preserve the class probabilities
of a frozen DINOv3 + Imagenette classifier. The pipeline extracts final patch activations, replaces
them with BSF reconstructions, and minimizes `KL(original || reconstructed)`. Only the BSF updates.
It combines [Goodfire's BSFs](https://arxiv.org/abs/2606.25234) with
[Apollo's end-to-end objective](https://arxiv.org/abs/2405.12241).

![Data flow and frozen/trainable components](docs/assets/architecture.svg)

**Results status:** implementation and preregistration are being published before real model runs.
No experimental advantage or significance is claimed. Measured tables and examples will be generated
from the completed run artifacts, including negative results and incomplete experiments.

| Condition | Test KL | Accuracy | Reconstruction | Status |
|---|---:|---:|---:|---|
| Block MSE baseline | — | — | — | Not run |
| Block KL from scratch | — | — | — | Not run |
| Block KL + 0.1 MSE | — | — | — | Not run |
| Scalar MSE / scalar KL | — | — | — | Not run |
| Identity / mean / untrained controls | — | — | — | Not run |

## Method and use

The teacher is a **new frozen ten-class patch classifier on DINOv3**, not a native DINO class head.
The selected Grassmannian implementation uses signed codes, QR-orthonormal blocks and block TopK.
Main settings are 4096 blocks × 3 coordinates, 16 active blocks. Scalar controls match total width
and 48 active coordinates. Local-MSE and KL runs share initializations and image order.
The complete scientific specification is in [the preregistration](docs/preregistration.md).

Requirements: Git, [uv](https://docs.astral.sh/uv/), Python 3.12 (managed by uv), an NVIDIA GPU with a
CUDA-13-compatible driver for the Windows lockfile, approximately 15 GB workspace storage, and
authorized access to [DINOv3](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
The tested target is an 8 GB RTX 5070 Laptop GPU; actual measurements are reported after execution.

```powershell
git clone https://github.com/BurnyCoder/end-to-end-block-sparse-featurizers.git
cd end-to-end-block-sparse-featurizers
uv sync --frozen
Copy-Item .env.example .env
uv run hf auth login
uv run pytest
uv run e2e-bsf run --profile smoke
uv run e2e-bsf run --profile full
```

Accept the model's access terms through its Hugging Face page. Existing `hf` login is supported;
an optional `HF_TOKEN` belongs only in ignored `.env`. Never paste credentials into commands/logs.
On Linux/macOS use `cp .env.example .env`; set `E2E_DEVICE=cpu` only for tests/small diagnostics.
Check the installed Torch build before using a different GPU/platform.

Individual phases use the same artifacts:

```powershell
uv run e2e-bsf prepare --profile smoke
uv run e2e-bsf train --profile smoke --condition block_kl --seed 100
uv run e2e-bsf evaluate --profile smoke
uv run e2e-bsf report --profile smoke
```

Images are hashed/grouped before splitting. The training split fits normalization and classifier;
development chooses the classifier and pilot settings. Extraction creates an approximately 8 GB
float32 full-dataset cache. Subsequent BSF training reads image batches from that cache and does
not rerun DINO. Predictions/code moments are streamed; complete dense latent datasets are not saved.
See [artifact contracts](docs/methodology.md) for cache/checkpoint details.

## Features and technologies

- PyTorch/CUDA, Transformers/DINOv3, torchvision/Imagenette, scikit-learn probe and grouped splits.
- Five matched training conditions, deterministic resume, partial-batch handling and memory backoff.
- Paired-seed inference, clustered image bootstrap, identity/mean/random controls and intervention plots.
- Timestamped sanitized terminal/file logs; complete predictions and machine-readable provenance.
- Generated Markdown reports, a PDF paper, and a static interactive GitHub Pages results explorer.
- Hugging Face checkpoint/model-card collection; original photos and gated backbone weights stay external.

Source revisions, reuse, and paper/code discrepancies are listed in the
[literature review](docs/literature-review.md). Full compute is bounded to 24 hours, with a separate
small engineering smoke run. No LLM API or paid cloud service is needed.

## Interpretation limits

Lower KL supports fidelity to this frozen ten-class readout. It does **not** establish that all
DINO information or visual concept manifolds survive. Images may overlap unknown backbone pretraining
data; this study holds out data for the probe/featurizer, not for original DINO training. The BSF uses
dense matrix operations, so block-sparse codes do not imply faster inference.
