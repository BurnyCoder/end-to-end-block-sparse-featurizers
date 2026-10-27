# Working instructions

This repository implements the human-specified plan in [docs/preregistration.md](docs/preregistration.md).
The separate block-6 follow-up is specified in [docs/downstream-preregistration.md](docs/downstream-preregistration.md).
It uses one seed, all-token replacement and disclosed test reuse; never apply its assumptions retroactively to the original study.
Keep changes small, modular, sourced, and tested before expensive runs. Use `uv`, Python 3.12,
the checkout's `.venv`, and ignored `.env` configuration. Work only in this checkout; use public
source links for the earlier workbench. Do not reuse pretrained rabbit BSFs in scratch conditions.

## Scientific invariants

- The teacher is frozen DINOv3 plus a frozen **patch-only** ten-class probe, not a native DINO classifier.
- Primary loss is teacher-to-inserted forward KL; no hidden MSE or reconstruction-error bypass.
- Split before fitting. Fit normalization/probe on training only, choose hyperparameters on development,
  and evaluate reserved test images after the frozen study schedule completes.
- Preserve whole-image boundaries. Pair seeds/data order, match scalar capacity, report actual parameters.
- Publish incomplete/negative results honestly. Never run extra seeds to manufacture significance.
- Do not commit credentials, original images, DINO weights, raw activation caches, or executable checkpoints.
- Keep source revisions, licenses, full metrics and resume state. Do not truncate any introduced LLM logs.

## Workflow

Run `uv run pytest`, then the real-image smoke pipeline. Publish code and preregistration before
running pretrained models. Use meaningful feature branches, PRs and merges. Generate results, the
paper and the static explorer from the same measured artifacts. Make one final correctness review.
The README is the entry point; methodology and results details belong in linked docs/reports.

## Commentary and sources

New modules/functions need local and global context with source links. Explain non-obvious operations
inline. Retain original notices for reused code. Log timestamped events to terminal and files, sanitize
credentials, and record the complete arrays in artifacts rather than relying on truncated tensor reprs.
