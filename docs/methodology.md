# Methodology and artifact contract

The prespecified scientific method and exact settings live in [preregistration](preregistration.md),
and source evidence in [literature review](literature-review.md). This page describes execution.

`prepare` downloads/hashes images, assigns groups and splits, extracts float32 DINO patches into a
disk-backed array, fits the clean probe and train-only normalization. Each cache row remains a full
image. Extraction progress is atomic and resumable; incompatible manifests cannot reuse a cache.

`train` initializes only a fresh BSF, reconstructs complete images' patches, restores their original
units and computes either local MSE, output KL, or the named mixed objective. Backbone weights and
the selected classifier never update. Effective batches include the final remainder. Checkpoints
contain model/optimizer/scheduler state, torch CPU/CUDA RNG states, deterministic data-order rule,
epoch/offset, normalization, probe, source/model revisions, and manifest/teacher hashes.

`evaluate` streams predictions and code moments. All per-image logits, errors and labels are retained
numerically; full dense codes are never concatenated. Stable rank is `trace(Cov(z))/lambda_max(Cov(z))`
conditional on a block being active. Singleton/constant blocks are excluded from the rank average
and the eligible count is reported. Map diagnostics use each image's eight highest-energy block maps,
normalized to unit spatial L2 norm before grid differences.

`report` consumes measured files to build reports, figures, a paper and a static results explorer.
It must not replace missing data with example numbers. The public repository holds reports and
small aggregate evidence; ignored `outputs/` contains input images, caches, predictions and model
artifacts. Publication exports a curated bundle to Hugging Face; it does not upload DINO weights,
photos or credentials. No LLM API is required by this pipeline.
