# Plan completion audit

Rechecked on 2026-10-27 after publication, against the human-specified plan and the
[preregistration](preregistration.md). Reviewed implementation and result release:
[`aea5c33`](https://github.com/BurnyCoder/end-to-end-block-sparse-featurizers/commit/aea5c33697050af99e4f5a8dec6063e406fa6f09).
The trained checkpoints remain bound to `c2adebc158be34482fedeb2efd337ca84171ac89`.

**Delivery is complete under the plan's exploratory fallback.** The 80% power target was not met;
the frozen allocation was five paired seeds and one epoch. No extra training is required or authorized
by this audit. An inconclusive hypothesis test is an empirical outcome, not an unfinished pipeline.

## Requirement-to-evidence checklist

| Plan requirement | Status and checked evidence |
|---|---|
| Python 3.12, uv, local environment, `.env`, committed lockfile and CUDA | Verified Python 3.12.13 in this checkout's `.venv`, ignored `.env`, tracked `uv.lock`, Torch 2.14.1+cu130 and available RTX 5070 Laptop GPU. [Installation](../README.md#method-and-use), [configuration](../.env.example). |
| Review and reuse the public workbench; preserve upstream identities and attribution | [Source review](literature-review.md), [reuse notices](../vendor/NOTICE.md) and Goodfire's retained license identify the reused source/recipes and paper/code differences. Goodfire and DINO revisions are bound in every checkpoint. No other local project was needed. |
| Released tied Grassmannian implementation, signed TopK, differentiable QR and learned gains | [Adapter](../src/e2e_bsf/featurizer.py) and [upstream output/gradient parity test](../tests/test_upstream_parity.py). The documented QR reuse preserves the released mathematics. |
| Imagenette-320, pinned 224-pixel processor, duplicate audit and stratified grouped split | [Data phase](../src/e2e_bsf/data.py); all 13,394 source images were hashed before fitting. Final counts are 7,575 train, 1,894 development and 3,875 test. All 50 smoke-preview test groups were excluded before the full study. Manifests and all source hashes accompany every public model bundle. |
| Float32 final patch cache, without CLS/register tokens or lost image boundaries | Verified disk-backed shape `[13344,196,768]` and dtype `float32`. [Extraction](../src/e2e_bsf/extraction.py) strips the five special tokens; [reconstruction](../src/e2e_bsf/featurizer.py) restores image grouping before pooling. |
| One frozen max-absolute pooled readout, C selection on development | All four declared C values were fitted on training images only. Selected C=10; no development refit. [Teacher phase](../src/e2e_bsf/teacher.py), public `teacher.json` and exact buffer comparisons in the [audit](../reports/study/final_audit.json). |
| Training-only positional means/RMS, inverted before the classifier | [Normalization implementation](../src/e2e_bsf/teacher.py), inversion tests and exact checkpoint-buffer comparisons. |
| Forward KL, temperature one, float32 logs, detached teacher and connected replacement | [Loss](../src/e2e_bsf/objectives.py), [training accumulation](../src/e2e_bsf/training.py) and [contract tests](../tests/test_core.py). Only BSF parameters enter Adam; reconstructed patches completely replace the original patches. |
| Cached versus complete image-to-output loss/gradient agreement | [Real-image verification](../reports/smoke/verification.json): zero measured patch and chart-gradient differences on the fixed smoke training pair. Identity KL was zero; zeroed patches changed outputs. This check used a small dictionary, not every full-study image. |
| All five from-scratch conditions and capacity-matched scalar control | All 25 main checkpoints and [individual reports](../reports/study/summary.md) exist. Main dimensions and actual parameter counts match the plan; scalar models are explicitly labeled tied signed featurizers. |
| Identity, mean replacement and untrained controls | All three were evaluated on the same 3,875 test images; [control scores](../reports/study/summary.md#controls) and complete local per-image arrays are retained. The untrained control uses the block architecture and seed 100. |
| Paired initialization/order, Adam, clipping, decay, microbatch search and remainder handling | [Training source](../src/e2e_bsf/training.py) uses identical architecture/seed initialization and deterministic image order across objectives, clip norm 1, cosine decay to 10%, microbatch 8 with OOM halving, and accumulation to 32. Each main run has 237 updates and 7,575 images, including its 23-image final batch. |
| Equal LR pilots, separate seeds, frozen replication and exposure | [All 30 pilots](../reports/study/pilot.json): five conditions x two LRs x seeds 0/1/2. [Frozen schedule](../reports/study/study_schedule.json): seeds 100-104, one epoch and all five conditions. The post-pilot allocation amendment is disclosed. |
| Original 24-hour deadline and stage budgets | Preparation and all pilots finished after 90.28 minutes. Allocation recovery delayed schedule freeze until 140.59 minutes after the original start; this did not reset the deadline. Main training totaled 9.326 hours. Final evaluations and controls completed before the original deadline. The [timing record](../reports/study/findings.md#allocation-deviations-and-interpretation-limits) and frozen schedule retain the evidence. |
| Primary paired effects, 95% intervals and image-level uncertainty | Every seed effect and per-model image-group bootstrap interval is retained in [results](../reports/study/results.json) and the paper. The audit independently recomputed KL, accuracy and the paired Student interval. Patches are not independent replicates. |
| Accuracy/agreement, reconstruction, activation/dead blocks, rank, coherence and costs | All named secondary metrics are in every run report and the machine-readable result file. Rank is explicitly the conditional covariance stable-rank proxy. Peak VRAM is PyTorch-allocated memory, not total device usage. |
| Development-selected blocks and fixed removal/swap interventions | [Diagnostics](../src/e2e_bsf/diagnostics.py) and [findings](../reports/study/findings.md#interventions-and-fixed-examples). Five seed-100 dictionaries, six fixed development examples and 20 fixed test intervention targets per condition. Random removal is approximately energy matched; inactive selected blocks are retained and disclosed. |
| CLI phases and short modular orchestration | `uv run e2e-bsf --help` exposes prepare/train/evaluate/report/run plus verify/publish; [orchestrator](../src/e2e_bsf/pipeline.py) imports separate phases. The complete real-image smoke and full-study workflows finished. |
| Bound, resumable checkpoints; streamed latent statistics | All local checkpoints contain BSF, readout, normalization, configuration, revisions, hashes, optimizer/scheduler, RNG and cursor state. [Resume test](../tests/test_training.py) is bitwise exact on its CPU fixture. Evaluation stores block moments and image-level predictions, not a dataset-sized dense latent array. Public safe-tensor bundles are inference exports; optimizer/RNG resume state stays local. |
| Focused tests, fixed-batch optimization and final correctness review | All **25 tests passed again**. Real-image fixed-batch KL decreased from 11.633245 to 0.027983. The independent audit was rerun over all 25 saved checkpoints and predictions without training or selecting new outcomes. |
| Repository/preregistration before the first model run; functional PRs and merges | PR #1 merged at 01:31:21 Europe/Berlin; the first smoke prepare command was logged at 01:31:39. Subsequent changes and final publication used [merged PRs #2-8](https://github.com/BurnyCoder/end-to-end-block-sparse-featurizers/pulls?q=is%3Apr+is%3Amerged). |
| Timestamped logs, complete metrics/predictions and credential exclusion | Logs and original resumable artifacts remain in ignored `outputs/`. Public run bundles retain complete test predictions, histories and provenance. The explicit [upload allowlist](../src/e2e_bsf/publication.py) excludes photos, gated DINO weights and credentials. No LLM API calls were introduced. |
| README, individual/consolidated reports, figures and Libor Burian paper | [README](../README.md), [experiment ledger](../reports/experiments.md), 25 per-run reports, four full-study figures and the seven-page [paper](../reports/study/paper.pdf) are published. The requested AI statement appears in the paper only. |
| Interactive GitHub Pages explorer and public model collection | The [live explorer](https://burnycoder.github.io/end-to-end-block-sparse-featurizers/) serves the same 25-run JSON and PDF as the repository. All 25 repositories are present in the [Hugging Face collection](https://huggingface.co/collections/BurnyCoder/end-to-end-bsfs-on-dinov3-full-6ae0a9d519d3d32881f784b8). Prior [publication verification](../reports/study/publication_verification.json) checks every uploaded file and two downloaded real-image inference roundtrips; collection membership and the live artifacts were checked again. |

## Qualifications, not hidden completion claims

- Pilot-estimated power was **77.47%, below 80%**. Six seeds were requested by the calculation;
  five fit the conservative forecast. One epoch is the frozen budget adaptation, not evidence of convergence.
- The primary mean improvement is positive, but its seed interval spans zero. The primary hypothesis
  remains inconclusive. The secondary scalar contrast and intervention observations do not change that claim.
- Removal controls use approximate energy matching, and interventions use one seed and 20 targets per
  condition. They do not demonstrate semantic correspondence or preservation of all DINOv3 information.
- Public models include BSF/readout/normalization weights and reproducibility metadata. They do not include
  gated DINO weights, original images or the local optimizer/RNG resume payloads.

This recheck found stale present-tense wording in the historical pilot report and experiment ledger.
Those descriptions were corrected and linked to the completed findings. No measured result, model,
split, seed, learning rate, exposure or original deadline was changed. The completed study's follow-up
schedule remains stopped.
