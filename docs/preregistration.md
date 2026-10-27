# Preregistered study: output-KL block-sparse vision featurizers

Registered 2026-10-27, before any real model run in this repository. Author: Libor Burian.
This is a bounded adaptation, not a replication of the full ImageNet study. Configuration and
data manifests become immutable before confirmatory runs. Runtime feasibility is not yet measured.

## Question and falsifiable hypothesis

Does training a released Grassmannian BSF from scratch with image-output forward KL preserve a
frozen DINOv3/Imagenette classifier better than training the identical BSF with local MSE?
Primary endpoint: arithmetic mean per-image `KL(original || inserted)` in nats on the reserved
test partition, averaged over paired initialization seeds. Primary contrast is local minus KL;
positive values favor KL. A nonpositive effect contradicts the directional hypothesis. A 95%
interval crossing zero is inconclusive. Ten percent relative reduction is the minimum practical
effect for pilot power analysis; it is not a guaranteed outcome or a reason to continue sampling.

## Fixed method

- DINOv3 ViT-B/16, revision `5931719e67bbdb9737e363e781fb0c67687896bc`, float32,
  pinned processor at 224 pixels, final `last_hidden_state[:,5:]`, 196 patches of width 768.
- Imagenette 320px, clean labels. Official training becomes group-aware 80/20 train/development
  using the first `StratifiedGroupKFold(5, shuffle=True, random_state=42)` fold. Official validation
  is the test partition. Exact SHA-256 and 64-bit perceptual hashes (Hamming distance <=4) form
  transitive duplicate groups before splitting; training images sharing a test group are excluded.
  This detects common near duplicates, not all possible semantic overlap. Existing backbone
  pretraining contamination cannot be ruled out; the holdout is for this featurizer study.
- Before the full pilot/confirmation, exclude all reserved-test duplicate groups previewed in the
  separate engineering smoke run. This pre-confirmation clarification keeps the final evaluation
  unseen after smoke diagnostics. The exclusion uses only image identities, never prediction quality.
- Pool as `abs(patches).amax(patch_axis)` and fit multinomial logistic regression with
  C in {0.01,0.1,1,10}, selected by development CE. Freeze the selected training-only fit.
  The clean teacher must exceed 20% development accuracy, twice the uniform-class chance level.
- Subtract training per-position mean and divide by one training RMS scale, restoring both before
  the head. No per-image or held-out normalization fitting; no CLS bypass.
- Goodfire implementation revision `219f121ea82d2b19200d1dac918396e6058d7eb9`: tied
  signed QR charts, unconstrained per-block gains initialized to one, hard per-patch block TopK.
  Main G=4096, b=3, k=16. Scalar G=12288, b=1, k=48. No AuxK, decoder residual bypass, or pretrained BSF.
- Five conditions: block MSE, block KL, block KL+0.1 normalized MSE, scalar MSE, scalar KL.
  Scalar is a tied signed Grassmannian ablation, not a ReLU SAE. Per-block gain counts differ;
  total and active scalar coordinates match, and actual parameter counts are reported.
- Adam, clip norm 1, initial LR {1e-4,3e-4} pilot grid, cosine decay to 10%, effective batch 32,
  microbatch 8 halved on OOM. Every incomplete effective batch is weighted by its actual image count.

## Budget and pilot decision rule

The full command has a 24-hour wall-time cap: preparation/pilot <=2h, optimization <=20h,
final evaluation <=2h. Smoke verification is a separate labeled engineering check. Pilot seeds
0,1,2 are excluded from confirmation. Every condition receives equal pilot exposure at each LR:
one epoch on a fixed 510-image training slice (51 per class), evaluated on a fixed 250-image
development slice (25 per class). Both slices are selected with seed 42 from their assigned splits;
normalization and the probe remain fitted on the full training partition. These short pilots only
estimate feasibility/variance; their variance may differ from longer confirmation runs.
Use pilot development KL to choose the initial LR separately for each condition, including MSE.
Compute paired pilot variance for the primary contrast and a two-sided paired-t power calculation
at alpha .05, power .8 and 10% of pilot local KL. Use `ceil(required)`, bounded to 5..10 seeds;
degenerate/insufficient pilot evidence uses 10 and an exploratory label.

Confirmatory seeds are consecutive from 100. Forecast total time by summing each condition's slowest
pilot image throughput including validation, multiplied by 1.25 for overhead. Fix the largest common epoch
count <=20 fitting the optimization allocation. If fewer than one epoch fits or pilot cannot finish,
record a feasibility failure and do not quietly shrink the architecture. If required replication
exceeds budget/cap, publish exploratory results and the estimated shortfall. Reaching a deadline
saves resumable state; incomplete runs are not counted as completed confirmatory pairs.

**Allocation amendment, 2026-10-27, during pilot and before schedule freeze/test evaluation:** the
initial conservative implementation applied the single slowest architecture's cost to all conditions.
Measured scalar QR runtime is roughly three times the block runtime. The revised forecast sums each
condition's own slowest pilot cost, retaining the same 25% allowance, common epoch count, replication
rule and 20-hour optimization cap. This is a timing-only correction; test outcomes remain unseen.

## Analysis and safeguards

Final checkpoints use the common fixed exposure, not the best test epoch. Complete all scheduled
training before test evaluation; pilot uses development only. Report per-seed differences and
Student 95% intervals, as well as 2000-replicate image-group bootstrap intervals. These characterize
different uncertainty sources; image bootstrap cannot replace independent training seeds. Show all
conditions and missing pairs. Extra analyses are labeled exploratory. Do not extend the schedule
because a p-value or interval is disappointing.

Secondary: CE increase, clean/inserted accuracy, agreement, normalized MSE, dead blocks, active
blocks/coordinates, conditional block-code stable rank, normalized-map total variation/Dirichlet
energy, throughput, elapsed time, and VRAM. Identity, positional-mean replacement and untrained
BSF are controls. Select eight high-energy blocks on development; show fixed ranked/random examples
and block removal, energy-matched random removal and predetermined within/across-class swaps.
These are causal perturbations of the composite classifier, not evidence of semantic concepts.

## Reporting limitations and references

Ten-way softmax supplies at most nine independent logit contrasts per image. Small KL can coexist
with lost geometry, transfer capability, or semantics. Final-layer cached training only differentiates
through the frozen pooling/head suffix. It does not demonstrate preservation of every DINO function.
Hard TopK supplies gradients only to selected coefficients. This implementation uses dense kernels
and makes no inference-speed claim. Max-abs pooling supplies gradients only to maxima.

Sources and source/code discrepancies: [literature review](literature-review.md). Reports must state
post-hoc changes, failures, all measurements, and uncertainty without promising significance.
