# Completed bounded vision study

The primary result is **inconclusive**. Block KL training reduced average held-out forward KL from
0.0375719 to 0.0350260 nats/image, a 6.78% reduction. The paired local-minus-KL effect is 0.00254596,
with a 95% Student interval of **[-0.00325909, 0.00835100]** across five training seeds. Three seeds
favored KL and two favored local MSE. The positive point estimate does not establish the directional
hypothesis because the interval includes zero. No additional runs were added to change this result.

The complete [condition table, individual seed effects, figures and run reports](summary.md) and
[machine-readable measurements](results.json) are the numerical source. The [paper](paper.pdf) includes
per-seed image-group bootstrap intervals. These condition on each fitted model; they do not replace
independent training-seed uncertainty. See the [prespecified analysis](../../docs/preregistration.md)
and the [SciPy paired-interval definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html).

## What completed

All 30 pilots and all 25 main runs completed. Main seeds were 100–104, with one epoch per condition,
7,575 training images per run, an effective batch of 32 and microbatches of eight. Each run made 237
optimizer updates, including the final 23-image batch. Every final model was evaluated on the same
3,875 test images after all scheduled training completed. Development contained 1,894 images.
The [frozen schedule](study_schedule.json), [completion status](study_status.json) and
[independent artifact audit](final_audit.json) preserve the allocation, hashes and source identities.

The selected clean classifier used C=10, development cross-entropy 0.0376775 and development accuracy
98.9968%. The clean test accuracy was 99.0452%. This classifier and its training-only normalization
were identical in every main checkpoint. The architecture and objective follow the
[Goodfire implementation](https://github.com/goodfire-ai/block-sparse-featurizer/tree/219f121ea82d2b19200d1dac918396e6058d7eb9)
and [Apollo forward-KL implementation](https://github.com/ApolloResearch/e2e_sae/blob/83b4add4d2652a4e5c6775a0e2b752897c871a87/e2e_sae/losses.py),
with the documented [paper/code differences](../../docs/literature-review.md).

## Positive and negative findings

- **Readout fidelity versus reconstruction:** block KL had lower average output divergence, but
  normalized patch MSE rose from 0.54515 to 0.78728 (44.41%). Its mean classification accuracy was
  99.37%, compared with 98.49% for block MSE. Accuracy was a secondary outcome, not the training loss.
- **Mixed objective:** adding 0.1 normalized MSE yielded mean KL 0.0334491 and NMSE 0.76812. It retained
  slightly more reconstruction fidelity than pure KL, while remaining substantially worse than local
  MSE on reconstruction. These are descriptive comparisons, not additional primary hypothesis tests.
- **Scalar comparison:** scalar KL reduced mean divergence from 0.254093 to 0.0309295 and raised mean
  accuracy from 90.74% to 99.05%. The paired scalar effect is 0.223164 nats/image, with a descriptive
  95% seed interval [0.206989, 0.239339]. NMSE increased from 0.66437 to 0.79378. This secondary interval
  is not multiplicity-adjusted. It does not establish that block structure is unnecessary in general.
- **Geometry:** block KL's normalized-map total variation and Dirichlet energy were higher than block
  MSE's (8.678 versus 7.090, and 1.058 versus 0.657). Conditional stable rank was 1.895 versus 1.797.
  All trained conditions used their specified active-coordinate budgets, and no block was dead across
  the entire test set. These measurements do not demonstrate semantic manifold recovery.
- **Controls:** identity insertion had KL zero; mean replacement had KL 3.71488; the untrained block
  featurizer had KL 0.734338. They establish reference behavior for this readout. The identity control's
  accuracy is the clean classifier's accuracy, not an assumption of perfect classification.

## Interventions and fixed examples

Eight blocks were selected using development energy. The plots show fixed development examples;
the explorer exposes all six retained examples and eight selected blocks for seed 100 in each condition.
An empty map means the selected block is inactive on that image. Examples and block sets were not
replaced to make maps look more interpretable.

Removal and predetermined same-class/other-class swaps used 20 test targets per condition, with only
seed 100. These are exploratory interventions on the measured classifier, not replicated semantic
causal effects. The random removal control selects among the eight nearest-energy active alternatives;
it is approximate matching, not exact matching. For block KL, mixed block KL, and scalar KL respectively,
the selected block had zero energy on 3, 4, and 4 of the 20 targets. Such targets were retained. Thus the
selected and random intervention can differ in whether a nonzero contribution is removed.

For block MSE, selected removal raised mean KL by 0.0393335, while same-class swaps lowered it by
0.0114726. For block KL, selected removal lowered mean KL by 0.0006901. Signs differ across conditions;
removing a block is not guaranteed to harm this readout. Complete donor identities, energies, baseline
KL and modified KL are in `diagnostics` in [results.json](results.json). No semantic correspondence
between independently learned dictionaries is assumed.

## Allocation, deviations and interpretation limits

The [pilot report](pilot.md) records every LR trial and the allocation failure. The pilot-based power
calculation requested six seeds. Its conservative forecast required 22.24 hours for six seeds versus
18.53 hours for five. A disclosed post-pilot amendment implemented the human-specified exploratory
fallback before any main training or final test evaluation. Estimated power at five was 77.47%, not
the target 80%, and that estimate itself comes from only three short pilot pairs.

All main runs used the same one-epoch exposure. LR was selected by the same development-KL search
for each condition: 3e-4 except scalar MSE at 1e-4. Consequently, objective comparisons describe each
objective with its selected LR, not an isolated objective change at one identical LR. The narrow LR
grid and short training cannot establish convergence or an architecture's best achievable result.
Faster measured execution was not used to add seeds or epochs after freezing the schedule.

Training totaled 33,574.375 seconds (9.326 hours); the largest PyTorch-allocated peak was 733,257,216
bytes (699.3 MiB), excluding other applications and CUDA allocations outside PyTorch. The study began
at 01:02:10 UTC, all training ended at 12:42:54 UTC, final evaluations and controls ended at 12:54:34 UTC,
and the first complete report finished at 13:01:16 UTC on 2026-10-27. The original hard deadline was
2026-10-28 01:02:10 UTC. Reporting, visual inspection and publication followed without changing the models.

Preserving this frozen ten-class readout does not establish preservation of all DINOv3 information.
The final patch insertion and max-absolute pooling are specific to this composite teacher. Unknown
overlap with DINO pretraining cannot be excluded; the duplicate screen only isolates the study's own
splits under its SHA-256/pHash rule. Findings are limited to this dataset, readout, architecture and
compute allocation. A study with longer training or other readouts would require a new registration.

## Final correctness review

`uv run python scripts/audit_study.py` independently checked all 25 checkpoints and recomputed forward
KL and accuracy from saved float32 logits using float64 SciPy operations. It confirmed the paired
Student interval, equal image exposure, source/model pins, parameter counts, teacher and normalization
buffer equality, manifest/checkpoint hashes, duplicate-group split isolation and identical test images
and teacher logits. It also confirmed that all training completed before final test evaluation.
The [audit record](final_audit.json) passed. All 25 focused tests pass, including deterministic resume,
released-implementation parity and the added uncertainty-display tests.

The safe model bundles include the trained BSF, fitted classifier and normalization, manifests,
complete test predictions, histories, pilot evidence, frozen schedule and source metadata. They omit
original photos, gated DINO weights and executable optimizer checkpoints. Local checkpoints retain
optimizer, scheduler and RNG state; the original bounded study is complete and must not be extended.

All 25 bundles are in the [public collection](https://huggingface.co/collections/BurnyCoder/end-to-end-bsfs-on-dinov3-full-6ae0a9d519d3d32881f784b8).
`uv run python scripts/audit_publication.py` anonymously verified collection membership and the Git/LFS
content identity of every uploaded file. Fresh downloads of the block-KL and scalar-KL seed-100 bundles
reproduced the fixed first development image's original and inserted probabilities within the declared
`atol=1e-5, rtol=1e-4` tolerance. Exact maximum errors and Hub commits are in the
[publication verification record](publication_verification.json). This was an inference audit, not additional training.
