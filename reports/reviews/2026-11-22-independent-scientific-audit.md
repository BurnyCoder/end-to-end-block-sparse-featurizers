# Independent scientific audit, 2026-11-22

**Scope.** This is an independent, read-only audit of the completed final-layer study, the stopped block-6
follow-up and every claim in the README, `docs/` and `reports/`. Earlier audit records were not used as evidence.
No model was trained or run for inference. The audit recomputed numbers from saved per-image predictions,
pilot records, frozen schedules and logs, compared the code with its pinned upstream sources, and checked every
external citation against its live source. Each number below comes from
[`scripts/recheck_claims.py`](../../scripts/recheck_claims.py), which writes
[`2026-11-22-claim-recheck.json`](2026-11-22-claim-recheck.json). Reproduce it with
`uv run python scripts/recheck_claims.py` (complete checks need the ignored local `outputs/` artifacts).

## Verdict

- **No error invalidates the reported findings.** Every audited number matches its artifact. The KL orientation,
  architecture, normalization, splits, controls and statistics agree with their sources and the preregistration.
  Test data were opened only after all scheduled training had finished.
- **The primary hypothesis was evaluated correctly as inconclusive.** The preregistration says an interval crossing
  zero is inconclusive. The paired interval is [-0.00325909, 0.00835100] nats/image (t = 1.22, 4 df, p = 0.29).
- **Three qualifications were missing and are now added.** (1) The design was less sensitive than its pilot power
  estimate implied. (2) The point estimate is below the registered minimum practical effect. (3) The secondary
  scalar contrast also includes a learning-rate difference.
- **Several precision and citation errors were corrected.** None of them changes a measured result.

## What was recomputed

| Claim (where stated) | Recomputed from | Result |
|---|---|---|
| Block KL vs block MSE KL 0.0350260 vs 0.0375719, 6.78% lower; NMSE +44.41% (README, findings, paper) | 25 `test_predictions.npz` arrays, float64 | Match |
| Paired effect 0.00254596, 95% CI [-0.00325909, 0.00835100]; 3 of 5 seeds favor KL | `scipy.stats.ttest_rel` on per-seed means | Match |
| Scalar effect 0.223164, CI [0.206989, 0.239339]; controls 0 / 3.71488 / 0.734338 KL | `results.json`, prediction arrays | Match |
| Pilot LRs (3e-4 except scalar MSE 1e-4), d = 1.62632, power at 5 = 0.774707, 6 seeds requested | All 30 pilot records, `statsmodels` `TTestPower` | Match |
| Forecast 18.53 h (5 seeds) vs 22.24 h (6 seeds); training 9.326 h; peak 699.3 MiB | Frozen schedule, run records | Match |
| Start 01:02:10, last pilot at 90.28 min, schedule frozen at 140.59 min, last training 12:42:54, first test evaluation 12:43:30, evaluation complete 12:54:34, report 13:01:16 UTC | Status file and original run logs | Match; test evaluation strictly after training |
| One duplicate pair among 13,394 images; 7,575/1,894/3,875 split; 50 smoke test groups excluded; no group crosses splits | `duplicate_audit.json`, manifests | Match (the pair is two training images, both in development) |
| Probe C = 10, development CE 0.0376775, accuracy 98.9968%; clean test accuracy 99.0452% | `teacher.json`, predictions | Match; C = 10 is the edge of the registered grid (see minor notes) |
| Intervention means (block MSE removal +0.0393335, same-class swap -0.0114726; block KL removal -0.0006901) and 3/4/4 inactive selected blocks | `diagnostics` in `results.json` | Match |
| Parameter counts 9,441,280 / 9,449,472 (smoke 73,760 / 73,824) | 4096·768·3 + 4096; 12288·768 + 12288 | Match |
| Sweep: 15 complete, 1 trained-only, 34 unstarted pilots; KL below MSE in 4/4 matched comparisons; development accuracy 11.6–30.0% | Sweep `results.json` | Match |
| Sweep preparation 68 min (extraction 26.5, statistics 23.4, hashing 17.7); verification passed 19:37:46; stop 20:00:02 | Committed sweep logs | Match (26.95 min from the command timestamp, including setup) |
| Batch-geometry diagnosis: maximum final-state difference 0.02734375, RMS 9.90854e-5, teacher-logit 1.04904e-5 | `batch_numerics_diagnosis.json` | Match |

## Methods compared with their sources

- **Featurizer.** The vendored [Goodfire files at 219f121](https://github.com/goodfire-ai/block-sparse-featurizer/tree/219f121ea82d2b19200d1dac918396e6058d7eb9)
  match upstream byte-for-byte apart from line endings. The adapter computes the same tied code
  `z_g = gamma_g x D_g^T`, the same block TopK by L2 norm and the same decoder. It reuses one QR per forward,
  which does not change the mathematics. The paper/code differences in the literature review are real:
  [Fel et al., Appendix D](https://arxiv.org/html/2606.25234v1) describes one shared positive scalar and QR
  re-projection every 20 steps, while the released code uses per-block unconstrained gains and QR on every forward.
- **Objective.** `F.kl_div(log q, log p, log_target=True)` summed over classes equals `KL(p_teacher || q_inserted)`.
  This matches the executable [Apollo `LogitsKLLoss`](https://github.com/ApolloResearch/e2e_sae/blob/83b4add4d2652a4e5c6775a0e2b752897c871a87/e2e_sae/losses.py),
  whose docstring states the reverse direction. Apollo's downstream loss is raw `F.mse_loss` with `total_coeff / len(hook_positions)`.
  The sweep's `beta * mean over six layers` implements the same rule.
- **Map metrics.** Total variation and Dirichlet energy over the eight highest-energy unit-L2 block maps follow
  [Fel et al., Appendix G.3](https://arxiv.org/html/2606.25234v1) exactly. Stable rank `trace(C)/lambda_max(C)` is
  the standard stable rank of the centered code matrix. The repository correctly calls it a proxy.
- **Normalization and NMSE.** The scale is the training RMS over all centered coordinates, so NMSE is the fraction
  of variance unexplained relative to training positional means. The mean-replacement control's test NMSE of
  0.99943 is consistent with that.
- **Statistics.** The Student interval, the 2,000-draw percentile image-group bootstrap and the paired-t power rule
  match the preregistration. The bootstrap intervals in the paper are per model, not per paired difference, and the
  documents never infer significance from their overlap.
- **Backbone and data.** The pinned model revision matches the [Hugging Face model record](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
  The only official [DINOv3 classification head](https://github.com/facebookresearch/dinov3) targets ViT-7B/16,
  which justifies fitting a ViT-B readout. Imagenette-320 has 9,469 training and 3,925 validation images, which
  sum to the 13,394 hashed images.

## Issues found and corrections

**Major (interpretation; corrected).**

1. *Design sensitivity overstated.* The 77.47% power estimate came from three 510-image pilot pairs whose
   paired-difference SD was 6.1% of local-MSE KL. In the main study it was 12.4%. At the realized SD, five seeds
   have 28% power for the registered 10% effect, and 15 seeds would be needed for 80%. The pilots' 24.0% mean
   reduction also overstated the main-study 6.78%. This check uses the registered effect, not the observed one,
   because observed-effect power only restates the p-value ([Hoenig and Heisey, 2001](https://doi.org/10.1198/000313001300339897)).
   *Added to:* findings, README, paper (computed by `statistics.design_sensitivity`).
2. *Minimum practical effect.* The 6.78% point estimate is below the preregistered 10% minimum practical effect.
   The interval spans -8.7% to +22.2% of local-MSE KL. *Added to:* findings, README, paper.
3. *Normality dependence with five pairs.* An exact sign-flip permutation test on five pairs cannot fall below
   p = 2/32 = 0.0625 (observed p = 0.375; exact Wilcoxon p = 0.4375). Any significant five-seed result would
   therefore have depended on the t normality assumption. *Added to:* findings, paper.

**Minor (corrected).**

4. The scalar MSE/KL contrast pairs a 1e-4 learning rate with a 3e-4 learning rate. The findings disclosed this
   only in general terms; the README, findings and paper now state it next to the scalar result.
5. The phrase "approximately energy-matched random removal" was unquantified. Where the selected block was active,
   the random block's energy ranged from 0.12 to 1.25 times the selected block's. Findings and paper now state it.
6. "Fifty fixed-batch updates reduced KL from X to Y" (smoke ledger, sweep notes, sweep paper): each recorded value
   is the loss before that update, so Y follows 49 updates. The wording is corrected.
7. Sweep allocation. The interruption cost about 31 minutes, but at the logged rates the 50-pilot grid needed about
   51–55 minutes against about 51 minutes available without it. The two-hour budget had essentially no slack.
   This post-hoc estimate is now in the execution notes.
8. Literature review. The metric audit is now named ([Chanin, 2026](https://arxiv.org/abs/2605.18229)) and the
   DINOv3 paper ([Siméoni et al., 2025](https://arxiv.org/abs/2508.10104)) is cited. "Seed instability" is narrowed to
   the seed-to-seed variation for some toy manifolds that [Jerpelea and Ananthram, Section 3.2](https://arxiv.org/html/2608.27515v1) report.
   [Karvonen (2025)](https://arxiv.org/abs/2503.17272) is described as KL+MSE fine-tuning.
9. Both papers now carry the requested author line, a date and an AI statement that distinguishes the original
   Codex-written work from this Claude Code audit. Both PDFs were re-rendered from their committed `results.json`.
   Page counts are unchanged (8 and 5).

**Noted, not changed (no error).** The selected probe C = 10 is the largest value in the registered grid, and
development cross-entropy was still falling (0.0444 at C = 1). The teacher is defined by this procedure, so this
does not affect validity. The 3e-4 learning rate was also the upper end of the two-point grid for four of five
conditions, which the documents already list as a limitation.

## Additional post-hoc evidence

Within-seed paired image-group bootstrap intervals for local-minus-KL are conditional on each trained pair and
use 2,000 draws with seed 42. They exclude zero for seeds 102 [0.00354, 0.01401] and 104 [0.00090, 0.01146], and
include zero for seeds 100 [-0.00798, 0.00294], 101 [-0.00454, 0.00673] and 103 [-0.00673, 0.00471]. Variation
between training seeds, not test-image sampling, therefore dominates the primary uncertainty. This supports the
registered choice of seeds as the inferential unit.

## Remaining limitations that need new runs

These cannot be fixed without new runs. Each would need a new registration:

- adequate replication (about 15 seeds for the 10% effect at the realized variance);
- longer training and a wider learning-rate grid;
- a learning-rate-matched scalar comparison;
- exact energy-matched intervention controls;
- the unmeasured block-6 sweep.

The study does not establish semantic concept recovery or preservation of all DINOv3 information.

## Citation check

All of the following resolved, and their content matches how the repository uses them:

- Papers: [Braun et al., 2024](https://arxiv.org/html/2405.12241v2) (Table 1: L0 69.4 vs 27.5 at CE increase 0.145 vs 0.144, Sections 2.4 and 3.1–3.2),
  [Fel et al., 2026](https://arxiv.org/abs/2606.25234), [Karvonen, 2025](https://arxiv.org/abs/2503.17272),
  [Jerpelea and Ananthram, 2026](https://arxiv.org/abs/2608.27515), [Chanin, 2026](https://arxiv.org/abs/2605.18229),
  [SAEBench](https://proceedings.mlr.press/v267/karvonen25a.html), [Card et al., 2020](https://aclanthology.org/2020.emnlp-main.745/),
  [Lones](https://arxiv.org/abs/2108.02497), [Lipton and Steinhardt](https://arxiv.org/abs/1807.03341).
- Code and data: [Goodfire code](https://github.com/goodfire-ai/block-sparse-featurizer/tree/219f121ea82d2b19200d1dac918396e6058d7eb9),
  [Apollo code](https://github.com/ApolloResearch/e2e_sae/tree/83b4add4d2652a4e5c6775a0e2b752897c871a87),
  [the previous workbench logger](https://github.com/BurnyCoder/block-sparse-featurizer-experiments/tree/6af61910c3ad2ddc4320bff76fe607b6ede13c18) (verbatim),
  [Imagenette](https://github.com/fastai/imagenette).
- Documentation: [PyTorch batched-computation note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations),
  [statsmodels `TTestPower`](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.html),
  [SciPy `ttest_rel`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html).
- Published artifacts: the [Hugging Face collection](https://huggingface.co/collections/BurnyCoder/end-to-end-bsfs-on-dinov3-full-6ae0a9d519d3d32881f784b8)
  and both [explorer](https://burnycoder.github.io/end-to-end-block-sparse-featurizers/) [pages](https://burnycoder.github.io/end-to-end-block-sparse-featurizers/sweep/).

A web search found no prior output-KL-trained BSF, consistent with the literature review's explicit non-novelty-claim wording.
