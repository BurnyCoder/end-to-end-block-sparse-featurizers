# Downstream sweep execution notes

These post-registration notes preserve the [original scientific specification](downstream-preregistration.md).
They do not change its grid, losses, seeds or deadlines.

## Batch geometry verification repair, 2026-10-27

Preparation started at 18:00:02 UTC. Extraction, training-only statistics and cache hashing completed
at 19:08:01 UTC. Verification stopped at 19:08:07 UTC before any optimization: the eight-image cache
was compared with a two-image online batch. One block-8 coordinate exceeded the unchanged combined
absolute/relative tolerance, with absolute difference 0.000127792. The 19:31 UTC monitor detected the stop.

A read-only check on the first training images compared batch sizes two and eight. At eight, all seven
raw states (blocks 6–12), teacher logits and identity-suffix logits matched exactly. At two, the maximum
final-state difference was 0.02734375 on large-magnitude coordinates, RMS difference 0.0000990854,
and teacher-logit maximum difference 0.0000104904. Identity downstream MSE remained below 1.24e-7.
Full layer measurements are retained in `outputs/sweeps/downstream-v1/batch_numerics_diagnosis.json`.

This agrees with [PyTorch's documented batch-versus-slice floating-point behavior](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations).
Verification now uses the cache's extraction batch size and only designated training images. Tolerances,
cached tensors, objectives and the full online-gradient comparison remain unchanged. A regression test
checks this batch-selection contract. This is a verification repair, not a parameter or outcome selection.

The original preparation deadline remains **20:00:02 UTC** and the final deadline remains **18:00:02 UTC
on 2026-10-28**. The interruption consumes the existing budget; it does not restart the clock. If the complete
equal pilot grid cannot finish in that allocation, main training will not begin and the allocation failure
will be reported. The completed final-layer study remains unchanged.

## Verification passed; preparation allocation failed

The repaired real-image check passed at 19:37:46 UTC. All seven states and the two objective values
matched exactly at batch size eight; the maximum BSF-gradient difference was 1.78814e-7. Identity KL
was zero, while mean replacement gave KL 4.66014. Fifty fixed-batch optimizer updates reduced recorded
training KL from 2.29414 to 0.0000396106. This validates the connected implementation on eight training
images; it is not held-out effectiveness evidence.

Pilots resumed at 19:38 UTC. The original preparation deadline stopped execution at **20:00:02 UTC**
on 2026-10-27, during development evaluation of `k008_downstream_b0.0001`, LR 0.0003. Fifteen trials
completed both training and development evaluation. The 16th completed 510 training images but has
no saved development endpoint. Thirty-four pilot trials never started. No main schedule or learning
rates were frozen; all 25 seed-200 main runs, new test evaluations and k=16 interventions are unperformed.

Preparation took approximately 68 minutes: extraction 26.5, training-only statistics 23.4, and final
hashing 17.7 minutes, plus loading and setup. The verification interruption and its detection/repair
consumed more of the original allocation. The initial GPU feasibility timing did not establish the
end-to-end cost of these CPU/storage phases. No stage deadline, seed, grid or exposure was changed.

The [independent checkpoint audit](../reports/downstream-sweep/audit.json) checks all 16 started pilots,
their shared initialization and image order, finite weights, unchanged readout/normalization, isolated
split groups, original deadlines, and absence of any main/test result. It rechecks cache size/mtime
against extraction hashes rather than rehashing the entire 58 GB. The audit's scope is preparation;
it cannot validate a primary effect that was never measured.

The [report](../reports/downstream-sweep/summary.md), [paper](../reports/downstream-sweep/paper.pdf) and
[explorer](https://burnycoder.github.io/end-to-end-block-sparse-featurizers/sweep/) retain every pilot result,
missing trial and recorded training curve. No LR is retrospectively selected from the partial grid,
and no bootstrap or sparsity claim is made. No eligible completed main bundle exists, so publication
stops before creating a Hugging Face collection. Local pilot checkpoints and caches remain available.

The next research step would require a separately documented, explicitly authorized allocation or
amendment. Existing engineering artifacts can support that decision, but cannot authorize extending
the present experiment or replace the missing registered measurements.

## Final correctness and presentation review

All 37 tests passed, including the new incomplete-report and no-model-publication regressions.
The independent audit passed for every started pilot. All five PDF pages and three figures were
visually inspected. The browser check found all 50 pilot statuses, 16 selectable training histories,
and no visible main-result panels or browser errors. Changing the selection displayed the incomplete
16th trial's 16 recorded updates. These checks verify the delivered implementation and failure record;
they do not supply the missing scientific measurements.
