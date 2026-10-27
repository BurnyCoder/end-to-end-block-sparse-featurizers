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
