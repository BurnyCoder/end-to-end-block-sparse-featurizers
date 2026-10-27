# Full-size pilots and allocation recovery

This is the historical allocation record written before main training. The bounded study has since
completed; see the [final findings](findings.md) and [plan completion audit](../../docs/plan-completion.md).

All 30 pilots completed before the preparation deadline: five conditions, two learning rates and
three paired seeds (0, 1, 2). Each used 510 training and 250 development images for one epoch.
At the time of this allocation decision, the final test split had not been evaluated.
[Every trial is retained](pilot.json), including
development KL, elapsed time and microbatch size. Shared methodology and sources are in the
[preregistration](../../docs/preregistration.md) and [methodology](../../docs/methodology.md).

| Condition | Selected LR | Mean development KL, nats/image | Slowest pilot, seconds |
|---|---:|---:|---:|
| Block MSE | 0.0003 | 0.360137 | 91.703 |
| Block KL | 0.0003 | 0.273871 | 91.016 |
| Block KL + 0.1 MSE | 0.0003 | 0.272218 | 92.328 |
| Scalar MSE | 0.0001 | 0.280861 | 286.797 |
| Scalar KL | 0.0003 | 0.086295 | 295.031 |

LR selection minimizes mean pilot development KL for every condition. These selected pilot scores
are feasibility and selection evidence; they are not independent estimates of final performance.
Scalar training was slower on this hardware. The cause is consistent with its threefold larger
batch of QR factorizations, but no controlled hardware profiling experiment was performed.

The prespecified power calculation uses the paired pilot differences' standard deviation and an
effect of 10% of local-MSE KL (0.0360137 nats), rather than the observed mean improvement. Its
standardized effect estimate is 1.62632. The two-sided paired-t calculation at alpha 0.05 requires
six seeds for 80% power; estimated power at five is 0.774707. This estimate is uncertain because
only three short pilot pairs determine variance. See the
[statsmodels paired-t documentation](https://www.statsmodels.org/dev/generated/statsmodels.stats.power.TTestPower.power.html).

Summing each condition's slowest observed cost, scaling to 7,575 training plus 1,894 development
images and retaining a 25% allowance predicts 22.24 hours for six seeds and one epoch, or 18.53 hours
for five. The allocator stopped before creating a schedule because the six-seed request exceeded
the optimization budget. The [original failure status](allocation_failure.json) is preserved.
No main run or reserved test evaluation preceded this failure.

The user-specified plan permits an exploratory study when adequate replication cannot fit. The
recovery implemented that alternative explicitly: use the largest feasible count between five
and the requested count, or fail if fewer than five fit. For these measurements, the registered
main stage was seeds 100–104, one common epoch, all five conditions, the LRs above and unchanged
architectures. Each run receives all 7,575 training images once. This post-pilot allocation amendment
is disclosed in the preregistration. Its estimated power falls short of the target by 2.53 percentage
points; one epoch also limits conclusions about convergence. No extra training will be added for
significance or because execution finishes early.

The original start is 2026-10-27 01:02:10 UTC. Optimization must end by 2026-10-27 23:02:10 UTC,
and the entire measured study must end by 2026-10-28 01:02:10 UTC. Recovery consumes remaining wall
time; it does not reset either deadline. The frozen schedule and resumable checkpoints retain
the actual source revision and split/teacher identities.

Evaluation originally repeated QR for every image batch although weights were fixed. It now computes
frames once per evaluation call, under
[PyTorch no-grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html), and recomputes them
on the next call. Training still differentiates through QR on every forward. Both block widths have
output-parity and stale-frame regression tests. Original pilot timing estimates remain the allocation
basis; faster evaluation does not authorize extra exposure. The complete suite passes 23 tests.
A [real-image CUDA comparison](evaluation_frame_parity.json) used seven development images with an
incomplete final batch and both full-size architectures. Cached and freshly recomputed QR frames
produced exactly equal logits (maximum absolute difference zero); no parameters were updated.
