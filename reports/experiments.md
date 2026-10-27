# Experiment ledger

This ledger links complete measured reports; it does not select favorable seeds or runs.
Definitions: **KL** is the original-to-reconstructed class-distribution divergence, in nats per image;
**NMSE** is squared patch reconstruction error divided by the training RMS scale squared;
**block** is a group of jointly selected signed coordinates; **active** means selected by block TopK.
The teacher is the fixed DINOv3 plus Imagenette logistic readout, not a native DINO classifier.
See the [method](../docs/preregistration.md) and [source review](../docs/literature-review.md).

| Experiment | Domain and data | Architecture / active coordinates | Training | Completion and evidence |
|---|---|---|---|---|
| Engineering smoke, seed 100 | Imagenette: 200 train, 50 dev, 50 test images | 32 × 3 blocks / 12 coordinates; scalar 96 × 1 / 12 | Adam; 3 epochs; five MSE/KL/mixed conditions; no curriculum | All five complete; [scores, curves, maps and limitations](smoke/summary.md) |
| Registered full study | Imagenette, grouped 80/20 official training; reserved validation less engineering-preview groups | 4096 × 3 / 48; scalar 12288 × 1 / 48 | Equal LR pilot; freeze 5–10 seeds and ≤20 common epochs before test | Not started at ledger creation; durable live status is `outputs/study/study_status.json` |

## Engineering findings, 2026-10-27

The smoke run is a real image-to-output integration check. It uses deliberately tiny dictionaries
to test the entire public CLI cheaply. It is separate from the registered effectiveness experiment.
All five trained conditions started from scratch. Their scores are in the linked report rather than
duplicated here. The low inserted-model accuracy is a negative result, and one seed cannot estimate
training-seed uncertainty. Relative to local MSE, KL modestly improves readout fidelity while worsening
patch reconstruction; this illustrates why both metrics are necessary, without establishing generality.

On a fixed pair of real training images, the online and cached patch tensors, KL, and BSF gradients
matched exactly in float32. Identity replacement gave zero KL. Replacing patches with zeros gave
KL 19.498661, confirming the classifier uses the intervened path. Fifty fixed-batch optimization steps
reduced KL from 11.633245 to 0.027980; the trajectory was not monotonic. These are implementation checks,
not held-out performance claims. Raw measurements are in `outputs/smoke/verification.json`.

Initial cache extraction stopped at image 284 after a Windows sharing violation during atomic JSON
replacement. We added bounded retries only for Windows sharing/access errors, wrote a regression test,
and resumed the same cache to 300 images. We retained atomic replacement rather than copying a partial
file. This fix affects artifact reliability, not model equations. The complete CLI then finished.

A final review exposed another interruption edge case: model resume was bitwise correct, but a crash
after the last checkpoint could leave duplicate updates or a partial JSON line in the plotted history.
A failing interruption test reproduced this. Resume now archives all original history, retains only
the checkpointed update prefix, and appends recovered training to that verified prefix. This changes
artifact recovery, not training mathematics or the prespecified analysis. The preparation process was
restarted to load this fix before full pilot training; the original wall-time deadline was retained.

The duplicate screen found one perceptually close pair among all 13,394 source images and no
training/validation overlap under the prespecified SHA-256/pHash rule. This cannot certify absence of
all semantic duplicates. Engineering test images and their detected duplicate groups are excluded
from the full final test set, so inspected examples cannot enter the confirmatory endpoint.

## Analysis boundaries

Hyperparameters for the full experiment follow the preregistration, including equal learning-rate
search, paired initialization and data order, common image exposure, a bounded compute budget and
pilot-based power calculation. Pilot seeds never enter the final paired contrast. No test result
chooses a learning rate, epoch count, seed count or visualization block. Semantic recovery and broad
DINO information preservation remain untested. The full study may be exploratory if power or compute
requirements cannot be met; significance is never an acceptance requirement.
