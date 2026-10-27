"""Global: fixed teacher and normalization. Local: clean-feature logistic readout.

Extraction follows the existing BSF workbench; probe fitting follows
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.
"""

import numpy as np  # Accumulate normalization in float64 for numerical accuracy.
import torch  # Differentiable readout on reconstructed patches.
from torch import nn  # Buffers travel with the frozen modules.
from sklearn.linear_model import LogisticRegression  # Reuse an established optimizer.
from sklearn.metrics import log_loss, accuracy_score  # Standard supervised probe metrics.
from threadpoolctl import threadpool_limits  # Prevent BLAS thread oversubscription.

from .io import save_checkpoint, write_json  # Bind learned preprocessing and head to artifacts.
from .logging_utils import log_event  # Preserve all pilot candidate results.


class Normalization(nn.Module):
    """Local: invertible affine transform; global: fixed train-only preprocessing."""

    def __init__(self, mean, scale):
        """Store the positional mean and scalar RMS as non-trainable checkpoint buffers."""
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        if not torch.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Normalization scale must be finite and positive")

    @classmethod
    def fit(cls, features, indices, batch_size=32):
        """Local: two streamed passes; global: never fit normalization on held-out images."""
        indices = np.asarray(list(indices), dtype=np.int64)  # Freeze the fit population.
        if len(indices) < 2:
            raise ValueError("Need at least two training images")
        total = np.zeros(features.shape[1:], dtype=np.float64)  # One accumulator per position.
        for start in range(0, len(indices), batch_size):
            total += np.asarray(features[indices[start:start + batch_size]], dtype=np.float64).sum(0)
        mean = total / len(indices)  # Fit only the designated training population.
        sse = 0.0  # Sum all squared deviations before reducing.
        for start in range(0, len(indices), batch_size):
            chunk = np.asarray(features[indices[start:start + batch_size]], dtype=np.float64)
            sse += float(np.square(chunk - mean).sum())
        scale = np.sqrt(sse / (len(indices) * mean.size))  # E||normalized patch||² = d.
        return cls(mean.astype("float32"), scale)

    def forward(self, patches):
        """Normalize without updating training statistics."""
        return (patches - self.mean) / self.scale

    def inverse(self, patches):
        """Restore the exact coordinate system on which the frozen head was fitted."""
        return patches * self.scale + self.mean


class FrozenProbe(nn.Module):
    """Local: max-absolute patch pooling; global: define the ten-class original model."""

    def __init__(self, weight, bias):
        """Register fixed classifier coefficients while retaining input gradients."""
        super().__init__()
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, patches):
        """Pool magnitude across patches; CLS and register tokens have no path here."""
        return patches.abs().amax(dim=1) @ self.weight.T + self.bias


def fit_teacher(features, rows, root, logger):
    """Local: fit four probe candidates; global: freeze one clean teacher for all conditions."""
    pooled = np.empty((len(rows), features.shape[-1]), dtype=np.float32)  # Tiny image-level matrix.
    for start in range(0, len(rows), 32):
        pooled[start:start + 32] = np.abs(features[start:start + 32]).max(axis=1)
    train = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    dev = np.array([i for i, row in enumerate(rows) if row["split"] == "dev"])
    labels = np.array([row["label"] for row in rows])  # Supervision trains only this fixed head.
    candidates = []  # Retain every candidate's development evidence.
    best = None  # Choose strictly by development cross-entropy.
    with threadpool_limits(limits=4):  # Small linear algebra need not use every CPU thread.
        for regularization in (0.01, 0.1, 1.0, 10.0):
            estimator = LogisticRegression(C=regularization, solver="lbfgs", max_iter=2000, tol=1e-6)
            estimator.fit(pooled[train], labels[train])  # Neither development nor test is fitted.
            probabilities = estimator.predict_proba(pooled[dev])
            metrics = {"C": regularization, "dev_ce": float(log_loss(labels[dev], probabilities)),
                       "dev_accuracy": float(accuracy_score(labels[dev], probabilities.argmax(-1))),
                       "iterations": int(estimator.n_iter_.max())}
            candidates.append(metrics)
            log_event(logger, "probe_candidate", metrics)
            if best is None or metrics["dev_ce"] < best[0]:
                best = (metrics["dev_ce"], estimator, metrics)  # Deterministic first tie winner.
    estimator = best[1]  # Freeze the selected candidate without refitting on development data.
    if best[2]["dev_accuracy"] <= 0.2:
        raise RuntimeError("Clean teacher is not useful (>20% development accuracy required)")
    normalization = Normalization.fit(features, train)  # Training-only fit after split assignment.
    probe = FrozenProbe(estimator.coef_, estimator.intercept_)
    state = {"probe": probe.state_dict(), "normalization": normalization.state_dict(),
             "classes": estimator.classes_.tolist(), "pooling": "absolute_max", "selection": best[2]}
    save_checkpoint(root / "teacher.pt", state)
    write_json(root / "teacher.json", {"candidates": candidates, "selected": best[2],
                                       "fit_images": len(train), "selection_images": len(dev)})
    return probe, normalization
