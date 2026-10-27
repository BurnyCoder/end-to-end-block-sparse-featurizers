"""Global: one measured evidence object for every presentation; local: complete curves and uncertainty.

Sources: NumPy paired resampling; https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html.
"""

from pathlib import Path
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..io import read_json, write_json
from ..logging_utils import log_event
from .analysis import bootstrap_columns, paired_curves, threshold_budgets, normalized_auc, pareto_indices


def table(rows):
    """Give every operating point the same table columns, with no chosen best model."""
    lines = ["| Objective | k | Active coordinates | KL | CE increase | Accuracy | Insertion NMSE | Downstream MSE | Alive blocks |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['label']} | {row['k']} | {row['active_coordinates']:.3g} | {row['kl']:.6g} | {row['ce_delta']:.6g} | {row['accuracy']:.2%} | {row['insertion_nmse']:.5g} | {row['downstream_mse']:.6g} | {row['alive_blocks']} |")
    return "\n".join(lines)


def figures(evidence, destination):
    """Display all curves on honest common axes; image intervals are explicitly conditional."""
    destination.mkdir(parents=True, exist_ok=True)
    rows = evidence["runs"]
    if not rows:
        return
    labels = list(dict.fromkeys(row["label"] for row in rows))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, metric, title in zip(axes.flat, ("kl", "ce_delta", "insertion_nmse", "downstream_mse"),
                               ("Output KL (nats/image)", "Cross-entropy increase (nats/image)", "Insertion normalized MSE", "Mean downstream raw MSE")):
        for label in labels:
            values = sorted([row for row in rows if row["label"] == label], key=lambda row: row["k"])
            x, y = [3 * row["k"] for row in values], [row[metric] for row in values]
            line, = ax.plot(x, y, "o-", label=label, linewidth=1)
            for a, b, row in zip(x, y, values):
                ci = row["image_ci95"][metric]
                ax.vlines(a, ci[0], ci[1], color=line.get_color(), alpha=.6)
        ax.set_xscale("log", base=2)
        ax.set(xlabel="Active coordinates per token", ylabel=title)
        ax.axhline(0, color="gray", linewidth=.5)
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("One-seed full grid | 95% paired image-group intervals, conditional on fitted models")
    fig.savefig(destination / "curves.png", dpi=160)
    plt.close(fig)
    from ..reporting import plot_maps
    plot_maps([dict(item, condition=item["cell"]) for item in evidence["diagnostics"]], destination,
              {item["cell"]: item["label"] for item in evidence["diagnostics"]})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for row in [row for row in rows if row["k"] == 16]:
        for ax, metric in zip(axes, ("layer_mse", "layer_nmse")):
            ax.plot(range(6, 13), np.asarray(row[metric])[:, 0], "o-", label=row["label"])
            ax.set(xlabel="Raw residual after block", ylabel="Raw MSE" if metric == "layer_mse" else "Training-variance-normalized MSE", ylim=(0, None))
    axes[0].legend(fontsize=8)
    fig.suptitle("k=16 | all 201 tokens | downstream penalty applies to blocks 7-12 only")
    fig.savefig(destination / "layers.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 2 * len(labels)), sharex=True, sharey=True, layout="constrained")
    for ax, label in zip(np.atleast_1d(axes), labels):
        for row in [row for row in rows if row["label"] == label]:
            history = evidence["histories"].get(row["cell"], [])
            ax.plot([h["images_seen"] for h in history], [h["kl"] for h in history], label=f"k={row['k']}", linewidth=.7)
        ax.set(title=label, ylabel="Batch KL", ylim=(0, None))
        ax.legend(fontsize=7, ncol=5)
    np.atleast_1d(axes)[-1].set_xlabel("Images seen; unsmoothed training batches")
    fig.savefig(destination / "training.png", dpi=140)
    plt.close(fig)


def report(spec, logger):
    """Rebuild presentations from saved predictions only; reporting never retrains or reevaluates."""
    root, destination = Path(spec.root), Path(spec.report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [root / "runs" / cell.key for cell in spec.cells() if (root / "runs" / cell.key / "test_summary.json").exists()]
    rows, predictions, diagnostics, histories = [], {}, [], {}
    reference_indices = reference_teacher = None
    for run in paths:
        row = read_json(run / "test_summary.json")
        with np.load(run / "test_predictions.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files if key != "block_stable_rank"}
        if reference_indices is None:
            reference_indices, reference_teacher = arrays["indices"], arrays["teacher_logits"]
        elif not np.array_equal(reference_indices, arrays["indices"]) or not np.array_equal(reference_teacher, arrays["teacher_logits"]):
            raise ValueError("Operating points use different images or teacher predictions")
        metrics = {key: arrays[key] for key in ("kl", "ce_delta", "accuracy", "agreement")}
        metrics.update(insertion_nmse=arrays["layer_nmse"][:, 0, 0], downstream_mse=arrays["layer_mse"][:, 1:, 0].mean(1))
        intervals = bootstrap_columns(np.column_stack(list(metrics.values())), arrays["groups"], spec.bootstrap_replicates)
        row["image_ci95"] = dict(zip(metrics, intervals.tolist()))
        predictions[row["cell"]] = arrays
        rows.append(row)
        histories[row["cell"]] = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
        if (run / "diagnostics.json").exists():
            diagnostics.append(read_json(run / "diagnostics.json"))
        (destination / f"{row['cell']}.md").write_text(
            f"# {row['label']}, k={row['k']}\n\nOne seed; reused benchmark; all metrics below are measured. "
            "See [shared methodology and comparison](summary.md).\n\n```json\n" + json.dumps(row, indent=2) + "\n```\n", encoding="utf-8")
    coordinates = [spec.block_size * k for k in spec.active_blocks]
    primary, secondary, thresholds = None, {}, {}
    complete = len(rows) == len(spec.cells())
    if complete:
        local = np.column_stack([predictions[f"k{k:03d}_mse"]["kl"] for k in spec.active_blocks])
        groups = predictions[spec.cells()[0].key]["groups"]
        for label in dict.fromkeys(row["label"] for row in rows):
            curve = sorted([row for row in rows if row["label"] == label], key=lambda row: row["k"])
            thresholds[label] = threshold_budgets([row["kl"] for row in curve], coordinates)
            proposed = np.column_stack([predictions[row["cell"]]["kl"] for row in curve])
            if label != "Local MSE":
                effect = paired_curves(local, proposed, coordinates, groups, spec.bootstrap_replicates)
                effect.update(local_auc=float(normalized_auc(local, coordinates).mean()),
                              proposed_auc=float(normalized_auc(proposed, coordinates).mean()))
                if label == "Output KL":
                    primary = effect
                else:
                    secondary[label] = effect
    narrative = "The full grid is not yet evaluated; no complete-curve inference is available."
    if primary:
        lo, hi = primary["ci95"]
        narrative = (f"Local-minus-output-KL normalized curve area: {primary['mean_improvement']:.6g} nats/image; "
                     f"conditional 95% image-group interval [{lo:.6g}, {hi:.6g}]. ")
        narrative += ("The observed nonpositive effect contradicts the directional advantage for these runs. " if primary["mean_improvement"] <= 0 else
                      "The point estimate favors output KL for this trained seed. ")
        narrative += ("The interval spans zero; the image-level comparison is inconclusive. " if lo <= 0 <= hi else "")
        narrative += "One seed and reused test data do not establish robustness across initializations or a fresh confirmatory result."
    pareto = {metric: [rows[i]["cell"] for i in pareto_indices([[row[metric], row["kl"]] for row in rows])]
              for metric in ("active_coordinates", "insertion_nmse", "downstream_mse")} if rows else {}
    evidence = {"label": "Exploratory block-6 full-grid sweep; one seed; previously analyzed Imagenette test set",
                "spec": spec.to_dict(), "status": read_json(root / "study_status.json"),
                "schedule": read_json(root / "study_schedule.json") if (root / "study_schedule.json").exists() else None,
                "runs": rows, "primary": primary, "secondary": secondary, "thresholds": thresholds, "pareto": pareto,
                "narrative": narrative, "coordinates": coordinates, "histories": histories, "diagnostics": diagnostics,
                "split_counts": read_json(root / "manifest.json")["counts"],
                "controls": [read_json(path) for path in sorted((root / "controls").glob("*.json"))]}
    write_json(destination / "results.json", evidence)
    figures(evidence, destination / "figures")
    body = ("# Downstream reconstruction and sparsity sweep\n\n" + evidence["label"] + "\n\n" + narrative + "\n\n" + table(rows) +
            "\n\n## Method and interpretation\n\nThe [preregistration](../../docs/downstream-preregistration.md) defines the full grid, "
            "raw downstream MSE, reused splits and fixed readout. All 201 tokens are replaced after block 6. "
            "The suffix and classifier remain frozen. Only development KL selects learning rates; final checkpoints receive equal exposure. "
            "The objective contrasts therefore include their development-selected learning rates. "
            "Intervals resample image groups conditional on one seed; they do not estimate training-seed uncertainty.\n\n"
            "## Measured sparsity thresholds\n\nSmallest measured active-coordinate budget meeting each KL threshold; null means unattained. "
            "No interpolation or extrapolation.\n\n```json\n" + json.dumps(thresholds, indent=2) + "\n```\n\n"
            "## Observed Pareto frontiers\n\nMinimize KL and each named quantity jointly among all measured points; ties remain. "
            "These are descriptive sample frontiers, without uncertainty-adjusted dominance claims.\n\n```json\n" +
            json.dumps(pareto, indent=2) + "\n```\n\n"
            "## Controls\n\n```json\n" + json.dumps(evidence["controls"], indent=2) + "\n```\n\n"
            "## Figures\n\n![All curves](figures/curves.png)\n\n![Layer reconstruction](figures/layers.png)\n\n"
            "![Learning curves](figures/training.png)\n\n![Fixed development maps](figures/maps.png)\n\n"
            "## Limits\n\nThis tests a ten-class readout, not all DINO information or semantic manifold recovery. "
            "Raw-MSE scales differ between layers; normalized diagnostics are reported separately and are not extra loss terms. "
            "Fixed maps and interventions at k=16 are exploratory; inactive blocks and approximate random matching are retained. "
            "All measured arrays, settings and histories are available in results.json and the individual model bundles.\n")
    (destination / "summary.md").write_text(body, encoding="utf-8")
    for filename in ("study_status.json", "study_schedule.json", "pilot.json", "verification.json", "cache.json", "publication.json",
                     "deviations.json", "batch_numerics_diagnosis.json"):
        if (root / filename).exists():
            shutil.copy2(root / filename, destination / filename)
    from .paper import write_paper
    write_paper(destination, evidence)
    site = Path("site/sweep")
    site.mkdir(parents=True, exist_ok=True)
    write_json(site / "data.json", evidence)
    shutil.copy2(destination / "paper.pdf", site / "paper.pdf")
    if (destination / "figures").exists():
        shutil.copytree(destination / "figures", site / "figures", dirs_exist_ok=True)
    log_event(logger, "sweep_report", {"runs": len(rows), "primary": primary, "directory": str(destination)})
    return evidence
