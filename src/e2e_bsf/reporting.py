"""Global: one measured evidence source. Local: tables, figures, Markdown and web data.

Sources: matplotlib and Plotly documented figure/export APIs; scipy Student intervals.
Missing or single-seed results are never presented as confirmatory findings.
"""

from collections import defaultdict
from pathlib import Path
import json
import shutil
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # Reports are generated without opening workstation windows.
import matplotlib.pyplot as plt

from .io import read_json, write_json
from .statistics import paired_summary
from .logging_utils import log_event

LABELS = {"block_mse": "Block MSE", "block_kl": "Block KL", "block_kl_mse": "Block KL + 0.1 MSE",
          "scalar_mse": "Scalar MSE", "scalar_kl": "Scalar KL"}


def aggregate_results(results):
    """Local: summarize all completed seeds; global: expose rather than hide missing runs."""
    grouped = defaultdict(list)
    for row in results:
        grouped[row["condition"]].append(row)
    summaries = []
    for condition, records in grouped.items():
        summary = {"condition": condition, "label": LABELS.get(condition, condition),
                   "seeds": [row["seed"] for row in records], "n_seeds": len(records),
                   "images_per_seed": records[0]["images"], "parameters": records[0]["parameters"]}
        for metric in ("kl", "accuracy", "teacher_accuracy", "agreement", "nmse", "ce_delta",
                       "active_blocks", "active_coordinates", "dead_fraction", "stable_rank", "map_tv", "map_dirichlet"):
            values = np.array([row[metric] for row in records if row.get(metric) is not None])
            summary[metric] = float(values.mean()) if len(values) else None
        kls = np.array([row["kl"] for row in records])
        half = float(stats.t.ppf(.975, len(kls) - 1) * stats.sem(kls)) if len(kls) > 1 else None
        summary["kl_seed_ci95"] = [summary["kl"] - half, summary["kl"] + half] if half is not None else None
        summaries.append(summary)
    local = {row["seed"]: row["kl"] for row in results if row["condition"] == "block_mse"}
    proposed = {row["seed"]: row["kl"] for row in results if row["condition"] == "block_kl"}
    return summaries, paired_summary(local, proposed)


def result_table(summaries):
    """Local: explicit units and sample sizes; global: shared README/report/paper evidence."""
    lines = ["| Condition | Seeds | Test KL (nats/image) | Accuracy | NMSE | Active blocks | Dead blocks |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['label']} | {row['n_seeds']} | {row['kl']:.6g} | {row['accuracy']:.2%} | {row['nmse']:.5g} | {row['active_blocks']:.3g} | {row['dead_fraction']:.2%} |")
    if not summaries:
        lines.append("| No completed test evaluations | 0 | — | — | — | — | — |")
    return "\n".join(lines)


def update_readme(report, label, narrative, table, controls):
    """Local: replace a bounded generated section; global: README never promises unmeasured results."""
    path = Path("README.md")  # Human-written setup and method sections remain intact.
    begin, end = "<!-- measured-results:start -->", "<!-- measured-results:end -->"
    original = path.read_text(encoding="utf-8")
    if begin not in original or end not in original:
        return  # A downstream user may intentionally remove the generated section.
    control_table = "\n".join(f"| {name} | {row['kl']:.6g} | {row['accuracy']:.2%} | {row['nmse']:.5g} |"
                              for name, row in controls.items())
    content = (f"{begin}\n\n**{label}**\n\n{narrative}\n\n{table}\n\n"
               "| Control | Test KL (nats/image) | Accuracy | NMSE |\n|---|---:|---:|---:|\n" + control_table +
               f"\n\n![Measured comparison]({report.as_posix()}/figures/comparison.png)\n\n"
               f"![Fixed development block maps]({report.as_posix()}/figures/maps.png)\n\n"
               f"[Detailed results and limitations]({report.as_posix()}/summary.md) · "
               f"[PDF paper]({report.as_posix()}/paper.pdf) · "
               "[Interactive explorer](https://burnycoder.github.io/end-to-end-block-sparse-featurizers/)\n\n" + end)
    before, remainder = original.split(begin, 1)  # Replace evidence only, not unrelated documentation.
    _, after = remainder.split(end, 1)
    path.write_text(before + content + after, encoding="utf-8")


def plot_evidence(summaries, histories, destination):
    """Local: simple honest axes; global: show all methods without cropping away failures."""
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    if summaries:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        labels = [row["label"].replace(" + ", "\n+ ") for row in summaries]
        axes[0].bar(labels, [row["kl"] for row in summaries], color="#347b88")
        for i, row in enumerate(summaries):
            interval = row["kl_seed_ci95"]
            if interval:
                axes[0].errorbar(i, row["kl"], yerr=[[row["kl"] - interval[0]], [interval[1] - row["kl"]]], color="black", capsize=4)
        axes[0].set_ylabel("Forward KL (nats / image), lower is better")
        axes[0].set_ylim(bottom=0)
        axes[0].tick_params(axis="x", rotation=25)
        axes[1].bar(labels, [row["nmse"] for row in summaries], color="#bd7343")
        axes[1].set_ylabel("Normalized patch MSE, lower is better")
        axes[1].set_ylim(bottom=0)
        axes[1].tick_params(axis="x", rotation=25)
        fig.suptitle("Held-out fidelity and reconstruction | error bars: 95% seed intervals where n > 1")
        fig.savefig(destination / "comparison.png", dpi=170)
        plt.close(fig)


def plot_maps(diagnostics, destination):
    """Local: fixed examples and normalized maps; global: show methods without semantic matching."""
    if not diagnostics:
        return
    fig, axes = plt.subplots(3, len(diagnostics), figsize=(3 * len(diagnostics), 7), squeeze=False, layout="constrained")
    for column, result in enumerate(diagnostics):
        for row, example in enumerate(result["examples"][:3]):
            maps = np.asarray(example["heatmaps"])
            selected = int(np.square(maps).sum((1, 2)).argmax())
            heat = maps[selected]
            heat = heat / max(1e-12, np.linalg.norm(heat))
            axes[row, column].imshow(heat, vmin=0, vmax=1, cmap="viridis")
            axes[row, column].set_title(f"{LABELS[result['condition']]}\nclass {example['label']}, block {example['block_ids'][selected]}", fontsize=8)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.suptitle("Fixed development examples | strongest selected block per image | unit spatial L2 norm")
    fig.savefig(destination / "maps.png", dpi=150)
    plt.close(fig)


def plot_history(histories, destination):
    """Local: complete recorded curves; global: avoid choosing favorable epochs."""
    if histories:
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        for name, records in histories.items():
            if records:
                ax.plot([row["images_seen"] for row in records], [max(0, row["kl"]) for row in records],
                        label=name, alpha=.75, linewidth=1)
        ax.set(xlabel="Training images seen", ylabel="Training batch forward KL (nats / image)", ylim=(0, None))
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(destination / "training.png", dpi=170)
        plt.close(fig)


def build_report(config, logger):
    """Local: derive every presentation from saved evidence; global: publish negative results too."""
    root = Path(config.root)
    report = Path("reports") / ("smoke" if config.profile == "smoke" else "study")
    report.mkdir(parents=True, exist_ok=True)
    summaries_paths = sorted((root / "runs").glob("*/test_summary.json"))
    results = [read_json(path) for path in summaries_paths if not path.parent.name.startswith("pilot_")]
    for row, path in zip(results, [path for path in summaries_paths if not path.parent.name.startswith("pilot_")]):
        run_path = path.parent / "run.json"
        row["runtime"] = read_json(run_path) if run_path.exists() else None  # Keep measured resource evidence with metrics.
    summaries, primary = aggregate_results(results)
    histories = {}
    for path in summaries_paths:
        history_path = path.parent / "history.jsonl"
        if history_path.exists() and not path.parent.name.startswith("pilot_"):
            histories[path.parent.name] = [json.loads(line) for line in history_path.read_text().splitlines()]
    controls = read_json(root / "controls_test.json") if (root / "controls_test.json").exists() else {}
    schedule = read_json(root / "study_schedule.json") if (root / "study_schedule.json").exists() else None
    status = read_json(root / "study_status.json") if (root / "study_status.json").exists() else {"status": "smoke"}
    label = "Engineering smoke experiment: one seed, small data and dictionary; no confirmatory claim." if config.profile == "smoke" else "Bounded preregistered study; interpret completion and power status below."
    if config.profile == "full" and (status["status"] != "complete" or not schedule or schedule["exploratory"]):
        label += " Exploratory: replication, power or completion criteria were not all met."
    narrative = "No paired test measurements are available."
    if primary["n_pairs"]:
        effect = primary["mean_improvement"]
        narrative = f"Mean paired local-minus-KL difference: {effect:.6g} nats/image over {primary['n_pairs']} seed(s). "
        narrative += "KL training had lower mean output divergence." if effect > 0 else "KL training did not improve mean output divergence."
        narrative += (f" Seed-level 95% interval: {primary['ci95']}." if primary.get("ci95")
                      else " A seed-level confidence interval cannot be estimated from one seed.")
        if primary.get("ci95") and primary["ci95"][0] <= 0 <= primary["ci95"][1]:
            narrative += " The interval includes zero; the directional comparison is inconclusive."
        if effect <= 0:
            narrative += " The observed nonpositive improvement contradicts the directional hypothesis."
    table = result_table(summaries)
    plot_evidence(summaries, histories, report / "figures")
    plot_history(histories, report / "figures")
    diagnostics = []
    from .diagnostics import inspect_run
    for path in summaries_paths:
        row = read_json(path)
        if row["seed"] == 100 and row["status"] == "complete":
            diagnostics.append(inspect_run(config, path.parent / "checkpoint.pt", logger))
    plot_maps(diagnostics, report / "figures")
    evidence = {"profile": config.profile, "label": label, "status": status, "schedule": schedule,
                "config": config.to_dict(), "summaries": summaries, "runs": results,
                "primary": primary, "controls": controls, "histories": histories, "diagnostics": diagnostics,
                "narrative": narrative, "source": "Measured artifacts; no synthetic display data."}
    write_json(report / "results.json", evidence)
    text = (f"# {'Smoke' if config.profile == 'smoke' else 'Full'} experiment report\n\n{label}\n\n{narrative}\n\n{table}\n\n"
            "## Method and interpretation\n\nSee the [preregistration](../../docs/preregistration.md) and "
            "[source review](../../docs/literature-review.md). Only BSF parameters train, from scratch. "
            "The frozen teacher is DINOv3 plus the fitted patch classifier. Lower KL is readout fidelity, "
            "not proof of semantic concepts or preservation of all DINO representations.\n\n"
            "## Controls\n\n" + "\n".join(f"- {name}: KL {value['kl']:.6g}, accuracy {value['accuracy']:.2%}, NMSE {value['nmse']:.6g}." for name, value in controls.items()) +
            "\n\n## Figures\n\n![Comparison](figures/comparison.png)\n\n![Training](figures/training.png)\n\n![Development maps](figures/maps.png)\n\n"
            "## Learning and limitations\n\nThe comparison separates behavior from reconstruction. "
            "A lower output KL can coincide with higher patch MSE or degraded map structure. "
            "The ten-class head constrains at most nine logit contrasts per image. "
            "Image-group intervals describe conditional evaluation variability; seed intervals describe "
            "training variability. The pilot variance is estimated from short runs. "
            "All run metrics, settings, uncertainty and intervention outputs are in results.json.\n")
    (report / "summary.md").write_text(text, encoding="utf-8")
    update_readme(report, label, narrative, table, controls)
    for path in summaries_paths:
        row = read_json(path)
        details = (f"# {path.parent.name}\n\nRecorded outcome of a prespecified condition; "
                   "configuration is stored with its checkpoint. This report contains no selected best example.\n\n"
                   f"```json\n{json.dumps(row, indent=2)}\n```\n\n"
                   "Interpretation and shared methodology: [summary](summary.md).\n")
        (report / f"{path.parent.name}.md").write_text(details, encoding="utf-8")
    site = Path("site")
    site.mkdir(exist_ok=True)
    write_json(site / "data.json", evidence)
    assets = site / "assets"
    assets.mkdir(exist_ok=True)
    for figure in (report / "figures").glob("*.png"):
        shutil.copy2(figure, assets / figure.name)
    from .paper import write_paper
    write_paper(report, evidence, table)
    shutil.copy2(report / "paper.pdf", site / "paper.pdf")
    from plotly.offline import get_plotlyjs
    if not (site / "plotly.min.js").exists():
        (site / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")  # Bundle the pinned library; no CDN runtime dependency.
    log_event(logger, "report_generated", {"directory": str(report), "runs": len(results), "primary": primary})
    return evidence
