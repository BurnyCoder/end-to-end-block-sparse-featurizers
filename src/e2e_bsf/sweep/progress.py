"""Global: disclose incomplete execution; local: render saved pilot artifacts without model access.

Sources: registered phase contract; https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html.
"""

import json
from pathlib import Path
import numpy as np
from ..io import read_json


def inventory(spec):
    """List every planned trial, keeping absent development endpoints explicitly unmeasured."""
    root = Path(spec.root)
    trials, main = [], []
    for cell in spec.cells():
        main_path = root / "runs" / cell.key / "run.json"
        main.append({"cell": cell.key, "status": read_json(main_path)["status"] if main_path.exists() else "not_started"})
        for lr in spec.learning_rates:
            name = f"pilot_{cell.key}_lr{lr:g}"
            run = root / "runs" / name
            state = read_json(run / "run.json") if (run / "run.json").exists() else {"status": "not_started"}
            development = read_json(run / "development.json") if (run / "development.json").exists() else None
            history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()] if (run / "history.jsonl").exists() else []
            trials.append({"name": name, "cell": cell.key, "label": cell.label, "k": cell.k, "lr": lr,
                           "seed": spec.pilot_seed, "state": state, "development": development, "history": history})
    return {"pilot_trials": trials, "pilot_planned": len(trials),
            "pilot_started": sum(row["state"]["status"] != "not_started" for row in trials),
            "pilot_completed": sum(row["state"]["status"] == "complete" and row["development"] is not None for row in trials),
            "main_runs": main, "main_started": sum(row["status"] != "not_started" for row in main)}


def narrative(progress, status):
    """Describe execution failure without interpreting unmeasured hypotheses as negative effects."""
    if status.get("status") == "incomplete" and progress["main_started"] == 0:
        return (f"Allocation failure before main training: {progress['pilot_completed']} of {progress['pilot_planned']} pilot trials completed; "
                f"{progress['pilot_started'] - progress['pilot_completed']} additional trial(s) started but did not complete. "
                "No main runs or main test evaluations were performed. No LR schedule was frozen. "
                "The primary AUC, sparsity thresholds and hypothesis remain unmeasured; this does not falsify or support the directional hypothesis. "
                f"Recorded stop: {status.get('error', 'unspecified failure')}. The original deadline was not extended.")
    return "The full grid is not yet evaluated; no complete-curve inference is available."


def pilot_table(trials):
    """Show missing trials as missing, while retaining every measured development score."""
    lines = ["| Cell | LR | Status | Images trained | Development KL | Development accuracy |",
             "|---|---:|---|---:|---:|---:|"]
    for row in trials:
        dev = row["development"]
        kl, accuracy = (f"{dev['kl']:.6g}", f"{dev['accuracy']:.2%}") if dev else ("unmeasured", "unmeasured")
        lines.append(f"| {row['cell']} | {row['lr']:g} | {row['state']['status']} | {row['state'].get('images_seen', 0)} | {kl} | {accuracy} |")
    return "\n".join(lines)


def observations(progress):
    """Summarize every available paired pilot comparison, without turning tuning data into a main result."""
    measured = [row for row in progress["pilot_trials"] if row["development"] is not None]
    if not measured:
        return "No complete pilot development endpoints are available."
    pairs = [(row, other) for row in measured if row["cell"].endswith("_mse") for other in measured
             if other["cell"].endswith("_kl") and (row["k"], row["lr"]) == (other["k"], other["lr"])]
    wins = sum(proposed["development"]["kl"] < local["development"]["kl"] for local, proposed in pairs)
    accuracy = [row["development"]["accuracy"] for row in measured]
    return (f"Completed pilot development accuracies range from {min(accuracy):.1%} to {max(accuracy):.1%}. "
            f"Pure KL has lower development KL in {wins}/{len(pairs)} completed comparisons matched on k and learning rate. "
            "These descriptive tuning-set comparisons use pilot seed 900, are incomplete across sparsity, and provide no main-study or seed-robustness conclusion.")


def write_pilot_reports(progress, destination):
    """Save individual started-trial records and the full 50-trial inventory, including missing cells."""
    directory = destination / "pilots"
    directory.mkdir(exist_ok=True)
    for row in progress["pilot_trials"]:
        if row["state"]["status"] == "not_started":
            continue
        body = (f"# Pilot: {row['name']}\n\nDevelopment-only seed {row['seed']}; not a main-study or test result. "
                "See the [full inventory and limitations](../pilots.md).\n\n```json\n" + json.dumps(row, indent=2) + "\n```\n")
        (directory / f"{row['name']}.md").write_text(body, encoding="utf-8")
    text = ("# Complete pilot inventory\n\nAll planned trials are shown. Development metrics tune learning rates and cannot establish "
            "held-out effectiveness. An incomplete grid does not permit freezing selected learning rates.\n\n" + pilot_table(progress["pilot_trials"]) + "\n")
    (destination / "pilots.md").write_text(text, encoding="utf-8")


def figures(evidence, destination):
    """Plot recorded pilot endpoints and the implementation check; neither is a test-set result."""
    import matplotlib.pyplot as plt
    progress = evidence["progress"]
    measured = [row for row in progress["pilot_trials"] if row["development"] is not None]
    if measured:
        fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
        positions = {row["cell"]: i for i, row in enumerate(progress["main_runs"])}
        for lr in evidence["spec"]["learning_rates"]:
            rows = [row for row in measured if row["lr"] == lr]
            ax.scatter([positions[row["cell"]] for row in rows],
                       [row["development"]["kl"] for row in rows], label=f"LR {lr:g}", marker="o" if lr == min(evidence["spec"]["learning_rates"]) else "x")
        labels = [row["cell"].replace("_downstream_b", "_ds") for row in progress["main_runs"]]
        ax.set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
        ax.set(xlim=(-.5, len(labels) - .5), ylim=(0, None), ylabel="Development KL (nats/image)",
               title=f"Pilot endpoints: {len(measured)}/{progress['pilot_planned']} complete | blanks are unmeasured")
        ax.legend()
        fig.savefig(destination / "pilot_endpoints.png", dpi=160)
        plt.close(fig)
    started = [row for row in progress["pilot_trials"] if row["history"]]
    if started:
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        for row in started:
            ax.plot([h["images_seen"] for h in row["history"]], [h["kl"] for h in row["history"]],
                    label=f"{row['cell']} LR {row['lr']:g}", linewidth=.8)
        ax.set(xlabel="Training images seen", ylabel="Unsmoothed batch KL", ylim=(0, None),
               title="Pilot training only | seed 900 | no main-study inference")
        ax.legend(fontsize=6, ncol=2)
        fig.savefig(destination / "pilot_training.png", dpi=150)
        plt.close(fig)
    verification = evidence.get("verification")
    if verification:
        curve = np.asarray(verification["fixed_batch_kl"])
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.plot(np.arange(len(curve)), curve)
        ax.set(xlabel="Optimizer update (loss before update)", ylabel="Fixed training-batch KL", ylim=(0, None),
               title="Implementation check only | eight training images | seed 901")
        fig.savefig(destination / "verification.png", dpi=160)
        plt.close(fig)
