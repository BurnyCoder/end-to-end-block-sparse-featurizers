"""Global: exploratory causal checks without semantic claims; local: fixed k=16 maps and swaps.

Adapted from the existing diagnostics.py; BSF map source: https://arxiv.org/html/2606.25234v1#A7.
"""

from pathlib import Path
import numpy as np
import torch
from ..featurizer import reconstruct
from ..objectives import forward_kl
from ..io import write_json
from ..logging_utils import log_event
from .cache import check_deadline
from .evaluation import load_cell


@torch.no_grad()
def inspect_cells(spec, ctx, logger, deadline=None):
    """Choose blocks using development patches; intervene on all 201 positions of each code."""
    dev, test = ctx.indices("dev", 250), ctx.indices("test")
    rows = ctx.rows
    targets = [int(i) for label in range(10) for i in [j for j in test if rows[int(j)]["label"] == label][:2]]
    for cell in [c for c in spec.cells() if c.k == 16]:
        run = Path(spec.root) / "runs" / cell.key
        destination = run / "diagnostics.json"
        if destination.exists():
            continue
        check_deadline(deadline)
        model, state = load_cell(run / "checkpoint.pt", ctx)
        frames = model.frames()
        atoms = frames.transpose(1, 2)
        energy = torch.zeros(model.n_groups, device=ctx.device)
        for start in range(0, len(dev), state["microbatch"]):
            check_deadline(deadline)
            clean, _, _ = ctx.batch(dev[start:start + state["microbatch"]])
            _, codes, _ = reconstruct(model, ctx.normalization, clean, frames)
            z = codes.reshape(len(clean), 201, model.n_groups, model.group_size)
            energy += z[:, 5:].square().sum((0, 1, 3))
        selected = energy.topk(8).indices
        examples = []
        for label in range(6):
            i = next(int(j) for j in dev if rows[int(j)]["label"] == label)
            clean, _, teacher = ctx.batch([i])
            modified, codes, _ = reconstruct(model, ctx.normalization, clean, frames)
            logits, _ = ctx.suffix(modified)
            heat = codes.reshape(201, model.n_groups, model.group_size)[5:, selected].norm(dim=-1).T.reshape(8, 14, 14)
            examples.append({"id": rows[i]["id"], "label": label, "block_ids": selected.cpu().tolist(),
                             "heatmaps": heat.cpu().tolist(), "original_probabilities": teacher.softmax(-1)[0].cpu().tolist(),
                             "modified_probabilities": logits.softmax(-1)[0].cpu().tolist()})
        rng, interventions = np.random.default_rng(42), []
        for i in targets:
            check_deadline(deadline)
            same = next(int(j) for j in test if rows[int(j)]["label"] == rows[i]["label"] and rows[int(j)]["group"] != rows[i]["group"])
            other = next(int(j) for j in test if rows[int(j)]["label"] == (rows[i]["label"] + 1) % 10)
            clean, _, teacher = ctx.batch([i, same, other])
            normalized = ctx.normalization(clean)
            reconstruction, codes = model(normalized.flatten(0, 1), frames)
            codes = codes.reshape(3, 201, model.n_groups, model.group_size)
            base = reconstruction.reshape_as(normalized)[:1]
            energies = codes[0].square().sum((0, 2))
            group = int(selected[energies[selected].argmax()])
            eligible = (torch.arange(model.n_groups, device=ctx.device) != group) & (energies > 0)
            distances = (energies - energies[group]).abs().masked_fill(~eligible, float("inf"))
            nearest = distances.topk(min(8, int(eligible.sum())), largest=False).indices
            random_group = int(nearest[int(rng.integers(len(nearest)))]) if len(nearest) else group
            contribution = codes[0, :, group] @ atoms[group]
            variants = {"baseline": base, "remove_selected": base - contribution,
                        "remove_random_matched": base - codes[0, :, random_group] @ atoms[random_group],
                        "swap_same_class": base - contribution + codes[1, :, group] @ atoms[group],
                        "swap_other_class": base - contribution + codes[2, :, group] @ atoms[group]}
            values = {name: float(forward_kl(ctx.suffix(ctx.normalization.inverse(value))[0], teacher[:1]).mean())
                      for name, value in variants.items()}
            interventions.append({"id": rows[i]["id"], "group": group, "selected_energy": float(energies[group]),
                                  "random_group": random_group, "random_energy": float(energies[random_group]),
                                  "same_class_donor": rows[same]["id"], "other_class_donor": rows[other]["id"], "kl": values})
        result = {"cell": cell.key, "label": cell.label, "k": cell.k, "seed": spec.seed,
                  "selection_split": "development", "selection_scope": "patch energy", "intervention_scope": "all 201 tokens",
                  "selected_blocks": selected.cpu().tolist(), "examples": examples, "interventions": interventions,
                  "caveat": "Fixed targets include inactive selected blocks; random matching is approximate; no semantic correspondence claim."}
        write_json(destination, result)
        log_event(logger, "sweep_diagnostics", {"cell": cell.key, "targets": len(interventions), "selected_blocks": result["selected_blocks"]})
