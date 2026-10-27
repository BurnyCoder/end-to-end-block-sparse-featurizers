"""Global: examine information beyond KL. Local: bounded feature maps and causal interventions.

Map definitions follow https://arxiv.org/html/2606.25234v1#A7. These diagnostics are
exploratory effects on the composite classifier; they do not establish semantic concepts.
"""

from pathlib import Path
import numpy as np
import torch

from .evaluation import load_model, load_teacher
from .featurizer import reconstruct
from .io import read_json, write_json
from .objectives import forward_kl


@torch.no_grad()
def inspect_run(config, checkpoint, logger):
    """Select blocks on development, then apply fixed test-image interventions."""
    root, checkpoint = Path(config.root), Path(checkpoint)
    destination = checkpoint.parent / "diagnostics.json"
    if destination.exists():
        return read_json(destination)  # Do not choose new examples after seeing results.
    model, state = load_model(checkpoint, config.device)
    probe, normalization = load_teacher(root, config.device)
    features = np.load(root / "patches.npy", mmap_mode="r")
    rows = read_json(root / "manifest.json")["images"]
    rng = np.random.default_rng(42)
    dev = np.concatenate([rng.permutation([i for i, row in enumerate(rows) if row["split"] == "dev" and row["label"] == label])[:25]
                          for label in range(10)])
    energy = torch.zeros(model.n_groups, device=config.device)
    for start in range(0, len(dev), state["microbatch"]):
        patches = torch.from_numpy(np.array(features[dev[start:start + state["microbatch"]]], copy=True)).to(config.device)
        _, codes, _ = reconstruct(model, normalization, patches)
        energy += codes.square().sum((0, 2))  # Aggregate development energy before choosing blocks.
    selected = energy.topk(min(8, model.n_groups)).indices
    examples = [next(int(i) for i in dev if rows[int(i)]["label"] == label) for label in range(6)]
    maps = []
    for i in examples:
        patches = torch.from_numpy(np.array(features[i:i + 1], copy=True)).to(config.device)
        modified, codes, _ = reconstruct(model, normalization, patches)
        heat = codes[:, selected].norm(dim=-1).T.reshape(len(selected), 14, 14)
        maps.append({"id": rows[i]["id"], "label": rows[i]["label"], "split": "dev",
                     "block_ids": selected.cpu().tolist(), "heatmaps": heat.cpu().tolist(),
                     "original_probabilities": probe(patches).softmax(-1)[0].cpu().tolist(),
                     "modified_probabilities": probe(modified).softmax(-1)[0].cpu().tolist()})
    tests = [i for i, row in enumerate(rows) if row["split"] == "test"]
    targets = [i for label in range(10) for i in [j for j in tests if rows[j]["label"] == label][:2]]
    atoms = model.atoms()
    interventions = []
    for i in targets:
        same = next((j for j in tests if rows[j]["label"] == rows[i]["label"] and rows[j]["group"] != rows[i]["group"]), None)
        different = next((j for j in tests if rows[j]["label"] == (rows[i]["label"] + 1) % 10), None)
        if same is None or different is None:
            continue  # Explicitly record fewer eligible pairs rather than inventing a donor.
        ids = [i, same, different]
        patches = torch.from_numpy(np.array(features[ids], copy=True)).to(config.device)
        normalized = normalization(patches)
        reconstructed, codes = model(normalized.flatten(0, 1))
        codes = codes.reshape(3, 196, model.n_groups, model.group_size)
        reconstructed = reconstructed.reshape_as(normalized)
        energies = codes[0].square().sum((0, 2))
        group = int(selected[energies[selected].argmax()])
        alternatives = torch.arange(model.n_groups, device=config.device)
        eligible = (alternatives != group) & (energies > 0)
        distance = (energies - energies[group]).abs().masked_fill(~eligible, float("inf"))
        # Pick randomly from the eight closest-energy active alternatives using the fixed RNG.
        nearest = distance.topk(min(8, int(eligible.sum())), largest=False).indices
        random_group = int(nearest[int(rng.integers(len(nearest)))]) if len(nearest) else group
        original_logits = probe(patches[:1])
        base = reconstructed[:1]
        contribution = codes[0, :, group] @ atoms[group]
        random_contribution = codes[0, :, random_group] @ atoms[random_group]
        variants = {"baseline": base, "remove_selected": base - contribution,
                    "remove_random_matched": base - random_contribution,
                    "swap_same_class": base - contribution + codes[1, :, group] @ atoms[group],
                    "swap_other_class": base - contribution + codes[2, :, group] @ atoms[group]}
        values = {name: float(forward_kl(probe(normalization.inverse(value)), original_logits).mean())
                  for name, value in variants.items()}
        interventions.append({"id": rows[i]["id"], "group": group, "selected_energy": float(energies[group]),
                              "random_group": random_group, "random_energy": float(energies[random_group]),
                              "same_class_donor": rows[same]["id"], "other_class_donor": rows[different]["id"],
                              "kl": values})
    result = {"condition": state["condition"], "seed": state["seed"],
              "selection_split": "dev", "selection_images": len(dev), "selected_blocks": selected.cpu().tolist(),
              "examples": maps, "interventions": interventions,
              "interpretation": "Exploratory perturbations of a ten-class readout; no semantic causal claim."}
    write_json(destination, result)
    return result
