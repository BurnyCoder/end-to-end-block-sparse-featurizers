"""Global: prevent image leakage. Local: reproducible download, duplicate groups, splits.

Sources: torchvision.datasets.Imagenette; https://github.com/JohannesBuchner/imagehash;
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html.
"""

from collections import Counter  # Summarize exclusions without hiding images.
from pathlib import Path  # Relative image paths make manifests portable.
import hashlib  # Exact file identity complements perceptual similarity.
import imagehash  # Reuse perceptual DCT hashing rather than inventing an embedding.
import numpy as np  # Vectorized Hamming distances avoid a quadratic Python inner loop.
from PIL import Image, ImageOps  # Normalize EXIF orientation before hashing.
from sklearn.model_selection import StratifiedGroupKFold  # Preserve duplicate groups.

from .io import read_json, write_json  # Atomic reusable manifests.
from .logging_utils import log_event  # Audit split and exclusion counts.


def assign_splits(rows, seed=42):
    """Local: group-aware 80/20 split; global: test-related groups never enter fitting."""
    rows = [dict(row) for row in rows]  # Do not mutate the caller's source manifest.
    test_groups = {row["group"] for row in rows if row["official_split"] == "val"}
    eligible = []  # Only official training images without test overlap can be fitted.
    for i, row in enumerate(rows):
        row["split"] = "test" if row["official_split"] == "val" else "excluded"
        if row["official_split"] == "train" and row["group"] not in test_groups:
            row["split"] = "train"
            eligible.append(i)
    labels = [rows[i]["label"] for i in eligible]  # Stratification preserves class coverage.
    groups = [rows[i]["group"] for i in eligible]  # Near duplicates remain in one fold.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, dev = next(splitter.split(eligible, labels, groups))  # Exactly one prespecified fold.
    for position in dev:
        rows[eligible[position]]["split"] = "dev"
    return rows


def prepare_manifest(config, logger):
    """Local: hash all source images; global: no model is fitted before split audit."""
    from torchvision.datasets import Imagenette  # Import lazily for offline unit tests.

    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return read_json(manifest_path)["images"]  # Persisted assignments are immutable.
    data_root = root.parent / "data"  # Share source images between smoke and full experiments.
    data_root.mkdir(parents=True, exist_ok=True)
    downloaded = (data_root / "imagenette2-320").exists()
    training = Imagenette(data_root, split="train", size="320px", download=not downloaded)
    testing = Imagenette(data_root, split="val", size="320px", download=False)
    samples = [(path, label, split) for dataset, split in ((training, "train"), (testing, "val"))
               for path, label in dataset._samples]  # Torchvision provides verified archive content.
    parent = list(range(len(samples)))  # Union-find records transitive duplicate components.
    hashes = np.empty(len(samples), dtype=np.uint64)  # One 64-bit perceptual hash per image.
    exact = {}  # SHA-256 equality handles byte-identical files independently of pHash.
    rows, matches = [], []  # Keep every detection for later audit.

    def find(index):
        """Local: path compression; global: transitive near duplicates form one unit."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, (path, label, split) in enumerate(samples):
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()  # Small compressed image files.
        with Image.open(path) as opened:
            phash = int(str(imagehash.phash(ImageOps.exif_transpose(opened).convert("RGB"))), 16)
        hashes[i] = phash
        neighbors = np.flatnonzero(np.bitwise_count(hashes[:i] ^ hashes[i]) <= 4).tolist()
        if sha in exact:
            neighbors.append(exact[sha])  # Exact matches are included even if decoding changes.
        for j in sorted(set(neighbors)):
            parent[find(i)] = find(j)  # Join without inspecting labels or model outcomes.
            matches.append([j, i, int((int(hashes[j]) ^ phash).bit_count())])
        exact[sha] = i
        relative = Path(path).relative_to(data_root).as_posix()
        rows.append({"id": relative, "label": int(label), "official_split": split,
                     "sha256": sha, "phash": f"{phash:016x}", "group": i})
        if (i + 1) % 1000 == 0:
            log_event(logger, "hash_progress", {"images": i + 1, "matches": len(matches)})
    for i, row in enumerate(rows):
        row["group"] = find(i)  # Materialize final connected components before splitting.
    assigned = assign_splits(rows, config.seed)
    selected = [row for row in assigned if row["split"] != "excluded"]
    if config.profile == "smoke":  # Deterministic, stratified real-image slice, never a headline study.
        rng = np.random.default_rng(config.seed)
        subset = []
        for split, count in (("train", 20), ("dev", 5), ("test", 5)):
            for label in range(10):
                candidates = [row for row in selected if row["split"] == split and row["label"] == label]
                chosen = rng.permutation(len(candidates))[:count]
                subset.extend(candidates[int(index)] for index in chosen)
        selected = subset
    selected.sort(key=lambda row: row["id"])  # Stable cache row IDs independent of filesystem listing.
    write_json(root / "duplicate_audit.json", {"phash_hamming_threshold": 4, "pairs": matches,
                                              "all_images": assigned})
    write_json(manifest_path, {"seed": config.seed, "profile": config.profile,
                               "counts": dict(Counter(row["split"] for row in selected)),
                               "images": selected})
    log_event(logger, "manifest", {"selected": len(selected),
                                   "counts": dict(Counter(row["split"] for row in selected)),
                                   "excluded_train_overlap": sum(row["split"] == "excluded" for row in assigned)})
    return selected
