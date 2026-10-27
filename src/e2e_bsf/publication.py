"""Global: reproducible public model artifacts. Local: curated safe bundles and HF collection.

Sources: https://huggingface.co/docs/huggingface_hub/guides/upload;
https://huggingface.co/docs/huggingface_hub/guides/collections.
Only BSFs, their fitted readout/statistics and measured outputs are uploaded.
"""

from pathlib import Path
import shutil
from safetensors.torch import save_file
from huggingface_hub import HfApi

from .io import load_checkpoint, read_json, write_json, digest
from .logging_utils import log_event

BUNDLE_FILES = ("model.safetensors", "config.json", "manifest.json", "duplicate_audit.json", "teacher.json",
                "study_schedule.json", "study_status.json", "pilot.json", "metrics.json", "test_predictions.npz",
                "history.jsonl", "diagnostics.json", "run.json", "GOODFIRE_LICENSE", "LICENSE", "README.md")


def stage_bundle(config, checkpoint):
    """Local: export weights without an executable pickle; global: avoid shipping raw inputs/secrets."""
    checkpoint = Path(checkpoint)
    state = load_checkpoint(checkpoint)
    stage = Path(config.root) / "publication" / checkpoint.parent.name
    stage.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for prefix, field in (("bsf", "model"), ("probe", "probe"), ("normalization", "normalization")):
        tensors.update({f"{prefix}.{key}": value.detach().cpu().contiguous() for key, value in state[field].items()})
    save_file(tensors, str(stage / "model.safetensors"))
    metadata = {key: state[key] for key in ("format_version", "architecture", "condition", "seed", "config",
                "model_id", "model_revision", "bsf_revision", "source_revision", "manifest_sha256", "teacher_sha256",
                "epoch", "updates", "status")}
    metadata.update(pooling="absolute_max", weights_sha256=digest(stage / "model.safetensors"))
    write_json(stage / "config.json", metadata)
    for filename in ("manifest.json", "duplicate_audit.json", "teacher.json", "study_schedule.json", "study_status.json", "pilot.json"):
        source = Path(config.root) / filename  # Public image identities and fitted-selection metadata.
        if source.exists():
            shutil.copy2(source, stage / filename)  # Reproduction needs assignments, not just their digest.
    metrics = read_json(checkpoint.parent / "test_summary.json")
    write_json(stage / "metrics.json", metrics)
    for filename in ("test_predictions.npz", "history.jsonl", "diagnostics.json", "run.json"):
        source = checkpoint.parent / filename
        if source.exists():
            shutil.copy2(source, stage / filename)
    shutil.copy2("vendor/goodfire/LICENSE", stage / "GOODFIRE_LICENSE")
    shutil.copy2("LICENSE", stage / "LICENSE")
    report = Path("reports") / ("smoke" if config.profile == "smoke" else "study") / "results.json"
    evidence = read_json(report) if report.exists() else None
    scope = (evidence["label"] + " " + evidence["narrative"]) if evidence else "Interpret this individual artifact without an aggregate effectiveness claim."
    card = ("---\nlanguage: en\nlicense: mit\ntags:\n- sparse-autoencoder\n- interpretability\n- dinov3\n- block-sparsity\n"
            "base_model: facebook/dinov3-vitb16-pretrain-lvd1689m\n---\n\n"
            f"# {state['condition']} / seed {state['seed']}\n\n"
            "Grassmannian block-sparse featurizer trained from scratch for a fixed DINOv3/Imagenette patch classifier. "
            f"Profile: **{config.profile}**; completed epochs: **{state['epoch']}**. "
            "Scalar artifacts are tied signed width-one featurizers, not conventional ReLU SAEs.\n\n"
            f"**Study-level interpretation:** {scope}\n\n"
            f"Measured held-out KL: **{metrics['kl']:.8g} nats/image**; accuracy: **{metrics['accuracy']:.4%}**; "
            f"normalized patch MSE: **{metrics['nmse']:.8g}**. Full metrics and per-image logits accompany this model.\n\n"
            "## Use\n\nInstall the [project](https://github.com/BurnyCoder/end-to-end-block-sparse-featurizers) "
            "with `uv sync --frozen`, download this repository with `hf download`, then:\n\n"
            "```python\nfrom PIL import Image\nfrom e2e_bsf.inference import VisionFeaturizer\n"
            "model = VisionFeaturizer(\"path/to/downloaded/model\")\n"
            "probabilities = model.predict([Image.open(\"image.jpg\").convert(\"RGB\")])\n```\n\n"
            "The artifact contains BSF weights, fixed normalization and the trained classifier. "
            "It excludes DINO weights and photos. Authorized DINO access and its separate license are required; "
            "the MIT license here does not relicense the backbone. Class order follows torchvision Imagenette.\n\n"
            "## Limits and provenance\n\nLower KL only measures preservation of this ten-class readout. "
            "It does not establish semantic concepts, full DINO fidelity, or inference acceleration. "
            "`config.json` binds revisions, architecture, objective, seed, preprocessing and training identities. "
            "`manifest.json` and `duplicate_audit.json` preserve image assignments and exclusion evidence. "
            "`teacher.json` records classifier selection. `model.safetensors` is for inference; the local training "
            "checkpoint additionally retains optimizer/RNG state.\n\n"
            "Sources: [BSF paper](https://arxiv.org/abs/2606.25234), "
            "[end-to-end SAE paper](https://arxiv.org/abs/2405.12241), "
            "[Goodfire code](https://github.com/goodfire-ai/block-sparse-featurizer), "
            "[preregistration](https://github.com/BurnyCoder/end-to-end-block-sparse-featurizers/blob/main/docs/preregistration.md).\n")
    (stage / "README.md").write_text(card, encoding="utf-8")
    return stage


def publish(config, logger):
    """Local: upload an allowlisted bundle; global: preserve all completed conditions, not just winners."""
    api = HfApi()
    account = api.whoami()["name"]
    if account != "BurnyCoder":
        raise RuntimeError(f"Expected BurnyCoder, authenticated as {account}; refusing wrong-account publication")
    root = Path(config.root)
    records = []
    candidates = sorted((root / "runs").glob("*/test_summary.json"))
    if not candidates:
        raise RuntimeError("No evaluated completed models are available to publish")
    collection = api.create_collection(title=f"End-to-End BSFs on DINOv3 - {config.profile}",
                                        namespace=account, exists_ok=True,
                                        description="From-scratch KL/local/mixed objectives with matched scalar controls; read cards for scope.")
    for summary in candidates:
        if summary.parent.name.startswith("pilot_"):
            continue
        stage = stage_bundle(config, summary.parent / "checkpoint.pt")
        repo = f"{account}/e2e-bsf-dinov3-imagenette-{config.profile}-{summary.parent.name.replace('_', '-')}"
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(repo_id=repo, folder_path=stage, allow_patterns=list(BUNDLE_FILES),
                          commit_message="Publish measured BSF weights, readout, provenance and outputs")  # Explicit Hub allowlist excludes incidental files.
        api.add_collection_item(collection_slug=collection.slug, item_id=repo, item_type="model", exists_ok=True)
        records.append({"repository": repo, "url": f"https://huggingface.co/{repo}"})
        write_json(root / "publication.json", {"collection": collection.url, "models": records})
        log_event(logger, "published_model", records[-1])
    return {"collection": collection.url, "models": records}
