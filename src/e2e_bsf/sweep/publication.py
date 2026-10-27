"""Global: usable public models without private inputs; local: explicit safe-bundle allowlist.

Sources: existing publication.py; https://huggingface.co/docs/huggingface_hub/guides/upload.
"""

from pathlib import Path
import shutil
from safetensors.torch import save_file
from huggingface_hub import HfApi
from ..config import MODEL_ID, MODEL_REVISION, BSF_REVISION
from ..io import read_json, write_json, load_checkpoint, digest
from ..logging_utils import log_event

FILES = ("model.safetensors", "config.json", "manifest.json", "duplicate_audit.json", "source_teacher.json",
         "study_schedule.json", "study_status.json", "pilot.json", "verification.json", "cache.json",
         "metrics.json", "test_predictions.npz", "history.jsonl", "diagnostics.json", "run.json",
         "deviations.json", "batch_numerics_diagnosis.json",
         "GOODFIRE_LICENSE", "LICENSE", "README.md")


def check_access():
    """Read account capability without printing tokens or creating public placeholders."""
    account = HfApi().whoami()
    if account["name"] != "BurnyCoder":
        raise RuntimeError("Hugging Face must be authenticated as BurnyCoder; run uv run hf auth login")
    role = account.get("auth", {}).get("accessToken", {}).get("role")
    if role == "read":
        raise RuntimeError("The active Hugging Face token is read-only; use a write-capable credential in ignored .env or hf login")
    return {"account": account["name"], "credential_role": role, "note": "Fine-grained repository permission is confirmed by the upload stage"}


def stage_bundle(spec, cell):
    """Only curated tensors and evidence enter the staging directory; no DINO weights or raw states."""
    root = Path(spec.root)
    run = root / "runs" / cell.key
    state = load_checkpoint(run / "checkpoint.pt")
    if state["status"] != "complete":
        raise ValueError("Publish completed models only")
    stage = root / "publication" / cell.key
    stage.mkdir(parents=True, exist_ok=True)
    tensors = {f"{prefix}.{name}": value.detach().cpu().contiguous()
               for prefix, field in (("bsf", "model"), ("probe", "probe"), ("normalization", "normalization"))
               for name, value in state[field].items()}
    save_file(tensors, str(stage / "model.safetensors"))
    metadata = {key: state[key] for key in ("format_version", "architecture", "cell", "recipe", "identity", "seed",
                                           "epoch", "updates", "images_seen", "source_revision", "initial_sha256")}
    metadata.update(model_id=MODEL_ID, model_revision=MODEL_REVISION, bsf_revision=BSF_REVISION,
                    insertion_block=6, sequence_scope="all_201", pooling="absolute_max", test_reuse=True,
                    weights_sha256=digest(stage / "model.safetensors"))
    write_json(stage / "config.json", metadata)
    metrics = read_json(run / "test_summary.json")
    write_json(stage / "metrics.json", metrics)
    for name in FILES:
        candidate = run / name if (run / name).exists() else root / name
        if name not in ("README.md", "config.json", "model.safetensors", "metrics.json") and candidate.exists():
            shutil.copy2(candidate, stage / name)
    shutil.copy2("vendor/goodfire/LICENSE", stage / "GOODFIRE_LICENSE")
    shutil.copy2("LICENSE", stage / "LICENSE")
    card = ("---\nlanguage: en\nlicense: mit\nbase_model: facebook/dinov3-vitb16-pretrain-lvd1689m\ntags:\n- interpretability\n- block-sparsity\n- dinov3\n---\n\n"
            f"# {cell.label}, k={cell.k}, seed 200\n\n"
            "Grassmannian BSF trained from scratch after DINOv3 block 6, replacing all 201 tokens. "
            "The frozen suffix applies blocks 7–12 and terminal normalization before a frozen patch-only Imagenette classifier. "
            f"Completed exposure: {state['epoch']} epoch(s), {state['images_seen']} training images.\n\n"
            f"Measured test KL: {metrics['kl']:.8g} nats/image; accuracy: {metrics['accuracy']:.4%}; "
            f"insertion normalized MSE: {metrics['insertion_nmse']:.6g}; downstream raw MSE: {metrics['downstream_mse']:.6g}.\n\n"
            "**Exploratory:** one shared training seed and a previously analyzed Imagenette test set. Image intervals "
            "condition on these fitted models and cannot establish seed robustness. This is a tied signed featurizer, not a ReLU SAE. "
            "Lower KL or reconstruction error does not establish concepts, preservation of all DINO information, or acceleration.\n\n"
            "## Inference\n\nInstall the [project](https://github.com/BurnyCoder/end-to-end-block-sparse-featurizers) with `uv sync --frozen`, "
            "download this bundle, and obtain authorized access to the separately licensed DINO model.\n\n"
            "```python\nfrom PIL import Image\nfrom e2e_bsf.inference import IntermediateVisionFeaturizer\n"
            "model = IntermediateVisionFeaturizer('path/to/downloaded/bundle')\n"
            "result = model.predict([Image.open('image.jpg').convert('RGB')])\n```\n\n"
            "## Reproducibility\n\nThe bundle contains safe BSF/readout/normalization tensors and complete image predictions, "
            "histories, frozen schedule, split identities and source revisions. Raw activation caches, photos, gated backbone "
            "weights and executable optimizer checkpoints are excluded. Local training checkpoints retain resume state. "
            "Raw downstream MSE is averaged equally over blocks 7–12; its coefficient is in config.json. "
            "There is no additional local reconstruction penalty in the KL+downstream conditions.\n\n"
            "Sources: [preregistration](https://github.com/BurnyCoder/end-to-end-block-sparse-featurizers/blob/main/docs/downstream-preregistration.md), "
            "[end-to-end dictionary learning](https://arxiv.org/html/2405.12241v2), [BSFs](https://arxiv.org/abs/2606.25234).\n")
    (stage / "README.md").write_text(card, encoding="utf-8")
    return stage


def publish(spec, logger):
    """Upload every completed cell to its own collection, preserving failures for resumable publication."""
    check_access()
    root, api = Path(spec.root), HfApi()
    collection = api.create_collection(namespace="BurnyCoder", title="DINOv3 BSFs - downstream reconstruction sweep",
        description="25 operating points; one seed; all-token block-6 insertion; reused Imagenette benchmark; read limitations.", exists_ok=True)
    records = []
    for cell in spec.cells():
        if not (root / "runs" / cell.key / "test_summary.json").exists():
            continue
        stage = stage_bundle(spec, cell)
        repo = f"BurnyCoder/e2e-bsf-dinov3-downstream-{cell.key.replace('_', '-')}-seed200"
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=False)
        commit = api.upload_folder(repo_id=repo, folder_path=stage, allow_patterns=list(FILES),
                                   commit_message="Publish measured block-6 sweep model and reproducibility evidence")
        api.add_collection_item(collection_slug=collection.slug, item_id=repo, item_type="model", exists_ok=True)
        records.append({"cell": cell.key, "repository": repo, "url": f"https://huggingface.co/{repo}", "commit": commit.oid})
        write_json(root / "publication.json", {"collection": collection.url, "models": records})
        log_event(logger, "sweep_published", records[-1])
    if not records:
        raise RuntimeError("No completed evaluated sweep models are available")
    return {"collection": collection.url, "models": records}
