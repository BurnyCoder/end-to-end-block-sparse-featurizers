"""Global: verify public artifacts; local: compare Hub objects and exercise two downloaded bundles.

Sources: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api;
https://git-scm.com/book/en/v2/Git-Internals-Git-Objects. This audit does not fit any parameters.
"""

from datetime import datetime, timezone  # Timestamp external verification separately from the study.
import hashlib  # Recompute Git blob identities for ordinary files and SHA-256 for large files.
from pathlib import Path  # Restrict downloads and outputs to this checkout.
import shutil  # Preserve the exact publication manifest alongside the report.
from huggingface_hub import HfApi, snapshot_download  # Use public Hub APIs without exposing credentials.
import torch  # Compare downloaded inference outputs with the fixed development example.

from e2e_bsf.extraction import image_batch
from e2e_bsf.inference import VisionFeaturizer
from e2e_bsf.io import read_json, write_json
from e2e_bsf.publication import BUNDLE_FILES


def audit():
    """Verify all published files and two representative architectures without changing the models."""
    root = Path("outputs/study")
    publication = read_json(root / "publication.json")
    api = HfApi(token=False)  # Publication must be usable anonymously; only DINO itself is gated.
    collection = api.get_collection(publication["collection"].split("/collections/")[1])
    expected = {entry["repository"] for entry in publication["models"]}
    assert len(expected) == 25 and {item.item_id for item in collection.items} == expected
    staged = {f"BurnyCoder/e2e-bsf-dinov3-imagenette-full-{path.name.replace('_', '-')}": path
              for path in (root / "publication").iterdir() if path.is_dir()}
    verified = []
    for repository in sorted(expected):
        info = api.model_info(repository, files_metadata=True)
        assert not info.private
        stage = staged[repository]
        files = {entry.rfilename: entry for entry in info.siblings}
        local = {path.name for path in stage.iterdir() if path.is_file() and path.name in BUNDLE_FILES}
        assert set(files) == local | {".gitattributes"}  # No photos, secrets or executable checkpoints.
        for name in sorted(local):
            payload = (stage / name).read_bytes()
            remote = files[name]
            assert remote.size == len(payload)
            if remote.lfs:
                assert remote.lfs.sha256 == hashlib.sha256(payload).hexdigest()
            else:
                blob = b"blob " + str(len(payload)).encode() + b"\0" + payload
                assert remote.blob_id == hashlib.sha1(blob).hexdigest()  # Git's content-addressed format.
        verified.append({"repository": repository, "commit": info.sha, "files_verified": sorted(local),
                         "weights_sha256": read_json(stage / "config.json")["weights_sha256"]})
        print(datetime.now(timezone.utc).isoformat(), "verified", repository, flush=True)
    roundtrips = []
    rows = read_json(root / "manifest.json")["images"]
    for run in ("block_kl_seed100", "scalar_kl_seed100"):
        repository = f"BurnyCoder/e2e-bsf-dinov3-imagenette-full-{run.replace('_', '-')}"
        revision = next(record["commit"] for record in verified if record["repository"] == repository)
        directory = snapshot_download(repository, revision=revision, allow_patterns=["config.json", "model.safetensors"],
                                      local_dir=root / "download_verification" / run, token=False)
        example = read_json(root / "runs" / run / "diagnostics.json")["examples"][0]
        index = next(i for i, row in enumerate(rows) if row["id"] == example["id"])
        assert rows[index]["split"] == "dev"  # A fixed development example avoids new test-based selection.
        model = VisionFeaturizer(directory)
        outputs = model.predict(image_batch(rows, [index], root.parent / "data"))
        record = {"run": run, "image": example["id"], "normalized_mse": outputs["normalized_mse"]}
        for field, reference in (("original", "original_probabilities"), ("inserted", "modified_probabilities")):
            expected_probabilities = torch.tensor(example[reference])
            torch.testing.assert_close(outputs[field][0], expected_probabilities, atol=1e-5, rtol=1e-4)
            record[f"{field}_maximum_absolute_error"] = float((outputs[field][0] - expected_probabilities).abs().max())
        roundtrips.append(record)
        del model  # Release the backbone before testing the other architecture.
        torch.cuda.empty_cache()
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
              "collection": publication["collection"], "models": verified, "downloaded_inference": roundtrips}
    write_json(Path("reports/study/publication_verification.json"), result)
    shutil.copy2(root / "publication.json", "reports/study/publication.json")
    print("Public collection, all file identities and both real-image inference roundtrips passed.", flush=True)


if __name__ == "__main__":
    audit()  # Explicit invocation keeps external verification out of ordinary offline unit tests.
