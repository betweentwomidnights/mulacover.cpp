"""Download pinned generator and style-encoder inputs inside the reference image."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

models = {
    "generator": ("HeartMuLa/MuLaCover", "bbbaef2b31835c2ef5172ff528dbe325230d46fb"),
    "style": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
}
root = Path("models")
root.mkdir(exist_ok=True)
for name, (repo, revision) in models.items():
    snapshot = snapshot_download(repo, revision=revision, max_workers=2,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    (root / (name + ".json")).write_text(json.dumps(dict(repo=repo, revision=revision, snapshot=snapshot), indent=2) + "\n")
    print(name + " ready: " + snapshot, flush=True)
