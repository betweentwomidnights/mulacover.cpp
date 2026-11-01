"""Verify downloaded oracle payload hashes, shapes and finite tensors without Torch."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
m = json.loads((a.reference / "manifest.json").read_text())
count = [0]; size = [0]
def verify(root, files):
    for name, item in files.items():
        path = root / item["file"]; payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != item["sha256"]: raise ValueError(f"Hash mismatch: {path}")
        dtype = np.dtype(item["stored_dtype"])
        if len(payload) != math.prod(item["shape"]) * dtype.itemsize: raise ValueError(f"Shape mismatch: {path}")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(np.frombuffer(payload, dtype)).all(): raise ValueError(f"Nonfinite tensor: {path}")
        count[0] += 1; size[0] += len(payload)
verify(a.reference, m["preprocessing"])
cases = []
for case in m["cases"]:
    verify(a.reference / case["name"], case["files"])
    expected = {f"frame{frame}.logits.{q}" for frame in range(3) for q in range(8)}
    if not expected.issubset(case["files"]): raise ValueError("Missing codebook logits")
    for frame in range(3):
        for layer in range(28):
            for kind in ("k", "v"):
                if f"frame{frame}.cache.backbone.{layer}.{kind}" not in case["files"]: raise ValueError("Missing backbone KV prefix")
        for q in range(8):
            raw_key = f"frame{frame}.raw_logits.{q}"
            raw_item = case["files"][raw_key]
            raw = np.fromfile(a.reference / case["name"] / raw_item["file"], np.float32).reshape(raw_item["shape"])
            logits_item = case["files"][f"frame{frame}.logits.{q}"]
            logits = np.fromfile(a.reference / case["name"] / logits_item["file"], np.float32)
            # For BF16, guided arithmetic rounds after each operation. Raw and
            # guided logits are both frozen; do not replace that with FP32 math.
            if case["batch"] == 1 and not np.array_equal(raw.reshape(-1), logits): raise ValueError("Conditional logits changed")
            if case["batch"] == 2 and m["generator_dtype"] == "float32":
                guided = raw[1:] + (raw[:1] - raw[1:]) * np.float32(case["guidance"])
                if not np.array_equal(guided.reshape(-1), logits): raise ValueError("CFG arithmetic changed")
    if not case["replay_bit_exact"] or not case["cache_tail_zero"]: raise ValueError("Upstream replay/cache check failed")
    cases.append(dict(name=case["name"], frames=case["frames"], batch=case["batch"],
                      tensors=len(case["files"]), replay_bit_exact=True, cache_tail_zero=True,
                      peak_allocated_bytes=case["peak_allocated_bytes"]))
report = dict(passed=True, upstream_revision=m["upstream_revision"], models=m["models"],
              versions=m["versions"], generator_dtype=m["generator_dtype"], style_dtype=m["style_dtype"],
              files_verified=count[0], payload_bytes=size[0], cases=cases,
              manifest_sha256=hashlib.sha256((a.reference / "manifest.json").read_bytes()).hexdigest())
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(f"Verified {count[0]} payloads ({size[0]} bytes): {a.reference}")
