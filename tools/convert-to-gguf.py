#!/usr/bin/env python3
"""Strict unquantized MuLaCover conversion, including exact upstream RoPE caches."""
import argparse
import json
from pathlib import Path
import sys
import torch
import numpy as np
from gguf import GGUFWriter

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--upstream", type=Path, required=True)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
sys.path.insert(0, str(a.upstream / "src"))
from mulacover import MuLaCover
source = json.loads(a.source.read_text())
model, loading = MuLaCover.from_pretrained(source["snapshot"], dtype=torch.float32,
    device_map="cpu", local_files_only=True, output_loading_info=True)
if any(loading[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
    raise ValueError(f"Non-strict checkpoint load: {loading}")
model.eval().requires_grad_(False)
required = dict(backbone_flavor="llama-3B", decoder_flavor="llama-300M", text_vocab_size=128256,
    audio_vocab_size=8197, audio_num_codebooks=8, muq_dim=512, qwen_dim=1024, peft_layer_num=8, train_tag=True)
if any(getattr(model.config, k) != v for k, v in required.items()):
    raise ValueError("Unsupported generator configuration")
a.output.parent.mkdir(parents=True, exist_ok=True)
writer = GGUFWriter(str(a.output), "mulacover", use_temp_file=True)
writer.add_uint32("mulacover.format_version", 1)
writer.add_string("mulacover.checkpoint_revision", source["revision"])
writer.add_string("mulacover.weight_precision", "F32")
shapes = {}
def add(name, value):
    value = np.ascontiguousarray(value.detach().cpu().float().numpy())
    if not np.isfinite(value).all(): raise ValueError(f"Nonfinite tensor: {name}")
    writer.add_tensor(name, value)
    shapes[name] = list(value.shape)
state = model.state_dict()
if len(state) != 659: raise ValueError(f"Unexpected state count: {len(state)}")
for name, value in sorted(state.items()):
    if name == "audio_head":
        for q in range(7): add(f"audio_head.{q}.weight", value[q].T)
    else: add(name, value)
ropes = {}
for name, module in model.named_modules():
    if hasattr(module, "is_cache_built") and hasattr(module, "cache"):
        d = module.dim
        cache = module.cache[:8192].reshape(-1, d)
        # Some cross/adaptor caches cover 5000 positions. Choose a full 8192
        # cache for each head dimension, and verify equal overlapping entries.
        if d in ropes and not torch.equal(ropes[d][:len(cache)], cache[:len(ropes[d])]):
            raise ValueError(f"Inconsistent RoPE caches: {name}")
        if d not in ropes or len(cache) > len(ropes[d]): ropes[d] = cache
if set(ropes) != {12, 64, 128, 384}: raise ValueError(f"Unexpected head dimensions: {ropes.keys()}")
for d, cache in ropes.items():
    if len(cache) < 8192:
        from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
        full = Llama3ScaledRoPE(d, max_seq_len=8192, base=500000, scale_factor=32).cache.reshape(8192, d)
        if not torch.equal(full[:len(cache)], cache): raise ValueError("RoPE regeneration changed values")
        cache = full
    add(f"rope.cache.{d}", cache)
if len(shapes) != 669: raise ValueError("Unexpected runtime tensor count")
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
a.output.with_suffix(".conversion.json").write_text(json.dumps(dict(checkpoint_revision=source["revision"],
    dtype="float32", tensor_count=len(shapes), config=required, tensors=shapes,
    transformations=["audio_head split into seven transposed matmul weights", "exact upstream nonpersistent RoPE caches retained"]), indent=2) + "\n")
print(f"Wrote {a.output}: {len(shapes)} F32 tensors", flush=True)
