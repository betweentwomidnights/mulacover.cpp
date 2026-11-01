#!/usr/bin/env python3
"""Pinned upstream MIDI/prompt/style and teacher-forced generator cache fixtures."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import mido


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    a = p.parse_args()
    generator_dtype = getattr(torch, a.dtype)
    sys.path.insert(0, str(a.upstream / "src"))
    from mulacover import MuLaCover, MuLaCoverConfig, MuLaCoverGenConfig, MuLaCoverGenPipeline
    import mulacover.modeling as modeling
    from tokenizers import Tokenizer
    torch.set_num_threads(4)
    torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(upstream_revision=subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip(),
        versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "torchtune", "tokenizers", "numpy")},
        generator_dtype=a.dtype, style_dtype="float32", seed=1701, cases=[])
    sources = {name: json.loads(Path("models", name + ".json").read_text()) for name in ("generator", "style")}
    manifest["models"] = {name: {k: v for k, v in source.items() if k != "snapshot"} for name, source in sources.items()}
    generator = Path(sources["generator"]["snapshot"])
    style = Path(sources["style"]["snapshot"])
    # Original, short musical material; no external recording is required.
    def midi(name, notes, channel=0):
        file = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack(); file.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        events = []
        for start, pitch, duration in notes:
            events.extend([(start * 120, 1, pitch), ((start + duration) * 120, 0, pitch)])
        previous = 0
        for tick, on, pitch in sorted(events):
            track.append(mido.Message("note_on" if on else "note_off", channel=channel,
                                      note=pitch, velocity=90 if on else 0, time=tick - previous))
            previous = tick
        path = a.output / (name + ".mid"); file.save(path); return path
    melody = midi("melody", [(0, 60, 4), (4, 64, 4), (8, 67, 4), (12, 72, 4)])
    chord = midi("chord", [(0, n, 8) for n in (48, 52, 55)] + [(8, n, 8) for n in (53, 57, 60)])
    drums = midi("drum", [(t, 36 if t % 8 == 0 else 38, 1) for t in (0, 4, 8, 12)], 9)
    lyrics = "[Verse]\nA lantern swings beside the rain\nWe count the lights along the train\n"
    tags = " Warm Acoustic Pop, Soft Drums, Clear Voice "
    paths = dict(mulacover=generator, qwen=style)
    devices = dict.fromkeys(("mulacover", "qwen", "codec", "transcriptor"), torch.device("cuda"))
    dtypes = dict.fromkeys(devices, generator_dtype); dtypes["qwen"] = torch.float32
    pipeline = MuLaCoverGenPipeline(paths, devices, dtypes, Tokenizer.from_file(str(generator / "tokenizer.json")),
        MuLaCoverGenConfig.from_file(generator / "gen_config.json"), MuLaCoverConfig.from_pretrained(generator), lazy_load=True)
    files = {}
    root = a.output
    def save(name, value):
        if isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                if isinstance(item, torch.Tensor): save(name + f".{i}", item)
            return
        if not isinstance(value, torch.Tensor): return
        dtype = str(value.dtype)
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16: value = value.float()
        array = np.ascontiguousarray(value.numpy())
        extension = {np.dtype("float32"): "f32", np.dtype("int64"): "i64", np.dtype("bool"): "bool"}.get(array.dtype)
        if not extension: raise ValueError(f"Unsupported fixture dtype: {array.dtype}")
        path = root / (name + "." + extension)
        payload = array.tobytes(); path.write_bytes(payload)
        files[name] = dict(file=path.name, shape=list(array.shape), source_dtype=dtype,
                           stored_dtype=str(array.dtype), sha256=hashlib.sha256(payload).hexdigest())
    with torch.inference_mode():
        pipeline._load_qwen()
        normalized = tags.strip().lower()
        for key, value in pipeline._qwen_tokenizer([normalized], padding=True, truncation=False, return_tensors="pt").items():
            save("style.input." + key, value)
        hook = pipeline._qwen.register_forward_hook(lambda m, i, o: save("style.hidden", o.last_hidden_state))
        # Execute actual upstream preprocessing, including MIDI conversion and
        # prompt construction. The style encoder is released before generation.
        prepared = pipeline.preprocess(dict(lyrics=lyrics, tags=tags, melody_midi=melody, chord_midi=chord, drum_midi=drums),
                                       cfg_scale=1.5, max_audio_length_ms=240)
        hook.remove()
        for key, value in prepared.items(): save("input." + key, value)
        (root / "request.json").write_text(json.dumps(dict(lyrics=lyrics, tags=tags, normalized_tags=normalized), indent=2) + "\n")
        manifest["preprocessing"] = files
        files = {}
        model, loading = MuLaCover.from_pretrained(generator, device_map="cuda", dtype=generator_dtype,
                                                   local_files_only=True, output_loading_info=True)
        if any(loading[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Checkpoint did not load strictly: {loading}")
        model.eval().requires_grad_(False)
        teacher = torch.tensor([[0, 8191, 7, 42, 314, 4096, 123, 777],
                                [11, 17, 23, 29, 31, 37, 41, 43],
                                [60, 64, 67, 72, 4095, 4097, 8189, 2]], dtype=torch.long, device="cuda")
        for cfg in (1.0, 1.5):
            name = "conditional" if cfg == 1 else "cfg-1.5"
            root = a.output / name; root.mkdir(exist_ok=True); files = {}
            batch = 1 if cfg == 1 else 2
            inputs = {key: value[:batch].to("cuda") for key, value in prepared.items()}
            for key in ("pianoroll", "drum_pianoroll", "chord"): inputs[key] = inputs[key].to(generator_dtype)
            model.setup_caches(batch)
            save("teacher_frames", teacher)
            hooks = []
            step_index = [0]
            for module_name, module in model.named_modules():
                if module_name in ("backbone.layers.0", "backbone.layers.13", "backbone.layers.27", "backbone.norm",
                                   "peft_layers.0", "peft_layers.13", "peft_layers.27", "tag_mlp_layers.13", "tag_mlp_layers.27"):
                    hooks.append(module.register_forward_hook(lambda m, i, o, n=module_name: save(f"frame{step_index[0]}.stage." + n, o)))
                elif module_name in ("projection", "decoder"):
                    hooks.append(module.register_forward_hook(lambda m, i, o, n=module_name: save(f"frame{step_index[0]}.local.{call[0] % 8}." + n, o)))
            original_sampler = modeling.sample_topk
            original_cfg = model._sample_with_cfg
            logits_by_run = []
            for run in range(2):
                model.reset_caches()
                call = [0]
                logits = []
                def sampler(value, topk, temperature):
                    index = call[0]; call[0] += 1
                    logits.append(value.detach().clone())
                    if run == 0: save(f"frame{index // 8}.logits.{index % 8}", value)
                    return teacher[index // 8, index % 8].reshape(1, 1).to(torch.int32)
                modeling.sample_topk = sampler
                def cfg_sampler(value, *args):
                    if run == 0: save(f"frame{call[0] // 8}.raw_logits.{call[0] % 8}", value)
                    return original_cfg(value, *args)
                model._sample_with_cfg = cfg_sampler
                current = dict(inputs)
                try:
                    with (torch.autocast("cuda", dtype=generator_dtype) if generator_dtype != torch.float32 else nullcontext()):
                        for step in range(3):
                            step_index[0] = step
                            sample = model.generate_frame(**current, temperature=1.0, topk=250, cfg_scale=cfg, first_step=step == 0)
                            if not torch.equal(sample[0].long(), teacher[step]): raise RuntimeError("Teacher forcing failed")
                            valid = int(current["input_pos"].max().item()) + 1
                            if run == 0:
                                for group in ("backbone", "decoder"):
                                    for layer, block in enumerate(getattr(model, group).layers):
                                        cache = block.attn.kv_cache
                                        active = valid if group == "backbone" else 8
                                        for kind in ("k", "v"):
                                            tensor = getattr(cache, kind + "_cache")
                                            save(f"frame{step}.cache.{group}.{layer}.{kind}", tensor[:, :, :active])
                                            if torch.count_nonzero(tensor[:, :, active:]).item():
                                                raise RuntimeError("Cache tail was modified beyond the active prefix")
                            tokens = torch.zeros(batch, 1, 9, dtype=torch.long, device="cuda")
                            tokens[:, 0, :8] = sample
                            mask = torch.ones_like(tokens, dtype=torch.bool); mask[..., -1] = False
                            current.update(tokens=tokens, tokens_mask=mask, input_pos=current["input_pos"][:, -1:] + 1,
                                           qwen_indices=None, muq_indices=None, muq_embedding=None)
                finally:
                    modeling.sample_topk = original_sampler
                    model._sample_with_cfg = original_cfg
                logits_by_run.append(logits)
                if run == 0:
                    for h in hooks: h.remove()
            if not all(torch.equal(x, y) for x, y in zip(*logits_by_run)):
                raise RuntimeError("Teacher-forced logits differ after cache reset and replay")
            case = dict(name=name, guidance=cfg, frames=3, batch=batch, replay_bit_exact=True,
                        cache_tail_zero=True, files=files, peak_allocated_bytes=torch.cuda.max_memory_allocated())
            (root / "case.json").write_text(json.dumps(case, indent=2) + "\n")
            manifest["cases"].append(case)
            print(f"Frozen {name}: {len(files)} tensors; exact reset/replay", flush=True)
            del logits_by_run
            gc.collect()
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
