# Official upstream full-song FP32 control

The user hears guitar-melody vocalization in the native FP32 instrumental-lane
run as well as the original Q4 laptop run. This control executes the pinned
official PyTorch generator and codec on the same request, to test whether that
behavior also occurs through the upstream implementation.

On November 2, 2026, the official pipeline completed successfully and sampled
EOS after 861 accepted frames, producing 68.88 seconds of finite 48kHz stereo.
Fresh preprocessing matched all 11 frozen native-input tensors byte-for-byte.
Generation took 370.93 seconds and codec execution took 46.44 seconds, excluding
weight loading. This run does not perform a second full upstream replay.

Torch peak allocated CUDA memory was 29,342,910,976 bytes during generation and
9,975,715,840 bytes during codec execution. These are allocator measurements for
this process, not total physical GPU usage. The reference image ID is
`sha256:70947194330272d2aed83dfa51923165ef013a32b0a03f9be8afb2d65957524b`.

The captured float waveform has peak 1.1252625, RMS 0.1722574, and 12 individual
channel samples above full scale. The default PCM16 export is retained unchanged.
The comparison files under `refdata/covers/billiegasm-upstream-listening/` use one
shared constant gain of 0.79203825 (-2.02508 dB), preserving relative levels while
bringing the highest peak to -1 dBFS. No limiting or edits are applied. The
original native F32 output remains 70 seconds; the EOS difference is retained.
The user hears the guitar melody played by horns, without vocalizations, in this
official output. That accepts the instrumental request for this particular run,
not every prompt or seed. The follow-up [matched-Philox control](PHILOX_PARITY.md)
now reproduces all generated codes/EOS natively; strict decoded waveform error
remains a separately recorded failure.

The source remains MuLaCover `e27fa7fa0da7c9817879cc6210cc5693f9327946`, with
generator `bbbaef2b31835c2ef5172ff528dbe325230d46fb`, Qwen
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, and HeartCodec
`f889dab0532cfa4bf459f2a3367eb6d346b8eeda`. The codec snapshot is the same source
used by native HeartCodec's SQ/flow conversion. No source checkout is modified.

`reference/run-upstream-audio.py` calls the actual upstream `preprocess`,
`_forward`, and `postprocess` methods. It freshly regenerates prompt/style and
symbolic tensors and requires all 11 shapes, dtypes and hashes to match the
prepared native instrumental-lane control. It preserves the official sampler,
model cache setup and codec implementation. Hooks only observe progress and
capture the float decoder output before the official WAV export.
The executed harness is preserved as `harness-used.py` with its hash in the
compact report. The final helper moves its observation-only progress hook from
the backbone root to its first layer; the root is bypassed by `generate_frame`.
That logging correction and recording the helper's source hash do not change
the preprocessing, generator, sampler, codec or export operations used here.

Settings remain `[Instrumental]`, `warm acoustic pop, soft drums, instrumental,
no vocals`, 875 requested frames (70 seconds), CFG 1.5, temperature 1 and top-k
250. The instrumental lane has 137 notes; the same 160 voiced chord notes and
zero drum conditioning are retained. Generator, Qwen and codec run in FP32,
with TF32 disabled and deterministic Torch algorithms enabled.

The Torch seed is reset to 1701 after generator weights load, and independently
to 1701 after codec weights load. These seeds do not reproduce the native
SplitMix64 sampling or C++ codec noise. The official generator uses its default
8,192-position cache, while the native control uses 1,024. Therefore this is a
behavioral control, not matched-noise or sample-for-sample parity. A different
sample alone cannot establish a port defect; matched-noise/logit investigation
would be needed to attribute a discrepancy. Existing longer-context prefill
activation gate failures remain open.

The original upstream `postprocess` saves `audio-official.wav` using soundfile's
default PCM16 encoding. The harness also captures the same waveform directly into
`audio-f32.wav`, without clipping, normalization or model changes. Raw generated
tokens are saved frame-major as `codes.i32`. EOS is respected rather than padded.

## Reproduction

Run from the Spark host after preparing the original full-song request. All
Python/model execution takes place in the dedicated reference Docker image;
production services and shared GGML pins remain unchanged.

```bash
docker run --rm --gpus all --user 1000:1000 --network none \
  -e USER=heartcodec -e LOGNAME=heartcodec -e HOME=/workspace \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v /home/kev/dev/mulacover.cpp:/workspace \
  -v /home/kev/dev/heartcodec.cpp:/codec:ro \
  -v /home/kev/dev/_heartcodec_refs:/refs:ro -w /workspace \
  heartcodec-reference:20261101 /opt/sa3-venv/bin/python reference/run-upstream-audio.py \
  --upstream /refs/MuLaCover \
  --prepared refdata/covers/billiegasm-full-acoustic/prepared \
  --request refdata/covers/billiegasm-full-acoustic/request.json \
  --codec-source /codec/models/full-decoder/source.json \
  --codec-snapshot /codec/.cache/huggingface/hub/models--HeartMuLa--HeartCodec-oss-20260123/snapshots/f889dab0532cfa4bf459f2a3367eb6d346b8eeda \
  --output refdata/covers/billiegasm-full-acoustic/official-upstream-f32
```

Use a fresh output directory for another run. Model execution requires no source
audio upload: the original text/MIDI request already resides in the development
mount. Compact evidence is recorded in `validation/upstream-audio-control.json`;
raw audio, tensors and checkpoints remain outside Git. Listener assessment is
separate from successful execution and prepared-input equality.
