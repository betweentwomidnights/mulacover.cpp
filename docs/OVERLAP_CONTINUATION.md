# Experimental overlap continuation

The first Love Is Blue past-only overlap experiment produced 24.16 seconds of
new music instead of the baseline 0.08 seconds. The user heard a coherent,
useful continuation. This is listening acceptance of one source/seed/preset,
not a duration guarantee or a general musical-quality result.

Follow-up (2026-11-08): the user reports that results remain hit or miss and
premature endings are still severe. Keep this option experimental and off by
default; [issue #6](https://github.com/betweentwomidnights/mulacover.cpp/issues/6)
tracks the unresolved problem independently of the implementation merge.

## Native staging and ownership

Negotiate `mulacover_get_overlap_api(1)` from `mulacover_overlap.h`. Existing
job structs, tables, behavior and unsigned seed defaults remain frozen. The
new config defaults to zero overlap. Set `overlap_frames=8` for the tested
0.64-second overlap (each frame is 80ms). Allowed values are 0--8192, clamped
to `guide_frames-1` so at least one teacher-forced frame remains.

The job encodes exactly the original past context, then teacher-forces the
guide except its final overlap frames. It samples replacement frames for that
tail and continues beyond the original cursor. Decoding includes those
replacement frames; the new-audio result crops at the **original cursor**.
The requested new-audio ceiling and cache capacity do not move. No future
audio is read, and no EOS suppression, minimum duration, padding or stretching
is applied. EOS inside the overlap or before a complete new frame returns the
existing no-new-audio error without publishing a partial result.

The new result owns its nested `job` allocations. Free it through the overlap
table's `result_free`, which is repeatable, before unloading the library.
Inputs and observers are borrowed only for the synchronous call. Cancellation
remains cooperative between backend operations. The existing priming stage
uses the name `regenerating overlap` during the sampled past frames.

In this capability, `job.guide_frames` means the logical decoded prefix before
the new-audio boundary. Its last `staging.overlap_frames` codes are generated,
not the original encoder IDs. `model_guide_frames` identifies the genuinely
teacher-forced prefix. Never use the entire exported `guide.i32` as an encoder
parity fixture for an overlap run. The full preview changes that guide tail;
the suffix begins at the same cursor. Noncausal codec decoding can also change
the waveform of earlier teacher-forced frames.

Overlap frames consume RNG draws. With the tested Torch CUDA cap 288, eight
frames consume 256 Philox offset units; EOS also consumes a sampled draw.
This is an explicit staging change, not an identical-stream continuation.

## CLI and server

`mulacover-continue --overlap-frames 8` selects the new capability. Omitted/zero
uses the existing job call. The HTTP options field is `overlap_frames` (integer,
default zero). Completed metadata records effective `regenerated_overlap_frames`
and `model_guide_frames`, alongside the existing cursor, new frames and offsets.
The advanced UI exposes an unchecked experimental checkbox for Continue audio.
The basic Transcribe + remix UI keeps its existing behavior: one provided-score
overlap probe still ended early and has no listening approval.

## Controls and limits

Matched frozen-condition Q4 controls, with unchanged sampling and EOS rules:

| Case | Baseline new audio | RNG offset only | Eight-frame overlap after cursor |
|---|---:|---:|---:|
| Billiegasm, lofi, seed 17442424470364887450, context 26.72s | 8.64s | 29.60s | 20.80s |
| Love Is Blue, warm acoustic, seed 1701, context 20s | 0.08s | 0.16s | 24.16s |

All these runs ended naturally before their respective ceilings. The offset-only
control skips 256 offset units without replacing the guide tail. It shows that
the Billiegasm improvement cannot be attributed solely to tail regeneration.
For Love Is Blue, the tested offset alone does not reproduce the improvement.
One full-source provided-score Love overlap run yielded only 7.12 seconds after
the original cursor (7.76 sampled seconds including overlap), without a listening
assessment. This does not solve remix length adherence.

A separate future-encoder-context Billiegasm control produced 16.32 new seconds;
it is research evidence only and is not used by this past-only feature. Encoder
codes change broadly with context, so padding/edge effects remain a hypothesis.
This work does not prove a root cause or eliminate the need for an upstream
raw-audio control. No LoRA was trained. See `SERVER_LIFECYCLE_EOS.md` for the
scoped cleanup investigation and `validation/lifecycle-overlap.json` for checks.
Original strict waveform/encoder and expanded activation failures remain;
shared GGML remains at coordinated `e3db5210`.
