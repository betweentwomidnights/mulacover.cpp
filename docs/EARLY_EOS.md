# Early-ending investigation

Status (2026-11-08): **open, severe usability problem**. The user reports that
continuation remains hit or miss after the overlap experiment. The successful
listening example does not establish reliable extension or source-length remix.
Track follow-up in [issue #6](https://github.com/betweentwomidnights/mulacover.cpp/issues/6).
Merging the working native implementation does not resolve this issue.

The reported Billiegasm case is reproducible: `[Instrumental]`, `lofi hip hop`,
seed **17442424470364887450**, guidance 1.5, temperature 1, top-k 250, 26.76s
requested context (26.72s / 334 complete frames), and a 30s new-audio ceiling.

| Matched Q4 control | Optimizations off | Device cache + workspaces on |
| --- | ---: | ---: |
| Billiegasm, exact reported settings | 8.64s new, EOS | 8.64s new, EOS |
| Love Is Blue, 20s guide, warm acoustic preset, seed 1701 | 0.08s new, EOS | 0.08s new, EOS |
| Love Is Blue, frozen full-source SheetSage2 score, lofi preset, 20s guide, seed 1701 | 0.08s new, EOS | 0.08s new, EOS |

All generated codes are byte-identical between each pair. The first two controls
run the native raw-PCM job, including CUDA encoding and stereo decoding: guide
codes, continuation WAVs and full WAVs also match byte-for-byte. Prepared text/
style is frozen between paths. The score control freezes the newly transcribed
MIDI, prepared score and guide and performs generator-only inference. It is not
a byte comparison against the user's expired browser session.

These runs rule out the optimization as the cause of these matched early endings.
They do not establish a universal optimization guarantee or resolve the underlying
short-output behavior. All three stop on sampled EOS, not cache exhaustion. No
forced minimum, EOS masking, padding, time stretching or RNG change was applied.
The upstream pipeline also stops if any sampled codebook value is at least 8193;
the native terminal rule is unchanged.

A **native unquantized F32 generator** control on Spark receives the exact frozen
Billiegasm prepared inputs/guide, seed and Torch CUDA cap 288. It ends after
**10.4s** of new audio rather than the 30s ceiling. This runs in a dedicated
HeartCodec Docker container. It is not a fresh upstream PyTorch comparison or an
upstream parity claim; F32 and laptop Q4 also run on different GPUs, so this does
not isolate quantization from hardware. Early endings also occur in this native
unquantized case. A further upstream comparison is needed before attributing
the remaining behavior specifically to the model, conditioning or native math.

Two separate server issues were corrected:

- A 132-second remix requires at least 1792 rounded cache positions with a short
  prompt. The former 1536-position default rejected it before inference. The
  default is now 2048; errors report required/available positions and explain
  that shortening the guide does not reduce a whole-source remix target. A
  2048-position / 455-row provided-score control has a sampled whole-device peak
  of 4804 MiB on the 8123 MiB laptop GPU. This does not validate every long score
  or the maximum 8192-position API capacity.
- The earlier `/p:OutDir` staging build silently linked CMake's original library
  paths, including old server-support metadata. Its optimization code was
  present, but health incorrectly identified the older source commit. CMake
  now honors `CMAKE_RUNTIME_OUTPUT_DIRECTORY`; normal linking rebuilds current
  libraries. Health/finished-job metadata expose optimization requests, and the
  UI identifies optimized/standard mode plus the build commit in its badge.
  These fields describe requested flags; native profiling records actual hits.

The canonical Windows build output has been restored to `build-windows-cuda/bin`.
Compact hashes, settings and limits are in
`docs/validation/early-eos-audit.json`. Existing strict waveform/encoder and
expanded activation failures remain unchanged; shared GGML stays at `e3db5210`.
