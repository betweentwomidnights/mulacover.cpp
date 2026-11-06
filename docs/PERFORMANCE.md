# Generator performance investigation

This investigation keeps coordinated GGML `e3db5210` unchanged. Profiling and the
candidate score cache are local MuLaCover code. The cache is **opt-in**, pending
broader review and long-song/memory tests; normal builds keep the existing path.
No math precision, quantization policy, frozen ABI layout, sampler or EOS policy
changes. The UI/server random-seed change is independent (see SERVER.md).

## Measured baseline

`MULACOVER_PROFILE=1` prints per-role JSON to stderr at context destruction.
Roles are embedding, backbone and local decoder. Times cover host graph building,
forward expansion/support checks/allocation, input uploads, synchronized backend
compute, and output readback. These are stage timings, not per-kernel GPU timings.
GGML's synchronous graph-compute API already synchronizes the backend; profiling
adds no extra GPU synchronization. Library defaults produce no profile output.

On the RTX 5070 Laptop, Q4_K_M, CFG 1.5, 17 prompt positions, 540 symbolic rows,
capacity 256, 12 teacher frames with two reset/replays (24 frames):

| Initial baseline role | Graphs | Host build ms | Allocation/preparation ms | Compute ms |
| --- | ---: | ---: | ---: | ---: |
| Backbone including score adaptors | 48 | 791.4 | 105.1 | 2206.6 |
| Embedding | 48 | 3.1 | 3.0 | 38.9 |
| Local decoder | 336 | 66.1 | 30.0 | 653.5 |

The backbone is the largest measured cost. Every frame currently executes eight
symbolic adaptors per CFG branch, with two transformer layers each, despite
unchanged score/style inputs. Each frame also builds seven separate local-decoder
graphs per branch. Graph metadata/allocator construction is measurable overhead.

## Candidate: cache invariant symbolic contexts

Set `MULACOVER_SCORE_CACHE=1` **before creating a generator context**. On the first
frame each branch computes all eight cumulative adaptor outputs using existing
GGML operations. Exact F32 bytes are retained on the host and uploaded as inputs
for later frames. No model computations are moved to CPU. The CPU storage costs
101.25 MiB for CFG with 540 symbolic rows; graph-buffer high-water increased from
52,042,112 to 85,219,712 bytes in this workload. This is not a device-peak measure;
the frozen memory API's KV-cache count does not include host score storage.

The cache belongs to a session. Begin/reset, condition replacement and failure
rewind invalidate it. Conditional/unconditional branches remain separate.
Nonzero experimental tempo conditioning bypasses it, because score masks can
depend on the current position. The default is off; set 0 or omit the variable
for the original path. `MULACOVER_PROFILE=1` reports actual cache hits/misses.

The first short timing was 4.77s -> 3.76s (21%), but repeated paired runs give the
more useful result: **median 4.541s -> 4.004s, 11.8% less generator time**. Three
paired trials alternate order. A trial that might overlap another GPU check was
excluded. Model loading is excluded; cold score preparation and reset/replay are
included. A longer 64-frame/two-replay diagnostic was 24.347s -> 20.879s (14.2%);
that run also writes late-frame diagnostic tensors. These are not full-pipeline
or all-workload speedup claims. A 128-row sampled test improved only about 3%.

Validation on the candidate:

- Spark Docker F32 conditional/CFG: all 600 tensors pass the unchanged rules;
  exact replay/cache-tail checks pass. The two original CFG flat-absolute
  activation diagnostics remain failed; no new waiver is introduced.
- Q4 baseline/candidate: all 100 exported tensors at frame 11, and all 100 at
  frame 63, are byte-identical. Warm-cache reset/replay is exact.
- Real provided 128-row score: 12 guide + 20 sampled frames, seed UINT64_MAX,
  Torch CUDA cap 288. All 256 codes, EOS state and sampler offset 640 match.
- Replacing a warmed condition with a changed score changes logits; reset of
  that condition and restoration of the original condition reproduce their
  respective cold logits exactly.
- Windows CUDA static/CPU DLL CTests pass. This does not establish CPU model
  accuracy, Metal/Vulkan performance, maximum-cache memory fit, long-song parity
  or a general musical-quality improvement.

Full scalar gates and existing upstream waveform/encoder limits remain as
recorded. Raw tensors, inputs and builds stay outside Git. Compact evidence is
in `docs/validation/generator-performance.json`.

For Spark, `reference/run-spark.sh` forwards the two diagnostic environment flags
into Docker. Set `MULACOVER_PARITY_OUTPUT` to a fresh directory when comparing
cache modes; the GGML checkout must still match the exact coordinated pin.

## Next targets

1. Keep invariant adaptor/cross-attention K/V and style-affine results on device,
   avoiding repeated projections and the prototype's host transfers. Measure the
   additional persistent VRAM against the laptop budget before adoption.
2. Reuse graph metadata, workspaces and allocations for repeated one-frame
   shapes. Current rebuilding cost is large; use existing GGML APIs first.
3. Add a dedicated guide-priming path that updates backbone KV without unused
   local-head logits. The existing teacher-frame API must still return all its
   documented outputs; optimize the job through an explicit primitive instead.
4. Explore CFG batching and active-KV attention ranges after the above. Those
   change matrix shapes/reduction behavior and need separate numerical checks.

Randomizing seeds does not fix or suppress EOS. An EOS investigation should hold
style/score/guide constant across fixed seeds and inspect terminal-head logits.
The earlier 30-second/empty-style versus 29.76-second/accepted-style comparison
changes two inputs and cannot isolate a cause of early stopping.
