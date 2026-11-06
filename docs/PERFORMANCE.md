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

## Device projections and reusable workspaces

Two further experiments are independently opt-in at generator context creation:

- `MULACOVER_DEVICE_CACHE=1` retains the eight style-affine results, both
  cross-attention K/V projections (musical attention and gate), and the last
  cumulative score context directly on the selected backend. Cold preparation
  uses the original GGML math and device-to-device copies. Warm frames use these
  tensors without CPU readback/upload of score contexts. This flag takes
  precedence over the earlier host score cache. Nonzero experimental tempo
  conditioning bypasses both score caches.
- `MULACOVER_GRAPH_REUSE=1` reuses nine session-owned metadata arenas and graph
  allocators: embedding, backbone and seven local-head slots. This follows
  ACE-Step's `ggml_reset` arena pattern using existing APIs. Graphs are still
  rebuilt and allocation plans are reserved on every call: equal tensor shapes
  and node counts do not guarantee equal lifetimes when debug outputs, views or
  persistent inputs change. Allocator buffers retain their high-water storage.
  Slots are borrowed by one synchronous graph at a time, and die before the
  backend. No backend graph plan or CUDA graph capture is introduced.

Both flags default **off**; no public ABI layout or EOS/RNG policy changes.
Begin/reset/failure rewind frees the device score cache. Conditional and
unconditional branches own separate tensors. Metadata/allocator slots survive
reset, but store no caller pointers or reusable output handles. The optional parity CLI
condition-reuse probe now also changes style, symbolic shape and CFG branch
count, then checks cold-result restoration.

Three short paired trials on the same 540-row workload above, rotating order:

| Path | Median generator seconds | Less time than original |
| --- | ---: | ---: |
| Original | 4.226 | — |
| Earlier host score cache | 3.795 | 10.2% |
| Device projections only | 2.129 | 49.6% |
| Workspaces only | 3.805 | 10.0% |
| Device projections + workspaces | 2.009 | 52.5% |

These timings exclude loading, include two cold preparations/reset-replays, and
write no diagnostic tensors. The existing Windows configuration has
`GGML_CUDA_GRAPHS=OFF`; the gain does not rely on enabling CUDA graph capture.
These are generator-only results for one laptop/workload, not end-to-end DAW
speedups. The two gains do not add independently.

For 540-row CFG, persistent device score storage is **226,832,384 bytes
(216.32 MiB)**. Combined reusable graph buffers retain about 47.7 MiB in the
short timed case, plus 46.2 MiB of CPU metadata arenas. This replaces the host
cache's 101.25 MiB of score vectors, but increases persistent device storage.
The private profile report exposes those costs separately from the frozen
memory API's existing per-graph high-water count.

Checks: all 600 Spark Docker FP32 tensors pass existing rules (the same
two original CFG flat-absolute diagnostics remain false). All 100 exported Q4
tensors at frame 11 are byte-identical to the original for device-only,
workspace-only and combined paths; the uncached refactor also matches the prior
baseline. The combined experimental tempo/local-window path bypasses score
caching and matches all 100 original frame-2 tensors. Reset/replay and
score/style/shape/branch replacement probes pass. Windows CUDA static and Spark
Docker CTests pass 15/15; Windows CPU DLL tests pass 14/14. This is not CPU model
parity or new Metal/Vulkan validation.

The smaller real provided-score workload (128 rows, otherwise the same short
teacher/replay settings) improves from median **2.856s to 1.957s, 31.5% less
generator time**, across three paired runs. A separate 12-guide/20-sampled-frame
check, Torch CUDA cap 288 and UINT64_MAX seed, matches all 256 codes, EOS state
and sampler offset 640 with replay/terminal checks intact.

A longer fresh control uses 29.76 seconds / 372 guide frames, score-none 540
rows, native-prepared 25-position prompt/style, seed 1701 and a 90-second new
audio ceiling. Both paths emit 713 new frames / 57.04 seconds then EOS; all
**8,680 codes** and sampler offset 22,848 are byte-identical. Generator time with
checks is **188.371s -> 80.429s**. This is one diagnostic pair, not a statistical
long-song benchmark; nvidia-smi monitoring runs in both, and brief CUDA smoke
tests overlap the beginning of the baseline diagnostic. No decode is performed.
Whole-device sampled peak (including desktop/other processes) is **4,579 MiB ->
4,688 MiB**, out of 8,123 MiB, at capacity 1536. Combined allocator storage
retains 56,745,472 bytes in this case. These results establish this laptop case;
maximum API capacity and 5,000-row scores remain unvalidated.

The long comparison is against the **fresh matched uncached control**. Its
native-prepared Qwen bytes differ from the historical cached upstream listening
fixture, although guide codes are identical. Neither equality to that historical
output nor a cause for its different ending is asserted here. Existing strict
upstream waveform limits are unchanged. Compact evidence is recorded separately
in `docs/validation/generator-device-workspaces.json`.

## Next targets

1. Review the opt-in device-cache/workspace implementations and extend memory
   coverage to longer scores and maximum API capacities before default adoption.
2. Reuse complete one-frame graph construction/plans where shapes, lifetimes and
   external pointers are explicitly stable; the current workspace experiment
   retains buffers/metadata but still rebuilds graphs and reserves their plans.
3. Add a dedicated guide-priming path that updates backbone KV without unused
   local-head logits. The existing teacher-frame API must still return all its
   documented outputs; optimize the job through an explicit primitive instead.
4. Explore CFG batching and active-KV attention ranges after the above. Those
   change matrix shapes/reduction behavior and need separate numerical checks.

Randomizing seeds does not fix or suppress EOS. An EOS investigation should hold
style/score/guide constant across fixed seeds and inspect terminal-head logits.
The earlier 30-second/empty-style versus 29.76-second/accepted-style comparison
changes two inputs and cannot isolate a cause of early stopping.
