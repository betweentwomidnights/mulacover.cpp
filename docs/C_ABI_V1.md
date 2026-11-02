# Teacher-forced generator ABI v1

The public C header is `include/mulacover.h`. Resolve `mulacover_get_api(1)`, reject
null, and check `api->size >= MULACOVER_API_V1_SIZE`. The table has library lifetime.
Unsupported versions return null. Set every argument/result `size = sizeof(value)`;
configuration and frame initializers preserve larger caller tails. The v1 table
prefix ends at `get_memory`. Direct C exports remain available.

`create` loads a strict F32 MuLaCover GGUF using the exact `cpu` or `cuda` backend;
there is no silent fallback. `session_begin` copies prepared prompt, style and
symbolic buffers and clears persistent state. The initial frame consumes the
entire prompt; later frames consume the preceding eight teacher-forced audio
codes at one new backbone position. Caches are bounded by `cache_capacity`
(default 8192). `session_reset` keeps conditioning and rewinds to the first frame.
Calls on one context must be serialized.

Inputs are frame-major arrays: prompt `[T,9]` int32 and mask bytes; Qwen `[1024]`,
MuQ `[512]`; piano `[S,256]`, drum `[S,128]`, chord `[S,12]`; and symbolic padding
mask `[S]` with one meaning padding. Prompt audio IDs are 0..8196 and text IDs
0..128255. Every mask byte is 0 or 1; at least one symbolic key must be unmasked.
The Qwen/MuQ placeholder indices must be distinct positions inside the prompt.
Preprocessing/tokenizer/Qwen inference are caller responsibilities in this version.

`teacher_frame` takes eight RVQ codes in 0..8191. It computes all heads and echoes
the supplied codes. It does not sample or process EOS. Returned `raw_logits` are
`[8,branches,8197]` in conditional-then-unconditional order. `logits` are `[8,8197]`
after CFG. Guidance <=1 uses only the conditional branch; guidance >1 uses two
branches and `U + scale*(C-U)`. `cache_position` is the active backbone prefix
length after the frame.

Frame arrays are library-owned. Initialize/free through this library, release a
previous result before reusing it, and keep the library loaded until results are
freed. Results remain valid after session reset or context destruction. Invalid
frame arguments preserve session state. Execution/dump failures clear caches and
rewind the session; resume from the initial frame after such an error. Replacing
conditioning preserves the previous session if replacement cache allocation fails.
Changing branch count may temporarily require both old and new cache allocations.

`get_memory` reports owned model/cache buffers and the maximum graph allocation so
far, excluding backend/driver workspace. An optional dump directory serializes
debug states, all logits and logical expanded KV prefixes; cache tails are checked
for zero writes. The parity CLI consumes this same C ABI and exercises both reset
replay and result ownership.

Downstream source builds use `add_subdirectory` and link `mulacover::mulacover`.
The library depends on `heartcodec::heartcodec` and reuses the existing `ggml` or
`ggml::ggml` target. The default standalone path builds from the sibling HeartCodec
source pin. This prototype does not yet install an exported MuLaCover CMake package.
