# Source and model attribution

This repository implements native MuLaCover generator primitives and reference
tooling following the sibling sa3.cpp / heartcodec.cpp architecture. The MIT
license at the repository root covers this project's own contributions; upstream
components retain their respective licenses.

| Component or reference | Source / terms |
| --- | --- |
| HeartMuLa/MuLaCover at `e27fa7fa0da7c9817879cc6210cc5693f9327946` | Apache-2.0 source; copy retained in `LICENSES/MuLaCover-Apache-2.0.txt`. Model architecture, symbolic conditioning, sampling and reference pipeline are attributed to its authors. |
| MuLaCover checkpoint weights and generated outputs | Separate upstream terms retained in `LICENSES/MuLaCover-model-and-output-terms.txt`; the source MIT license does not replace these terms. No weights or audio are bundled. |
| betweentwomidnights/heartcodec.cpp | Linked sibling dependency, MIT project source with its own third-party inventory and RNG notices. |
| betweentwomidnights/ggml | Shared pinned dependency supplied by HeartCodec; retain that fork's license when distributing it. No second GGML copy is bundled here. |
| betweentwomidnights/sa3.cpp | MIT source; library ownership, capability negotiation, parity and Philox development patterns. |
| PyTorch / torchtune / Transformers / Qwen | External reference and model dependencies, not vendored source or checkpoints. Retain their applicable terms when supplying those dependencies. |

The reference scripts execute separately checked-out pinned upstream source.
Model/checkpoint retrieval and distribution are separate from this source-only
repository. See the exact pinned checkpoint terms before distributing weights
or generated outputs.
