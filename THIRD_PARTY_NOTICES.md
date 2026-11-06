# Source and model attribution

This repository implements native MuLaCover generator primitives and reference
tooling following the sibling sa3.cpp / heartcodec.cpp architecture. The MIT
license at the repository root covers this project's own contributions; upstream
components retain their respective licenses.

| Component or reference | Source / terms |
| --- | --- |
| HeartMuLa/MuLaCover at `e27fa7fa0da7c9817879cc6210cc5693f9327946` | Apache-2.0 source; copy retained in `LICENSES/MuLaCover-Apache-2.0.txt`. Model architecture, symbolic conditioning, sampling and reference pipeline are attributed to its authors. |
| HeartMuLa/heartlib at `5858ca8f1ffe58d62be7c007ea55f1033c5ca456` | External Apache-2.0 reference source for the audio-guide encoder experiment. `HeartCodec-oss-encoder` is separately pinned to `e6020510612d6653b73d6cd54393ef80929e3c81`; source, checkpoint and output terms remain separate. No encoder source or weights are vendored here. |
| MuLaCover checkpoint weights and generated outputs | Separate upstream terms retained in `LICENSES/MuLaCover-model-and-output-terms.txt`; the source MIT license does not replace these terms. No weights or audio are bundled. |
| betweentwomidnights/heartcodec.cpp | Linked sibling dependency, MIT project source with its own third-party inventory and RNG notices. |
| betweentwomidnights/ggml | Shared pinned dependency supplied by HeartCodec; retain that fork's license when distributing it. No second GGML copy is bundled here. |
| betweentwomidnights/sa3.cpp | MIT source; library ownership, capability negotiation, parity and Philox development patterns. |
| betweentwomidnights/yue2.cpp | MIT source retained in `LICENSES/YuE2-MIT.txt`; native UTF-8/NFC and tokenizer splitting structure adapted for the conditioning frontend. Generated tables use the pinned reference runtimes. The transcription model remains an external sibling dependency. |
| betweentwomidnights/acestep.cpp | MIT source retained in `LICENSES/ACE-Step-MIT.txt`; Qwen3 encoder graph and lifecycle reference. Its weight loader/backend fallback and tokenizer approximations are not imported. |
| PyTorch / torchtune / Transformers / Qwen | External reference and model dependencies, not vendored source or checkpoints. Retain their applicable terms when supplying those dependencies. |

The reference scripts execute separately checked-out pinned upstream source.
Model/checkpoint retrieval and distribution are separate from this source-only
repository. See the exact pinned checkpoint terms before distributing weights
or generated outputs.
