# GGML baseline inventory

Read-only inventory on November 1, 2026. No pins or other model checkouts were
changed to align these repositories.

| Consumer | Recorded GGML gitlink at its HEAD | Working GGML HEAD |
| --- | --- | --- |
| heartcodec.cpp | `07b77b0112cc30861b49116b057bac33e819ee06` | same, clean |
| sa3.cpp | `b77ae790e628270af46a8af1eca38441c23f1252` | `4ad3b30bfe452efc5f4f9c0e111e017d9970292a` |
| yue2.cpp | `b77ae790e628270af46a8af1eca38441c23f1252` | same, clean |

The initial MuLaCover native parity baseline uses HeartCodec's existing `07b77b0`
checkout, not a claim that every consumer is aligned. The Spark source is mounted
read-only, its full SHA/clean status is checked by the harness, and the executable
reports its linked GGML commit. These results establish parity on this pin only.

Selecting a common baseline, handling sa3's checkout state, reviewing any required
backend changes, and rerunning affected parity suites must be coordinated before
cross-repository adoption. Do not automatically checkout or update another
consumer's submodule. Any essential shared change needs an isolated branch and
reviewable PR first; no new GGML PR or commit was needed for this milestone.
