# GGML baseline inventory

Updated November 1, 2026 after the user selected the shared baseline. HeartCodec's
recorded gitlink and local/Spark working checkouts use
`b77ae790e628270af46a8af1eca38441c23f1252`. MuLaCover reuses that single target.
The remote branch is `feature/shared-sa3-acestep-v0.17`; its name is historical.
Use the exact SHA as authority. This commit's own CMake metadata reports 0.27.0.

| Consumer | Recorded GGML gitlink at its HEAD | Working GGML HEAD |
| --- | --- | --- |
| heartcodec.cpp | `b77ae790e628270af46a8af1eca38441c23f1252` | same, clean, laptop and Spark |
| sa3.cpp | `b77ae790e628270af46a8af1eca38441c23f1252` | `4ad3b30bfe452efc5f4f9c0e111e017d9970292a` |
| yue2.cpp | `b77ae790e628270af46a8af1eca38441c23f1252` | same, clean |

sa3/yue2 were inspected only. Their source, gitlinks and working checkouts were
not modified. Do not automatically repair sa3's checkout state.

Historical native parity used `07b77b0112cc30861b49116b057bac33e819ee06`.
The current harness defaults to clean `b77ae79`, checks the full source SHA and
the executable's linked SHA, and keeps earlier results unchanged. See
[the new baseline report](GGML_B77_ALIGNMENT.md) for the CUDA precision failure
and the isolated review candidate. No consumer pin includes that candidate.
