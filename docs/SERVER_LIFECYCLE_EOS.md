# Repeated-request and early-EOS investigation

The user's reports of worsening endings were tested with two fixed anchors in
one long-lived diagnostic server process. Billiegasm/lofi/seed
17442424470364887450 produced 8.64 new seconds; Billiegasm/warm acoustic/seed1701
with 29.76 seconds of context reached its 16-second ceiling without EOS.

Between the repeated anchors, the same process ran a different source, a
cooperative cancellation during priming, an intentional failure after style
preparation, and a full-source SheetSage2 transcription job. Prepared Qwen,
symbolic conditioning, guide IDs, generated IDs and both decoded WAVs were
byte-identical for each anchor before and after those jobs. Fresh-process
anchors also reproduced the same conditioning, codes and continuation WAVs.
The user's older server reproduced the same failing Billiegasm continuation WAV.
An initially rejected admission control is retained separately; only its corrected
replacement actually exercised the post-style worker failure.

This sequence did **not** reproduce progressive output degradation or demonstrate
a general absence of leaks. Private memory fluctuated after warming up rather
than rising on every request; whole-device usage was 757 MiB at the sampled
terminal checkpoints after warmup and release (including desktop/other processes).
The test is a bounded sequence, not a long soak or per-allocation leak proof.
Native ownership and HTTP worker joining were also inspected; no model session
is intentionally reused across jobs.

The opt-in `MULACOVER_EOS_TRACE=1` diagnostic records guided per-head terminal
logits/rank and EOS probability after the existing temperature/top-k filtering.
It consumes no RNG draws and records the first sampled frame, EOS and ceiling.
At the failing Billiegasm ending (108 new frames), codebook 0's EOS probability
was 0.999484658, rank 1; all eight heads sampled 8193. This is a strong ending
prediction in that trajectory, not an isolated low-probability special-token
draw. The good anchor's last ceiling frame had EOS outside the top 250.

`mulacover-server --diagnostic-dir DIR` optionally writes a unique local directory
per job with resolved options, prepared conditioning, guide/generated IDs and
metadata. It defaults off, is configured by the launcher only, and writes no raw
source WAV. Incomplete directories identify how far cancelled/failed jobs got.
These files contain user conditioning and are not automatically expired; manage
the local directory explicitly. No diagnostic data is uploaded or checked in.
In overlap mode the saved guide includes generated replacement frames; consult
the staging metadata before comparing encoder IDs.

The promising response is explicit past-only overlap regeneration, documented in
`OVERLAP_CONTINUATION.md`. It preserves natural EOS and does not silently retry
until an attractive duration appears. The ceiling is still only a ceiling: the
model is not given a separate duration-conditioning tensor. Training a duration
or continuation adapter remains a later option requiring verified MuLaCover
support and curated positive continuation examples as well as true endings.
