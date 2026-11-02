# Deferred installed-GGML packaging observation

A local unapplied diff is saved in ignored
`build-artifacts/ggml-installed-cuda-dependencies.patch`; it is not committed,
published, or adopted. The installed static CUDA package omitted dynamic CUDA
runtime/cuBLAS dependencies and, when enabled, NCCL. The existing in-tree target
already declares these correctly, so this parity milestone does not require a
GGML change or PR.

HeartCodec's local and Spark GGML working trees were restored to commit
`07b77b0112cc30861b49116b057bac33e819ee06`; all consumer pins remain unchanged.
Native parity is rebuilt against that source pin, mounted read-only in Spark
Docker. Do not apply this patch or update pins as part of routine model work.
If installed-package support becomes necessary, first reproduce it in an isolated
GGML worktree, add focused package-consumer tests, and submit a draft PR for other
agents to review. Cross-repository adoption is a separate coordinated decision.
See `../GGML_PIN_INVENTORY.md`: the inspected repositories are not all currently
at the same commit, and this patch must not be treated as an alignment mechanism.
