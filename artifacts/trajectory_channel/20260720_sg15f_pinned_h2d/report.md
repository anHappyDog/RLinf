# SG-15F Pinned Receive and H2D

For the target per-rank live payload, direct pinned transport measured 1.42 ms
p50 versus 1.87 ms for direct pageable buffers, a 24.4% reduction. The raw
measurements are in `local.json`.

The fixed-frame E2E validates pinned Gloo receive together with uneven routing,
extra camera data and compressed consecutive frames. The real target Runner
completed a 10-step rollout and one Actor/Critic update with pinned receive and
OpenPI reusable pinned staging/non-blocking H2D enabled.

The action-dependent Env-policy loop does not permit overlap with a future
observation. H2D overlap is limited to CPU preparation/submission of subsequent
model-input fields.
