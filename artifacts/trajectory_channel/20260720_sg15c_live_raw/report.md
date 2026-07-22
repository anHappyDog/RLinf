# SG-15C Raw Live Policy Transport

The target per-rank payload contains `B=8`, two 256×256 RGB uint8 tensors and
one `[B, 8]` float32 state tensor, totalling 3,145,984 tensor bytes.

| path | sender p50/p99 | receiver p50/p99 |
|---|---:|---:|
| ChannelWorker two-hop | 5.28 / 5.90 ms | 5.10 / 6.34 ms |
| direct fixed-frame | 1.82 / 1.89 ms | 1.81 / 1.85 ms |

`local.json` contains the raw paired measurements. A standalone two-frame
cross-Worker correctness test passed. The real target Runner then completed a
10-step rollout and one Actor/Critic update with 32 LIBERO environments and
four Env, Rollout, and Actor ranks.

Live XOR/LZ4, pinned buffers, H2D overlap, extra camera views, variable batches,
and non-one-to-one rank mappings are intentionally outside this baseline.
