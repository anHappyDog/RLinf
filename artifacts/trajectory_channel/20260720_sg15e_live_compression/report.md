# SG-15E Live Image Compression

The paired workload uses `B=8`, two 256×256 RGB uint8 images and `[B, 8]`
float32 states. It runs three warmups and twenty measured frames.

| path | sender p50/p99 | receiver p50/p99 |
|---|---:|---:|
| ChannelWorker two-hop | 5.09 / 6.98 ms | 5.18 / 6.87 ms |
| direct raw | 1.61 / 1.89 ms | 1.61 / 2.00 ms |
| direct XOR+LZ4 | 2.80 / 2.95 ms | 2.80 / 2.95 ms |

The codec sends actual encoded views and falls back to raw when compression
does not reduce bytes. Unit tests validate keyframe and XOR-frame bitwise
restoration. Ray Worker E2E validates two consecutive compressed frames for
both 2 Env→1 Rollout and 1 Env→2 Rollout routing.

Compression remains disabled in the same-host target config because it is
slower than direct raw there. It is an explicit opt-in for placements whose
paired benchmark demonstrates a net latency gain.
