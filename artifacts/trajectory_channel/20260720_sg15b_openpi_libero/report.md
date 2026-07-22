# SG-15B Compression Pipeline Report

## Scope

The workload is the SG-15 target Storage shard: 8 slots × 48 chunk steps,
four 256×256 RGB image leaves and the major OpenPI tensor leaves. Raw payload
is 314,511,360 bytes. Images are deterministic synthetic tiles, not captured
LIBERO camera frames.

## Local round-trip p50

| implementation | raw | LZ4 | Zstd |
|---|---:|---:|---:|
| SG-15 block-then-pack | 13.14 ms | 223.47 ms | 313.70 ms |
| SG-15B direct-pack/reuse, 1 lane | 13.81 ms | 217.15 ms | 303.96 ms |
| SG-15B direct-pack/reuse, 2 lanes | 14.51 ms | 116.01 ms | 173.00 ms |
| SG-15B direct-pack/reuse, 4 lanes | 34.64 ms | 68.52 ms | not measured |

## Cross-host Gloo round-trip

| lanes | mode | p50 | p95 | p99 | wire MiB | effective GiB/s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | raw | 164.20 ms | 168.67 ms | 169.50 ms | 299.94 | 1.78 |
| 1 | LZ4 | 234.21 ms | 257.88 ms | 271.97 ms | 23.72 | 1.25 |
| 1 | Zstd | 317.70 ms | 320.71 ms | 320.78 ms | 15.85 | 0.92 |
| 2 | raw | 167.76 ms | 180.71 ms | 186.79 ms | 299.94 | 1.75 |
| 2 | LZ4 | 134.40 ms | 137.44 ms | 139.10 ms | 23.72 | 2.18 |
| 2 | Zstd | 191.90 ms | 192.70 ms | 192.89 ms | 15.85 | 1.53 |
| 4 | raw | 168.32 ms | 170.87 ms | 171.40 ms | 299.94 | 1.74 |
| 4 | LZ4 | 84.99 ms | 87.66 ms | 88.56 ms | 23.72 | 3.45 |
| 8 | LZ4 | 91.37 ms | 94.76 ms | 95.55 ms | 23.72 | 3.21 |

Four-lane LZ4 improves paired p50 by 49.5% and p99 by 48.3% relative to raw.
Eight lanes regress, so four lanes are the candidate configuration.

## Correctness and ownership

- Every warmup validates bitwise equality.
- Unit tests verify that a repeated schema reuses the same packed workspace
  address and performs zero new workspace allocations.
- Storage only reuses its workspace after synchronous Gloo send completes.
- Actor decompression uses independent codec contexts but allocates final
  batch-owned destination tensors.
- A real StorageWorker-to-Actor Worker compressed round-trip passed with
  codec threads.

## Decision and limitation

The example remains `compression.enabled: false`; `num_threads: 4` records the
measured candidate for placements where compression is enabled. The user has
separately accepted real LIBERO camera compressibility, but production default
enablement still requires a paired latency run on its target placement.
