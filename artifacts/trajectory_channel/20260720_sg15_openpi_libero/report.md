# SG-15 Trajectory Channel Performance Report

## Scope

One target Storage shard: 8 slots × 48 chunk steps, four 256×256 RGB image
leaves plus the major OpenPI tensor leaves. Payload: 299.94 MiB.
Images are deterministic spatially coherent synthetic tensors; byte count and
schema match the target profile, while compression ratios must not be presented
as real-camera distribution measurements.

## Codec and cross-host transfer

| mode | local p50 ms | local p95 ms | cross p50 ms | cross p95 ms | cross p99 ms | cross p99.9 ms | wire MiB | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 13.14 | 13.21 | 164.18 | 167.94 | 168.22 | 168.28 | 299.94 | 1.00x |
| lz4 | 223.47 | 225.57 | 258.06 | 307.80 | 313.25 | 314.48 | 23.72 | 12.64x |
| zstd | 313.70 | 405.35 | 335.64 | 348.47 | 350.79 | 351.31 | 15.85 | 18.92x |

| mode | local CPU p50 ms | local RSS Δ MiB | rank0 CPU p50 ms | rank1 CPU mean ms | rank0 RSS Δ MiB | rank1 RSS Δ MiB | effective GiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 486.63 | 300.91 | 60.74 | 164.36 | 0.00 | 325.95 | 1.78 |
| lz4 | 1273.54 | 316.63 | 1208.23 | 169.73 | 61.49 | 312.03 | 1.14 |
| zstd | 534.64 | 332.04 | 380.15 | 196.11 | 22.84 | 335.05 | 0.87 |

On this high-bandwidth `bjd_dev → bjd_dev_2` link, raw is the fastest mode even
though LZ4/Zstd greatly reduce bytes. The target config must therefore default
to raw; compression remains an opt-in placement/network decision.

## Live-loop isolation

| mode | probe p50 ms | probe p95 ms | probe p99 ms | probe p99.9 ms |
|---|---:|---:|---:|---:|
| baseline | 0.561 | 1.012 | 1.053 | 1.062 |
| inline | 1702.777 | 3049.184 | 3168.859 | 3195.778 |
| async | 1706.160 | 3040.567 | 3160.239 | 3187.162 |
| thread | 0.565 | 1.016 | 1.053 | 1.062 |
| process | 0.564 | 1.017 | 1.055 | 1.750 |

Inline and async-task compression block the event loop. Thread and independent
process avoid that head-of-line blocking in this run, but only the StorageWorker
process also provides ownership, failure, queue and memory isolation.

## Provisional regression gates

- Default transfer mode must be the fastest measured mode: `raw`.
- Dedicated-process live probe p99 delta: 0.002 ms; gate ≤ 0.500 ms.
- Future SGs must rerun paired raw/default A/B on the same hosts. A single Runner step time is not a performance baseline.

## Limitations

- Ten measured samples support an engineering regression check, not a universal p99.9 SLO.
- Gloo tensor transfer is the production data-plane primitive, but this tool does not include Ray RPC/control-plane overhead.
- CPU/RSS values are process-level measurements; `process_time()` sums CPU time
  across process threads, and shared-library allocators can retain memory between modes.
- Real-camera compression ratios and restricted-bandwidth break-even remain unmeasured.
