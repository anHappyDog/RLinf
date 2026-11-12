# Qwen3.5 Flow Matching Action Experts

Use this implementation to pretrain Qwen3.5 VLAs with [Knowledge Insulation](#pretrain-with-knowledge-insulation) and compare attention, Gated DeltaNet (GDN), and direct state readout for continuous actions. Qwen keeps its native multimodal backbone. The frozen-backbone FM workflows below are architecture controls; the KI entry trains the VLM and Expert through separate objectives. These standalone entries use native PyTorch distributed execution; integration with RLinf distributed SFT workers remains separate.

## Choose An Architecture

The architecture follows the [official Qwen3.5-2B configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json) and [native Transformers Qwen3.5 implementation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

The five configurations live in `examples/sft/config/qwen35_fm/`.

| Config filename | Expert | Cross-attention calls per evaluation | Hidden / FFN width |
| --- | --- | --- | --- |
| `qwen35_gqa6.yaml` | 6 bidirectional GQA blocks | 6 | 1024 / 4096 |
| `qwen35_gqa6_gdn1.yaml` | 6 GQA blocks, then 1 independent causal GDN block | 6 | 1024 / 4096 |
| `qwen35_gqa6_gdn2.yaml` | 6 GQA blocks, then 2 independent causal GDN blocks | 6 | 1024 / 4096 |
| `qwen35_gqa6_gdn3.yaml` | 6 GQA blocks, then 3 independent causal GDN blocks | 6 | 1024 / 4096 |
| `qwen35_gqa6_vlm_state.yaml` | 6 GQA blocks with fast-weight state readout | 6 | 1024 / 4096 |

The three additive GDN experts retain the complete GQA baseline: each GQA block contains action self-attention, cross-attention to VLM hidden states, and an FFN. Independently parameterized GDN blocks follow that six-block backbone. They process action features and include their own FFN; they neither replace GQA blocks nor share the pretrained VLM's GDN weights. V3 instead adds direct reads from the pretrained VLM's recurrent states to the six GQA blocks.

All presets use the same hidden size 1024, FFN size 4096, action horizon 16, action dimension 8, and 16 query / 4 KV heads. Additive GDN blocks also share 8 key/value heads and key/value head dimensions 128. Parameter counts follow naturally from the added modules; widths are not adjusted to reach a target budget. Keep data, seeds, normalization, optimization, and conditioning fixed when comparing these configurations.

Set `depth` to override the GQA backbone depth and `gdn_depth` to override the number of appended GDN blocks; defaults are six GQA blocks and the GDN count named by the variant. Set `action_self_attn: false` for the GQA cross-attention-only ablation. Each block receives the FM time through adaptive normalization. Action embeddings also include learned action-index positions and the time embedding.

## Encode And Predict Actions

Use an environment with PyTorch, Transformers 5.19.0, OmegaConf, and PyYAML. SFT additionally uses PyArrow, PyAV, NumPy, Pillow, and the native Qwen processor dependencies. Existing local model and processor files are required; `from_pretrained` enforces `local_files_only=True`.

Instantiate any of the five configurations by changing `config_path` below. The pretrained wrapper defaults to a frozen VLM; set `trainable=True` for VLM gradients.

```python
import torch
import yaml
from rlinf.models.embodiment.qwen35_fm.action_expert import ActionExpertConfig, build_action_expert
from rlinf.models.embodiment.qwen35_fm.policy import FlowMatchingPolicy
from rlinf.models.embodiment.qwen35_fm.vlm import Qwen35VLMWrapper

config_path = "examples/sft/config/qwen35_fm/qwen35_gqa6_vlm_state.yaml"
config = ActionExpertConfig(**yaml.safe_load(open(config_path))["expert"])
vlm = Qwen35VLMWrapper.from_pretrained(
    "/mnt/public/daibo/models/Qwen3.5-2B", trainable=False,
    torch_dtype=torch.bfloat16, attn_implementation="eager",
).to("cuda:0")
expert = build_action_expert(config, vlm.state_specs).to("cuda:0", torch.bfloat16)
policy = FlowMatchingPolicy(vlm, expert)
# image is a PIL image; native processed image/video/text kwargs also work.
condition = vlm.encode(
    observation=image, instruction="Pick up the cup.",
    extract_states=config.variant == "gqa6_vlm_state",
)
actions = policy.sample(condition, steps=10, method="euler")
```

`encode` returns token features and their validity mask. `predict_velocity(noisy_actions, timestep, condition)` evaluates every action position at one noise level. `loss(actions, condition, action_mask=...)` uses Gaussian noise and uniform time with `A_tau=(1-tau)*epsilon+tau*A`, target `A-epsilon`, and masked coordinate MSE. `sample` integrates from noise at time 0 to actions at time 1 using Euler or Heun. Normalize training actions before FM and invert that normalization before robot execution.

Each sampling call caches the six blocks' fixed VLM K/V projections locally; action GDN states are still initialized for every vector-field evaluation. GQA passes compact KV heads to native SDPA and prioritizes fused CUDA backends, including cuDNN for padded inputs. Unsupported shapes or devices use native SDPA fallback. KI batches preserve shared image tensors through pinning and device transfer; FM and CE denominators are reduced together, with metrics read on logging steps.

| Interface | Shape |
| --- | --- |
| Actions, noise, velocity | `[B,H,D_action]` |
| FM time | scalar or `[B]` for prediction; `[B]` for loss |
| Action embeddings | `[B,H,D_hidden]` |
| VLM final hidden states | `[B,L,2048]` for Qwen3.5-2B |
| VLM token mask | `[B,L]`, nonzero means valid |
| Training action mask | `[B,H]` or `[B,H,D_action]` |
| Expert token mask | `[B,H]` |
| Selected recurrent state | `gdn_states[layer]: [B,H_v,K,V]` |
| State read query / result | `[B,H,H_v,K]` / `[B,H,H_v,V]` |

## Read Native Multimodal Memory

Qwen3.5-2B has 24 text layers. Zero-based layers 3, 7, 11, 15, 19, and 23 use full attention with 8 query heads, 2 KV heads, and head dimension 256. The other 18 layers use GDN with 16 key heads, 16 value heads, key/value dimensions 128, and convolution width 4. Native full attention separates projected queries from its output gate. GDN has separate QKV, output gate, beta, and decay projections, followed by causal depthwise convolution, Q/K normalization, the gated delta rule, and gated RMS normalization.

The wrapper calls the native multimodal backbone, avoiding language logits. With `extract_states=True`, a fresh native cache requests final states. The supported Transformers 5.19 cache stores `layers[layer].recurrent_states[0]` separately from `conv_states[0]` and attention KV. The inspected older native cache exposes `recurrent_states[layer]`; extraction handles that layout too. State dimensions come from the actual text configuration and are checked against returned tensors.

The native matrix orientation is `[B,H_v,K,V]`: a normalized action query reads `einsum("bthk,bhkv->bthv", q, S)`. Each selected layer has separate learned query and output projections through a configurable `state_adapter_dim` bottleneck. Softmax layer weights combine readouts; a learned sigmoid gate injects the projected result as a residual before the block FFN. The gate starts at `sigmoid(-4)`, approximately 0.018. `state_injection: per_block` creates one adapter per block; `shared` reuses one adapter at every block. States are never flattened into VLM tokens.

Select layers with `Qwen35VLMWrapper(..., state_layers=[2,10,22])`, or leave the selection unset for all GDN layers. Fresh cache tensors keep their computation graph, including native `copy_` updates; trainable VLM states are not detached. Trainable state extraction requires gradient checkpointing disabled. Left padding is required for extraction because trailing padding would continue recurrent decay; the wrapper sets the native processor to left padding and rejects right-padded extraction.

The supported scope is `current_vlm_forward`. External old caches and `persistent_stream` are rejected until an incremental input and episode reset protocol is implemented. Action GDN blocks also create no persistent cache: each vector-field call starts independently, including every Euler/Heun evaluation. Native chunkwise execution is the GPU default; `gdn_kernel: reference` uses a differentiable token scan with the same state orientation. The CPU expert path uses the native unwrapped PyTorch chunk implementation when needed. Installed FLA can make native VLM CPU execution select a CUDA kernel; use a GPU for the native VLM tests in that environment. Causal convolution falls back to native PyTorch when `causal-conv1d` is unavailable. Optional PEFT adapter injection is exposed but has not been validated in this pilot.

For all 18 layers, retained FP32 recurrent matrices require `B*18*16*128*128*4` bytes, or 18 MiB per example. BF16 attention KV for six full-attention layers requires `B*L*6*2*2*256*2` bytes. Convolution buffers have shape `[B,6144,4]` per GDN layer. These analytical estimates exclude activations, allocator overhead, and temporary caches: selecting fewer returned states still creates native caches for all layers during extraction.

## Run Tests And A Bounded SFT Pilot

Run from the repository root, using the Python environment described above:

```bash
PYTHONPATH=. python -m pytest tests/unit_tests/test_qwen35_fm.py -q
QWEN35_CHECKPOINT=/mnt/public/daibo/models/Qwen3.5-2B \
  PYTHONPATH=. python -m pytest tests/unit_tests/test_qwen35_fm.py -q -k local_pretrained_multimodal_checkpoint
```

Native VLM tests require the validated Transformers 5.19.0 environment and a GPU when FLA is installed. The regular CI environment pins an older Transformers version; optional Qwen3.5 paths may be skipped or unsupported there. This feature does not upgrade the shared CI environment.

The first command exercises synthetic experts and a tiny native multimodal backbone; the environment variable additionally enables local checkpoint validation. Tests cover masks, reference recurrence, gradients, readout, numerical preservation, state mapping, state isolation, optimizer steps, and ODE shapes.

The pilot uses an existing LeRobot v3 MolmoBot conversion with native train/validation separation and verified `obs[i]` / `command[i+1]` alignment. Actions contain seven joint position targets and one gripper command; the loader preserves the 8D action and its supervision mask. The run selects four training episodes and two validation episodes, two horizon-16 chunks each: 8 training chunks and 4 validation chunks. The exterior camera is resized to 224×224. Normalization is fit only on selected training chunks. The source dataset metadata counts are recorded separately from the actual selected sample count.

```bash
PYTHONPATH=. python examples/sft/qwen35_fm_sft.py \
  --config examples/sft/config/qwen35_fm/qwen35_gqa6.yaml \
  --model-path /mnt/public/daibo/models/Qwen3.5-2B \
  --data-path /mnt/public/daibo/datasets/robot-data-pipeline/pickup_molmobot_full_native/datasets \
  --output results/qwen35_fm_pilot_new \
  --variants gqa6_gdn1,gqa6_gdn2,gqa6_gdn3 \
  --steps 10 --batch-size 2 --train-episodes 4 --val-episodes 2 \
  --chunks-per-episode 2 --image-size 224 --precision bfloat16 --device cuda:0 \
  --seed 42 --lr 1e-4 --cache-conditions \
  --hidden-size 128 --intermediate-size 512 --num-heads 4 --num-kv-heads 2 \
  --gdn-num-key-heads 2 --gdn-num-value-heads 2 \
  --gdn-key-head-dim 32 --gdn-value-head-dim 32 --state-layers 2 10 22
```

This reruns the three changed additive architectures under the existing small-width plumbing protocol, with a frozen pretrained VLM and randomly initialized experts. Each variant receives 10 AdamW updates and 20 training chunk draws. The earlier `gqa6` and `gqa6_vlm_state` pilot results remain applicable because their small-width configurations and implementations are unchanged; they are reused rather than retrained. Cached conditions are valid only with a frozen VLM; `--train-vlm` disables that option and requires separate matched runs. The output directory must be new. It receives the selected-data manifest, normalization, result JSON, and expert/optimizer checkpoints; trainable runs also save VLM weights. Offline FM loss on this subset does not establish closed-loop task success or demonstrate useful state memory.

## Count Parameters And Measure Cost

Run the benchmark with local model configuration to report exact parameter counts and state/cache estimates without loading weights:

```bash
PYTHONPATH=. python examples/sft/qwen35_fm_benchmark.py \
  --model-path /mnt/public/daibo/models/Qwen3.5-2B \
  --config-dir examples/sft/config/qwen35_fm --counts-only \
  --output results/qwen35_fm_counts.json
```

Measure expert forward time, forward/backward time, and peak GPU memory with identical conditioning shapes and hardware:

```bash
PYTHONPATH=. python examples/sft/qwen35_fm_benchmark.py \
  --model-path /mnt/public/daibo/models/Qwen3.5-2B \
  --config-dir examples/sft/config/qwen35_fm --device cuda:0 \
  --batch-size 2 --horizon 16 --sequence-length 128 \
  --hidden-size 128 --intermediate-size 512 --iterations 10 \
  --output results/qwen35_fm_benchmark.json
```

The benchmark uses synthetic conditioning with the actual VLM configuration. It measures expert cost, not image encoding or a complete control cycle. Keep reported parameter counts, cross-attention counts, state selection, precision, and timing scope alongside any comparison. Architecture implementation alone supplies no evidence that GDN or VLM state readout improves robot performance.

The uniform-width preset report is [parameter_counts.json](../../../../results/qwen35_fm_additive_20261112/parameter_counts.json); measured expert cost is [benchmark.json](../../../../results/qwen35_fm_additive_20261112/benchmark.json). The report records exact expert, trainable, and total parameter counts, head configurations, state dimensions, and analytical cache memory. VLM weights are tied and contain 2,213,241,664 parameters. The common GQA backbone and each additional GDN block use identical dimensions across variants.

The small-width pilot has different parameter counts. The updated additive runs record their actual data and configuration in [data_manifest.json](../../../../results/qwen35_fm_additive_20261112/molmobot_pilot/data_manifest.json) and their training outcomes in [results.json](../../../../results/qwen35_fm_additive_20261112/molmobot_pilot/results.json). Unchanged `gqa6` and `gqa6_vlm_state` results are reused from the [earlier pilot](../../../../results/qwen35_fm_20261112/molmobot_pilot/results.json). The earlier replacement-GDN results describe a different architecture and do not validate these additive experts.

## Train Frozen-Backbone Controls Across Two Nodes

Use `qwen35_fm_hybrid_sft` with PyTorch 2.11 on two nodes with four GPUs each. The `(replicate=2, shard=4)` FSDP2 mesh shards the trainable expert within each node and replicates it across nodes. The frozen native VLM runs independently on every rank. Set network interfaces and the RDMA device/GID for your hosts before launching; the current run records the working settings in its launcher.

Run the following on both nodes with a shared output directory and the same training arguments:

```bash
# Set NODE_RANK=0 on the first node and NODE_RANK=1 on the second.
NODE_RANK=0
MASTER_ADDR=10.208.38.28
PYTHONPATH=. python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29671 \
  -m examples.sft.qwen35_fm_hybrid_sft \
  --config examples/sft/config/qwen35_fm/qwen35_gqa6_vlm_state.yaml \
  --model-path /mnt/public/daibo/models/Qwen3.5-2B \
  --data-path /mnt/public/daibo/datasets/robot-data-pipeline/pickup_molmobot_full_native/datasets \
  --output results/qwen35_hybrid_new \
  --steps 1000 --max-seconds 1200 --batch-size 128 \
  --train-episodes 4096 --val-episodes 64 --chunks-per-episode 8 \
  --workers 4 --checkpoint-every 100 --log-every 5
```

`--batch-size` is per rank, so this command uses a global batch of 1,024. The current selection contains 32,387 training chunks and 512 validation chunks; normalization uses only the selected training actions. Each model stops at 1,000 updates or 1,200 seconds of training, then saves a distributed checkpoint and evaluates the native validation split. Setup and validation time are recorded separately.

Pass `--resume /absolute/path/to/checkpoint_XXXXXX` to restore expert, optimizer, sampler position, and rank RNG. Keep data, conditioning, optimization, precision, and batch settings fixed; the runner checks them against the saved manifest. `status.json`, `metrics.jsonl`, and `checkpoint_*/metadata.json` record actual progress. The earlier frozen-control queue used `gqa6_vlm_state`, then `gqa6_gdn3`, then `gqa6`; its retained receipt is [queue_status.json](../../../../results/qwen35_hybrid_20261112T122822Z/queue_status.json).

## Pretrain With Knowledge Insulation

Use `examples/sft/qwen35_ki_pretrain.py` to pretrain the VLA with a trainable Qwen backbone, a continuous action expert, and authentic visual-language co-training. The frozen-backbone FM runs above remain architecture controls. [Knowledge Insulation (KI)](https://arxiv.org/html/2505.23705v1) separates the learning signals rather than freezing the VLM:

| Update type | Loss | Parameters updated |
| --- | --- | --- |
| Robot | FAST action-token CE | Native Qwen, including vision and GDN layers |
| Robot | Continuous action FM | Action Expert and its state-readout adapters |
| Visual-language | Answer-token CE | Native Qwen |

Every fourth update is a VL update (`--vl-every 4`); the other three combine robot CE and FM. Each component is globally averaged over its own valid targets. FM reads detached prefix hidden states and GDN matrices, while the trainable readout still receives gradients. A separate prefix-only native forward runs in evaluation mode under `no_grad`; the teacher-forced CE forward runs with training gradients and checkpointing. FAST targets never enter the FM prefix. This extra prefix forward is deliberate: the final GDN state after a teacher-forced FAST suffix would leak target actions. Native Qwen attention and recurrent kernels are retained.

### Prepare The Data And Token Targets

The current MolmoBot selection contains 32,387 training chunks and 512 validation chunks, each `[16,8]`: seven joint targets in radians and the native 0–255 gripper command. The current state remains 9D: seven arm joints and two gripper-driver positions. FM uses training-only mean/std normalization. FAST uses independent training-only 1st/99th-percentile bounds mapped to `[-1,1]`, and encodes only the leading fully valid timesteps. The nine state coordinates use their own training quantiles and become nine integer text bins in `[0,255]`.

The four external views are `exterior`, `randomized_gopro_analogue_1`, `randomized_zed2_analogue_1`, and `randomized_zed2_analogue_2`, in that fixed order. Each has a camera label in the prefix and is resized to 224×224. The wrist view is available in the dataset but is outside this four-view recipe. Native observation/action alignment is preserved.

Use the official [FAST tokenizer](https://huggingface.co/physical-intelligence/fast), pinned here to `ec4d7aa71691cac0b8bed6942be45684db2110f4`. The downloaded tokenizer/code assets are approximately 682 KiB. The bridge adds 2,048 code symbols plus begin/end markers: tokenizer length changes from 248,077 to 250,127 and native embedding rows from 248,320 to 250,127. Existing rows, multimodal token IDs, and tied input/output weights are preserved at initialization; the original Qwen checkpoint files stay unchanged.

VL co-training uses [Robo2VLM-1](https://huggingface.co/datasets/keplerccc/Robo2VLM-1), a robot-related visual question-answer dataset. The local subset has 71 of 262 training shards, containing 183,748 rows, plus the complete 6,676-row official test index. The [official exporter](https://github.com/KeplerC/robo2VLM/blob/main/scripts/create_huggingface_dataset.py) splits questions rather than protecting episode-directory groups; the local audit found 1,536 groups crossing its train/test boundary. The loader excludes every official-test group, then uses a stable seed-42 hash for a 5% group holdout. The selected train/validation sets contain 8,192/512 examples from 2,674/143 exporter groups and have zero shared resized-RGB hashes. These groups are sanitized directory names recovered from IDs, not verified native physical episodes; this does not establish scene or object OOD separation. This VQA source also differs from the KI paper's broad web-data mixture.

Prepare both caches once on CPU, choosing a new output root:

```bash
KI_RUN_ROOT=results/qwen35_ki_new
PYTHONPATH=. python -m examples.sft.qwen35_fm_image_cache \
  --data-path /mnt/public/daibo/datasets/robot-data-pipeline/pickup_molmobot_full_native/datasets \
  --output "$KI_RUN_ROOT/robot_images4" --workers 12 \
  --train-episodes 4096 --val-episodes 64 --chunks-per-episode 8 \
  --horizon 16 --image-size 224 \
  --cameras observation.images.exterior \
    observation.images.randomized_gopro_analogue_1 \
    observation.images.randomized_zed2_analogue_1 \
    observation.images.randomized_zed2_analogue_2
PYTHONPATH=. python -m examples.sft.qwen35_ki_data \
  --data-path /mnt/public/daibo/datasets/robo2vlm-1/data \
  --output "$KI_RUN_ROOT/vl_cache" --workers 8 \
  --train-limit 8192 --val-limit 512 --image-size 224 --seed 42
```

The first command decodes robot views with the native timestamp rule and checks selected RGB pixels against the uncached loader. The second builds group-disjoint VQA metadata and image memmaps. Builders require new output paths; reuse complete caches for subsequent variants. Prepared inputs for this session live under `results/qwen35_ki_20261112/robot_images4` and `results/qwen35_ki_20261112/vl_cache`; their metadata records the actual selection. No VLM features or recurrent states are cached across optimizer updates.

### Run A Two-Node KI Smoke

Use PyTorch 2.11 and Transformers 5.19.0 on two nodes with four GPUs each. The FSDP2 mesh is `(replicate=2, shard=4)` and shards the trainable VLM as well as the Expert. The command below uses this session's verified RDMA mapping; adjust the device, GIDs, and address for other hosts, and choose a free rendezvous port. Run it on both nodes with different `NODE_RANK` and the same shared paths:

```bash
# Set NODE_RANK=0 locally and NODE_RANK=1 on nxb_dev_2.
NODE_RANK=0
MASTER_ADDR=10.208.38.28
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA='=mlx5_1'
export NCCL_IB_GID_INDEX=$((5 + 4 * NODE_RANK))
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
KI_RUN_ROOT=results/qwen35_ki_new
PYTHONPATH=. python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29675 \
  -m examples.sft.qwen35_ki_pretrain \
  --config examples/sft/config/qwen35_fm/qwen35_gqa6_vlm_state.yaml \
  --model-path /mnt/public/daibo/models/Qwen3.5-2B \
  --fast-path /mnt/public/daibo/models/physical-intelligence-fast-tokenizer \
  --data-path /mnt/public/daibo/datasets/robot-data-pipeline/pickup_molmobot_full_native/datasets \
  --vl-data-path /mnt/public/daibo/datasets/robo2vlm-1/data --vl-format robo2vlm \
  --image-cache "$KI_RUN_ROOT/robot_images4" \
  --vl-image-cache "$KI_RUN_ROOT/vl_cache" --output "$KI_RUN_ROOT/smoke_v3" \
  --train-episodes 4096 --val-episodes 64 --chunks-per-episode 8 \
  --vl-limit 8192 --vl-val-limit 512 --vl-every 4 \
  --cameras observation.images.exterior \
    observation.images.randomized_gopro_analogue_1 \
    observation.images.randomized_zed2_analogue_1 \
    observation.images.randomized_zed2_analogue_2 \
  --image-size 224 --batch-size 1 --workers 2 --steps 4 --max-seconds 600 \
  --vlm-lr 2.5e-5 --expert-lr 1e-4 --warmup-steps 50 --schedule-steps 1000 \
  --checkpoint-every 100 --log-every 1
```

This checks four updates—three robot updates and one VL update—with a global batch of eight. It is a wiring smoke, not a production batch-size recommendation. Measure a full four-camera KI step before choosing the common batch for `gqa6_vlm_state`, `gqa6_gdn3`, and `gqa6`; frozen-VLM throughput and memory measurements do not apply to trainable-Qwen KI.

FP32 master parameters and AdamW states are retained, computation uses BF16, and gradient reductions and losses use FP32, following [FSDP2's mixed-precision contract](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html). VLM and Expert have separate optimizers and separate norm-1 clipping. The defaults use VLM/Expert peak LR `2.5e-5`/`1e-4`, AdamW β=`(0.9,0.95)`, ε=`1e-8`, and weight decay `1e-10`. The VLM LR and AdamW settings reference [OpenPI's optimizer configuration](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/optimizer.py); the Expert LR and the short-run 50-update warmup/cosine schedule are explicit experimental choices. `--schedule-steps 1000` fixes the decay horizon, ending at 0.1× peak LR. The requested uniform FM time sampling is retained, whereas the KI paper uses a transformed Beta distribution; this recipe is not an exact reproduction of its hyperparameters.

Run `PYTHONPATH=. python -m pytest tests/unit_tests/test_qwen35_fm.py -q -k ki_` for KI gradient routing, prefix isolation, native masked CE equivalence, FAST mapping, and multimodal CE checks. Set `QWEN35_FAST_TOKENIZER` if the reviewed local tokenizer lives elsewhere. Eleven KI tests and an eight-rank, full-Qwen, four-update smoke passed in this session; that first smoke used one exterior view. Its [status receipt](../../../../results/qwen35_ki_20261112/smoke_v3/status.json) records the actual robot/VL counts and losses. Four-view production runs require their own measured results.

Distributed checkpoints contain Qwen, Expert, both optimizers, normalization, FAST mapping, sampler/update position, and every rank's RNG; the full-model smoke checkpoint is approximately 30 GB. Use `--resume /absolute/path/to/checkpoint_XXXXXX` with unchanged data, camera order, batch, precision, optimizer, and schedule protocol. Logs distinguish FM, action CE, and VL CE and report each validation denominator. These offline objectives validate learning and routing; they do not establish closed-loop success or retention of general web knowledge.

### Input preparation and logging

KI runs require verified `--image-cache` and `--vl-image-cache` inputs. Use `--allow-uncached-data` only for bounded diagnostics that intentionally decode source media. Numeric episode selection indexes each parquet once and slices the original rows; this does not change chunk selection, masks, normalization, or native action alignment. It does not provide a shared full-dataset numeric cache across ranks.

CPU collators validate the observation/FAST prefix contract and prepare CE positions before device transfer. Direct policy calls retain validation by default. The optimized batch path preserves detached FM conditioning and trainable FAST/VL CE.

Keep `--output` on the checkpoint storage (for XJC, `/mnt/h_public/daibo/<run>`). Rank zero writes training metrics and status under `/tmp` by default, or `--log-dir` on a local disk, and mirrors complete records to the output every `--log-sync-seconds` (default 60). `local_logs.json` records the local directory. A failed mirror retains local logs; a clean exit attempts a final sync. `step_seconds` excludes logging and checkpoint I/O; cumulative `logging_seconds`, `checkpoint_seconds`, and `end_to_end_updates_per_second` expose their cost.

The local `STOP_AFTER_UPDATE` path is recorded in `local_logs.json` and is checked each update; the same file under shared output is polled by the background mirror at the sync interval.
