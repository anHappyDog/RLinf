#!/usr/bin/env bash

set -euo pipefail

repo=${B1K_RL_REPO:-/mnt/public/daibo/timeline/0831/RLinf}
venv=${B1K_RL_VENV:-/mnt/public/daibo/venv/behavior_openpi}
model=${B1K_RL_MODEL:-/mnt/public/daibo/models/b1k_grounded_control_v01/pi05_official_train480_p2_sqrt_stage_step20000}
norm_stats_input=${B1K_RL_NORM_STATS:-/mnt/public/daibo/results/b1k_grounded_control_v01/eval/p2_train480_step6000_comet_native_v1/model/assets/behavior-1k/2025-challenge-demos/norm_stats.json}
if [[ -d $norm_stats_input ]]; then
  norm_stats_dir=$norm_stats_input
  norm_stats=$norm_stats_dir/norm_stats.json
else
  norm_stats=$norm_stats_input
  norm_stats_dir=$(dirname "$norm_stats")
fi
token_mapping=${B1K_RL_TOKEN_MAPPING:-/mnt/public/daibo/results/b1k_grounded_control_v01/oracle_2025_radio_lunchbox_microwave_train480_stride8_subgoal_v04_split_v1/structural_token_mapping.json}

: "${B1K_SUBPOOL_MANIFEST:?Set B1K_SUBPOOL_MANIFEST to the 20-state train manifest.}"
: "${B1K_SUBPOOL_RESULT_DIR:?Set B1K_SUBPOOL_RESULT_DIR to a fresh result directory.}"

max_steps=${B1K_RL_MAX_STEPS:-200}
save_interval=${B1K_RL_SAVE_INTERVAL:-5}
micro_batch_size=${B1K_RL_MICRO_BATCH_SIZE:-100}
global_batch_size=${B1K_RL_GLOBAL_BATCH_SIZE:-16000}
critic_global_batch_size=${B1K_RL_CRITIC_GLOBAL_BATCH_SIZE:-3200}
cache_critic_inputs=${B1K_RL_CACHE_CRITIC_INPUTS:-true}
max_attempts=${B1K_RL_MAX_ATTEMPTS_PER_STATE:-8}
state_cache_size=${B1K_RL_STATE_CACHE_SIZE:-20}
dynamic_batching=${B1K_RL_DYNAMIC_BATCHING:-false}
dynamic_batch_size=${B1K_RL_DYNAMIC_BATCH_SIZE:-5}
dynamic_batch_wait=${B1K_RL_DYNAMIC_BATCH_WAIT_SECONDS:-0.1}

export PATH="$venv/bin:$PATH"
export PYTHONPATH="$repo"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export EMBODIED_PATH="$repo/examples/embodiment"
export RAY_ADDRESS=${RAY_ADDRESS:-10.208.38.172:6379}
export TMPDIR=${TMPDIR:-/mnt/public/daibo/tmp}
export B1K_GROUNDED_TOKEN_MAPPING="$token_mapping"
export B1K_ASSET_FINGERPRINT=b1k-2fd66d5c-radio-bc82d211
export B1K_SUBPOOL_MODEL_PATH="$model"
export RLINF_REMOTE_COLLECTOR_TOKEN=${RLINF_REMOTE_COLLECTOR_TOKEN:-B1K_CROSSDC_RADIO_0909_V1}

"$venv/bin/python" - "$B1K_SUBPOOL_MANIFEST" "$norm_stats" "$micro_batch_size" "$global_batch_size" "$critic_global_batch_size" <<'PY'
import json
import sys
from pathlib import Path

manifest, norm_stats = map(Path, sys.argv[1:3])
micro_batch_size, global_batch_size, critic_global_batch_size = map(int, sys.argv[3:6])
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
assert len(rows) == 20, len(rows)
assert len({row["snapshot_id"] for row in rows}) == 20
assert all(row["pool_type"] == "canonical" for row in rows)
assert all(row["subtask_id"] == 1 for row in rows)
assert all(row["metadata"]["reward"]["max_steps"] == 1280 for row in rows)
assert all(row["metadata"]["reward"]["potential_terms"] == [] for row in rows)
assert all(json.loads(row["control_json"])["subgoal"] for row in rows)
assert all((manifest.parent / row["state_path"]).is_file() for row in rows)
assert norm_stats.is_file(), norm_stats
assert global_batch_size % (micro_batch_size * 4) == 0
assert critic_global_batch_size % (micro_batch_size * 4) == 0
print(
    "Validated 20 canonical states; "
    f"MBS={micro_batch_size}, policy GBS={global_batch_size}, "
    f"critic GBS={critic_global_batch_size}."
)
PY

remote_collectors='{enabled:true,auth_token_env:RLINF_REMOTE_COLLECTOR_TOKEN,response_compression:{codec:zlib,level:1,min_bytes:65536},endpoints:[{env_rank:0,ssh_host:gdb_4090_1,port:46100,local_port:47100,env_overrides:{video_cfg:{save_video:false}}},{env_rank:1,ssh_host:gdb_4090_1,port:46101,local_port:47101,env_overrides:{video_cfg:{save_video:false}}},{env_rank:2,ssh_host:gdb_4090_1,port:46102,local_port:47102,env_overrides:{video_cfg:{save_video:false}}},{env_rank:3,ssh_host:gdb_4090_1,port:46103,local_port:47103,env_overrides:{video_cfg:{save_video:false}}},{env_rank:4,ssh_host:gdb_4090_1,port:46104,local_port:47104,env_overrides:{video_cfg:{save_video:false}}},{env_rank:5,ssh_host:gdb_4090_1,port:46105,local_port:47105,env_overrides:{video_cfg:{save_video:false}}},{env_rank:6,ssh_host:gdb_4090_1,port:46106,local_port:47106,env_overrides:{video_cfg:{save_video:false}}},{env_rank:7,ssh_host:gdb_4090_1,port:46107,local_port:47107,env_overrides:{video_cfg:{save_video:false}}},{env_rank:8,ssh_host:gdb_4090_2,port:46100,local_port:47108,env_overrides:{video_cfg:{save_video:false}}},{env_rank:9,ssh_host:gdb_4090_2,port:46101,local_port:47109,env_overrides:{video_cfg:{save_video:false}}},{env_rank:10,ssh_host:gdb_4090_2,port:46102,local_port:47110,env_overrides:{video_cfg:{save_video:false}}},{env_rank:11,ssh_host:gdb_4090_2,port:46103,local_port:47111,env_overrides:{video_cfg:{save_video:false}}}]}'
trainer_env_config='[{node_ranks:0,env_vars:[{RLINF_REMOTE_COLLECTOR_TOKEN:B1K_CROSSDC_RADIO_0909_V1}]}]'

command=(
  "$venv/bin/python" examples/embodiment/train_embodied_agent.py
  --config-name behavior_subpool_ppo_openpi_pi05_hetero
  runner.max_steps="$max_steps"
  runner.save_interval="$save_interval"
  runner.val_check_interval=-1
  ++runner.enable_decoupled_mode="$dynamic_batching"
  runner.logger.log_path="$B1K_SUBPOOL_RESULT_DIR"
  cluster.num_nodes=3
  cluster.component_placement.env.node_group="'trainer,behavior'"
  cluster.component_placement.env.placement="'0-3:0-11,4-11:12-19'"
  "+cluster.node_groups.0.env_configs=$trainer_env_config"
  env.train.total_num_envs=20
  env.train.max_episode_steps=1280
  env.train.max_steps_per_rollout_epoch=1280
  env.train.skip_intermediate_obs_in_chunk=true
  +env.obs_compression.enable=true
  +env.obs_compression.codec=zlib
  +env.obs_compression.level=1
  +env.obs_compression.xor_delta=false
  +env.train.behavior.init_retry_count=5
  +env.train.behavior.init_retry_delay=5.0
  +env.train.behavior.init_retry_backoff=2.0
  env.train.subpool.fixed_subtask_id=1
  env.train.subpool.outcome_group_size=20
  env.train.subpool.outcome_snapshot_schedule=shuffled_round_robin
  env.train.subpool.sticky_outcome_snapshot=true
  env.train.subpool.pool_weights.canonical=1.0
  env.train.subpool.pool_weights.predecessor_success=0.0
  env.train.subpool.pool_weights.recovery=0.0
  env.train.subpool.dynamic_updates=false
  env.train.subpool.state_cache_size="$state_cache_size"
  env.train.subpool.skip_official_task_termination=true
  "+env.train.remote_collector=$remote_collectors"
  env.train.video_cfg.video_base_dir="$B1K_SUBPOOL_RESULT_DIR/video/train"
  +env.train.video_cfg.fps=15
  algorithm.reward_type=subtask_chunk_level
  algorithm.logprob_type=action_level
  algorithm.kl_beta=0.01
  algorithm.kl_penalty_type=low_var_kl
  algorithm.outcome_dynamic_sampling.group_size=20
  algorithm.outcome_dynamic_sampling.groups_per_update=20
  algorithm.outcome_dynamic_sampling.parallel_groups=false
  algorithm.outcome_dynamic_sampling.max_attempts_per_group="$max_attempts"
  algorithm.gamma=0.999686
  actor.model.precision=fp32
  rollout.model.precision=bf16
  actor.model.openpi_data.norm_stats_path="$norm_stats_dir"
  actor.model.openpi.value_vlm_mode=state_attention
  actor.micro_batch_size="$micro_batch_size"
  actor.global_batch_size="$global_batch_size"
  actor.critic_global_batch_size="$critic_global_batch_size"
  actor.cache_critic_inputs="$cache_critic_inputs"
  actor.policy_update_epochs=1
  actor.critic_update_epochs=5
  actor.enable_offload=false
  rollout.enable_offload=false
  rollout.dynamic_batching.enabled="$dynamic_batching"
  rollout.dynamic_batching.max_batch_size="$dynamic_batch_size"
  rollout.dynamic_batching.max_wait_seconds="$dynamic_batch_wait"
  actor.fsdp_config.sharding_strategy=shard_grad_op
  actor.fsdp_config.gradient_checkpointing=false
  actor.fsdp_config.forward_prefetch=true
  actor.fsdp_config.backward_prefetch=pre
  actor.fsdp_config.limit_all_gathers=false
  actor.fsdp_config.mixed_precision.param_dtype=bf16
  actor.fsdp_config.mixed_precision.reduce_dtype=fp32
  actor.fsdp_config.mixed_precision.buffer_dtype=bf16
  actor.optim.policy_clip_grad=5.0
  actor.optim.value_clip_grad=5.0
  actor.optim.critic_only=false
)

cd "$repo"
if [[ "${1:-}" == "--config-only" ]]; then
  shift
  exec "${command[@]}" "$@" --cfg job
fi
exec "${command[@]}" "$@"
