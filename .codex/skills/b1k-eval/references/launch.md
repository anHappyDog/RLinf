# B1K tmux launch patterns

Adapt the placeholders rather than copying an older experiment's checkpoint, episode, sidecar, or output path.

## Naming and outputs

Use short, unique names that identify the checkpoint and slot:

```text
policy tmux: b1k_<run>_<slot>_policy
B1K tmux:    b1k_<run>_<slot>_eval
server log:  <run-root>/server_<slot>.log
eval log:    <run-root>/<slot>/eval.log
metrics:     <run-root>/<slot>/metrics/*.json
videos:      <run-root>/<slot>/videos/*.mp4
eval view:   <run-root>/<slot>/subtask_eval_view
```

Create the output and cache directories before starting tmux. Refuse a session-name or port collision unless it is an exact, healthy reuse of the requested run.

## Policy server

Launch on either `nxb_4090` or a verified same-datacenter model host. Run from the COMET checkout:

```bash
tmux new-session -d -s "${POLICY_TMUX}" -c /mnt/public/daibo/repos/comet/openpi-comet \
  "env CUDA_VISIBLE_DEVICES=${POLICY_GPU} \
    PYTHONPATH=/mnt/public/daibo/timeline/0831/RLinf:/mnt/public/daibo/repos/comet/openpi-comet/src \
    HF_HOME=/opt/.cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=${TORCH_CACHE} \
    /mnt/public/daibo/venv/behavior_openpi/bin/python \
    /mnt/public/daibo/timeline/0831/RLinf/toolkits/b1k_grounded/serve_grounded_policy.py \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --token-mapping-path ${TOKEN_MAPPING} \
    --control-profile ${CONTROL_PROFILE_CLI} \
    --task-name ${TASK_NAME} \
    --port ${POLICY_PORT} 2>&1 | tee ${SERVER_LOG}"
```

`CHECKPOINT_DIR` is the COMET-native directory containing `model.safetensors` and `assets/.../norm_stats.json`. Use CLI enum spellings such as `P2_GROUND_SG`; the evaluator Hydra override uses the lowercase spelling such as `p2_ground_sg`.

Confirm the server process and listener. For split placement, test the route from `nxb_4090` to the model host's private IP and exact port before starting B1K. A short Python `socket.create_connection((host, port), timeout=5)` check is portable when `nc` is unavailable.

## B1K evaluator

Launch on `nxb_4090`; one evaluator uses one RTX GPU:

```bash
tmux new-session -d -s "${EVAL_TMUX}" -c /mnt/public/daibo/venv/behavior_openpi/BEHAVIOR-1K \
  "env CUDA_VISIBLE_DEVICES=${EVAL_GPU} \
    TMPDIR=/mnt/public/daibo/tmp \
    PYTHONPATH=/mnt/public/daibo/timeline/0831/RLinf:/mnt/public/daibo/repos/comet/openpi-comet/src \
    OMNI_KIT_ACCEPT_EULA=YES \
    OMNIGIBSON_DATA_PATH=/mnt/public/daibo/datasets/omni_data \
    OMNIGIBSON_DATASET_PATH=/mnt/public/daibo/datasets/omni_data/behavior-1k-assets \
    OMNIGIBSON_KEY_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson.key \
    OMNIGIBSON_ASSET_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson-robot-assets \
    OMNIGIBSON_APPDATA_PATH=${APPDATA_DIR} \
    /mnt/public/daibo/venv/behavior_openpi/bin/python \
    /mnt/public/daibo/timeline/0831/RLinf/toolkits/b1k_grounded/eval_grounded_subtasks.py \
    policy=websocket \
    task.name=${TASK_NAME} \
    env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
    eval_level=subtask \
    keep_running_after_success=false \
    write_video=true \
    run_episode_indices=${EPISODE_LIST} \
    subtask_index=${SUBTASK_START} \
    subtask_end_index=${SUBTASK_END} \
    demo_data_dir=/mnt/public/daibo/datasets/behavior-1k/2025-challenge-demos \
    instance_reward_mode=task \
    log_path=${SLOT_OUTPUT} \
    model.host=${POLICY_HOST} \
    model.port=${POLICY_PORT} \
    +grounded_control_sidecar=${SIDECAR} \
    +grounded_eval_view_dir=${SLOT_OUTPUT}/subtask_eval_view \
    +grounded_control_profile=${CONTROL_PROFILE_HYDRA} \
    +grounded_infer_missing_parts=true \
    +grounded_demo_calibration=false 2>&1 | tee ${EVAL_LOG}"
```

Quote Hydra list values such as `EPISODE_LIST='[10,20]'` so the remote shell does not alter them. Include `+grounded_infer_missing_parts=true` only when the selected profile/evaluator supports and needs it.

For colocated serving, set `POLICY_HOST=127.0.0.1`. For split serving, use the model host's private address reachable from `nxb_4090`, not its SSH alias unless that alias resolves on `nxb_4090`.

## Bounded verification and handoff

Use `tmux has-session`, `tmux capture-pane`, exact-PID inspection, the server/eval log tails, the listening port, and `nvidia-smi`. On `nxb_4090`, also inspect `/proc/<evaluator-pid>/environ` and require the exact value `TMPDIR=/mnt/public/daibo/tmp`; a tmux command that merely intended to set it is insufficient evidence. Stop after confirming healthy startup; first-request policy compilation and the full rollout are not part of the startup wait.

Return both attach commands with their hosts, for example:

```bash
ssh <policy-host> -t 'tmux attach -t <policy-tmux>'
ssh nxb_4090 -t 'tmux attach -t <eval-tmux>'
```

Report the server log, evaluator log, run root, metrics directory, and video directory even when metrics/videos have not appeared yet.
