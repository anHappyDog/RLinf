# Remote collector deployment

## Environment

Set these independently on each collector host. Use a random shared token and
do not print it into logs or manifests.

```bash
export TMPDIR=/mnt/public/daibo/tmp
export OMNIGIBSON_DATA_PATH=/mnt/public/daibo/datasets/omni_data
export OMNIGIBSON_DATASET_PATH=/mnt/public/daibo/datasets/omni_data/behavior-1k-assets
export OMNIGIBSON_KEY_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson.key
export OMNIGIBSON_ASSET_PATH=/mnt/public/daibo/datasets/omni_data/omnigibson-robot-assets
export OMNI_KIT_ACCEPT_EULA=YES
export RLINF_REMOTE_COLLECTOR_TOKEN=<shared-secret>
```

## Start one daemon per GPU

Use a run-specific prefix and log directory. The audited OmniGibson source must
precede the editable installation.

```bash
repo=/mnt/public/daibo/timeline/0831/RLinf-b1k-singleenv
b1k=/mnt/public/daibo/timeline/0831/BEHAVIOR-1K-b1k-rl-singleenv/OmniGibson
python=/mnt/public/daibo/venv/behavior_openpi/bin/python
run_id=<unique-run-id>

"$python" toolkits/b1k_grounded/manage_remote_collectors.py start \
  --python "$python" \
  --repo "$repo" \
  --omnigibson-path "$b1k" \
  --log-dir "/mnt/public/daibo/results/b1k_grounded_control_v01/remote_collectors/$run_id/$(hostname)" \
  --session-prefix "b1k_${run_id}" \
  --gpus 0,1,2,3 \
  --ports 46100,46101,46102,46103 \
  --replace
```

Run `status` with the same prefix/GPU/port lists. Also inspect each listening
port and daemon log; tmux status alone does not prove initialization.

## Endpoint mapping

Each daemon belongs exclusively to one run. Assign a unique endpoint to every
global logical EnvWorker rank:

```yaml
remote_collector:
  enabled: true
  auth_token_env: RLINF_REMOTE_COLLECTOR_TOKEN
  response_compression: {codec: zlib, level: 1, min_bytes: 65536}
  endpoints:
    - env_rank: 8
      ssh_host: gdb_4090_1
      port: 46100
      local_port: 47100
      env_overrides: {video_cfg: {save_video: false}}
```

Keep one scene per endpoint. The global `env_rank` must match Ray placement,
and trainer-side `local_port` values must not collide.

## Metrics and failure semantics

Monitor:

```text
env/time/remote_collector_handler
env/time/remote_collector_transport
env/remote_collector/response_mib
env/remote_collector/response_raw_blob_mib
env/remote_collector/response_compression_ratio
env/remote_collector/reconnects
```

Handler time is remote simulation/service time. Transport includes encoding,
the SSH tunnel, and transfer. Persistent reconnects are an error signal. If the
driver dies after mutating an environment, replace the daemon before a new run;
the old simulator state is not resumable training state.

