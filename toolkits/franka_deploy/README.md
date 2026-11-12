# Franka canonical pickup deployment

[中文](README.zh-CN.md)

This toolkit serves canonical Pi0.5 checkpoints through the existing isolated
Polymetis setup on `uav001`. It also retains the legacy Pi0 and MolmoBot F2
adapters. These configurations describe this workcell, not generic robot defaults.

The deployed code directory is `/home/infini/daibo/code/franka_deploy`. The existing
model environment is `runtime/python.sh`; hardware uses
`runtime/polymetis_fr3/python.sh`. Both are relative to `/home/infini/daibo`.
They use separate environments and leave other users' installations unchanged.
The Polymetis binaries and private device-source dependencies already installed
under `runtime/polymetis_fr3` are required; this directory is not an installer.

## Checkpoint contract

`models/pi05_franka_pickup_sft_step4549/actor` must contain
`model_state_dict/full_weights.pt` and
`upstream_assets/{train_config.pkl,canonical_qtarget.json}`. The loader restores
all `core.*` weights strictly, preserving the model's selective BF16/FP32 layout.

- State is measured `qpos7` in radians plus physical aperture in metres. All eight
  fields use checkpoint Molmo quantiles, then official Pi0.5 state tokenization
  and padding. Width is not masked or converted to a simulated driver angle.
- RGB external and wrist images use the training crop and 224×224 letterbox.
  The third image slot is zero with a false image mask. Task text is unchanged.
- The model predicts H16×32. The first eight dimensions are decoded with the
  checkpoint action quantiles. Each joint delta adds the same observation qpos
  once to recover absolute targets. Closedness stays continuous: 0 open, 1 closed.
- The client maps closedness to `(1 - closedness) * 0.085` metres. Hardware joint
  limits remain enforced. The Pi0.5 adapter adds no F2 action scaling or threshold.
- `--execute-horizon 8` executes the first eight targets, then replans using a new
  observation. Both server and client must specify the same value; 16 is also
  supported. The requested action period is 0.066 seconds; measured loop timing
  includes inference and is recorded in `ticks.jsonl`.
- `--seed` initializes the inference process once. Reconnecting clears buffered
  actions, not the random stream. Restart the model process for a repeat from the
  same seed. Prompt changes require a new connection.

## Left-arm test

Keep an already running left controller service; do not start a second instance
on port 50061. Otherwise, with the operator present and left FCI enabled, run:

```fish
cd /home/infini/daibo
runtime/polymetis_fr3/python.sh code/franka_deploy/polymetis_server/launch.py \
    --config code/franka_deploy/polymetis_server/fr3.yaml --execute
```

Stop any previous model process using port 18080 or occupying the GPU, then start:

```fish
cd /home/infini/daibo
env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    taskset -c 0-5,16-31 runtime/python.sh \
    -m franka_deploy.canonical_tick_server \
    --model-kind pi05-qtarget \
    --checkpoint models/pi05_franka_pickup_sft_step4549 \
    --action-dt 0.066 --execute-horizon 8 --denoise-steps 10 --seed 0 --port 18080
```

After the model reports `Serving`, run in another fish terminal:

```fish
cd /home/infini/daibo
set run_dir runs/pi05_step4549_left_(date +%Y%m%d_%H%M%S)
runtime/polymetis_fr3/python.sh code/franka_deploy/molmobot_f2_polymetis_client.py \
    --hardware code/franka_deploy/configs/molmobot_f2_polymetis.yaml \
    --policy-kind pi05-qtarget --execute-horizon 8 \
    --endpoint ws://127.0.0.1:18080 --execute --wrist-rotation 180 \
    --prompt "Pick up the orange cube" --max-seconds 20 \
    --output "$run_dir" --save-images --record-video
```

The first Enter opens the gripper and moves to the fixed five-second home; the
second starts policy actions. Inspect `post_home_preview` before the second Enter
and ensure the target is visible in the wrist view. Connecting may activate the
gripper before the first Enter. The 180-degree image rotation retains the current
workcell test setting; it does not rotate joint 7 and is not a training transform.

The arm waits for home completion without downloading its unused 1 kHz state log.
No automatic arm recovery is performed. A reported fault needs operator handling.
Video recording uses independent threads and the camera caches throughout home,
waiting and policy execution. `videos/{external,wrist}.mp4` use the configured
camera FPS; repeated cached frames are possible. Timestamp sidecars retain capture
and write times. The wrist recording has the same rotation as the policy input.
Normal exit and Ctrl+C finalize both videos; hard process termination may not.

The right-arm configurations are separate and use port 50063. Keep the left-arm
files unchanged when using `configs/molmobot_f2_polymetis_right.yaml` and
`polymetis_server/fr3_right.yaml`.
