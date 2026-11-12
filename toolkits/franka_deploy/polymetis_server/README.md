# Isolated FR3 Polymetis deployment

All new dependencies live in `/home/infini/daibo/runtime/polymetis_fr3`.
Existing Franky launchers, environments and system configuration are unchanged.
The F2 model server continues to use its existing environment on port 18081.
The CPU client uses the private `device_source` copy of RLinf's device modules;
its changes replace the worker logger with standard Python logging and keep
device lock files inside this private runtime.
The original RLinf tree is not modified.

## Start the physical arm service

Run only when the operator is present, FCI is enabled and the Franky client has exited.
This starts physical torque control and applies the DROID FR3 collision thresholds
(40 Nm / 40 N arrays). The client below opens the gripper and homes the arm.

```fish
cd /home/infini/daibo
runtime/polymetis_fr3/python.sh code/franka_deploy/polymetis_server/launch.py --execute
```

The dedicated service binds to `127.0.0.1:50061`. Logs and owned process IDs are
written under `runs/polymetis_fr3/`. Ctrl-C stops its hardware client and server.
No system service, sudo or automatic recovery is used.

## Run the F2 client in another terminal

This command first opens to 85 mm, moves to the official home over 5 seconds,
then waits for Enter before model actions. The 20-second limit applies to model
execution after that prompt; it is a short physical test, not the paper's full
900-step evaluation protocol.

```fish
cd /home/infini/daibo
set run_dir runs/molmobot_f2_polymetis_green_(date +%Y%m%d_%H%M%S)
runtime/polymetis_fr3/python.sh code/franka_deploy/molmobot_f2_polymetis_client.py \
  --execute \
  --hardware code/franka_deploy/configs/molmobot_f2_polymetis.yaml \
  --endpoint ws://127.0.0.1:18081 \
  --prompt "Pick up the green cube" \
  --max-seconds 20 \
  --output "$run_dir" --save-images
```

## Return to Franky

Exit the Polymetis client and stop the Polymetis service before using the existing
Franky commands. Do not run both physical arm backends concurrently.

## Configuration

`fr3.yaml` publishes the exact saved controller URDF and its FR3 limits. Controller
Kq/Kqd/Kx/Kxd use the pinned DROID FR3 configuration recorded in `sources`; the client
calls `start_joint_impedance()` without overrides. The service retains the upstream
software safety controller, with DROID FR3 bounds checked against the device URDF. Actual
server gains, model hash and execution timings are recorded in each trial trace.
The serial gripper uses the official Polymetis 2F-85 conversion: open/closed encoder
3/230, speed 0.085 m/s to rSP=249 and force 10 percent to rFR=25.

The physical cameras remain RealSense rather than the ZED Mini / ZED 2(i) used
by the published deployment. Their intrinsics and mounting are separate hardware
differences; matching image dimensions does not calibrate those quantities.
