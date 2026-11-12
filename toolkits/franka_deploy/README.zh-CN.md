# Franka canonical pickup 部署

[English](README.md)

本工具用于 uav001 上已有的独立 Polymetis 部署，也保留旧 Pi0 和 MolmoBot F2 接口。
配置描述当前工位，不是通用机器人默认值。工作目录为 `/home/infini/daibo`；
代码位于 `code/franka_deploy`，模型使用 `runtime/python.sh`，硬件使用
`runtime/polymetis_fr3/python.sh`。依赖已有的私有 Polymetis 二进制和 device-source，
本目录不包含环境安装器，也不修改其他用户的环境。

## 模型接口

`models/pi05_franka_pickup_sft_step4549/actor` 下需要：

- `model_state_dict/full_weights.pt`
- `upstream_assets/train_config.pkl`
- `upstream_assets/canonical_qtarget.json`

加载器严格恢复全部 `core.*` 参数，保留模型选择性 BF16/FP32 布局。
State 为 qpos7（rad）＋真实夹爪宽度（米），八维全部有效；使用 checkpoint 的
Molmo quantile 统计，再执行官方 Pi0.5 state tokenization 和 padding。
不能继续置零宽度，也不能将它转换成仿真夹爪驱动角。

两路 RGB 使用训练时裁剪和224×224 letterbox，第三槽补零并设置 False mask，task 不加前缀。
模型预测 H16×32，只解码前8维。前7维反归一化后，每一帧都加回同一次观测的 qpos，
恢复绝对关节目标；不是累积 delta，也不能重复加 qpos。夹爪保持连续闭合度，0开、1闭，
控制端映射为 `(1 - closedness) * 0.085` 米。不额外套用 F2 的缩放或二值化规则。

服务和客户端均指定 `--execute-horizon 8` 时，每次预测16步、执行前8步后用新观测重规划；
也支持两端同时指定16。请求动作间隔为0.066秒，实际循环包含推理等待，记录于 `ticks.jsonl`。
`--seed` 仅初始化一次服务随机流；重连会清动作缓存，不重置随机流。
需要从相同 seed 重复时重启模型进程；更换 prompt 需要新连接。

## 左臂启动

已有左臂控制服务运行时不要重复启动50061端口。否则，操作员在现场、左臂 FCI 已开启后运行：

```fish
cd /home/infini/daibo
runtime/polymetis_fr3/python.sh code/franka_deploy/polymetis_server/launch.py \
    --config code/franka_deploy/polymetis_server/fr3.yaml --execute
```

先结束占用18080端口或显存的旧模型进程，再启动模型：

```fish
cd /home/infini/daibo
env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    taskset -c 0-5,16-31 runtime/python.sh \
    -m franka_deploy.canonical_tick_server \
    --model-kind pi05-qtarget \
    --checkpoint models/pi05_franka_pickup_sft_step4549 \
    --action-dt 0.066 --execute-horizon 8 --denoise-steps 10 --seed 0 --port 18080
```

显示 `Serving` 后，另开 fish 终端：

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

第一次 Enter 打开夹爪并执行固定5秒 home；第二次 Enter 开始模型动作。
第二次 Enter 前查看 `post_home_preview`，确认目标出现在腕部画面。
连接时可能在第一次 Enter 前激活夹爪。腕部旋转180°保留当前工位测试设置，
这是图像补偿，不是转动第7关节，也不是训练预处理。

回 home 后只等待控制器结束，不下载无用的1kHz控制日志。没有自动机械臂故障恢复，
故障由操作员处理。持续录像在独立线程读取相机缓存，覆盖 home、等待和执行过程；
输出为 `videos/external.mp4` 与 `videos/wrist.mp4`，按相机配置帧率写入，可能重复缓存帧。
时间戳 sidecar 保留采集和写入时间，腕部视频与模型输入使用相同旋转。
正常结束和 Ctrl+C 会封装视频，强制杀进程不保证封装完成。

右臂使用独立配置 `configs/molmobot_f2_polymetis_right.yaml` 与
`polymetis_server/fr3_right.yaml`，端口50063，不需要改动左臂配置。
