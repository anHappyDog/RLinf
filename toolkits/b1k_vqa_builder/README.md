# B1K VQA 数据构建工具

这个工具把 B1K grounded-control sidecar 转换成 RLinf 可直接读取的多选
VQA Parquet。Ground truth 始终来自 `control_json` 中的类别、部件、角色和
grounding；Qwen3.8-27B 只负责问题改写、受控干扰项选择和视觉质检，不能修改
正确标签。

流水线包含五步：

1. 从三路 MP4 精确读取 `frame_index`，选择可见像素最多的视角并绘制红框。
2. 根据确定性标注生成 object、part、role 和 skill 四类候选问题。
3. 通过 SGLang 调用 Generator，改写问题并生成语义接近的 hard negatives。
4. 通过第二次、隐藏 GT 的调用让 Judge 独立答题并检查歧义。
5. 过滤不清晰或 Judge 答错的样本，嵌入 JPEG bytes 并导出三个 Parquet split。

## 目录

```text
toolkits/b1k_vqa_builder/
├── build_candidates.py       # 抽帧、画框、确定性 GT 与候选项
├── run_sglang.py             # Generator / Judge，支持断点续跑
├── finalize_dataset.py       # 过滤并导出 RLinf Parquet
├── validate_dataset.py       # schema、答案、图片及 split 泄漏检查
├── make_review_sheet.py      # 生成人工抽检 HTML
├── launch_sglang_qwen38_27b.sh
├── run_pipeline.sh
├── configs/
└── prompts/
```

默认结果写到 `/mnt/public/daibo/results/b1k_vqa_v01`，不会修改 grounded-control
sidecar 或原始 B1K 视频。

## 1. 安装离线构建依赖

在用于生成数据的 Python 环境中运行：

```bash
cd /mnt/public/daibo/timeline/0831/RLinf
python -m pip install -r toolkits/b1k_vqa_builder/requirements.txt
```

SGLang 可以安装在另一个环境或容器中，只要 OpenAI-compatible API 能从构建
进程访问即可。

## 2. 启动 Qwen3.8-27B

四张 A100 使用 tensor parallel 4：

```bash
cd /mnt/public/daibo/timeline/0831/RLinf
bash toolkits/b1k_vqa_builder/launch_sglang_qwen38_27b.sh \
  /path/to/Qwen3.8-27B
```

默认监听 `0.0.0.0:30000`。启动后检查服务：

```bash
curl http://127.0.0.1:30000/v1/models
```

如果 `/v1/models` 返回的模型 ID 不是 `Qwen/Qwen3.8-27B`，把
`configs/b1k_vqa_qwen38_27b.yaml` 中的 `sglang.model` 改成服务返回的 ID。
配置默认关闭 thinking，以减少数据生成成本并提高 JSON 稳定性。
Generator 会优先参考 B1K 同类标签；当原始词表缺少合适负例时，可以提出同语义
层级的新负例，例如为 `radio` 生成 `speaker`。正确标签仍由 sidecar 固定，盲测
Judge 会过滤同义项或多解问题。若只允许 B1K 词表，把
`sglang.restrict_distractors_to_pool` 改为 `true`。

## 3. 运行完整流水线

先检查以下路径是否符合实际环境：

- `source.sidecar_path`
- `source.dataset_root`
- `output_root`
- `sglang.base_url`
- `sglang.model`

然后运行：

```bash
cd /mnt/public/daibo/timeline/0831/RLinf
bash toolkits/b1k_vqa_builder/run_pipeline.sh \
  toolkits/b1k_vqa_builder/configs/b1k_vqa_qwen38_27b.yaml
```

`run_sglang.py` 逐条追加成功记录；中断后重新执行同一命令时会跳过已有 ID，
只处理尚未完成的记录。失败项写入相邻的 `*.errors.jsonl`。
本机服务默认忽略 `HTTP_PROXY`/`ALL_PROXY`；只有远程 API 确实依赖环境代理时才把
`sglang.use_environment_proxy` 设为 `true`。

也可以逐步运行：

```bash
TOOL=/mnt/public/daibo/timeline/0831/RLinf/toolkits/b1k_vqa_builder
CFG=${TOOL}/configs/b1k_vqa_qwen38_27b.yaml

python ${TOOL}/build_candidates.py --config ${CFG}
python ${TOOL}/run_sglang.py --config ${CFG} --mode generate
python ${TOOL}/run_sglang.py --config ${CFG} --mode judge
python ${TOOL}/finalize_dataset.py --config ${CFG}
python ${TOOL}/validate_dataset.py --config ${CFG}
python ${TOOL}/make_review_sheet.py \
  --dataset-dir /mnt/public/daibo/results/b1k_vqa_v01/dataset \
  --output /mnt/public/daibo/results/b1k_vqa_v01/review.html
```

## 4. 先做无模型 smoke test

候选生成不依赖 SGLang。使用至少三个 task，以满足三路 group split：

```bash
TOOL=/mnt/public/daibo/timeline/0831/RLinf/toolkits/b1k_vqa_builder
CFG=${TOOL}/configs/b1k_vqa_qwen38_27b.yaml
SMOKE=/tmp/b1k_vqa_smoke

python ${TOOL}/build_candidates.py \
  --config ${CFG} \
  --output-root ${SMOKE} \
  --limit 100

python ${TOOL}/finalize_dataset.py \
  --config ${CFG} \
  --input ${SMOKE}/intermediate/candidates.jsonl \
  --dataset-dir ${SMOKE}/dataset \
  --allow-unjudged

python ${TOOL}/validate_dataset.py \
  --config ${CFG} \
  --dataset-dir ${SMOKE}/dataset
```

## 输出格式

```text
b1k_vqa_v01/
├── images/                    # 红框 JPEG，中间产物
├── intermediate/
│   ├── candidates.jsonl      # 确定性候选
│   ├── generated.jsonl       # Generator 结果
│   ├── judged.jsonl          # Judge 结果
│   └── *.errors.jsonl
├── dataset/
│   ├── train/part-00000.parquet
│   ├── validation/part-00000.parquet
│   ├── test/part-00000.parquet
│   └── manifest.json
└── review.html
```

Parquet 的训练核心列为：

```text
image: struct<bytes: binary, path: string>
question: string
choices: list<string>
correct_answer: string     # A/B/C/D
correct_label: string
solution: string
```

`image.bytes` 存 JPEG，而不是普通路径字符串，兼容 RLinf 当前的
`Robo2VLMDataset` 和 `Robo2VLMSFTDataset`。SFT 与 GRPO 的 data/reward 配置片段
分别在 `configs/rlinf_sft_data.yaml` 和 `configs/rlinf_grpo_data.yaml`。

## 数据质量与切分

默认配置使用 `pilot_50ep_best3`，因为它从三个时间点中选择可见性最好的帧。
此外还施加更严格的 VQA 门槛：至少 256 个可见像素、bbox 两边至少 12 px，
并过滤过大的框和稀疏 mask。
role 问题只从至少包含两个参数的 primitive 生成，避免单参数动作仅凭
`Current primitive` 就能猜出角色。

当前 sidecar 每个 task 只有一个 episode，因此默认按 `task_index` 做 40/5/5
切分。这是未见 task 泛化评估。将来每个 task 有多个 episode 后，应把
`splits.group_by` 改成 `episode_index`，以得到同 task、不同 episode 的评估；
仍然不能随机按 VQA 行切分。

最终验收至少包括：

- `validate_dataset.py` 无错误。
- 打开 `review.html`，每类人工检查至少 30 条。
- 运行一次不输入图像的 text-only baseline，确认准确率没有异常高于随机水平。
- 分 question type 报告 macro accuracy，避免高频类别掩盖小类失败。

重建同一个输出目录时，候选脚本要求显式传入 `--overwrite`。如果候选内容或
配置已经改变，Generator 与 Judge 也应使用 `--overwrite` 重新生成；更安全的
做法是为新配置使用新的 `output_root`。
