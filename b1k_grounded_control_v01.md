# BEHAVIOR-1K Grounded-Control VLA 实现规范 v0.1

> 目标：为 BEHAVIOR-1K 构建一个可被 Codex 直接实现的、结构化的 VLA conditioning 与训练/评测 pipeline。
>
> 本规范中的 `GroundedControlSpec`、结构 token、role taxonomy 与 serializer 是**项目设计**，不是 BEHAVIOR-1K 官方格式。数据应从 BEHAVIOR-1K annotation / simulator replay 中解析后映射到本规范。

---

## 0. 项目目标

当前阶段**不重新预训练完整 π0.6 foundation model**。以已有的 π0.5 / BEHAVIOR-adapted VLA checkpoint 为起点，先回答一个核心问题：

> 在 BEHAVIOR-1K 中，给 low-level VLA 提供 Grounded Subgoal（正确 skill + 相关 object + spatial grounding）是否显著优于只提供 global goal 或 Simple Subgoal？

第一阶段优先完成 **Oracle Study**：

1. `Direct`：overall goal → action
2. `SimpleSG-Oracle`：GT subgoal / skill / object names → action
3. `GroundSG-Oracle`：GT subgoal / skill / object names + GT object grounding → action
4. `PartGroundSG-Oracle`：在 GroundSG 基础上加入可可靠获取的 part grounding

只有在 `GroundSG-Oracle >> SimpleSG-Oracle` 时，才进入第二阶段训练 Grounded High-Level Predictor。

第二阶段训练统一 VLM 的语义能力：

- Grounded subgoal generation
- object / part grounding
- task-state / progress prediction
- 可选 FAST auxiliary action-token objective

推理时不得依赖 simulator privileged state；privileged information 只用于生成训练 target 和 Oracle evaluation condition。

---

# 1. 总体模型接口

在时间步 `t`，Action Expert 的语义 condition 定义为：

\[
C_t = (G,\; S_t,\; K_t,\; \mathcal{A}_t)
\]

其中：

- `G`：episode-level overall goal
- `S_t`：当前 macro subgoal，可选
- `K_t`：当前 canonical atomic skill
- `A_t`：当前 skill 的 typed arguments

\[
\mathcal{A}_t = \{A_t^{(1)}, A_t^{(2)}, ..., A_t^{(N_t)}\}
\]

**不要假设一个 skill 只有一个 object。**

每个 argument：

\[
A_i = (role_i,\ entity_i,\ qualifier_i,\ grounding_i,\ part_i)
\]

即：

- semantic role
- entity identity / textual category
- optional qualifier
- object-level grounding
- optional part + part grounding

最终 Action Expert 的 prefix 由：

\[
[\text{image tokens},\ \text{control condition tokens},\ \text{state tokens}]
\]

经过 VLM 得到 prefix KV cache，然后 noisy action + flow timestep 作为 Action Expert suffix。

---

# 2. Goal / Subgoal / Skill 的严格定义

## 2.1 Goal

时间尺度：整个 episode。

例：

```text
turn on the radio
```

字段：

```python
goal: str
```

必须始终保留，除非专门做 `w/o goal` ablation。

## 2.2 Subgoal

时间尺度：macro stage / semantic stage。

例：

```text
press the power button on the radio
```

字段：

```python
subgoal: str | None
```

要求：

- 不得简单复制 `skill`
- 如果 BEHAVIOR annotation 中存在可靠的 coarse subtask，就直接使用
- 如果当前数据只有 atomic skill annotation，则第一阶段可以令 `subgoal=None`
- 后续如需构造 subgoal，必须由独立的数据规则/annotation 映射生成，不要默认为 LLM 自由生成

## 2.3 Skill

时间尺度：atomic control segment。

例：

```text
press
pick up from
place in
open door
```

字段：

```python
skill: str
```

要求：

- 使用 BEHAVIOR annotation 中的 canonical skill taxonomy
- 不允许同义词漂移，例如训练中不要同时使用 `pick`, `pick up`, `grasp`
- skill value 使用自然语言 tokenizer，不为每个 skill 分配独立 special token

---

# 3. Argument 定义

## 3.1 为什么使用 Argument，而不是固定 Object

不同 skill 的 object 数量不同：

```text
press(button)
pick_up_from(cup, table)
place_in(plate, sink)
place_on_next_to(bowl, table, plate)
```

因此统一表示为：

```python
arguments: list[EntityArgument]
```

一个 object/entity 对应一个 `EntityArgument`。

## 3.2 Role Taxonomy

v0.1 固定以下 role：

```python
class Role(Enum):
    TARGET = ...
    MANIPULATED = ...
    SOURCE = ...
    DESTINATION = ...
    REFERENCE = ...
    TOOL = ...
    OTHER = ...
```

语义：

### `MANIPULATED`
机器人正在抓取、搬运、移动、倒出、使用的主体 object。

### `SOURCE`
`MANIPULATED` object 当前来自的位置 / receptacle / support。

### `DESTINATION`
`MANIPULATED` object 应被放置到的位置 / receptacle / support。

### `REFERENCE`
不是直接 destination，但用于约束目标空间关系的 entity。

### `TARGET`
机器人直接施加动作的目标，但该 object 不一定被搬运，例如 button / switch / door / drawer / appliance。

### `TOOL`
用于作用于另一个 object 的工具。

### `OTHER`
当前无法可靠赋予上述语义的额外 argument。仅作为 fallback。

---

# 4. Skill Signature Registry

必须实现：

```python
SkillSignatureRegistry
```

其职责：

```text
canonical_skill
    ↓
ordered expected roles
```

例如：

```python
SKILL_SIGNATURES = {
    "move to": ["TARGET"],
    "pick up from": ["MANIPULATED", "SOURCE"],
    "place in": ["MANIPULATED", "DESTINATION"],
    "place on": ["MANIPULATED", "DESTINATION"],
    "place under": ["MANIPULATED", "REFERENCE"],
    "place in next to": ["MANIPULATED", "DESTINATION", "REFERENCE"],
    "place on next to": ["MANIPULATED", "DESTINATION", "REFERENCE"],
    "insert": ["MANIPULATED", "DESTINATION"],
    "attach": ["MANIPULATED", "DESTINATION"],
    "hang": ["MANIPULATED", "DESTINATION"],
    "open door": ["TARGET"],
    "close door": ["TARGET"],
    "open drawer": ["TARGET"],
    "close drawer": ["TARGET"],
    "open lid": ["TARGET"],
    "close lid": ["TARGET"],
    "press": ["TARGET"],
    "turn on switch": ["TARGET"],
    "turn off switch": ["TARGET"],
}
```

**重要：上表只作为初始 seed。**

必须写一个脚本枚举 BEHAVIOR-1K 全部 canonical skills，并打印每个 skill 下：

```text
skill_description
object_id
manipulating_object_id
spatial_prefix
memory_prefix
```

的真实样本。

然后人工核验所有 skill 的 signature，尤其：

- chop
- pour
- spray
- sweep
- wipe
- ignite
- tool-use 类 skill

不要凭名称猜 role。

---

# 5. `manipulating_object_id` 的映射规则

若 BEHAVIOR annotation 中：

```python
obj in manipulating_object_id
```

则优先：

```python
role(obj) = Role.MANIPULATED
```

剩余 object 根据 `SkillSignatureRegistry` 与 annotation order 赋 role。

如果 mapping 产生冲突：

- 不允许静默修复
- 输出 preprocessing error / warning
- 将该 segment 标记为 `ambiguous_signature`
- 默认不进入 GroundSG 训练集

---

# 6. EntityArgument 数据结构

```python
@dataclass
class EntityArgument:
    role: Role

    # semantic fields
    category_name: str
    instance_id: str | None  # privileged metadata only

    qualifier: str | None

    # object-level grounding
    groundings: dict[CameraID, Grounding2D]

    # optional object part
    part: PartArgument | None

    # optional annotation/debug metadata
    raw_object_id: str | None = None
```

其中：

- `category_name`：进入模型 prompt，例如 `"radio"`
- `instance_id`：仅 preprocessing / debugging，**不得进入部署 prompt**
- `qualifier`：如 `left`, `second left`, `top`, `the other` 等，但 v0.1 不自动把 memory expression 放入 action prompt
- `groundings`：每个 camera 的可见 grounding
- `part`：可选 part

---

# 7. Part 定义

```python
@dataclass
class PartArgument:
    name: str
    groundings: dict[CameraID, Grounding2D]
```

例：

```text
object = radio
part   = power button
```

或：

```text
object = cabinet
part   = second left door
```

规则：

1. part 必须属于某个 entity，不作为平级 argument
2. 只有能够从 part-level segmentation / prim hierarchy / reliable asset mapping 获取 GT 时，才生成 `part_bbox`
3. 如果只有 part text，没有可靠 part mask：
   - 保留 `part.name`
   - `part.groundings = {}`
4. 禁止根据 object bbox 人为估计 part bbox

---

# 8. Grounding 的正式定义

## 8.1 Grounding2D

```python
@dataclass
class Grounding2D:
    camera: CameraID
    bbox_xyxy: tuple[float, float, float, float]  # normalized [0,1]
    visible_pixels: int
    visible_fraction: float
    point_xy: tuple[float, float] | None = None   # normalized [0,1]
```

内部统一 bbox 顺序：

```text
xmin, ymin, xmax, ymax
```

全部 normalized 到 `[0, 1]`。

## 8.2 Object BBox

Object bbox 定义为：

> 当前 camera 下，该 object **实际可见像素**的 tight 2D bbox。

不是：

- projected 3D AABB
- object 完整几何边界
- hidden / occluded 部分的 privileged bbox

从 instance / part segmentation mask 计算：

\[
M_{o,c}(x,y)=1
\]

表示 pixel `(x,y)` 属于 object `o` 的某个 mesh/link。

如果 object 对应多个 mesh / links：

\[
M_{o,c} = \bigvee_{m \in meshes(o)} M_{m,c}
\]

然后：

\[
x_{min}=\min_{M=1} x,\quad y_{min}=\min_{M=1} y
\]

\[
x_{max}=\max_{M=1} x,\quad y_{max}=\max_{M=1} y
\]

若可见 pixel 数低于阈值，则视为该 camera 无 grounding。

配置：

```yaml
grounding:
  min_visible_pixels: 16
  min_visible_fraction: 0.0001
```

阈值必须可配置。

## 8.3 Part BBox

Part bbox 使用同一规则，但 mask 仅包含该 part 的 mesh / prim。

禁止用整个 object bbox 代替 part bbox。

## 8.4 Point Grounding

可选 point 定义优先使用 mask 内部稳定点，而不是简单 bbox center。

v0.1 推荐实现两种：

```python
PointMode.BBOX_CENTER
PointMode.MASK_CENTROID
```

默认：

```yaml
point_mode: mask_centroid
```

如果 centroid 不落在 mask 内，可以投影到最近的 foreground pixel。

后续可增加：
- distance-transform maximum
- contact / interaction point

第一阶段 Oracle Study 可同时记录 bbox + point，但 serializer profile 单独决定是否输入模型。

---

# 9. Multi-Camera Grounding

底层必须支持多 camera：

```python
groundings = {
    CameraID.HEAD: ...,
    CameraID.LEFT_WRIST: ...,
    CameraID.RIGHT_WRIST: ...,
}
```

但 v0.1 Action Prompt **每个 argument 只选择一个 primary grounding view**。

默认 selection：

\[
c_i^* = \arg\max_c visible\_fraction(i,c)
\]

即每个 object 独立选择可见面积比例最大的 camera。

配置：

```yaml
primary_view_policy: max_visible_fraction
```

后续允许：
- fixed head
- prefer wrist for precise skill
- multi-view serialization

如果所有 camera 都不可见：

```text
<no_grounding>
```

不得编码成 `0,0,0,0`。

---

# 10. Coordinate Tokenization

如果当前 π0.5/PaliGemma tokenizer 已存在：

```text
<loc0000> ... <loc1023>
```

直接使用。

内部 bbox 为：

```python
(xmin, ymin, xmax, ymax)  # [0,1]
```

序列化为：

```text
<loc_ymin><loc_xmin><loc_ymax><loc_xmax>
```

即 PaliGemma location 顺序：

```text
ymin, xmin, ymax, xmax
```

量化：

\[
q(v)=round(1023 \cdot clamp(v,0,1))
\]

必须写 round-trip unit test。

---

# 11. Structural Token 定义

优先复用当前 tokenizer / checkpoint 中**确认未使用的 reserved token IDs**。

不要先假设具体 ID。

初始化时：

```python
ReservedTokenAllocator(tokenizer)
```

必须：

1. 枚举 reserved / unused IDs
2. 验证它们没有被当前 OpenPI tokenizer pipeline 占用
3. 验证 input embedding 与 LM head shape 已包含这些 IDs
4. 建立固定 mapping
5. 将 mapping 写入 checkpoint/config
6. resume 时禁止 mapping 改变

v0.1 logical tokens：

```text
<goal>
<subgoal>
<skill>

<arg>
<end_arg>

<role_target>
<role_manipulated>
<role_source>
<role_destination>
<role_reference>
<role_tool>
<role_other>

<object>
<qualifier>
<part>

<view_head>
<view_left_wrist>
<view_right_wrist>

<object_bbox>
<part_bbox>
<point>
<no_grounding>

<end_control>
```

规则：

- Structure：reserved token ID
- Semantic values：普通 tokenizer vocabulary
- Geometry：已有 `<locXXXX>` token

不要为 `press`、`radio`、`sink`、`pick up from` 等语义值分配专用 special token。

---

# 12. GroundedControlSpec 数据结构

```python
@dataclass
class GroundedControlSpec:
    goal: str
    subgoal: str | None
    skill: str | None
    arguments: list[EntityArgument]

    episode_id: str | None = None
    segment_id: int | None = None
    timestep: int | None = None
```

Dataset 中保存 `GroundedControlSpec`，**不要直接保存最终 prompt/token string**。

最终输入由 serializer profile 动态生成。

---

# 13. Serializer Grammar

正式 grammar：

```text
CONTROL :=
    <goal> TEXT
    [<subgoal> TEXT]
    [<skill> TEXT]
    ARG*
    <end_control>

ARG :=
    <arg>
    ROLE
    <object> TEXT
    [<qualifier> TEXT]
    [OBJECT_GROUNDING]
    [PART]
    <end_arg>

ROLE :=
      <role_target>
    | <role_manipulated>
    | <role_source>
    | <role_destination>
    | <role_reference>
    | <role_tool>
    | <role_other>

OBJECT_GROUNDING :=
    VIEW
    <object_bbox>
    LOC LOC LOC LOC
    [<point> LOC LOC]
    | <no_grounding>

PART :=
    <part> TEXT
    [PART_GROUNDING]

PART_GROUNDING :=
    VIEW
    <part_bbox>
    LOC LOC LOC LOC
    [<point> LOC LOC]

VIEW :=
      <view_head>
    | <view_left_wrist>
    | <view_right_wrist>

LOC :=
    <loc0000> ... <loc1023>
```

---

# 14. Prompt 示例

## 14.1 Direct

```text
<goal>
turn on the radio
<end_control>
```

## 14.2 SimpleSG Oracle

```text
<goal>
turn on the radio

<subgoal>
press the power button on the radio

<skill>
press

<arg>
<role_target>
<object>
radio
<part>
power button
<end_arg>

<end_control>
```

## 14.3 GroundSG Oracle

```text
<goal>
turn on the radio

<subgoal>
press the power button on the radio

<skill>
press

<arg>
<role_target>
<object>
radio

<view_head>
<object_bbox>
<loc0212><loc0317><loc0671><loc0723>

<part>
power button

<view_head>
<part_bbox>
<loc0411><loc0583><loc0457><loc0632>

<end_arg>

<end_control>
```

## 14.4 Two-object skill

```text
<goal>
put the dishes in the sink

<subgoal>
place the plate inside the sink

<skill>
place in

<arg>
<role_manipulated>
<object>
plate
<view_head>
<object_bbox>
<loc...><loc...><loc...><loc...>
<end_arg>

<arg>
<role_destination>
<object>
sink
<view_head>
<object_bbox>
<loc...><loc...><loc...><loc...>
<end_arg>

<end_control>
```

---

# 15. Serializer / Condition Profiles

## P0 — Direct

包含：

```text
Goal
```

用途：`global task → action`。

## P1 — SimpleSG

包含：

```text
Goal
Subgoal
Skill
Argument roles
Object names
Part names
```

不包含 bbox / point。

用途：`SimpleSG Oracle`。

## P2 — GroundSG

包含 P1，加：

```text
primary-view object bbox
optional point
```

默认不加 part bbox。

用途：`GroundSG Oracle`。

## P3 — PartGroundSG

包含 P2，加：

```text
part bbox / part point
```

只对具有可靠 part GT 的 sample 开启。

## P4 — PredictedGroundSG

格式与 P2/P3 **完全一致**，但字段值来自 VLM predictor。

Action Expert 不知道 condition 是 GT 还是 predicted。

---

# 16. Action Expert Prefix 定义

最终 prefix：

```text
[head image tokens]
[left wrist image tokens]
[right wrist image tokens]
[GroundedControlSpec serialized tokens]
[state/proprio tokens]
```

由 VLM forward：

\[
H_{prefix}, KV_{prefix} = VLM(prefix)
\]

Action Expert suffix：

```text
noisy action chunk
flow timestep
```

每次 flow denoising step 复用 `KV_prefix`。

---

# 17. High-Level Predictor 与 Action Expert 的接口

High-Level Predictor 输出必须直接符合同一个 control grammar。

输入：

```text
images
overall goal
optional history
```

输出：

```text
<subgoal> ...
<skill> ...
<arg> ...
...
<end_control>
```

然后重新构造 action prefix：

```text
images
+
goal
+
predicted structured control
+
state
```

再生成新的 prefix KV cache。

**禁止：**

- 生成 GroundSG 后继续使用生成前的旧 prefix cache
- 让 high-level 输出自由文本 reasoning，再用 heuristic parser 转结构
- Oracle 与 predicted 使用不同 prompt format

---

# 18. Preprocessing Pipeline

实现：

```text
BEHAVIOR episode
    ↓
load episode annotation
    ↓
split by skill/subtask segment
    ↓
replay / load segmentation
    ↓
resolve object IDs → simulator prims / categories
    ↓
SkillSignatureRegistry
    ↓
assign argument roles
    ↓
extract per-camera object masks
    ↓
extract object bbox / point
    ↓
extract reliable part mask/bbox when available
    ↓
GroundedControlSpec
    ↓
save structured dataset / index
```

建议模块：

```text
b1k_grounded/
├── schema.py
├── tokens.py
├── skill_registry.py
├── annotation_parser.py
├── entity_resolver.py
├── grounding_extractor.py
├── part_resolver.py
├── serializer.py
├── profiles.py
├── dataset_builder.py
├── validation.py
└── tests/
```

---

# 19. 数据派生视图

同一个 trajectory / segment 后续应可生成多个 training view。

## 19.1 Action View

输入：

```text
RGB / depth
proprio
serialized control condition
```

target：

```text
action chunk
```

loss：

\[
L_{FM}
\]

可选：

\[
L_{FAST}
\]

## 19.2 Grounded High-Level View

输入：

```text
RGB / depth
overall goal
optional history
```

target：

```text
Subgoal
Skill
Arguments
Grounding
```

loss：

\[
L_{GroundHL}=CE
\]

第一阶段 Oracle Study **不需要实现该训练任务**。

## 19.3 Progress / State View

第二阶段再实现。

输入：

```text
RGB / depth
goal
optional history
```

target 由 BDDL / object states 自动生成：

```text
which goal predicates are satisfied
whether current skill completed
relevant object state
```

loss：

\[
L_{Progress}=CE
\]

不要第一阶段混入 generic VQAv2 / caption 数据。

---

# 20. 第一阶段：Oracle Study

## 20.1 目标

回答：

\[
GroundSG\ Oracle \stackrel{?}{\gg} SimpleSG\ Oracle
\]

在 BEHAVIOR-1K 上是否成立。

## 20.2 第一阶段只需要

实现：

- B1K annotation parser
- SkillSignatureRegistry
- object/part grounding extractor
- GroundedControlSpec
- structural token allocator
- serializer profiles P0/P1/P2/P3
- action SFT dataloader
- rollout/eval conditioning switch

暂时不需要：

- GroundSG predictor
- memory architecture
- progress head
- full π0.6 reproduction
- external VQA/web data

## 20.3 模型

从已有 π0.5 / BEHAVIOR-adapted checkpoint 起步。

Oracle Study 要保证不同 condition profile 的模型训练预算可比。

第一版建议两种实现方式择一：

### Option A：分别训练 P0/P1/P2

最干净，适合论文 ablation。

### Option B：一个模型 condition dropout

一个模型随机接受 P0/P1/P2 条件。

工程成本低，但不同 profile 之间可能互相影响。

**默认优先 Option A。**

---

# 21. 第一阶段评测

首先做 skill-reset evaluation，而不是只看 full-task success。

对于每个 GT skill segment：

```text
reset to / replay near skill start state
    ↓
provide GT condition
    ↓
execute action policy
    ↓
evaluate skill completion
```

得到：

```text
skill type
num trials
success count
conditional success rate
```

重点比较：

```text
P0 Direct
P1 SimpleSG Oracle
P2 GroundSG Oracle
P3 PartGroundSG Oracle
```

核心差值：

\[
\Delta_{semantic}=S_{P1}-S_{P0}
\]

\[
\Delta_{ground}=S_{P2}-S_{P1}
\]

\[
\Delta_{part}=S_{P3}-S_{P2}
\]

如果 `Delta_ground` 明显且稳定为正，再进入第二阶段。

---

# 22. 第二阶段：Learned Grounded High-Level Predictor

训练：

\[
(o_t, goal, history) \rightarrow GroundedControlSpec_t
\]

至少输出：

```text
Subgoal
Skill
arguments
object grounding
```

然后评估：

1. `OracleSG + OracleGround`
2. `OracleSG + PredGround`
3. `PredSG + OracleGround`
4. `PredSG + PredGround`

这样可以分解：

- planner error
- grounding error
- low-level execution error

---

# 23. 第三阶段：Progress / Memory

只有第二阶段确认瓶颈后再加入。

训练 B1K-native progress supervision：

```text
current goal predicates
current skill completion
object states relevant to task
```

然后比较：

```text
current observation only
subgoal history
short visual memory
structured progress memory
```

不要一开始把 memory 与 GroundSG、action adaptation 同时改变。

---

# 24. 必须实现的 Unit Tests

## Token Tests

- structural token IDs 不与已有使用 token 冲突
- checkpoint save/load 后 mapping 不变
- `<loc>` token 全部可 encode/decode
- bbox quantization round trip

## Annotation Tests

- 每个 canonical skill 都有 signature 或被明确标记 unsupported
- argument 数量与 signature 匹配
- manipulating object 的 role 一致
- ambiguity 不静默吞掉

## Grounding Tests

随机可视化 100+ samples：

```text
RGB
+ GT object mask
+ computed bbox
+ computed point
+ object/role text
```

必须人工 sanity check。

## Serializer Tests

- 同一个 `GroundedControlSpec` 可生成 P0/P1/P2/P3
- 不存在字段不会生成 `none`
- multiple arguments order deterministic
- same sample serialize deterministic

## Train/Inference Consistency

Oracle 与 predicted path：

```text
GroundedControlSpec -> serializer -> prefix
```

必须共用完全相同代码。

---

# 25. 必须提供的 Debug Visualization

实现一个 offline visualization script。

输入：

```text
episode_id
segment_id
timestep
```

输出一张图或 HTML：

```text
RGB image
object bbox
part bbox
point
role
object name
subgoal
skill
final serialized prompt
token IDs
```

这是第一阶段最重要的 debug 工具之一。

---

# 26. Config 建议

```yaml
condition:
  profile: ground_sg

  include_goal: true
  include_subgoal: true
  include_skill: true
  include_arguments: true

  include_object_bbox: true
  include_object_point: false

  include_part: true
  include_part_bbox: false

  primary_view_policy: max_visible_fraction

grounding:
  bbox_mode: visible_tight_2d
  point_mode: mask_centroid
  min_visible_pixels: 16
  min_visible_fraction: 0.0001

tokens:
  structural_token_source: reserved_unused
  use_existing_loc_tokens: true

preprocessing:
  drop_ambiguous_signatures: true
  drop_missing_required_objects: true

training:
  action_loss: flow_matching
  fast_aux_loss_weight: 0.0

evaluation:
  profiles:
    - direct
    - simple_sg
    - ground_sg
    - part_ground_sg
```

---

# 27. Codex 实现顺序

严格按以下顺序实现。

### Milestone 1 — Schema

实现：

```text
GroundedControlSpec
EntityArgument
PartArgument
Grounding2D
Role
CameraID
```

并测试 JSON serialization。

### Milestone 2 — Skill Registry

扫描真实 BEHAVIOR annotations。

生成：

```text
skill_audit.json
```

包含每种 skill 的真实 object/role 样本。

人工确认后冻结 `SkillSignatureRegistry`。

### Milestone 3 — Grounding Extraction

从 segmentation / replay 提取：

```text
object mask
bbox
point
part mask if available
```

实现 debug visualization。

### Milestone 4 — Structural Token Layer

实现：

```text
ReservedTokenAllocator
ControlSerializer
P0/P1/P2/P3 profiles
```

### Milestone 5 — Dataset Builder

把 B1K trajectory 转成结构化 sample/index。

### Milestone 6 — OpenPI Integration

修改 tokenizer / transform，使：

```text
GroundedControlSpec
    ↓
token IDs
    ↓
VLM prefix
    ↓
Action Expert
```

保持原 action representation / flow loss 不变。

### Milestone 7 — Oracle Training

分别训练：

```text
P0
P1
P2
```

P3 只在 part grounding coverage 足够时训练。

### Milestone 8 — Skill-Level Evaluation

实现：

```text
per-skill conditional success
per-task skill breakdown
P0/P1/P2/P3 comparison
```

### Milestone 9 — Decide Next Step

如果 GroundSG gap 显著：

```text
implement GroundHL predictor
```

否则优先解决 low-level action / RL，而不是继续堆 planner。

---

# 28. 第一轮明确不做的事情

为了控制变量，v0.1 不做：

- π0.6 foundation-scale pretraining
- DROID/OXE/AgiBot/web-VQA 混合
- generic image captioning
- MEM/RMT
- long-term memory
- RL
- predicted GroundSG
- arbitrary free-form chain-of-thought
- object 3D pose privileged input
- simulator object-state privileged inference input

这些可以作为后续阶段，而不是第一轮 prerequisite。

---

# 29. 第一轮 Success Criteria

工程层：

- 100% canonical skill annotation 可解析或明确标记 unsupported
- GroundedControlSpec 可稳定生成
- bbox 可视化正确率人工抽检接近 100%
- P0/P1/P2 serializer deterministic
- training / inference 使用同一个 condition path
- OpenPI action SFT 正常收敛

研究层：

得到可信的：

```text
Direct
SimpleSG Oracle
GroundSG Oracle
PartGroundSG Oracle（若可用）
```

在：

```text
per-skill
per-task
full-task（后续）
```

上的结果。

第一阶段最重要的判断条件：

\[
S_{GroundSG\ Oracle} - S_{SimpleSG\ Oracle}
\]

是否足够大、是否跨多个 skill/task 稳定存在。

---

# 30. 最终设计原则

整个系统遵循三个分工：

\[
\boxed{\text{Structure = reserved structural token IDs}}
\]

\[
\boxed{\text{Semantics = pretrained natural-language vocabulary}}
\]

\[
\boxed{\text{Geometry = pretrained / fixed location tokens}}
\]

Action Expert condition 不是：

```text
one subgoal + one bbox
```

而是一个 typed grounded skill invocation：

\[
\boxed{
C_t=
\left[
G,\ S_t,\ K_t,\
\{
role_i,\ object_i,\ qualifier_i,\ bbox_i,\ part_i,\ partBBox_i
\}_{i=1}^{N_t}
\right]
}
\]

其中：

- `N_t` 由当前 skill 的 argument arity 决定
- 每个 entity 有独立 grounding
- grounding 与 camera 明确绑定
- object 与 part grounding 严格区分
- Oracle / predicted condition 共享同一个 schema 和 serializer
- simulator privileged information 只作为 label / Oracle，不作为最终部署输入

这份协议一旦进入实现阶段，除非发现 BEHAVIOR annotation 本身无法支持某个字段，否则不要随意修改 schema；新增能力优先通过 serializer profile / optional field 扩展，而不是重写数据格式。
