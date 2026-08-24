Embodied Data 接口
==================

具身数据接口将运行时通信与 Actor 训练数据分开：

.. code-block:: text

   EnvOutput
       -> PolicyInput (+ previous PolicyCompletion)
       -> PolicyOutput(actions)

   PolicyPart + EnvPart
       -> TrajectoryCollector
       -> Trajectory / episode shard / pipeline micro-batch

完整生命周期、数据所有权和执行模式语义参见
:doc:`../../concepts/trajectory_collector`。

环境与策略消息
--------------

``EnvOutput`` 是环境 reset 或执行 action 后的本地结果。``PolicyInput`` 将
observation 发送给 Rollout，并可携带上一个 action 的 ``PolicyCompletion``。
``PolicyOutput`` 是返回 Env 的纯 action 响应。

.. autoclass:: rlinf.data.schema.embodied_types.EnvOutput
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.PolicyInput
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.PolicyCompletion
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.PolicyOutput
   :members:
   :member-order: bysource

Trajectory Part
---------------

``PolicyPart`` 和 ``EnvPart`` 是 Actor channel 仅有的两种输入，并通过
``TrajectoryKey`` 配对。Channel routing 拆分 batch 时，``TrajectorySource``
保留 source size 和 offset。

.. autoclass:: rlinf.data.schema.embodied_types.TrajectoryKey
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.TrajectorySource
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.PolicyPart
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.EnvPart
   :members:
   :member-order: bysource

采集
----

``TrajectoryPlan`` 校验输出模式并推导 rollout geometry。所有具身模式都使用同一个
公共 ``TrajectoryCollector``。

.. autoclass:: rlinf.data.schema.trajectory_collector.TrajectoryMode
   :members:

.. autoclass:: rlinf.data.schema.trajectory_collector.TrajectoryPlan
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.trajectory_collector.TrajectoryCollector
   :members:
   :member-order: bysource

Actor 输出
----------

``Trajectory`` 保存非 pipeline Actor worker 使用的训练 tensor，典型布局为
``[T, B, ...]``。Boundary 和 bootstrap 字段可能包含 ``T + 1`` 个元素。

``ChunkStepResult`` 描述一个已配对 action chunk 所保留的数据。跨 chunk 的时序累积
属于 collector 的私有实现细节。

.. autoclass:: rlinf.data.schema.embodied_types.Trajectory
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.ChunkStepResult
   :members:
   :member-order: bysource
