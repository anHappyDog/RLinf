Agentic 数据接口
======================

本节介绍 RLinf 中在 **Megatron + SGLang 后端** 组合下，不同 Worker
之间进行数据传输所使用的关键 Agentic **数据结构**。
其中包含两个基本结构：``RolloutRequest`` 和 ``RolloutResult``。


RolloutRequest
---------------

.. autoclass:: rlinf.data.schema.agentic.requests.RolloutRequest
   :members: 
   :member-order: bysource

RolloutResult
-----------------------

.. autoclass:: rlinf.data.schema.agentic.types.RolloutResult
   :members: 
   :member-order: bysource
   
