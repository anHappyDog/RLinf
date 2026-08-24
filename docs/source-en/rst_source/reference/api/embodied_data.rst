Embodied Data Interface
=======================

The embodied data interface separates runtime communication from Actor-facing
training data:

.. code-block:: text

   EnvOutput
       -> PolicyInput (+ previous PolicyCompletion)
       -> PolicyOutput(actions)

   PolicyPart + EnvPart
       -> TrajectoryCollector
       -> Trajectory / episode shard / pipeline micro-batch

See :doc:`../../concepts/trajectory_collector` for the lifecycle, data
ownership, and execution-mode semantics.

Environment and policy messages
-------------------------------

``EnvOutput`` is the local result of environment reset or action execution.
``PolicyInput`` transports observations to Rollout and may carry the previous
action's ``PolicyCompletion``. ``PolicyOutput`` is the action-only response sent
back to Env.

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

Trajectory parts
----------------

``PolicyPart`` and ``EnvPart`` are the only Actor channel input types. They are
matched by ``TrajectoryKey``. ``TrajectorySource`` preserves source size and
offset when channel routing splits a batch.

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

Collection
----------

``TrajectoryPlan`` validates the configured output mode and derives rollout
geometry. ``TrajectoryCollector`` is the one public channel collector for every
embodied mode.

.. autoclass:: rlinf.data.schema.trajectory_collector.TrajectoryMode
   :members:

.. autoclass:: rlinf.data.schema.trajectory_collector.TrajectoryPlan
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.trajectory_collector.TrajectoryCollector
   :members:
   :member-order: bysource

Actor output
------------

``Trajectory`` contains the tensors used by non-pipeline Actor workers. Its
typical layout is ``[T, B, ...]``. Boundary and bootstrap fields may have
``T + 1`` entries.

``ChunkStepResult`` describes the data retained for one joined action chunk.
Sequence accumulation is a private collector implementation detail.

.. autoclass:: rlinf.data.schema.embodied_types.Trajectory
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.ChunkStepResult
   :members:
   :member-order: bysource
