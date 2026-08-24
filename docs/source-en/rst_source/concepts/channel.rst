Customize Data Flow with Channel Hooks
======================================

Use a ``Channel`` to move items between WorkerGroups. Add a ``Collector`` when
items must be transformed before enqueueing, and add a ``Dispatcher`` when each
output must be assigned to a specific consumer.

Hook Execution Model
--------------------

Every ``put`` follows one ordered path on the ``ChannelWorker``:

.. code-block:: text

   producer.put(item, key)
            |
            v
   Collector.collect(item, key)
            |
            |  zero or more (output_key, output_item) pairs
            v
   Dispatcher.route(output_item, output_key)
            |
            v
   shared queue or one consumer's private queue

The two hooks solve different problems:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Hook
     - Input
     - Decision
   * - ``Collector``
     - One item passed to ``put``
     - Drop, transform, merge, split, or re-key it.
   * - ``Dispatcher``
     - Each item emitted by the Collector
     - Keep it shared or assign it to one consumer.

Both hooks are synchronous and stateful. ``setup(ctx)`` runs once before
traffic starts. ``collect()`` and ``route()`` then run serially on the channel
worker, so their state does not need locks.

Register a Collector and Dispatcher
-----------------------------------

Define hooks in an importable module, for example
``my_project/channel_hooks.py``:

.. code-block:: python

   from rlinf.scheduler import (
       Collector,
       Dispatcher,
       register_collector,
       register_dispatcher,
   )


   @register_collector("flatten_batch")
   class FlattenBatchCollector(Collector):
       """Emit each sample in an incoming batch as one queue item."""

       def collect(self, item, key):
           for sample in item:
               yield key, sample


   @register_dispatcher("even")
   class EvenDispatcher(Dispatcher):
       """Assign items to consumers in a fixed rotation."""

       def setup(self, ctx):
           self.consumers = sorted(ctx.consumers)
           self.next_consumer = 0

       def route(self, item, key):
           if not self.consumers:
               return None
           consumer = self.consumers[
               self.next_consumer % len(self.consumers)
           ]
           self.next_consumer += 1
           return consumer

The decorators register each class under a lowercase name. ``Channel.create``
also accepts a hook instance, a hook class, or a ``"module:ClassName"`` path.
Passing the class directly is the simplest choice for application-local hooks:

.. code-block:: python

   from my_project.channel_hooks import EvenDispatcher, FlattenBatchCollector
   from rlinf.scheduler import Channel

   sample_channel = Channel.create(
       "Samples",
       collector=FlattenBatchCollector,
       dispatcher=EvenDispatcher,
       producers=[rollout_group],
       consumers=[actor_group],
   )

``rollout_group`` and ``actor_group`` are launched WorkerGroups. Channel setup
expands ``actor_group`` into stable consumer ids such as ``actor:0``,
``actor:1``, and ``actor:2`` and exposes them through ``ctx.consumers``.

If the registration module is imported in the ChannelWorker process, select a
hook by its registered name instead:

.. code-block:: python

   sample_channel = Channel.create(
       "Samples",
       collector="flatten_batch",
       dispatcher="even",
       producers=[rollout_group],
       consumers=[actor_group],
   )

Framework-integrated hooks normally use registered names. External modules can
always use ``"my_project.channel_hooks:FlattenBatchCollector"`` or pass the
class directly.

Follow One Batch Through the Hooks
----------------------------------

Assume one producer sends six samples:

.. code-block:: python

   sample_channel.put(["s0", "s1", "s2", "s3", "s4", "s5"])

The Collector converts one input batch into six queue items. The Dispatcher
then rotates over three consumers:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Consumer
     - First Assignment
     - Second Assignment
   * - ``actor:0``
     - ``s0``
     - ``s3``
   * - ``actor:1``
     - ``s1``
     - ``s4``
   * - ``actor:2``
     - ``s2``
     - ``s5``

Each Actor calls ``sample_channel.get()`` with the same queue key. The Channel
identifies the calling worker and reads its private queue. Assignment counts
differ by at most one when the number of outputs is not divisible by the
consumer count.

Choose the Queue Behavior
-------------------------

Use the defaults until data transformation or consumer ownership is required:

.. list-table::
   :header-rows: 1
   :widths: 29 31 40

   * - Dispatcher
     - Queue Ownership
     - Use It When
   * - ``None`` or ``"shared"``
     - One shared queue per key
     - Any consumer may process any item.
   * - ``"round_robin"``
     - One private queue per consumer
     - Assignment must be deterministic and even.
   * - ``"least_loaded"``
     - One private queue per consumer
     - Producers should balance total assignments.
   * - ``"least_loaded_stealing"``
     - Private queues with stealing
     - Slow consumers must not hold idle peers back.

A Dispatcher that returns ``None`` leaves an item in the shared queue. Returning
a consumer id assigns the item before any consumer calls ``get``. Override
``rebalance()`` only when a blocking consumer may steal from a peer.

Use Channel Keys and Weights
----------------------------

``put(item, key=...)`` and ``get(key=...)`` select an independent logical queue.
A Collector may change the key it received. A Dispatcher routes independently
within each output key.

Set ``weight`` on ``put`` and call ``get_batch(target_weight=...)`` when items
have different costs and consumers need approximately weighted batches. Hook
routing happens before weighted items enter their queues.

For exact method signatures, see :doc:`../reference/api/channel`. For a
production Collector that joins embodied rollout data, continue with
:doc:`trajectory_collector`.
