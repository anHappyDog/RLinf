Nsight Systems
==============================

This document introduces the ``cluster.nsight`` configuration in RLinf for
system-level profiling with NVIDIA Nsight Systems.

RLinf supports wrapping selected Ray worker groups with ``nsys profile`` so you
can collect traces for CUDA kernels, cuDNN, cuBLAS, NVTX ranges, and optionally
CPU-side runtime activity.


How To Enable It
------------------------------

In an embodied YAML, add the Nsight preset to ``defaults``:

.. code-block:: yaml

   defaults:
     - training_backend/fsdp@actor.fsdp_config
     - weight_syncer/patch_syncer@weight_syncer
     - nsight/default@cluster.nsight

The corresponding config files are:

- ``examples/embodiment/config/nsight/default.yaml``


Default Preset
------------------------------

The built-in default preset looks like this:

.. code-block:: yaml

   enabled: true
   worker_groups: [ActorGroup, RolloutGroup, EnvGroup, Actor, Rollout, Env]
   options:
     t: cuda,cudnn,cublas,nvtx,osrt
     sample: process-tree
     cpuctxsw: process-tree
     osrt-threshold: 1000

This preset targets the most common embodied compute and communication workers by default:

- ``ActorGroup``
- ``RolloutGroup``
- ``EnvGroup``
- ``Actor``
- ``Rollout``
- ``Env``

These names must match real worker group names such as ``actor.group_name`` and
``rollout.group_name``. They are not the component aliases ``actor`` or
``rollout``.


The ``enabled`` Flag
------------------------------

The ``enabled`` field is the main switch for Nsight wrapping:

.. code-block:: yaml

   cluster:
     nsight:
       enabled: false

When ``enabled: false``:

- RLinf does not wrap workers with ``nsys profile``
- RLinf does not reserve the default Nsight output directory
- the rest of the config can stay in place for later reuse

So there is no need to maintain a separate ``disabled.yaml``. You can keep the
same preset and override ``cluster.nsight.enabled: false`` in the main YAML.


How To Override Worker Groups
------------------------------

You can override the preset directly in the main YAML:

.. code-block:: yaml

   cluster:
     nsight:
       worker_groups: [EnvGroup, RolloutGroup, ActorGroup, Env, Rollout, Actor]

This is especially useful when you want to profile:

- compute workers such as ``ActorGroup`` or ``RolloutGroup``
- channel workers such as ``Env``, ``Rollout``, and ``Actor``
- environment workers such as ``EnvGroup``

If ``worker_groups`` is omitted, RLinf profiles all worker groups.

One subtle point is that ``ChannelWorker`` is not launched as a child process of
``ActorGroup`` or ``RolloutGroup`` ranks. In the current implementation,
``Channel.create(name)`` launches a separate worker group whose group name is
usually ``Env``, ``Rollout``, or ``Actor``. So profiling ``ActorGroup`` does
not automatically include the ``Actor`` channel worker. If you want channel-side
traces, add those channel group names explicitly to ``worker_groups``.


How To Override Nsight Options
------------------------------

``cluster.nsight.options`` maps directly to ``nsys profile`` CLI flags:

.. code-block:: yaml

   cluster:
     nsight:
       options:
         t: cuda,cudnn,cublas,nvtx,osrt
         sample: process-tree
         cpuctxsw: process-tree

Useful options include:

- ``t``: traced APIs such as ``cuda``, ``cudnn``, ``cublas``, ``nvtx``, and ``osrt``
- ``sample``: CPU sampling mode
- ``cpuctxsw``: CPU thread scheduling trace
- ``capture-range`` and ``capture-range-end``: restrict collection to NVTX or CUDA-profiler-controlled ranges
- ``o`` or ``output``: explicit output prefix

If you enable ``capture-range: nvtx``, make sure your code actually emits NVTX
ranges. Otherwise Nsight may generate an almost empty report.


Output Path
------------------------------

When ``cluster.nsight.enabled`` is true and you do not explicitly set ``o`` or
``output``, RLinf writes reports under:

.. code-block:: text

   runner.logger.log_path/runner.logger.experiment_name/nsights

For example:

.. code-block:: text

   ../results/libero_spatial_ppo_openpi/nsights/

If you want a custom path, set it explicitly:

.. code-block:: yaml

   cluster:
     nsight:
       options:
         o: /mnt/public/profiles/my_run/worker_trace


Recommended Workflow
------------------------------

For a first pass, the simplest setup is:

- start with ``nsight/default@cluster.nsight``
- keep ``enabled: true``
- use the preset as-is if you want both CUDA-side traces and CPU/channel-side runtime visibility
- avoid ``capture-range: nvtx`` until you have confirmed the target workers
  really emit NVTX ranges
