Franka Pickup Mixture SFT
=========================

Train π₀.₅ on all or selected sources in a curated Franka pickup collection. This recipe reads LeRobot v3.0 partitions and their shared catalog, prepares grouped training and validation splits, and fine-tunes one policy on native control schemas. It preserves each source's sampling rate and action meaning; deployment still requires the matching source controller and normalization statistics.

Overview
--------

Use the same loader for single-arm and dual-arm demonstrations, with a source profile specifying the camera mapping and sampling weight.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅

   .. grid-item-card:: Methods
      :text-align: center

      Full SFT · Masked Flow Matching

   .. grid-item-card:: Data
      :text-align: center

      LeRobot v3.0 · Shared Catalog

   .. grid-item-card:: Hardware
      :text-align: center

      2 NVIDIA GPUs · FSDP

Tasks
~~~~~

The collection combines pickup clips from real and simulated robots. Preparation accepts source-success, clean pickup annotation, and visually confirmed quality tiers; it excludes motion-only candidates and clips with unknown semantic core bounds.

.. list-table::
   :header-rows: 1
   :widths: 28 35 37

   * - Profile
     - Data
     - Default Weight
   * - ``molmobot``
     - Simulated single arm
     - 4
   * - ``habit``
     - Real dual arm
     - 4
   * - ``fmb`` · ``droid``
     - Real single arm
     - 1 each
   * - ``duobench_real`` · ``duobench_sim``
     - Real / simulated dual arm
     - 1 each
   * - ``gr00t_transport``
     - Reviewed simulated dual-arm setup clips
     - 1

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

A training sample uses the current camera frames and state to predict 50 future native actions. Context frames remain available in the source, while only supervised pickup anchors and valid future core actions contribute to the loss.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Item
     - Training Representation
   * - Observation
     - Up to three current RGB views resized to 224 × 224; missing views have a false image mask. Native state uses per-schema ``tanh((state - mean) / std)`` and is padded to 32 dimensions.
   * - Action
     - 50 × 32 tensor. Native 7 / 8 / 14 / 16-dimensional actions are normalized per schema. Padding, clip tails, and non-core targets have zero loss weight.
   * - Prompt
     - Task text plus schema, control mode, and native FPS; up to 200 tokens, including the discretized state input.
   * - History
     - ``data.num_obs_steps: 1``. This recipe uses the current observation; other values are rejected.
   * - Loss
     - Flow-matching error averaged over valid action elements in each sample, then averaged across the batch.

The 50-step horizon covers 5 seconds at 10 Hz and about 1.67 seconds at 30 Hz. The loader keeps native indices, does not interpolate actions, and does not scale delta commands. MolmoBot media runs at 15.15 Hz; its nominal 0.066-second control interval remains separate source metadata.

Prepare the Dataset
-------------------

Start with the converted collection root containing ``catalog/episodes.jsonl`` and its referenced partitions. Keep the original videos available: the partitions may contain relative links, and the loader reads the recorded video offset for each episode.

Run preparation from the repository root in the training environment:

.. code:: bash

   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/prepare_pickup_mixture.py \
     --catalog-root /path/to/unified_lerobot_v1 \
     --output-dir /path/to/prepared-pickup-mixture \
     --allow-development

This writes ``train.jsonl``, ``val.jsonl``, ``normalization.json``, and ``summary.json`` without copying media. ``--allow-development`` explicitly includes historical development clips in training, which allows the small pilot sources to participate. Omit it to train only on eligible non-development data, and select profiles with ``--sources molmobot habit``. Each selected profile must retain eligible training episodes.

The split is deterministic for ``--seed`` and ``--val-fraction``. Episodes sharing a scene, or a source group when a scene is unavailable, stay together. Development groups always stay in training. Statistics use training core rows only and are computed separately for each schema; validation rows never enter normalization. Actions use mean/std normalization. State tensors and the discretized state tokenizer both use ``tanh((state - mean) / std)`` to stay in the tokenizer's bounded range. Constant columns use std 1. Reuse the same schema statistics and transforms at inference; invert the action transform before sending commands to its controller.

For a short trial, add ``--max-train-episodes 2 --max-val-episodes 1``. These limits apply per profile after splitting. Inspect ``summary.json`` to see the actual eligible episodes and supervised anchors before launching training; small sources may contribute no validation episodes.

Installation
------------

Create an isolated environment using :doc:`OpenPI SFT <sft_openpi>` before preparing or training the mixture. Older OpenPI environments may contain LeRobot v2.1: this loader requires the v3 layout and ``lerobot.datasets.video_utils``. The versions verified for this recipe are:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Package
     - Verified Version
   * - torch
     - 2.6.0
   * - torchvision
     - 0.21.0
   * - transformers
     - 4.53.2
   * - lerobot
     - 0.4.3
   * - datasets
     - 4.1.1
   * - av
     - 15.1.0
   * - pyarrow
     - 24.0.0

After the base OpenPI installation, install the mixture loader packages and check the decoder import. ``--no-deps`` preserves the existing Torch and transformers stack; run this in the isolated environment with the base runtime dependencies already installed.

.. code:: bash

   uv pip install --python /path/to/isolated-env/bin/python --no-deps \
     'lerobot==0.4.3' 'datasets==4.1.1' 'av==15.1.0' 'pyarrow<26'
   /path/to/isolated-env/bin/python -c \
     'from lerobot.datasets.video_utils import decode_video_frames; print(decode_video_frames.__name__)'

The command should print ``decode_video_frames``. This v3 reader belongs to the new mixture loader; the existing OpenPI loaders retain their own dataset format requirements.

Download the Model
------------------

Use a π₀.₅ base checkpoint with 32-dimensional model actions. Follow the checkpoint download and conversion instructions in :doc:`sft_openpi` for the supported RLinf layout. Set ``actor.model.model_path`` to the checkpoint directory containing ``model.safetensors``; the loader also converts the key names of official OpenPI PyTorch checkpoints during loading.

The recipe loads the full base model and trains all parameters. FP32 master parameters and optimizer updates are combined with BF16 activations, FSDP ``full_shard``, and gradient checkpointing. It uses the mixture's ``normalization.json`` rather than statistics from an unrelated robot task.

Run It
------

Point both data paths at the prepared directory and launch the standard VLA SFT entry point. The supplied config places actor ranks on GPUs 0 and 1 and uses micro batch size 1 with global batch size 4.

.. code:: bash

   export EMBODIED_PATH="$PWD/examples/sft"
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/train_vla_sft.py \
     --config-path "$EMBODIED_PATH/config" \
     --config-name franka_pickup_sft_openpi_pi05 \
     data.train_data_paths=/path/to/prepared-pickup-mixture \
     data.val_data_paths=/path/to/prepared-pickup-mixture \
     actor.model.model_path=/path/to/pi05-base \
     runner.logger.log_path=/path/to/pickup-sft-results

The default run performs 1,000 optimizer steps, validates every 100 steps, and saves at step 1,000. For a short trial, append ``runner.max_steps=3 runner.val_check_interval=1 runner.save_interval=3``. ``data.samples_per_epoch`` controls the sampled training anchors per epoch, and ``data.eval_max_samples`` bounds validation cost.

The sampler first chooses a schema according to its configured weight, then chooses an eligible anchor from that schema. Source size therefore does not determine its mixture share. Adjust weights or camera mappings in ``examples/sft/config/data/franka_pickup_sources.yaml`` and rerun preparation to record the changed mixture.

Extend the Mixture
------------------

Add a source profile when another collection uses the same catalog and LeRobot layout. Give it a unique ``name``, a matching ``source_dataset``, an optional ``domain``, a positive ``weight``, and a mapping from supported camera slots to its image features. One profile must contain one control mode and FPS; split heterogeneous sources into separate profiles.

The shared loader reads the selected episodes, source video offsets, numeric schema, and supervision masks from the manifests. A new compatible source needs a profile rather than a source-specific loader branch. A new observation modality or physical action representation requires an explicit model adapter.

.. warning::

   Normalization and dimension padding let one model learn multiple native schemas. They do not establish a shared physical action space. Converting all sources to a target robot controller requires verified coordinate frames, units, gripper conventions, and kinematics before resampling or retargeting.

Visualization and Results
-------------------------

Read TensorBoard logs under ``runner.logger.log_path`` and the saved checkpoints under ``checkpoints/global_step_<N>/``. Validation uses a fixed seed, restores training RNG state, and disables image augmentation. Its flow loss measures prediction of held-out demonstrations; robot pickup success requires a separate closed-loop evaluation with the matching controller. See :doc:`Training Metrics </rst_source/reference/metrics>` for the shared logging conventions.
