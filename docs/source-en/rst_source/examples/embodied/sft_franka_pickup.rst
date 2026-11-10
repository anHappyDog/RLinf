Franka Pickup Mixture SFT
=========================

Train π₀.₅ on all or selected sources in a curated Franka pickup collection. This recipe reads LeRobot v3.0 partitions and their shared catalog, prepares grouped training, validation, and test splits, and fine-tunes one policy with explicit control adapters. Choose the legacy native-control profile or the v2 joint-relative profile, then reuse its action and observation transforms at inference. Each source retains its original sampling rate.

For a MolmoBot checkpoint, follow `Use the Official MolmoBot Model`_ below. That route preserves the checkpoint's original processor, normalization, and H16 action horizon.

Overview
--------

Use the same loader for single-arm and dual-arm demonstrations, with a source profile specifying the camera mapping and sampling weight.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅ · MolmoBot π₀

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
     - Legacy Default Weight
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
     - Up to three current RGB views resized to 224 × 224; missing views have a false image mask. State is encoded and normalized per schema, then padded to 32 dimensions. The legacy profile uses ``tanh((state - mean) / std)``; v2 joint profiles use quantiles and canonical grippers.
   * - Action
     - 50 × 32 tensor. Native 7 / 8 / 14 / 16-dimensional actions pass through the selected adapter and per-schema normalization. Padding, clip tails, and non-core targets have zero loss weight.
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

Use the v2 source configuration for verified DROID and DuoBench joint interfaces. Pass multiple converted roots when new pickup collections live alongside the original catalog:

.. code:: bash

   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/prepare_pickup_mixture.py \
     --catalog-root /path/to/unified_lerobot_v1 /path/to/duobench_pickup_v2 \
     --source-config examples/sft/config/data/franka_pickup_sources_v2.yaml \
     --output-dir /path/to/prepared-pickup-mixture-v2 \
     --action-horizon 50 --action-dim 32 \
     --sampling-unit episode --statistics-max-anchors 8192 \
     --allow-development

This writes ``train.jsonl``, ``val.jsonl``, ``test.jsonl``, ``normalization.json``, ``contract.json``, and ``summary.json`` without copying media. Use ``--sources duobench_real duobench_sim`` to select profiles. Each selected profile needs eligible training episodes. ``--allow-development`` includes historical pilot clips in training; omit it to exclude those clips.

The preparer honors catalog ``assigned_split`` values of ``train``, ``val``, and ``test``. Otherwise, ``--seed`` and ``--val-fraction`` split by scene, or by source group when a scene is unavailable. All clips from one original episode stay together, and development groups remain in training. The test manifest is reserved for independent evaluation; the SFT loader reads only train and validation. For a short trial, add ``--max-train-episodes 2 --max-val-episodes 1``. These limits apply per profile after splitting. Check the actual episode and anchor counts in ``summary.json`` before training.

V2 DROID and DuoBench profiles subtract the current observed joint position from every future joint target in the action chunk. All future steps share that reference; this is different from taking consecutive differences. Grippers map to 0=open and 1=closed and pass through normalization unchanged. Profiles with verified ``adapter`` or ``normalization`` entries compute statistics after this transformation, using valid supervised future targets only. Deterministic hash sampling selects at most 32 anchors per original episode and at most ``--statistics-max-anchors`` per schema. Quantile normalization uses q01/q99 without clipping. The sample count, transform, and normalization parameters are recorded in the manifest.

To retain the original native-action recipe, use ``examples/sft/config/data/franka_pickup_sources.yaml``. Profiles without adapter or normalization entries use action mean/std and state ``tanh((state - mean) / std)``, calculated from training core rows. The v2 source file selects DROID and DuoBench only; use the legacy file for other sources while their controller contracts are reviewed. Validation and test data never contribute statistics.

The prepared contract hashes the split manifests and normalization file. Adapter-aware preparation also binds the action horizon and model dimension; changing either requires new statistics. Each checkpoint saves normalization, summary, contract, and exposure records in ``mixture_assets``. Resume checks reject a different prepared mixture rather than silently continuing with changed action semantics.

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

With ``--sampling-unit episode``, the sampler chooses a schema by its configured weight, then an original source episode uniformly, then one of its eligible anchors uniformly. Multiple clips from one original episode share that episode's probability. ``--sampling-unit frame`` retains the legacy uniform-within-schema anchor sampling. Weights specify source shares, independent of source size; draws are with replacement. Change the selected source configuration and rerun preparation to record a different mixture.

For a remote host with a slow data link, materialize the exact model inputs before transfer. Set ``data.mixture.sample_cache_dir=/path/to/sample-cache`` to read its ``train`` and ``val`` subdirectories. Each subdirectory contains ``metadata.json`` with ``contract`` from the corresponding dataset's ``sample_cache_contract()`` and ``indices`` listing cached dataset indices. Store each unchanged ``dataset[index]`` dictionary as ``<index>.pkl.gz`` using gzip and pickle protocol 5. Create these artifacts from your prepared dataset and verify their hashes after transfer.

The cache preserves the original dataset length, sampling weights, source groups, numeric arrays, image values, tokens, and masks. Its contract binds the selected manifest rows, schemas, dimensions, token length, and preprocessing source hashes. Include every index requested by the chosen sampler seed, epochs, sample budget, and validation limit. A changed input contract or a missing cached index stops loading; regenerate the relevant cache when changing the recipe.

Match Inference to Training
---------------------------

Use the same prepared schema to turn native observations into model input and predicted chunks back into controller commands. In an evaluation configuration for a matching 16-dimensional DuoBench environment, set:

.. code:: yaml

   actor:
     model:
       action_dim: 16
       openpi:
         task: eval
         config_name: native_mixture
         action_horizon: 50
         model_action_dim: 32
         mixture:
           manifest_dir: /path/to/prepared-pickup-mixture-v2
           schema_id: duobench_sim

This installs ``MixturePolicyIO`` using the selected schema. ``manifest_dir`` also accepts a checkpoint's ``mixture_assets`` directory; its normalization hash is checked against the saved contract. Supply batched native ``states`` and ``task_descriptions``, plus HWC RGB views under model slots in ``images`` or the conventional ``main_images``, ``wrist_images``, and ``extra_view_images`` keys. RGB values are uint8 in [0, 255] or floating point in [0, 1]. The schema selects the required views. Training and inference share resizing, padding, prompt construction, state encoding, and normalization.

Action decoding removes padded dimensions, reverses normalization and gripper mapping, and adds the original observed joint positions to the relative chunk. The original state travels with the request explicitly; normalized state is never the joint reference. ``native_mixture`` online I/O currently supports ordinary ``task: eval``. ``rtc_enabled: true`` is rejected: a relative action chunk from a previous request must first be rebased to the new observation's joint reference. This numerical interface has CPU coverage; using it in a simulator requires verified state order, camera mapping, controller commands, and execution timing. Pickup success still requires closed-loop evaluation.

Extend the Mixture
------------------

Add a source profile when another collection uses the same catalog and LeRobot layout. Give it a unique ``name``, a matching ``source_dataset``, an optional ``domain``, a positive ``weight``, and a mapping from supported camera slots to its image features. One profile must contain one control mode and FPS; split heterogeneous sources into separate profiles. Specify ``adapter`` for reversible control changes and ``normalization`` for transformed statistics. Keep joint/state indices, gripper endpoints, units, and coordinate frames explicit in the source contract.

The shared loader reads the selected episodes, source video offsets, numeric schema, and supervision masks from the manifests. A new compatible source needs a profile rather than a source-specific loader branch. A new observation modality or physical action representation requires an explicit model adapter.

.. warning::

   Normalization and dimension padding let one model learn multiple native schemas. They do not establish a shared physical action space. Converting all sources to a target robot controller requires verified coordinate frames, units, gripper conventions, and kinematics before resampling or retargeting.

Visualization and Results
-------------------------

Read TensorBoard logs under ``runner.logger.log_path`` and the saved checkpoints under ``checkpoints/global_step_<N>/``. Validation uses a fixed seed, restores training RNG state, and disables image augmentation. Its flow loss measures prediction of held-out demonstrations; robot pickup success requires a separate closed-loop evaluation with the matching controller. See :doc:`Training Metrics </rst_source/reference/metrics>` for the shared logging conventions.

Use the Official MolmoBot Model
-------------------------------

Use ``franka_pickup_sft_openpi_official`` to fine-tune a MolmoBot checkpoint through the original OpenPI PyTorch core. This route uses the checkpoint's data transforms and normalization, absolute joint targets, and H16/D8 actions. Select native MolmoBot single-arm manifests or use the DROID native8 reader described below; other robot interfaces need their own explicit adapter.

Install the Official Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a separate Python 3.11 environment for the official model. The installer selects Torch 2.7.1 and pins the OpenPI, MolmoBot, and LeRobot source revisions used by this recipe:

.. code:: bash

   UV_TORCH_BACKEND=cu126 bash requirements/install.sh embodied \
     --venv .venv-openpi-official --model openpi_official --env dummy
   source .venv-openpi-official/bin/activate

The installer applies the original OpenPI transformers replacement to this environment. It installs the model and data dependencies with CPU JAX; Torch performs the model computation. The ``dummy`` target selects the SFT runtime. The Docker target for this environment is ``embodied-openpi-official``.

Download the checkpoint together with its training config and normalization assets:

.. code:: bash

   hf download allenai/MolmoBot-Pi0-DROID \
     --revision 48c8e2fc1868fd34dba1dd059278e7a2430a5383 \
     --local-dir /path/to/molmobot-pi0-droid

The model directory must retain ``assets/train_config.pkl`` and ``assets/molmobot/norm_stats.json``. The loader reads the trusted checkpoint config to reconstruct the original processor.

Train With the Checkpoint Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Point the official recipe at the prepared native manifests and the checkpoint directory:

.. code:: bash

   export EMBODIED_PATH="$PWD/examples/sft"
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/train_vla_sft.py \
     --config-path "$EMBODIED_PATH/config" \
     --config-name franka_pickup_sft_openpi_official \
     data.train_data_paths.target=/path/to/prepared-native-pickup \
     data.train_data_paths.auxiliary=/path/to/prepared-molmo-pickup \
     data.val_data_paths.target=/path/to/prepared-native-pickup \
     actor.model.model_path=/path/to/molmobot-pi0-droid \
     runner.logger.log_path=/path/to/official-pickup-sft-results

This model uses its original mean/std normalization and full-element flow-matching loss. The loader selects anchors with a complete supervised H16 future and records each sample's source and original episode for exposure accounting. Read the resulting sample counts before choosing a training budget.

Read DROID native8 manifests whose prepared schema in ``normalization.json`` declares ``native_control_mode: droid_absolute_joint_position_and_normalized_closure``. Both the state and action contain seven absolute joint positions followed by continuous gripper closure, with 0=open and 1=closed. The MolmoBot route retains its native state9/action8 layout and driver/actuator-command scales.

The DROID adapter requires a MolmoBot ``MlSpacesDatasetConfigFactory`` checkpoint with ``joint_pos_actions=True``. It skips only ``GripperStateTransform`` and ``GripperActionTransform``, since DROID already supplies normalized closure. All other checkpoint data transforms, statistical normalization, and model transforms remain in the original order. DROID and MolmoBot transform chains are cached separately. A shared storage format does not make different action interfaces interchangeable.

To add prepared DROID data to the named source mapping above, append ``+data.train_data_paths.droid=/path/to/prepared-droid``. Its schemas become ``droid/<schema_id>`` and use the existing auxiliary sampling stream. This reader support covers data preparation; actual SFT and closed-loop pickup performance require separate validation with the selected checkpoint.

The example uses ``openpi_cosine`` with peak learning rate 5e-5 and 1,000 warmup updates. The first optimizer update uses 5e-5 / 1001, about 4.995e-8. Each optimizer update advances the scheduler once. The configured schedule length is 300,000 updates; with ``min_lr: 5e-5``, the learning rate stays at the peak after warmup.

The named directories namespace their source schemas as ``target/molmobot`` and ``auxiliary/molmobot``. With ``data.official.sampling: target_exposure``, the target and auxiliary streams use independent random generators. Set ``target_samples_per_epoch`` and ``auxiliary_samples_per_epoch`` under ``data.official`` to control their counts; changing the auxiliary count preserves the target stream's sampled order and exposure within each epoch.

For an exposure-matched comparison, run the same number of complete epochs in target-only and mixed training, and report the additional optimizer updates used by the auxiliary data. A fixed ``runner.max_steps`` can end the runs at different points in their target streams; compare the saved source sample counts. Append ``runner.max_epochs=1 runner.max_steps=-1 data.official.target_samples_per_epoch=512 data.official.auxiliary_samples_per_epoch=0`` for target-only training. For mixed training, use the same overrides with ``data.official.auxiliary_samples_per_epoch=512``. With global batch size 4, these runs use 128 and 256 updates respectively, each with 512 target samples.

Adding auxiliary samples also adds optimizer updates and advances warmup further. Include an additional target-only control with the same number of updates as mixed training, and report the actual target sample counts. Record the learning rate used before each optimizer update. The existing ``learning_rate`` metric is read after the scheduler advances and therefore gives the next update's rate.

For a two-update target-only SFT integration test, set ``OPENPI_OFFICIAL_MODEL_PATH`` and ``OPENPI_OFFICIAL_DATA_PATH`` to the checkpoint and target data paths, then run ``bash tests/e2e_tests/sft/run_vla_sft.sh franka_pickup_sft_openpi_official_test`` with ``REPO_PATH`` set to the repository. The corresponding SFT CI job is enabled when repository variables with those two names point to assets on its runner.

Export for the Original Policy Loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export the trained actor to a new directory before loading it with the original OpenPI or MolmoBot policy:

.. code:: bash

   python -m rlinf.models.embodiment.openpi_official.checkpoint \
     --actor-dir /path/to/experiment/checkpoints/global_step_N/actor \
     --output-dir /path/to/exported-policy

The exporter reads ``full_weights.pt`` and ``upstream_assets`` from the actor directory, strictly restores the ``core.*`` tensors into the original CPU model, and saves ``model.safetensors`` with shared-weight metadata. It copies the original training config and data normalization statistics into ``assets`` and writes ``EXPORT.json`` after export completes. Keep this complete directory, including ``EXPORT.json``, when transferring the trained policy; the RLinf ``full_weights.pt`` keys cannot be used directly as an original-model safetensors file.

RLinf recognizes ``EXPORT.json`` and loads these weights through ``rlinf.models.embodiment.openpi_official.policy.load_exported_policy``. External OpenPI or MolmoBot services should use the same helper with their training config and exported directory. The upstream factory temporarily casts FP32 normalization-layer weights to BF16; the helper strictly restores exported tensors after their final dtypes are established, preserving the learned FP32 updates. Publisher checkpoints retain the original upstream loading path.
