# B1K RL machine inventory

Observed on 2026-09-15. Revalidate before every run; this is a routing aid, not
proof that a restarted machine still has the same mounts or GPUs.

| Role / SSH name | GPUs | Storage group | Runtime paths |
| --- | --- | --- | --- |
| Current trainer (`is-dc7kruhvtugdvm6w-devmachine-0`) | 4 × A100 80GB | NXB | canonical paths below |
| `nxb_4090` | 8 × RTX 4090D 24GB | NXB | same path strings and shared `/mnt/public` as trainer group |
| `gdb_4090_1` | 8 × RTX 4090 24GB | GDB | same path strings, different `/mnt/public` filesystem |
| `gdb_4090_2` | 4 × RTX 4090 24GB | GDB | shares `/mnt/public` with `gdb_4090_1` |

Canonical runtime paths:

```text
RLinf:        /mnt/public/daibo/timeline/0831/RLinf-b1k-singleenv
Python:       /mnt/public/daibo/venv/behavior_openpi/bin/python
B1K root:     /mnt/public/daibo/timeline/0831/BEHAVIOR-1K-b1k-rl-singleenv
OmniGibson:   /mnt/public/daibo/timeline/0831/BEHAVIOR-1K-b1k-rl-singleenv/OmniGibson
Data root:    /mnt/public/daibo/datasets/omni_data
Dataset:      /mnt/public/daibo/datasets/omni_data/behavior-1k-assets
Key:          /mnt/public/daibo/datasets/omni_data/omnigibson.key
Robot assets: /mnt/public/daibo/datasets/omni_data/omnigibson-robot-assets
Results:      /mnt/public/daibo/results/b1k_grounded_control_v01
```

The venv has an editable OmniGibson installation. Always prepend the canonical
`OmniGibson` path and verify `omnigibson.__file__`; otherwise NXB and GDB can use
different source with the same Python command.

The GDB RLinf deployment may not contain `.git`. Verify its source manifest;
do not report its revision from a copied marker alone.

