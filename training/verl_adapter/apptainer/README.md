# Apptainer wrappers — verl on Lonestar6

These scripts run our verl-related commands inside verl's official Docker image
(`verlai/verl:vllm017.latest`) via Apptainer (the HPC-compatible container
runtime — Docker itself is not allowed on TACC for security reasons).

## Why containers (vs conda)

verl's image ships torch 2.10 + vllm 0.17 + flash-attn + apex + TransformerEngine
all version-matched by the maintainers. Conda installs of these are fragile
(silent torch downgrades, ToS gates, multi-hour compiles). One `.sif` file =
identical bits everywhere; reproducible across nodes and machines.

## How it works

The image provides every heavy dep but **not verl itself**. We expose our
submodule's verl source (and the rest of our repo) via `PYTHONPATH` so
`import verl` and `import models...` find the right code:

```
host                                   inside container
$WORK/LLMasRNN/        ────bind────►  /work/<id>/jinrui/ls6/LLMasRNN/
$SCRATCH/              ────bind────►  /scratch/<id>/jinrui/
                       PYTHONPATH=    /work/.../LLMasRNN + /work/.../training/verl
                       HF_HOME=       /scratch/<id>/jinrui/hf_cache
```

## ⚠️ All apptainer commands need a COMPUTE NODE on TACC

TACC blocks apptainer on login nodes (the binary prints a banner and exits 0
without doing anything — silent failure). Run inside `idev` or `sbatch`.

`module load tacc-apptainer` must be re-issued every fresh idev session.

## First-time setup (do once per account)

```bash
ssh <user>@ls6.tacc.utexas.edu
idev -p gpu-a100-dev -N 1 -n 1 --gpus-per-node=3 -t 02:00:00 -A ASC25003
module load tacc-apptainer
cd $WORK/LLMasRNN
git submodule update --init --recursive

# Pull the verl image (~10-20 min, ~15 GB)
bash training/verl_adapter/apptainer/pull_image.sh

# Persist the .sif location for future shells
echo 'export VERL_SIF=$WORK/apptainer_images/verlai_verl_vllm017.latest.sif' >> ~/.bashrc
source ~/.bashrc
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `apptainer: command not found` | TACC hides it behind a module | `module load tacc-apptainer` (the helpers try this automatically) |
| `pull` hangs / fails on Docker Hub | rate limit or network | retry; if rate-limited, log in: `apptainer remote login docker://docker.io` |
| `import verl` fails inside container | submodule not initialized | `git submodule update --init --recursive` |
| `CUDA out of memory` | `gpu_memory_utilization` in YAML too high | edit `training/configs/rubric_v1.yaml` (judge) or `experiment_smoke.yaml` (RLN) and lower |
| Image pull works but `--nv` fails | NVIDIA driver libs not visible | usually a TACC issue; `module load cuda` before apptainer commands |
| Multi-vLLM OOM on cuda:0 | configs missing `cuda_visible_devices` | each model in YAML must explicitly pin its GPU |

## Files

- `pull_image.sh` — one-time `apptainer pull` (must run inside idev)
- `dump_trajectories.sh` — Phase B wrapper: runs `dump_trajectories_to_parquet.py` in container
- `start_judge_server.sh` — Phase C wrapper: `vllm serve` the rubric judge (config from rubric YAML)
- `_apptainer_common.sh` — shared helpers; sets `VERL_SIF`, `BIND_ARGS`, `ENV_ARGS`. Don't run directly; sourced by the others.
