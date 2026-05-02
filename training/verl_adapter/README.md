# verl Adapter Layer

Bridges our LLM-as-RNN training pipeline with verl (`training/verl/` submodule,
pinned at `release/v0.7.1`). Adapter code lives here, *not* inside `training/verl/`,
so we never touch the verl submodule.

## Phase status

| Phase | Goal | Status |
|---|---|---|
| **A** | Verify verl + GSM8K + LoRA work on Lonestar6 | ✅ done — Phase A scaffolding has been removed (no longer needed) |
| **B** | Convert our `TrajectoryStep` → verl-compatible parquet | ✅ done — smoke verified on 5 patients |
| **C** | Custom reward function (rubric judge) | ✅ scaffold landed — placeholder judge in YAML; swap to your SFT'd judge by editing one yaml block |
| **D** | Multi-node Lonestar6 training run end-to-end | not started |

## Directory layout

```
verl_adapter/
  README.md                                  # this file
  __init__.py
  dump_trajectories_to_parquet.py            # Phase B: TrajectoryStep -> parquet
  rubric_reward.py                           # Phase C: verl reward function
  smoke_phase_c.py                           # Phase C: end-to-end smoke (live judge + real parquet)
  test_rubric_reward.py                      # Phase C: 11 unit tests, mock judge, runs on Mac
  apptainer/
    README.md                                # first-time Apptainer setup notes
    _apptainer_common.sh                     # shared bind/env helpers
    pull_image.sh                            # one-time docker→sif pull
    dump_trajectories.sh                     # Phase B wrapper (runs the dumper in container)
    start_judge_server.sh                    # Phase C wrapper (vllm serve the judge)
    run_phase1.sh                            # Phase D launcher: GRPO training end-to-end
```

## Lonestar6 specifics

- **TACC project**: `ASC25003`
- **System**: Lonestar6, 3× A100 40GB per node
- **Storage**: `$WORK` for code + Apptainer images, `$SCRATCH` for data + HF cache, **never `$HOME`** (10 GB quota)
- **Apptainer is blocked on login nodes** — all `apptainer` commands must run inside `idev` (or via `sbatch`)
- **`module load tacc-apptainer`** must be re-run on every fresh idev session

## First-time setup (do once per account)

ssh into Lonestar6, then inside an `idev` session:

```bash
ssh <user>@ls6.tacc.utexas.edu
idev -p gpu-a100-dev -N 1 -n 1 --gpus-per-node=3 -t 02:00:00 -A ASC25003
module load tacc-apptainer
cd $WORK/LLMasRNN
git submodule update --init --recursive

# Pull the verl image (~10-20 min, ~15 GB)
bash training/verl_adapter/apptainer/pull_image.sh

# Persist the .sif location so future shells find it
echo 'export VERL_SIF=$WORK/apptainer_images/verlai_verl_vllm017.latest.sif' >> ~/.bashrc
source ~/.bashrc
```

## Phase B usage — dump trajectories to parquet

Inside an idev session with `tacc-apptainer` loaded:

```bash
# Smoke (5 patients, ~5-10 min)
bash training/verl_adapter/apptainer/dump_trajectories.sh \
    --config configs/experiment_smoke.yaml \
    --split train --max-patients 5

# Full train split
bash training/verl_adapter/apptainer/dump_trajectories.sh \
    --config configs/experiment_smoke.yaml \
    --data-path data/splits/cleaned_df_train_100.json --split train

# Same for val
bash training/verl_adapter/apptainer/dump_trajectories.sh \
    --config configs/experiment_smoke.yaml \
    --data-path data/splits/cleaned_df_val_100.json --split val
```

Output: `$SCRATCH/data/llm_as_rnn/<split>.parquet`. Schema mirrors verl's stock
GSM8K example so verl's data loader works without modification:
- `prompt` is a chat-message list (verl applies the chat template at training time)
- `data_source = "llm_as_rnn_memory_update"` is the routing tag for our reward function
- `reward_model.ground_truth` carries the full serialized TrajectoryStep so the reward function can decode `(h_prev, x_t, ŷ_t, e_t)` and render the rubric prompt

## Phase C usage — reward function + judge server

**The rubric YAML is the single source of truth.** `training/configs/rubric_v1.yaml`
carries:
- rubric content (dimensions, prompt template, score format)
- `judge:` block — model id, endpoint, sampling params
- `serve:` block — gpu, port, mem util, max_model_len, dtype

Both `rubric_reward.py` AND `start_judge_server.sh` read from this YAML, so they
can never drift. To swap rubric/judge:
- **Same yaml**: edit `judge.model` in place
- **Side-by-side**: copy to `rubric_v2.yaml`, update `RUBRIC_YAML_PATH` constant
  in `rubric_reward.py` (one line), pass `--rubric` flag to `start_judge_server.sh`

### Quick contract check (Mac, no GPU, no judge needed)

```bash
python training/verl_adapter/test_rubric_reward.py    # expect: 11/11 passed
```

Mocks the judge, validates JSON parsing, prompt rendering, all reward branches.

### End-to-end smoke (Lonestar6, real judge)

```bash
# tmux pane 1: judge server (foreground)
module load tacc-apptainer
bash training/verl_adapter/apptainer/start_judge_server.sh

# tmux pane 2: smoke against real parquet
module load tacc-apptainer
apptainer exec --nv \
    --bind /work:/work --bind /scratch:/scratch \
    --env PYTHONPATH=$WORK/LLMasRNN:$WORK/LLMasRNN/training/verl \
    $VERL_SIF \
    python -m training.verl_adapter.smoke_phase_c \
        --parquet $SCRATCH/data/llm_as_rnn/train.parquet --n 3
```

For each row it scores 3 mock candidates (good / bad-json / empty) so you can see
how the reward function differentiates. Health checks at the end assert
PARSE_FAIL_REWARD always fires for bad/empty, good rewards stay in [0, 1].

### Phase D: GRPO training end-to-end

`training/verl_adapter/apptainer/run_phase1.sh` is a self-contained launcher.
All experiment knobs (LR, KL coef, G, batch size, model name, LoRA rank, etc.)
are exposed at the **top of the file** — verl-style. It:

1. starts the rubric judge server in the background (config from `RUBRIC_YAML`)
2. polls until the judge endpoint is alive
3. launches verl GRPO with our reward function plugged in
4. cleans up the judge on exit (success/failure/Ctrl-C)

```bash
# inside idev with module load tacc-apptainer
bash training/verl_adapter/apptainer/run_phase1.sh

# or as a SLURM job (uncomment SBATCH headers in the script first):
sbatch training/verl_adapter/apptainer/run_phase1.sh

# new experiment? copy and edit the knobs at top:
cp training/verl_adapter/apptainer/run_phase1.sh \
   training/verl_adapter/apptainer/run_phase1_lr5e6.sh
vim training/verl_adapter/apptainer/run_phase1_lr5e6.sh   # edit LR=5e-6, EXPERIMENT_NAME=...
```

**GPU layout** (3-GPU node default):
- `cuda:0,1` → verl actor + rollout (TP=2)
- `cuda:2`   → judge server (set in `rubric_v1.yaml`'s `serve.gpu`)

## Audit logs

Every reward call writes one JSON-line to:
```
$SCRATCH/llm_as_rnn/rubric_audit/calls_<pid>.jsonl
```

(One file per worker PID — verl spawns several reward workers; PID-suffix
avoids concurrent-write races.)

Each row contains:
- `branch` — `judge_called` / `parse_fail` / `wrong_data_source` / `ground_truth_bad_json`
- `patient_id`, `visit_index` — joinable back to the parquet row
- `solution_str` — the **raw rollout** (the full text the policy generated)
- `candidate_h_t` — parsed `evolving_summary` (the input to the judge)
- `judge_prompt` — the **full rendered rubric prompt** sent to the judge
- `judge_response` — raw judge output text
- `reward` — final scalar in [0, 1]

Useful one-liners:
```bash
# All judge responses across workers, sorted by time
cat $SCRATCH/llm_as_rnn/rubric_audit/*.jsonl | jq -s 'sort_by(.ts)'

# Reward distribution (good for spotting collapse)
cat $SCRATCH/llm_as_rnn/rubric_audit/*.jsonl | jq -r '.reward' | sort | uniq -c

# Find candidates that got the lowest reward (and their judge prompts)
cat $SCRATCH/llm_as_rnn/rubric_audit/*.jsonl | jq 'select(.reward < 0.2)' | head

# How many of each branch fired (parse failures vs healthy judge calls)
cat $SCRATCH/llm_as_rnn/rubric_audit/*.jsonl | jq -r '.branch' | sort | uniq -c
```

The default location ($SCRATCH/...) is gitignored. Override by setting
`RR.AUDIT_DIR = Path("...")` programmatically (e.g. in tests).

## Migration notes

- 2026-04-25: switched primary install from conda to Apptainer (conda hit silent torch version downgrade → verl 0.7.1 `DTensorSpec` import fail). The verl Docker image version-matches torch + vllm + flash-attn upstream.
- 2026-04-27: removed Phase A scaffolding (`apptainer/{preprocess_gsm8k,smoke_gsm8k}.sh`, `slurm/`, `env/`) now that Phase A is verified. To re-verify on a new system, follow verl's own example at `training/verl/examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh`.
- 2026-04-27: rubric YAML became single source of truth — judge model + endpoint + serve config all in one file (no env vars).
