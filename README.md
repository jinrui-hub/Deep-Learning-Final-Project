## Research Context

This is a research project reproducing the paper **"LLM-as-RNN"** (arxiv 2601.13352). The core idea: convert a frozen LLM into a recurrent system by maintaining a **natural-language hidden state** (an "evolving summary") that updates at each timestep through feedback-driven text modifications. No model weights are changed — all "learning" happens through prompt rewriting.

The current focus is **MIMIC-IV discharge diagnosis prediction**: given a patient's sequential hospital visits, predict the discharge diagnosis for each visit. Evaluation uses **Acc@1** and **Acc@5** via an LLM-as-Judge (semantic equivalence, not string matching). Because this is a time-series task, scoring is computed **only on each patient's final visit** — earlier visits serve as context / hidden-state buildup and are not included in the aggregate metric (though their predictions and evaluations are still recorded for qualitative analysis).

## Configuration

The single config file is `configs/experiment1_seperateJudge.yaml`. It contains all hyperparameters, model settings (four-model architecture), and prompt templates (from paper Appendix C).

Key features of the current config:
- **Four-model architecture**: `prediction_model` + `memory_model` + `evaluation_model` + `final_judge_model`.
- **Dedicated baseline prompt templates**: `zero_shot_prompt` and `history_prompt` are explicit top-level templates (aligned to paper Appendix C.3).
- **Simplified prediction JSON**: each diagnosis is `{"name": "..."}` only — no `confidence` fields.

## Commands

```bash
# Run full experiment (all 4 methods, all patients)
python run_experiment.py --config configs/experiment1_seperateJudge.yaml

# Run specific methods only
python run_experiment.py --config configs/experiment1_seperateJudge.yaml --methods zero_shot,llm_as_rnn

# Run on specific patients
python run_experiment.py --config configs/experiment1_seperateJudge.yaml --patients 10001401

# Install dependencies
pip install -r requirements.txt
```

Set API keys via environment: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `PORTKEY_API_KEY`.

## Architecture

The system implements 4 methods that all share the same data pipeline and evaluation:

**The RLN (LLM-as-RNN) algorithm** runs a 3-phase loop per visit:
1. **Contextualization**: System prompt (with evolving summary h_{t-1}) + current visit → prediction
2. **Reflection**: evaluation model compares prediction to ground truth → evaluation feedback
3. **Memory Update**: LLM rewrites the evolving summary using feedback → h_t

**Three baselines** for comparison:
- `zero_shot` — current visit only, no history
- `full_history` — concatenate all prior visits (with their known diagnoses) as context
- `memprompt` — LLM-summarize each prior visit, concatenate summaries

**Shared infrastructure**: All methods produce `PredictionOutput` objects, which are evaluated by a shared `LLMJudge` and aggregated by `MetricsAggregator`.

### Key data flow

`data/mimic4_filtered_all_fields_{train,test}.json` → `MIMICDataLoader` (produces `PatientRecord` with `List[VisitData]`) → method runner → `List[PredictionOutput]` → `LLMJudge` → `EvaluationResult` → `MetricsAggregator` → `results/results.json`

For LLM-as-RNN, `RecurrentLanguageNetwork.forward()` also returns `prompt_states_per_visit`: a list where entry `i` is the `SystemPromptState` used to predict visit `i` (i.e., h_{t-1} captured *before* any memory update triggered by that visit). `run_experiment.py` forwards `evolving_summary` from each of those snapshots into `MetricsAggregator.add_result()` so the hidden-state trajectory ends up in `per_visit_detailed`.

### Four LLMs are used simultaneously

| Config key | Role | Phases |
|---|---|---|
| `prediction_model` | Generates diagnosis predictions; also handles `prompt_compression` if that path triggers | `prediction`, `prompt_compression` |
| `memory_model` | Rewrites the evolving summary during `prompt_update` | `memory_update` |
| `evaluation_model` | Compares prediction to ground truth on **intermediate** visits to produce feedback for memory update | `evaluation` (intermediate visits only) |
| `final_judge_model` **(required)** | Scores each patient's **last visit** for Acc@1 / Acc@5 — the only evaluation that counts toward aggregate metrics | `final_judge` |

For baselines (`zero_shot`, `full_history`, `memprompt`), intermediate-visit evaluations are done by `evaluation_model` when configured (for qualitative logging only). The last visit is always evaluated by `final_judge_model`. For LLM-as-RNN, the last-visit evaluation from `forward()` is **not** reused for scoring — `final_judge_model` re-evaluates it independently.

## Backends

The `backend` field in each model's config section selects the LLM backend. Five are supported; pick by *where the model runs* and *how often you re-run the experiment*.

- **`anthropic` / `openai` (hosted API)** — Zero setup, no GPU needed. Use for: initial development on Mac, small-scale runs, and the intermediate evaluation model. Requires `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. Cost scales with token volume.

- **`portkey` (Portkey AI gateway)** — Routes to hosted models (e.g., Claude, GPT) via the Portkey API. Requires `PORTKEY_API_KEY`. Currently used for `final_judge_model`. Includes built-in retry logic.

- **`vllm` (in-process, local GPU)** — `VLLMInterface` loads the model into the Python process via `from vllm import LLM`. Use for: **a single end-to-end benchmark run on a GPU machine**. Downside: the model reloads (~30s–2min) every time you invoke `run_experiment.py`, so it's wasteful during iteration. Set `tensor_parallel_size` to match available GPUs; tune `gpu_memory_utilization` and `max_model_len` for your hardware.

- **`openai` backend → vLLM OpenAI-compatible server (recommended for iteration)** — Start `vllm serve <model> --port 8000` once as a long-running process, then set `backend: openai`, `base_url: http://localhost:8000/v1`, `api_key: EMPTY` in the config. `OpenAIInterface` already supports custom `base_url`, so no code changes are needed. Model stays resident across runs — big speedup while tuning prompts.

- **`huggingface` (in-process, transformers)** — Fallback when vLLM isn't installable. Slower than vLLM; use only if vLLM cannot run on the target hardware.

**Platform note**: vLLM requires a CUDA GPU (Linux). On macOS/darwin, use hosted APIs or ssh to a GPU box. **Batching note**: `VLLMInterface.batch_generate()` exists but is not currently wired into the method runners in `baselines/` or `models/rln_core.py` — they call `generate()` one visit at a time, so vLLM's throughput advantage is partly unrealized. Batching across patients/visits would require refactoring the runners.

## Outputs

Running `run_experiment.py` produces **console output** and **eight files** under `<output_dir>/<run_id>/` where:
- `<output_dir>` is `data.output_dir` in the config (default `results/`)
- `<run_id>` is either `--run-name` (if provided) or `run_<12-char-sha>` computed from a config fingerprint

Files written (all append-only except `results.json`):
- `manifest.json` — run identity, fingerprint, config snapshot, creation time. Written once on first run.
- `results.json` — aggregate metrics + per-visit detail (full rewrite after every completed `(method, patient)` pair, so the file always reflects the latest state).
- `completed.jsonl` — append-only list of `(method, patient_id)` pairs that finished successfully. Used to skip work on resume.
- `per_visit_detailed.jsonl` — append-only, one row per visit, same schema as `results.json["per_visit_detailed"]`. Rewritten once on startup to drop orphan rows from crashed pairs.
- `calls_prediction.jsonl` — append-only, one row per prediction-model call.
- `calls_memory.jsonl` — append-only, one row per memory-model call (`prompt_update` / memory writing).
- `calls_judge.jsonl` — append-only, one row per intermediate-evaluation call. Historical filename kept for compatibility.
- `calls_final_judge.jsonl` — append-only, one row per final-judge-LLM call (last-visit scoring).

## Resume and run identity

**Fingerprint-based run id**: on startup, `run_experiment.py:compute_fingerprint()` hashes the scientifically-relevant config fields (`prediction_model`, `memory_model`, `evaluation_model`, `final_judge_model`, `rln`, `methods`, `templates`, `data.input_path`) and uses the first 12 hex chars as the default `run_id`. Any change to those fields produces a new `run_id` → a new output directory → no accidental mixing of experiments. Runtime-only fields (`output_dir`, GPU settings) are NOT in the fingerprint.

**Manifest check**: on every run, `run_experiment.py:resolve_run_dir()` compares the current fingerprint to the one stored in `<run_dir>/manifest.json`. Mismatch aborts with an actionable error unless `--force` is passed.

**Resume flow**:
1. Load `completed.jsonl` into a set of `(method, patient_id)` pairs that are fully done.
2. Rewrite `per_visit_detailed.jsonl` to keep only rows belonging to completed pairs (drops orphans from crashed runs).
3. Replay those rows into `MetricsAggregator.results` so the final aggregate includes prior work.
4. In the main loop, skip any `(method, patient_id)` already in the completed set.
5. After each pair finishes all its visits successfully, append to `completed.jsonl` and rewrite `results.json`.

**Human-readable runs**: pass `--run-name my_ablation_v2` to override the fingerprint-derived id. The manifest still stores the fingerprint, so fingerprint mismatch checking still applies. Use named runs when you want multiple iterations of the same experiment (e.g., temperature sweep) to live in separate directories.

**Crash semantics**:
- Mid-call crash → the call's record may not be written (we only log on successful return or handled exception). Partially-processed pair re-runs from scratch on resume.
- Mid-pair crash → orphan per-visit rows may be on disk without a completion marker. Startup cleanup drops them; pair re-runs from scratch.
- Between pairs → next run resumes cleanly at the next un-done pair.

**Console** — method/visit progress logs, an aggregate summary table (headlined "scored on last visit per patient"), and final LLM usage stats (`get_usage_stats()` for prediction, memory, intermediate evaluation, and final judge models). Usage stats are logged only, not persisted to disk.

**`results.json`** — four top-level keys:
- `overall` — per-method `{acc_at_1, acc_at_5, n_patients}`. Scored on each patient's **final visit only**. `n_patients` (not `n_visits`) is the denominator.
- `per_patient` — per patient, per method: `{acc_at_1, acc_at_5, last_visit_id}` where each acc is 0.0 or 1.0 — the last-visit outcome, not an average across that patient's visits.
- `per_visit` — lightweight row per *every* visit (not just last): `{method, patient_id, visit_id, primary_correct, any_top5_correct}`. Useful for plotting accuracy vs. visit index.
- `per_visit_detailed` — rich row per every visit, with fields for qualitative analysis:
  - `ground_truth` — diagnoses the intermediate evaluation model compared against
  - `prediction` — `{primary_diagnosis, top_5_diagnoses, raw_output}` from the prediction LLM
  - `evaluation` — `{diagnosis_evaluation, raw_output}` from the intermediate evaluation model, including its full explanation
  - `improvement_suggestions` — intermediate evaluation model's free-text guidance
  - `evolving_summary_before` — **LLM-as-RNN only**, the hidden state h_{t-1} used for this visit. `null` for baselines. Walking this field across a patient's visits gives the memory-evolution trace.

**`calls_prediction.jsonl` / `calls_memory.jsonl` / `calls_judge.jsonl`** — one JSON record per line, produced by the `LLMInterface` base-class logging wrapper (`_install_generate_logger()`). Every `generate()` call on any backend is captured automatically — no subclass changes needed to add a new backend.

Each record contains:
- Call data: `timestamp`, `model`, `latency_s`, `temperature`, `max_tokens`, full `system_prompt`, full `prompt`, full `response` (raw unparsed string), `error` (null on success).
- Context tags: `method`, `patient_id`, `visit_id`, `visit_index` (1-based), `total_visits`, `is_last_visit` (bool), `phase`.

The `is_last_visit` flag is the key for linking a call to the scored visit: `is_last_visit == true && phase == "prediction"` gives exactly the N records (one per patient) that fed `overall.acc_at_1` / `overall.acc_at_5`. The `phase` tag distinguishes call types:
- `prediction` — the main diagnosis call
- `evaluation` — RLN intermediate evaluation / reflection (lives in `calls_judge.jsonl`, which keeps its historical filename)
- `final_judge` — last-visit scoring evaluation (lives only in `calls_final_judge.jsonl`)
- `memory_update` — LLM-as-RNN only: evolving-summary rewrite (lives in `calls_memory.jsonl`)
- `prompt_compression` — optional prompt compression call, still uses the prediction-side model path in Stage 1
- `memprompt_summarize` — MemPrompt only: per-visit summarization

Visit tagging is set in `models/rln_core.py` (RLN forward pass) and in each baseline's `run_patient()` in `baselines/*.py`. To add new tags from application code, use `llm.set_call_context(**tags)` / `llm.clear_call_context()` — merges are per-key, so you can override one field without clobbering others.

**Known caveats**:
- Last-visit scoring in `MetricsAggregator._last_visit_per_patient()` relies on insertion order (visits appended chronologically). If the data loader ever returns visits out of order, it will silently score the wrong visit — sort by `VisitData.timestamp` if making the loader more flexible.
- The post-final-visit `SystemPromptState` (after the last memory update) is computed but not persisted anywhere.
- Running on macOS with `backend: vllm` will fail — use a hosted API or ssh to a GPU box (see Backends section).

## Important Paths

- `configs/experiment1_seperateJudge.yaml` — the single config file with four-model architecture, all hyperparameters, model settings, and **all prompt templates** (from paper Appendix C).
- `data/mimic4_filtered_all_fields.json` — full dataset: 7128 patients, 37536 visits (raw, contains some empty diagnoses).
- `data/cleaned_df.json` — **cleaned dataset**: 6488 patients, 33484 visits. Filtered by removing any patient with at least one visit where `discharge_diagnosis` is empty. Use this for experiments. Sample patient ID: `10001401`.
- `data/splits/cleaned_df_val_100.json` — 100 patients. Validation set, randomly sampled from `cleaned_df.json` (seed=42). Use for quick evaluation runs.
- `data/splits/cleaned_df_train_100.json` — 100 patients. Training set, randomly sampled from the remaining 6388 patients (seed=42).
- `data/splits/cleaned_df_test_6288.json` — 6288 patients. Test set, remaining patients after train split.
- `models/rln_core.py` — core algorithm + shared dataclasses (`VisitData`, `PredictionOutput`, `EvaluationResult`)
- `models/prompt_manager.py` — `SystemPromptState` management: init, update (the "recurrent step"), and compression
- `models/llm_interface.py` — LLM backend abstraction with `create_llm_interface()` factory. Also has `safe_json_parse()` / `extract_json_from_text()` used everywhere for parsing LLM JSON output.
- `baselines/visit_format.py` — shared helpers `format_current_visit_block()` / `format_prior_visit_block()` used by all three baseline runners to serialize `VisitData` into prompt text.
- `results/` — output directory for experiment results (created at runtime)

## Key Design Decisions

- **Prompt templates live in YAML, not code.** All templates (`system_prompt_init`, `prediction_prompt`, `evaluation_prompt`, `prompt_update`, `prompt_compression`, `zero_shot_prompt`, `history_prompt`, `visit_summary_prompt`) are in `configs/experiment1_seperateJudge.yaml`. The `PromptManager` reads them from a `templates` dict.
- **`prompt_update` and `prompt_compression` are intentionally split in Stage 1.** `prompt_update` always uses `memory_model`; `prompt_compression` remains on the prediction-side model path for now.
- **Baselines use dedicated prompt templates.** `zero_shot` uses `zero_shot_prompt`; `full_history` and `memprompt` both use `history_prompt`. These templates are distinct from the RLN's `prediction_prompt`/`system_prompt_init`. Baseline runners receive a `PatientRecord` (not `List[VisitData]`) so they can access patient-level metadata.
- **`baselines/visit_format.py` is the single serialization point** for `VisitData` → prompt text in baselines. `format_current_visit_block()` serializes the current visit (no ground truth); `format_prior_visit_block()` serializes prior visits including their ground-truth diagnoses. Do not inline visit serialization elsewhere.
- **Prediction JSON has no `confidence` fields.** Both the `prediction_prompt` and baseline prompt templates ask for `{"name": "..."}` objects only — no `confidence` key. Parsers downstream do not expect it.
- **Discharge diagnosis parsing is non-trivial.** The `MIMICDataLoader._extract_ground_truth()` in `data_loader.py` handles varied MIMIC formats (simple strings, multi-line with labels, parenthesized expressions spanning lines). Changes here affect all evaluation.
- **The `VisitData` dataclass expects structured fields** (`vital_signs: Dict`, `lab_results: Dict`) but MIMIC data embeds these as free text in `pertinent_results`. The loader passes empty dicts for vitals/labs and relies on `pertinent_results` carrying the information.
- **LLM-as-RNN reuses its internal evaluations for intermediate visits only.** During `RecurrentLanguageNetwork.forward()`, evaluations are generated for the memory update loop and reused for intermediate-visit logging. The **last visit is always re-evaluated** by `final_judge_model` in `run_experiment.py`'s eval loop — the RLN's own last-visit evaluation (from `evaluation_model`) is discarded for scoring purposes.
- **Scoring is last-visit-only; execution is full-sequence.** `MetricsAggregator.compute_metrics()` only aggregates each patient's final visit, but all four methods still *run* on every visit — LLM-as-RNN needs the sequence to build up its hidden state, and the baselines need the visits as context/summarization input. Do not "optimize" by skipping earlier visits; you'll break the time-series setup.

## Coding Standards

- Use type hints on all function signatures
- Include docstrings for classes and public methods
- Use pytest-style tests
- Follow existing patterns: dataclasses for data structures, `safe_json_parse()` for all LLM output parsing, `LLMInterface` abstraction for any new backend

## Training Pipeline (`training/`)

This repo has TWO RL training stacks. Knowing which is which avoids confusion.

| Stack | Location | Status | Use it when |
|---|---|---|---|
| **Hand-rolled GRPO** | `training/collect/`, `training/reward/`, `training/train/`, `training/scripts/` | **Reference / legacy** — kept on `feat/grpo-training` | Reading code to understand the algorithm; sanity-checking verl's numbers |
| **verl integration** | `training/verl/` (submodule) + `training/verl_adapter/` | **Active path** — on `feat/grpo_verl` | Real training on Lonestar6 (multi-GPU, FSDP, on-policy LoRA sync) |

We migrated to verl for: FSDP/Megatron sharding (needed for 7B+), on-policy LoRA sync (verl auto-syncs trained LoRA back into vLLM rollout worker — fixes our Phase 1.0 off-policy gap), Ray-based multi-node, and verl's tested vllm/torch/flash-attn version pinning.

### Branch organization

| Branch | Purpose | What's on it |
|---|---|---|
| `main` | Inference baselines (paper reproduction) | 4 methods + eval pipeline; **no training code** |
| `feat/grpo-training` | Hand-rolled GRPO trainer (legacy) | Inference + custom GRPO loss / collect / score / artifact_writer |
| `feat/grpo_verl` | verl-based training (current focus) | Inference + verl submodule + verl_adapter (Phase A/B/C/D scaffold) |

All three branches share the **4-model architecture**. Use short-lived `feat/<short-name>` branches for new features and merge to `feat/grpo_verl` when ready. Don't keep multi-month feature branches.

### verl Integration (`training/verl_adapter/`)

The adapter layer sits OUTSIDE verl (which is a pinned submodule at `release/v0.7.1` so we never modify upstream). Phased delivery:

| Phase | Goal | Status (2026-04-26) |
|---|---|---|
| **A** | Run verl's stock GSM8K demo on Lonestar6 — proves env works | ✅ passed (reward variance + bounded KL + LoRA on-policy sync working) |
| **B** | `TrajectoryStep` → verl-compatible parquet | ✅ scaffold + `experiment_smoke.yaml`; smoke verified |
| **C** | Custom reward function (rubric judge) | ✅ scaffold landed — verl-compatible `compute_score` in `training/verl_adapter/rubric_reward.py`. Rubric YAML is single source of truth (rubric content + judge model + serve config all in one file); `start_judge_server.sh` and reward function read from the same YAML so they can't drift. **Audit logging**: every reward call writes a JSONL row (rollout text + judge prompt + judge response + reward) under `$SCRATCH/llm_as_rnn/rubric_audit/calls_<pid>.jsonl` — one file per worker PID. 15/15 sanity tests pass on Mac (mock judge + audit + parser-strict-mode regression). |
| **D** | Single-node GRPO training launcher (multi-node deferred) | ✅ launcher landed — `training/verl_adapter/apptainer/run_phase1.sh`. Self-contained: starts judge server in background, waits for endpoint, launches verl GRPO with our reward function, cleans up on exit. **All GRPO knobs (LR, KL, G, batch, LoRA rank, epochs, etc.) at the TOP of the file** (verl-style). Single-node 3-GPU layout: cuda:0,1 → actor+rollout (TP=2), cuda:2 → judge. Multi-node Ray cluster setup deferred. |

### Lonestar6 setup (TACC)

**Project**: `ASC25003`. **System**: Lonestar6 with 3× A100 40GB per node. **Storage**: `$WORK` for code + Apptainer images, `$SCRATCH` for data + HF cache, **never `$HOME`** (10 GB quota).

**Important TACC quirks** (learned the hard way during Phase A):
- Apptainer is **blocked on login nodes** (the wrapper exits 0 silently, looks like success but creates no file). All `apptainer` commands must run inside `idev` or `sbatch`.
- `module load tacc-apptainer` must be re-run on every fresh idev session (modules don't persist across sessions).
- vLLM model loading on cuda:0 by default — if multiple vLLM instances are needed, each model in YAML must explicitly set `cuda_visible_devices: "<idx>"` to avoid OOM.

**Install path**: Apptainer (verl's official Docker image — torch 2.10 + vllm 0.17 + flash-attn pre-built and version-matched). One-time `apptainer pull`, then ~1s startup per run. (We tried conda first, hit 3 different version-mismatch breakages, removed the conda scaffolding 2026-04-27.)

### Quick-start: running on Lonestar6 from scratch

```bash
# === ssh to login node, get the code (do once) ===
ssh <user>@ls6.tacc.utexas.edu
cd $WORK
git clone --recursive https://github.com/<user>/LLMasRNN.git    # --recursive pulls verl submodule
cd LLMasRNN
git checkout feat/grpo_verl                                     # or main for inference

# === every working session: get a compute node ===
idev -p gpu-a100-dev -N 1 -n 1 --gpus-per-node=3 -t 02:00:00 -A ASC25003
module load tacc-apptainer                                       # required every fresh idev

# === one-time on this account: pull verl image (~10-20 min, ~15 GB) ===
bash training/verl_adapter/apptainer/pull_image.sh
echo 'export VERL_SIF=$WORK/apptainer_images/verlai_verl_vllm017.latest.sif' >> ~/.bashrc
source ~/.bashrc

# === Phase B: dump our trajectories to parquet ===
bash training/verl_adapter/apptainer/dump_trajectories.sh \
    --config configs/experiment_smoke.yaml \
    --split train \
    --max-patients 5
# Output: $SCRATCH/data/llm_as_rnn/train.parquet

# === Phase C: start judge server + smoke reward function ===
# Edit training/configs/rubric_v1.yaml to set judge.model first.
bash training/verl_adapter/apptainer/start_judge_server.sh                 # tmux pane 1
apptainer exec --nv \
    --bind /work:/work --bind /scratch:/scratch \
    --env PYTHONPATH=$WORK/LLMasRNN:$WORK/LLMasRNN/training/verl \
    $VERL_SIF \
    python -m training.verl_adapter.smoke_phase_c \
        --parquet $SCRATCH/data/llm_as_rnn/train.parquet --n 3              # tmux pane 2
```

(Phase A smoke scripts have been removed — Phase A is verified. To re-verify on a
new system, follow verl's own example at
`training/verl/examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh`.)

### Smoke vs full configs

- `configs/experiment1_seperateJudge.yaml` — full settings; vLLM `gpu_memory_utilization: 0.9`, `max_model_len: 8192`. For real inference + final paper runs.
- `configs/experiment_smoke.yaml` — Phase B/C smoke; `gpu_memory_utilization: 0.4`, `max_model_len: 4096`. References experiment1's templates via `templates_config:` (no duplication). Default `data.input_path = train_100`, so the dumper just works without `--data-path`.

### Phase C smoke lesson (2026-04-29) — judge format compliance is everything

First Phase C smoke used `OpenRubrics/RubricARM-8B-Judge` as a placeholder
judge with our scalar prompt template (asks for a single 1-9 digit). All 3
"good" candidates got reward = 0.2222 (= 2/9), perfectly consistent. Looked
like a healthy result.

**Audit log saved us**: dumping the `judge_response` field showed RubricARM
was outputting verbose CoT analysis ("1. Factual faithfulness... 2. Compression
efficiency...") that got truncated mid-dimension by `max_tokens=64`. Our
parser was using `re.findall(r"[1-9]")` and grabbing the LAST [1-9] digit —
which happened to be the dimension number "2" in "2. Compression efficiency".
The reward 0.2222 was a **coincidence**, not a real quality signal.

**Two fixes pushed**:
  1. Parser now requires "Score: N" pattern as primary signal; falls back to
     short-text-only digit; otherwise returns `JUDGE_NO_DIGIT_REWARD = 0.5`.
     **Refuses to extract digits from long verbose text** — no more false
     positives on dimension numbers.
  2. `judge.max_tokens` bumped 64 → 1500 in `rubric_v1.yaml` so CoT-style
     judges can finish; prompt template updated to instruct "Score: N" on
     final line.

**Takeaways for SFT judge prep**:
  - SFT data MUST end with `\nScore: N\n` — same format as the prompt
    template, otherwise the parser falls back to 0.5.
  - Parser now PREFERS strict format over loose digit-grabbing; this is the
    safer default for production.
  - Always read audit logs before trusting reward numbers — a "consistent"
    reward can hide a parsing bug.
  - RubricARM-8B is a PAIRWISE judge (Response A vs B); won't ever output
    a clean digit even with infinite tokens. Don't try to "fix" it; SFT a
    proper scalar judge instead.

### Open decisions (will block when relevant)

- **Reward model for Phase C**: candidates are scalar local SFT judge (simplest) vs `OpenRubrics/RubricARM-8B-Judge` (pairwise — would need a tournament adapter to convert pairwise wins into a scalar reward, but eliminates the need to train our own judge). Decision deferred until Phase B parquets are validated and we eyeball reward distribution on a few candidates.
- **SFT warm-up before GRPO**: open question. If first GRPO epoch shows mass JSON-parse failures from the memory-update head, we'd need an SFT warm-up pass before GRPO. Current plan: try direct GRPO first, fall back to SFT-then-GRPO if reward variance collapses.
- **`training/memory_grpo/`** (legacy scaffold from a pre-verl GRPO attempt) is still on `feat/grpo_verl` from the 4-model commit. Superseded by `training/verl_adapter/`; safe to delete in a cleanup pass.

### Important verl-side paths

- `training/verl/` — verl submodule, pinned at `release/v0.7.1`. **Never modified.** Update by bumping the submodule pointer (`cd training/verl && git checkout <new-tag> && cd .. && git add training/verl && git commit`).
- `training/verl_adapter/dump_trajectories_to_parquet.py` — Phase B dumper: runs the 4-model RLN forward, writes verl-compatible parquet (chat-format prompt + serialized `TrajectoryStep` as `reward_model.ground_truth`).
- `training/verl_adapter/apptainer/_apptainer_common.sh` — sourced by every Apptainer wrapper; sets `VERL_SIF`, `BIND_ARGS` (binds `/work` and `/scratch`), `ENV_ARGS` (PYTHONPATH includes repo root + verl submodule, HF_HOME under `$SCRATCH`).
- `training/types.py` + `training/collect/collect_trajectories.py` — shared `TrajectoryStep` dataclass + the function that runs RLN forward and emits steps. Used by both the hand-rolled trainer (legacy) and the verl Phase B dumper.

### Hand-rolled GRPO trainer (legacy, on `feat/grpo-training`)

Reference implementation. ~50 LOC custom GRPO loss in `training/train/grpo_loss.py`: group-relative advantage (no critic) + PPO-clipped surrogate + KL anchor to frozen base via `model.disable_adapter()` (no second 3B copy in memory). One-epoch driver: `bash training/scripts/epoch.sh --config training/configs/grpo_phase1.yaml --epoch N --resume-from latest`. Per-epoch persistence: `policy_lora/`, `optimizer.pt`, `meta.json`, `train_metrics.jsonl`, plus full per-call audit JSONLs. Designed for single-GPU 3B + LoRA. Will not scale to 7B+ — that's why we moved to verl.
