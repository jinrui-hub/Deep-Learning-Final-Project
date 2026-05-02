"""
LLM-as-RNN: Main Experiment Runner
Reproduces MIMIC-IV discharge diagnosis prediction experiments.

Usage:
    python run_experiment.py --config configs/experiment.yaml
    python run_experiment.py --config configs/experiment.yaml --methods zero_shot,llm_as_rnn
    python run_experiment.py --config configs/experiment.yaml --patients 10001401,10004235
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import yaml

from data_loader import MIMICDataLoader, PatientRecord
from models.llm_interface import create_llm_interface, LLMInterface
from models.prompt_manager import PromptManager
from models.rln_core import RecurrentLanguageNetwork, VisitData, PredictionOutput
from baselines.zero_shot import ZeroShotBaseline
from baselines.full_history import FullHistoryBaseline
from baselines.memprompt import MemPromptBaseline
from evaluation.llm_judge import LLMJudge
from evaluation.metrics import MetricsAggregator


# Config fields that define run identity. Any change here = new fingerprint =
# fresh run directory (unless the user passes --run-name or --force).
FINGERPRINT_FIELDS = [
    "prediction_model",
    "memory_model",
    "evaluation_model",
    "final_judge_model",
    "rln",
    "methods",
    "templates",
]


def compute_fingerprint(config: dict) -> tuple[str, dict]:
    """Hash the scientifically-relevant config fields. Returns (hex_digest, payload)."""
    payload = {field: config.get(field) for field in FINGERPRINT_FIELDS}
    payload["data_input_path"] = config.get("data", {}).get("input_path")
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:12]
    return digest, payload


def resolve_run_dir(config: dict, args) -> tuple[str, str, dict]:
    """Compute the run_id, make the run directory, and write/verify its manifest.

    Returns (run_dir, run_id, fingerprint_payload). Aborts with a clear message
    on a fingerprint mismatch unless --force was passed.
    """
    fingerprint, payload = compute_fingerprint(config)
    run_id = args.run_name or f"run_{fingerprint}"
    base_dir = config.get("data", {}).get("output_dir", "results/")
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            existing = json.load(f)
        if existing.get("fingerprint") != fingerprint:
            if not args.force:
                logger.error(
                    f"Fingerprint mismatch in {run_dir}:\n"
                    f"  existing fingerprint: {existing.get('fingerprint')}\n"
                    f"  current fingerprint:  {fingerprint}\n"
                    f"The config at {args.config} does not match the prior run stored "
                    f"in this directory. Refusing to resume to avoid mixing experiments.\n"
                    f"Options:\n"
                    f"  1. Pass --run-name <new_name> to start a fresh run.\n"
                    f"  2. Pass --force to overwrite the manifest (keeps existing files — risky).\n"
                    f"  3. Revert the config change."
                )
                sys.exit(1)
            logger.warning(f"--force: overwriting manifest fingerprint in {run_dir}")
            existing["fingerprint"] = fingerprint
            existing["config_snapshot"] = payload
            existing["forced_at"] = time.time()
            with open(manifest_path, "w") as f:
                json.dump(existing, f, indent=2, default=str)
        else:
            logger.info(f"Resuming run {run_id} (fingerprint {fingerprint} matches)")
    else:
        manifest = {
            "run_id": run_id,
            "fingerprint": fingerprint,
            "config_snapshot": payload,
            "created_at": time.time(),
            "config_path": os.path.abspath(args.config),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info(f"Started new run {run_id} (fingerprint {fingerprint}) in {run_dir}")

    return run_dir, run_id, payload


def load_completed_pairs(run_dir: str) -> set:
    """Load the set of (method, patient_id) pairs already fully processed."""
    path = os.path.join(run_dir, "completed.jsonl")
    if not os.path.exists(path):
        return set()
    pairs = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pairs.add((row["method"], row["patient_id"]))
    return pairs


def mark_pair_complete(run_dir: str, method: str, patient_id: str, n_visits: int) -> None:
    """Append a completion marker for (method, patient_id)."""
    path = os.path.join(run_dir, "completed.jsonl")
    with open(path, "a") as f:
        f.write(
            json.dumps(
                {
                    "method": method,
                    "patient_id": patient_id,
                    "n_visits": n_visits,
                    "timestamp": time.time(),
                }
            )
            + "\n"
        )
        f.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_config(config: dict, methods: list[str]) -> None:
    """Validate renamed model fields and llm_as_rnn requirements."""
    if "judge_model" in config:
        logger.error(
            "Config key 'judge_model' has been renamed to 'evaluation_model'. "
            "Update the YAML and rerun."
        )
        sys.exit(1)

    if "llm_as_rnn" not in methods:
        return

    if not config.get("evaluation_model"):
        logger.error(
            "llm_as_rnn requires 'evaluation_model' in config. "
            "Intermediate feedback is now configured via evaluation_model."
        )
        sys.exit(1)

    if not config.get("memory_model"):
        logger.error(
            "llm_as_rnn requires 'memory_model' in config. "
            "Memory updates now run through a dedicated memory_model."
        )
        sys.exit(1)


def run_zero_shot(
    patient: PatientRecord,
    prediction_llm: LLMInterface,
    templates: dict,
) -> list[PredictionOutput]:
    """Run zero-shot baseline for a patient."""
    baseline = ZeroShotBaseline(prediction_llm, templates)
    return baseline.run_patient(patient)


def run_full_history(
    patient: PatientRecord,
    prediction_llm: LLMInterface,
    templates: dict,
    token_budget: int,
) -> list[PredictionOutput]:
    """Run Full History Concatenation baseline for a patient."""
    baseline = FullHistoryBaseline(prediction_llm, templates, token_budget)
    return baseline.run_patient(patient)


def run_memprompt(
    patient: PatientRecord,
    prediction_llm: LLMInterface,
    templates: dict,
    token_budget: int,
) -> list[PredictionOutput]:
    """Run MemPrompt baseline for a patient."""
    baseline = MemPromptBaseline(prediction_llm, templates, token_budget)
    return baseline.run_patient(patient)


def run_llm_as_rnn(
    patient: PatientRecord,
    prediction_llm: LLMInterface,
    memory_llm: LLMInterface,
    evaluation_llm: LLMInterface,
    templates: dict,
    rln_config: dict,
) -> tuple[list[PredictionOutput], list, list]:
    """Run LLM-as-RNN method for a patient.

    Returns (predictions, evaluations, prompt_states_per_visit) where
    prompt_states_per_visit[i] is the hidden state h_{t-1} used for visit i.
    """
    prompt_manager = PromptManager(
        templates=templates,
        memory_llm=memory_llm,
        compression_llm=prediction_llm,
        max_prompt_length=rln_config.get("max_prompt_length", 4096),
        compress_every_n=rln_config.get("compress_every_n", 5),
    )
    rln = RecurrentLanguageNetwork(
        prediction_llm=prediction_llm,
        evaluation_llm=evaluation_llm,
        memory_llm=memory_llm,
        prompt_manager=prompt_manager,
        config=rln_config,
    )

    initial_state = prompt_manager.initialize_prompt(
        patient_id=patient.subject_id,
        age="Unknown",
        gender=patient.sex,
    )

    predictions, evaluations, prompt_states, _final_state = rln.forward(
        patient_visits=patient.visits,
        initial_prompt_state=initial_state,
        evaluate_intermediate=True,
    )

    return predictions, evaluations, prompt_states


def main():
    parser = argparse.ArgumentParser(description="LLM-as-RNN MIMIC-IV Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods to run (e.g., zero_shot,llm_as_rnn)",
    )
    parser.add_argument(
        "--patients",
        type=str,
        default=None,
        help="Comma-separated patient IDs to process (default: all)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Human-readable run id. If omitted, a fingerprint-derived id is used.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow resuming into a run directory whose stored config fingerprint "
             "differs from the current config. Risky; see error message for details.",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    templates = config["templates"]
    rln_config = config.get("rln", {})
    token_budget = rln_config.get("token_budget", 4096)

    # Determine which methods to run
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = config.get("methods", ["zero_shot", "full_history", "memprompt", "llm_as_rnn"])

    logger.info(f"Methods to run: {methods}")
    validate_config(config, methods)
    use_rln = "llm_as_rnn" in methods

    # Resolve run directory (handles fingerprint check + manifest).
    run_dir, run_id, _ = resolve_run_dir(config, args)

    # Initialize LLMs
    pred_config = config["prediction_model"]
    prediction_llm = create_llm_interface(
        backend=pred_config["backend"],
        model_name=pred_config["model_name"],
        config=pred_config,
    )
    logger.info(f"Prediction LLM: {pred_config['model_name']} ({pred_config['backend']})")

    evaluation_llm = None
    evaluation_config = config.get("evaluation_model", {})
    if evaluation_config:
        evaluation_llm = create_llm_interface(
            backend=evaluation_config["backend"],
            model_name=evaluation_config["model_name"],
            config=evaluation_config,
        )
        logger.info(
            "Evaluation LLM: %s (%s)",
            evaluation_config["model_name"],
            evaluation_config["backend"],
        )

    memory_llm = None
    if use_rln:
        memory_config = config["memory_model"]
        memory_llm = create_llm_interface(
            backend=memory_config["backend"],
            model_name=memory_config["model_name"],
            config=memory_config,
        )
        logger.info(
            "Memory LLM: %s (%s)",
            memory_config["model_name"],
            memory_config["backend"],
        )

    final_judge_config = config.get("final_judge_model")
    if not final_judge_config:
        logger.error(
            "final_judge_model is required in config. This model scores "
            "Acc@1/Acc@5 on each patient's last visit."
        )
        sys.exit(1)
    final_judge_llm = create_llm_interface(
        backend=final_judge_config["backend"],
        model_name=final_judge_config["model_name"],
        config=final_judge_config,
    )
    logger.info(f"Final Judge LLM: {final_judge_config['model_name']} ({final_judge_config['backend']})")

    # Enable append-only streaming for crash-safe logging.
    prediction_llm.start_call_log_stream(os.path.join(run_dir, "calls_prediction.jsonl"))
    if evaluation_llm is not None:
        evaluation_llm.start_call_log_stream(os.path.join(run_dir, "calls_judge.jsonl"))
    if memory_llm is not None:
        memory_llm.start_call_log_stream(os.path.join(run_dir, "calls_memory.jsonl"))
    final_judge_llm.start_call_log_stream(os.path.join(run_dir, "calls_final_judge.jsonl"))

    # Load data
    data_path = config["data"]["input_path"]
    loader = MIMICDataLoader(data_path)
    patients = loader.load_patients()

    # Filter patients if specified
    if args.patients:
        patient_ids = set(p.strip() for p in args.patients.split(","))
        patients = [p for p in patients if p.subject_id in patient_ids]
        logger.info(f"Filtered to {len(patients)} patients: {[p.subject_id for p in patients]}")

    # Initialize evaluation
    evaluation = (
        LLMJudge(evaluation_llm, templates["evaluation_prompt"])
        if evaluation_llm
        else None
    )
    final_judge = LLMJudge(final_judge_llm, templates["evaluation_prompt"])
    aggregator = MetricsAggregator()

    # Resume: load completed pairs, clean orphan per-visit rows, replay the rest.
    completed_pairs = load_completed_pairs(run_dir)
    per_visit_path = os.path.join(run_dir, "per_visit_detailed.jsonl")
    if completed_pairs:
        kept = aggregator.rewrite_stream_file(per_visit_path, completed_pairs)
        replayed = aggregator.load_from_stream(per_visit_path)
        logger.info(
            f"Resume: {len(completed_pairs)} (method, patient) pairs already done; "
            f"replayed {replayed} per-visit rows from {per_visit_path} "
            f"(kept {kept}, dropped any orphans)"
        )
    aggregator.enable_streaming(per_visit_path)

    # Run experiments
    for patient_idx, patient in enumerate(patients):
        logger.info(
            f"\n{'='*60}\n"
            f"Patient {patient_idx+1}/{len(patients)}: {patient.subject_id} "
            f"({len(patient.visits)} visits)\n"
            f"{'='*60}"
        )

        for method in methods:
            if (method, patient.subject_id) in completed_pairs:
                logger.info(
                    f"SKIP {method} / {patient.subject_id} (already completed in {run_id})"
                )
                continue

            logger.info(f"\n--- Running {method} for patient {patient.subject_id} ---")

            # Tag subsequent LLM calls so the JSONL audit trail is searchable.
            prediction_llm.clear_call_context()
            prediction_llm.set_call_context(
                method=method, patient_id=patient.subject_id, phase="prediction"
            )
            if evaluation_llm is not None:
                evaluation_llm.clear_call_context()
                evaluation_llm.set_call_context(
                    method=method, patient_id=patient.subject_id, phase="evaluation"
                )
            if memory_llm is not None:
                memory_llm.clear_call_context()
                memory_llm.set_call_context(
                    method=method, patient_id=patient.subject_id, phase="memory_update"
                )

            predictions = []
            rln_evaluations = None
            rln_prompt_states = None

            if method == "zero_shot":
                predictions = run_zero_shot(patient, prediction_llm, templates)

            elif method == "full_history":
                predictions = run_full_history(patient, prediction_llm, templates, token_budget)

            elif method == "memprompt":
                predictions = run_memprompt(patient, prediction_llm, templates, token_budget)

            elif method == "llm_as_rnn":
                if evaluation_llm is None or memory_llm is None:
                    logger.error(
                        "LLM-as-RNN requires both evaluation_model and memory_model. "
                        "Skipping."
                    )
                    continue
                predictions, rln_evaluations, rln_prompt_states = run_llm_as_rnn(
                    patient,
                    prediction_llm,
                    memory_llm,
                    evaluation_llm,
                    templates,
                    rln_config,
                )
            else:
                logger.warning(f"Unknown method: {method}, skipping.")
                continue

            # Evaluate predictions
            for pred_idx, (pred, visit) in enumerate(zip(predictions, patient.visits)):
                is_last = pred_idx == len(patient.visits) - 1

                if is_last:
                    # Last visit scored by the dedicated final judge model.
                    final_judge_llm.set_call_context(
                        method=method,
                        patient_id=patient.subject_id,
                        visit_id=visit.visit_id,
                        visit_index=pred_idx + 1,
                        total_visits=len(patient.visits),
                        is_last_visit=True,
                        phase="final_judge",
                    )
                    eval_result = final_judge.evaluate(pred, visit.ground_truth_diagnoses)
                elif method == "llm_as_rnn" and rln_evaluations and pred_idx < len(rln_evaluations):
                    # For LLM-as-RNN, reuse evaluations from the forward pass
                    eval_result = rln_evaluations[pred_idx]
                elif evaluation is not None:
                    eval_result = evaluation.evaluate(pred, visit.ground_truth_diagnoses)
                else:
                    logger.warning(
                        "No evaluation_model available, skipping intermediate evaluation "
                        "for %s",
                        visit.visit_id,
                    )
                    continue

                evolving_summary_before = None
                if method == "llm_as_rnn" and rln_prompt_states and pred_idx < len(rln_prompt_states):
                    evolving_summary_before = rln_prompt_states[pred_idx].evolving_summary

                aggregator.add_result(
                    method=method,
                    patient_id=patient.subject_id,
                    visit_id=visit.visit_id,
                    eval_result=eval_result,
                    prediction=pred,
                    ground_truth=visit.ground_truth_diagnoses,
                    evolving_summary_before=evolving_summary_before,
                )

            # Pair completed successfully — mark it done and checkpoint results.
            mark_pair_complete(run_dir, method, patient.subject_id, len(patient.visits))
            completed_pairs.add((method, patient.subject_id))
            aggregator.save_results(run_dir)

    # Print final summary. Files were all written incrementally.
    aggregator.print_summary()
    aggregator.save_results(run_dir)
    aggregator.close_stream()
    prediction_llm.close_call_log_stream()
    if evaluation_llm is not None:
        evaluation_llm.close_call_log_stream()
    if memory_llm is not None:
        memory_llm.close_call_log_stream()
    final_judge_llm.close_call_log_stream()

    # Print usage stats
    pred_usage = prediction_llm.get_usage_stats()
    logger.info(f"\nPrediction LLM usage: {pred_usage}")
    if evaluation_llm:
        evaluation_usage = evaluation_llm.get_usage_stats()
        logger.info(f"Evaluation LLM usage: {evaluation_usage}")
    if memory_llm:
        memory_usage = memory_llm.get_usage_stats()
        logger.info(f"Memory LLM usage: {memory_usage}")
    final_judge_usage = final_judge_llm.get_usage_stats()
    logger.info(f"Final Judge LLM usage: {final_judge_usage}")


if __name__ == "__main__":
    main()
