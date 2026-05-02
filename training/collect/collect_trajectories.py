"""
Step 1 of the GRPO epoch loop: trajectory collection.

Runs the standard `RecurrentLanguageNetwork.forward()` over the training split,
then materializes one `TrajectoryStep` per (patient, visit) pair where a memory
update happened AND a downstream visit exists for scoring (i.e. visit indices
0 .. n-2 for an n-visit patient).

Stays in memory; the periodic checkpoint writer handles disk persistence.
"""

import json
import logging
from typing import Dict, List, Optional

from data_loader import MIMICDataLoader, PatientRecord
from models.prompt_manager import PromptManager
from models.rln_core import RecurrentLanguageNetwork

from training.types import TrajectoryStep

logger = logging.getLogger(__name__)


def _patient_history_at(
    visit_index: int,
    patient_id: str,
    visits,
    prompt_manager: PromptManager,
) -> str:
    """Reconstruct the patient_history string as it would have looked at visit i's
    memory-update step.

    Mirrors what `PromptManager.update_prompt` does internally: it calls
    `_update_patient_history` for visit i+1 BEFORE building the history summary,
    so visits 0..i are folded into the summary at the i-th update.
    """
    saved = prompt_manager.patient_diagnoses_history.get(patient_id)
    try:
        prompt_manager.patient_diagnoses_history[patient_id] = []
        for k in range(visit_index + 1):
            prompt_manager._update_patient_history(patient_id, k + 1, visits[k])
        return prompt_manager._build_history_summary(patient_id)
    finally:
        if saved is None:
            prompt_manager.patient_diagnoses_history.pop(patient_id, None)
        else:
            prompt_manager.patient_diagnoses_history[patient_id] = saved


def _build_trajectory_steps_for_patient(
    patient: PatientRecord,
    rln: RecurrentLanguageNetwork,
    prompt_manager: PromptManager,
    predictions,
    evaluations,
    prompt_states,
) -> List[TrajectoryStep]:
    """Assemble TrajectoryStep records for one patient after `forward()` returns.

    A step is emitted only when (a) visit i triggered a memory update — i.e.
    i < n-1 — and (b) we have a recorded evaluation e_i. Forward emits one
    evaluation per intermediate visit, so they line up: evaluations[i] is e_i.
    """
    visits = patient.visits
    n = len(visits)
    if n < 2:
        return []

    steps: List[TrajectoryStep] = []
    for i in range(n - 1):
        if i >= len(evaluations):
            break  # safety: forward bailed early on this patient
        h_prev = prompt_states[i]
        h_curr = prompt_states[i + 1]

        visit_summary = rln._create_visit_summary(visits[i], predictions[i], evaluations[i])
        patient_history = _patient_history_at(i, patient.subject_id, visits, prompt_manager)

        update_prompt = prompt_manager.templates["prompt_update"].format(
            current_evolving_summary=h_prev.evolving_summary,
            evaluation_feedback=json.dumps(rln._evaluation_to_dict(evaluations[i]), indent=2),
            visit_summary=visit_summary,
            patient_history=patient_history,
        )

        steps.append(
            TrajectoryStep(
                patient_id=patient.subject_id,
                visit_index=i,
                update_prompt=update_prompt,
                h_prev_evolving=h_prev.evolving_summary,
                h_prev_full_prompt=h_prev.full_prompt,
                h_curr_evolving_ref=h_curr.evolving_summary,
                visit=visits[i],
                next_visit=visits[i + 1],
                prediction=predictions[i],
                evaluation=evaluations[i],
                age="Unknown",
                gender=patient.sex,
                primary_conditions="Unknown",
            )
        )
    return steps


def collect_trajectories(
    data_path: str,
    rln: RecurrentLanguageNetwork,
    prompt_manager: PromptManager,
    patient_filter: Optional[List[str]] = None,
    max_patients: Optional[int] = None,
) -> List[TrajectoryStep]:
    """Run RLN forward over patients and materialize trajectory steps in memory.

    Args:
        data_path: JSON file with MIMIC patients (e.g. cleaned_df_train_100.json).
        rln: RecurrentLanguageNetwork wired with the CURRENT policy as its
             memory-update LLM and the predictor as its prediction LLM.
        prompt_manager: PromptManager shared with `rln`.
        patient_filter: optional whitelist of subject_ids.
        max_patients: hard cap on number of patients processed this epoch.

    Returns:
        Flat list of TrajectoryStep across all patients (one entry per
        memory-update event).
    """
    loader = MIMICDataLoader(data_path)
    patients = loader.load_patients()
    if patient_filter is not None:
        wanted = set(patient_filter)
        patients = [p for p in patients if p.subject_id in wanted]
    if max_patients is not None:
        patients = patients[:max_patients]

    all_steps: List[TrajectoryStep] = []
    for p_idx, patient in enumerate(patients, start=1):
        logger.info(
            f"[collect] patient {p_idx}/{len(patients)} id={patient.subject_id} "
            f"visits={len(patient.visits)}"
        )

        initial_state = prompt_manager.initialize_prompt(
            patient_id=patient.subject_id,
            age="Unknown",
            gender=patient.sex,
        )
        try:
            predictions, evaluations, prompt_states, _final = rln.forward(
                patient_visits=patient.visits,
                initial_prompt_state=initial_state,
                evaluate_intermediate=True,
            )
        except Exception as e:
            logger.error(f"[collect] forward failed for patient {patient.subject_id}: {e}")
            continue

        steps = _build_trajectory_steps_for_patient(
            patient, rln, prompt_manager, predictions, evaluations, prompt_states
        )
        all_steps.extend(steps)
        logger.info(f"[collect]   -> {len(steps)} trajectory steps")

    logger.info(f"[collect] total trajectory steps: {len(all_steps)}")
    return all_steps
