"""
Recurrent Language Network (RLN) Core Module
Main implementation of the RLN algorithm
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json

from models.llm_interface import LLMInterface, safe_json_parse
from models.prompt_manager import PromptManager, SystemPromptState

logger = logging.getLogger(__name__)


@dataclass
class VisitData:
    """Represents a single patient visit"""

    visit_id: str
    timestamp: str
    chief_complaint: str
    vital_signs: Dict[str, float]
    lab_results: Dict[str, float]
    medications: List[str]
    procedures: List[str]
    ground_truth_diagnoses: List[str]
    ground_truth_treatments: Optional[List[str]] = None
    history_present_illness: Optional[str] = None
    past_medical_history: Optional[str] = None
    social_history: Optional[str] = None
    family_history: Optional[str] = None
    physical_exam: Optional[str] = None
    pertinent_results: Optional[str] = None
    hospital_course: Optional[str] = None
    allergies: Optional[str] = None
    admission_medications: Optional[List[str]] = None


@dataclass
class PredictionOutput:
    """Represents model prediction for a visit"""

    visit_id: str
    top_5_diagnoses: List[Dict[str, Any]]
    primary_diagnosis: Dict[str, Any]
    raw_output: str 


@dataclass
class EvaluationResult:
    """Represents evaluation of a prediction"""

    visit_id: str
    diagnosis_evaluation: Dict[str, Any]
    raw_output: str = ""
    improvement_suggestions: str = ""


class RecurrentLanguageNetwork:
    """
    Recurrent Language Network for clinical prediction

    The RLN processes sequential patient visits, maintaining a hidden state
    (system prompt) that evolves based on textual gradients from evaluation feedback.
    """

    def __init__(
        self,
        prediction_llm: LLMInterface,
        evaluation_llm: LLMInterface,
        memory_llm: LLMInterface,
        prompt_manager: PromptManager,
        config: Dict[str, Any],
    ):
        """
        Args:
            prediction_llm: LLM for making clinical predictions
            evaluation_llm: LLM for evaluating predictions
            memory_llm: LLM for updating evolving memory
            prompt_manager: Manager for system prompts
            config: RLN configuration
        """
        self.prediction_llm = prediction_llm
        self.evaluation_llm = evaluation_llm
        self.memory_llm = memory_llm
        self.prompt_manager = prompt_manager
        self.config = config

        # RLN parameters
        self.update_frequency = config.get("update_frequency", "every_visit")
        self.update_after_n = config.get("update_after_n_visits", 1)

        logger.info("RecurrentLanguageNetwork initialized")

    def forward(
        self,
        patient_visits: List[VisitData],
        initial_prompt_state: SystemPromptState,
        evaluate_intermediate: bool = True,
    ) -> Tuple[List[PredictionOutput], List[EvaluationResult], List[SystemPromptState], SystemPromptState]:
        """
        Forward pass through all visits for a patient

        Args:
            patient_visits: List of visits in chronological order
            initial_prompt_state: Initial system prompt state
            evaluate_intermediate: Whether to evaluate intermediate predictions

        Returns:
            Tuple of (predictions, evaluations, prompt_states_per_visit, final_prompt_state).
            prompt_states_per_visit[i] is the state (h_{t-1}) used when predicting visit i,
            captured BEFORE any memory update triggered by that visit.
        """
        logger.info(
            f"Starting forward pass for patient {initial_prompt_state.patient_id} "
            f"with {len(patient_visits)} visits"
        )

        predictions = []
        evaluations = []
        prompt_states_per_visit: List[SystemPromptState] = []
        current_prompt_state = initial_prompt_state

        total_visits = len(patient_visits)
        for visit_idx, visit in enumerate(patient_visits):
            visit_num = visit_idx + 1
            logger.info(
                f"Processing visit {visit_num}/{total_visits}: {visit.visit_id}"
            )

            # Stamp visit_id on every subsequent LLM call so records can be
            # grouped per visit across prediction, evaluation, and memory update.
            visit_tags = {
                "visit_id": visit.visit_id,
                "visit_index": visit_num,
                "total_visits": total_visits,
                "is_last_visit": visit_num == total_visits,
            }
            self.prediction_llm.set_call_context(**visit_tags)
            self.evaluation_llm.set_call_context(**visit_tags)
            self.memory_llm.set_call_context(**visit_tags)

            prompt_states_per_visit.append(current_prompt_state)

            # Step 1: Generate prediction
            self.prediction_llm.set_call_context(phase="prediction")
            prediction = self._generate_prediction(visit, current_prompt_state)
            predictions.append(prediction)

            # Step 2: Evaluate and update (except for last visit during training)
            is_last_visit = visit_idx == len(patient_visits) - 1

            if evaluate_intermediate and not is_last_visit:
                # Evaluate prediction
                self.evaluation_llm.set_call_context(phase="evaluation")
                evaluation = self._evaluate_prediction(visit, prediction)
                evaluations.append(evaluation)

                # Update system prompt based on evaluation
                if self._should_update(visit_num):
                    self.memory_llm.set_call_context(phase="memory_update")
                    visit_summary = self._create_visit_summary(visit, prediction, evaluation)
                    current_prompt_state = self.prompt_manager.update_prompt(
                        current_state=current_prompt_state,
                        evaluation_feedback=self._evaluation_to_dict(evaluation),
                        visit_summary=visit_summary,
                        visit_number=visit_num,
                        visit_data=visit,
                        prediction=prediction,
                    )
                    logger.info(f"Updated prompt after visit {visit_num}")

        logger.info(
            f"Forward pass complete. Generated {len(predictions)} predictions, "
            f"{len(evaluations)} evaluations (last visit left to final_judge)"
        )

        return predictions, evaluations, prompt_states_per_visit, current_prompt_state

    def _generate_prediction(
        self, visit: VisitData, prompt_state: SystemPromptState
    ) -> PredictionOutput:
        """
        Generate clinical prediction for a single visit

        Args:
            visit: Visit data
            prompt_state: Current system prompt state

        Returns:
            PredictionOutput
        """
        # Format prediction prompt - include ALL available information for fair comparison
        visit_dict = {
            "chief_complaint": getattr(visit, "chief_complaint", None),
            "vital_signs": getattr(visit, "vital_signs", {}),
            "lab_results": getattr(visit, "lab_results", {}),
            "medications": getattr(visit, "medications", []),
            "procedures": getattr(visit, "procedures", []),
            "history_present_illness": getattr(visit, "history_present_illness", None),
            "past_medical_history": getattr(visit, "past_medical_history", None),
            "social_history": getattr(visit, "social_history", None),
            "family_history": getattr(visit, "family_history", None),
            "physical_exam": getattr(visit, "physical_exam", None),
            "pertinent_results": getattr(visit, "pertinent_results", None),
            "hospital_course": getattr(visit, "hospital_course", None),
            "allergies": getattr(visit, "allergies", None),
            "admission_medications": getattr(visit, "admission_medications", None),
        }

        prediction_prompt = self.prompt_manager.format_prediction_prompt(
            visit_data=visit_dict, system_prompt=prompt_state.full_prompt
        )

        # Generate prediction
        try:
            raw_output = self.prediction_llm.generate(
                prompt=prediction_prompt,
                system_prompt=prompt_state.full_prompt,
                temperature=0.7,
                max_tokens=2048
            )

            # Parse output
            parsed_output = safe_json_parse(
                raw_output,
                default={
                    "top_5_diagnoses": [],
                    "primary_diagnosis": {},
                },
            )

            return PredictionOutput(
                visit_id=visit.visit_id,
                top_5_diagnoses=parsed_output.get("top_5_diagnoses", []),
                primary_diagnosis=parsed_output.get("primary_diagnosis", {}),
                raw_output=raw_output,
            )

        except Exception as e:
            logger.error(f"Prediction generation failed for visit {visit.visit_id}: {e}")
            # Return empty prediction
            return PredictionOutput(
                visit_id=visit.visit_id,
                top_5_diagnoses=[],
                primary_diagnosis={},
                raw_output=str(e),
            )

    def _evaluate_prediction(
        self, visit: VisitData, prediction: PredictionOutput
    ) -> EvaluationResult:
        """
        Evaluate a prediction against ground truth using the intermediate evaluation LLM

        Args:
            visit: Visit with ground truth
            prediction: Model prediction

        Returns:
            EvaluationResult
        """
        logger.debug(f"Evaluation LLM scoring prediction for visit {visit.visit_id}")

        # Format evaluation prompt
        evaluation_prompt = self.prompt_manager.templates["evaluation_prompt"].format(
            prediction=json.dumps(
                {
                    "top_5_diagnoses": prediction.top_5_diagnoses,
                    "primary_diagnosis": prediction.primary_diagnosis,
                },
                indent=2,
            ),
            ground_truth_diagnoses=", ".join(visit.ground_truth_diagnoses)
        )

        # Generate evaluation
        try:
            raw_output = self.evaluation_llm.generate(
                prompt=evaluation_prompt,
                temperature=0.3,
                max_tokens=1536  # Need enough tokens for detailed evaluation
            )

            parsed_eval = safe_json_parse(raw_output, default={})

            improvement_text = self._extract_improvement_suggestion(
                parsed_eval.get("improvement_suggestions", "")
                if isinstance(parsed_eval, dict)
                else ""
            )

            return EvaluationResult(
                visit_id=visit.visit_id,
                diagnosis_evaluation=parsed_eval.get("diagnosis_evaluation", {}) if isinstance(parsed_eval, dict) else {},
                improvement_suggestions=improvement_text,
                raw_output=raw_output,
            )

        except Exception as e:
            logger.error(f"Evaluation failed for visit {visit.visit_id}: {e}")
            return EvaluationResult(
                visit_id=visit.visit_id,
                diagnosis_evaluation={},
                improvement_suggestions="",
                raw_output=str(e),
            )

    def _should_update(self, visit_num: int) -> bool:
        """Determine if prompt should be updated at this visit"""
        if self.update_frequency == "every_visit":
            return True
        elif self.update_frequency == "every_n_visits":
            return visit_num % self.update_after_n == 0
        else:
            return False

    def _create_visit_summary(
        self, visit: VisitData, prediction: PredictionOutput, evaluation: EvaluationResult
    ) -> str:
        """
        Create a comprehensive summary of the visit for prompt update

        Args:
            visit: Visit data with ground truth
            prediction: Model's prediction
            evaluation: Intermediate evaluation result

        Returns:
            Formatted summary string
        """
        if isinstance(prediction.primary_diagnosis, dict):
            primary_diag = prediction.primary_diagnosis.get('name', 'Unknown')
        else:
            primary_diag = str(prediction.primary_diagnosis) if prediction.primary_diagnosis else "Unknown"

        if isinstance(prediction.top_5_diagnoses, list) and prediction.top_5_diagnoses:
            top_5 = ', '.join([d.get('name', 'Unknown') if isinstance(d, dict) else str(d) for d in prediction.top_5_diagnoses[:5]])
        else:
            top_5 = "None"

        visit_details_block = self._format_visit_details(visit)

        summary = f"""
        Visit ID: {visit.visit_id}
        Timestamp: {visit.timestamp}
        Chief Complaint: {visit.chief_complaint}

        VISIT DETAILS:
        {visit_details_block}

        WHAT WAS PREDICTED:
        - Primary Diagnosis: {primary_diag}
        - Top-5 Diagnoses: {top_5}

        ACTUAL GROUND TRUTH:
        - Diagnoses: {', '.join(visit.ground_truth_diagnoses)}

        EVALUATION INSIGHTS: {self._format_eval_summary(evaluation)}

        LEARNING OPPORTUNITY:
        Use the evaluation insights to update understanding of this patient's clinical patterns and improve future predictions.
        """
        return summary.strip()

    def _evaluation_to_dict(self, evaluation: EvaluationResult) -> Dict[str, Any]:
        """Convert EvaluationResult to dictionary"""
        return {
            "diagnosis_evaluation": evaluation.diagnosis_evaluation,
            "improvement_suggestions": evaluation.improvement_suggestions,
        }

    def predict_single_visit(
        self, visit: VisitData, prompt_state: SystemPromptState
    ) -> PredictionOutput:
        """
        Make prediction for a single visit (inference mode)

        Args:
            visit: Visit data
            prompt_state: Current system prompt state

        Returns:
            PredictionOutput
        """
        return self._generate_prediction(visit, prompt_state)

    @staticmethod
    def _clean_field(value: Any, max_len: int = 600) -> str:
        """Normalize and trim free-text fields to keep prompts compact"""
        if value is None:
            return "Not recorded"
        text = str(value).strip()
        return text[:max_len] if len(text) > max_len else text

    def _format_visit_details(self, visit: VisitData) -> str:
        """
        Build a rich, compressed snapshot of a visit (aligned with compress_summary_baseline formatting)
        """
        chief_complaint = self._clean_field(getattr(visit, "chief_complaint", None))
        history_present_illness = self._clean_field(
            getattr(visit, "history_present_illness", None)
        )
        past_medical_history = self._clean_field(
            getattr(visit, "past_medical_history", None)
        )
        social_history = self._clean_field(getattr(visit, "social_history", None))
        family_history = self._clean_field(getattr(visit, "family_history", None))
        physical_exam = self._clean_field(getattr(visit, "physical_exam", None))
        pertinent_results = self._clean_field(
            getattr(visit, "pertinent_results", None)
        )
        hospital_course = self._clean_field(getattr(visit, "hospital_course", None))
        allergies = self._clean_field(getattr(visit, "allergies", None))

        procedures = getattr(visit, "procedures", None)
        procedures_str = "; ".join(procedures[:5]) if procedures else "None recorded"

        admission_meds = getattr(visit, "admission_medications", None)
        admission_meds = admission_meds if admission_meds else getattr(visit, "medications", None)
        meds_str = (
            ", ".join(admission_meds[:10])
            if isinstance(admission_meds, list)
            else (str(admission_meds) if admission_meds else "None recorded")
        )

        vitals = visit.vital_signs if isinstance(visit.vital_signs, dict) else {}
        vitals_str = ", ".join(f"{k}: {v}" for k, v in vitals.items()) or "None recorded"

        labs = visit.lab_results if isinstance(visit.lab_results, dict) else {}
        labs_str = ", ".join(f"{k}: {v}" for k, v in labs.items()) or "None recorded"

        gt_diagnoses = getattr(visit, "ground_truth_diagnoses", [])

        parts = [
            f"Visit ID: {visit.visit_id}",
            f"Date: {visit.timestamp}",
            f"Chief Complaint: {chief_complaint}",
            f"History of Present Illness: {history_present_illness}",
            f"Past Medical History: {past_medical_history}",
            f"Social History: {social_history}",
            f"Family History: {family_history}",
            f"Physical Exam: {physical_exam}",
            f"Pertinent Results: {pertinent_results}",
            f"Hospital Course: {hospital_course}",
            f"Allergies: {allergies}",
            f"Vital Signs: {vitals_str}",
            f"Lab Results: {labs_str}",
            f"Procedures: {procedures_str}",
            f"Medications on Admission: {meds_str}",
            f"Discharge Diagnoses: {', '.join(gt_diagnoses) if gt_diagnoses else 'None'}",
        ]

        return "\n".join(parts)

    def _build_eval_insights(self, parsed_eval: Dict[str, Any], visit: VisitData) -> str:
        """Flatten evaluation into a single concise string (for logging/prompt updates)"""
        return str(parsed_eval).strip() if parsed_eval else ""

    def _format_eval_summary(self, evaluation: EvaluationResult) -> str:
        """Create a concise evaluation summary from available fields"""
        parts = []
        if evaluation.diagnosis_evaluation:
            parts.append(f"Diagnosis eval: {evaluation.diagnosis_evaluation}")
        if evaluation.improvement_suggestions:
            parts.append(f"Suggestions: {evaluation.improvement_suggestions}")
        if not parts and evaluation.raw_output:
            return evaluation.raw_output[:400]
        return " ".join(parts) if parts else "Not provided"

    @staticmethod
    def _extract_improvement_suggestion(value: Any) -> str:
        """Normalize improvement suggestions into a single sentence string"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            # Join list elements into one sentence
            return "; ".join(str(v).strip() for v in value if v).strip()
        if isinstance(value, dict):
            # Flatten key/value pairs into a brief sentence
            items = []
            for k, v in value.items():
                if isinstance(v, list):
                    items.append(f"{k}: " + "; ".join(str(x).strip() for x in v if x))
                else:
                    items.append(f"{k}: {str(v).strip()}")
            return "; ".join(items).strip()
        return str(value).strip()

    def batch_predict(
        self,
        visits: List[VisitData],
        prompt_states: List[SystemPromptState],
    ) -> List[PredictionOutput]:
        """
        Batch prediction for multiple visits (different patients)

        Args:
            visits: List of visit data
            prompt_states: Corresponding system prompt states

        Returns:
            List of PredictionOutput
        """
        logger.info(f"Batch predicting for {len(visits)} visits")

        predictions = []
        for visit, prompt_state in zip(visits, prompt_states):
            pred = self.predict_single_visit(visit, prompt_state)
            predictions.append(pred)

        return predictions
