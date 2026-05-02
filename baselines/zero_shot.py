"""
Zero-Shot Baseline
Predicts discharge diagnosis using only the current visit, no patient history.
Uses the paper's Zero-Shot prompt template (Appendix C.3).
"""

import logging
from typing import List, Dict, Any

from models.llm_interface import LLMInterface, safe_json_parse
from models.rln_core import VisitData, PredictionOutput
from data_loader import PatientRecord

logger = logging.getLogger(__name__)


def _join_list(value, limit: int = 10) -> str:
    if isinstance(value, list) and value:
        return ", ".join(str(v) for v in value[:limit])
    if value:
        return str(value)
    return "None recorded"


def _format_vitals(vitals: Dict[str, Any]) -> str:
    if not isinstance(vitals, dict) or not vitals:
        return "None recorded"
    return ", ".join(f"{k}: {v}" for k, v in vitals.items())


def _format_labs(labs: Dict[str, Any]) -> str:
    if not isinstance(labs, dict) or not labs:
        return "None recorded"
    return ", ".join(f"{k}: {v}" for k, v in labs.items())


class ZeroShotBaseline:
    """Zero-shot: predict from current visit alone, no history."""

    def __init__(self, prediction_llm: LLMInterface, templates: Dict[str, Any]):
        self.prediction_llm = prediction_llm
        self.templates = templates

    def predict(self, visit: VisitData, age: str, gender: str) -> PredictionOutput:
        """Generate prediction for a single visit with no history context."""
        meds = visit.admission_medications or visit.medications
        prompt = self.templates["zero_shot_prompt"].format(
            age=age,
            gender=gender,
            chief_complaint=visit.chief_complaint or "Not recorded",
            vital_signs=_format_vitals(visit.vital_signs),
            lab_results=_format_labs(visit.lab_results),
            medications=_join_list(meds),
            procedures=_join_list(visit.procedures, limit=5),
        )

        try:
            raw_output = self.prediction_llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2048,
            )

            parsed = safe_json_parse(raw_output, default={"top_5_diagnoses": [], "primary_diagnosis": {}})

            return PredictionOutput(
                visit_id=visit.visit_id,
                top_5_diagnoses=parsed.get("top_5_diagnoses", []),
                primary_diagnosis=parsed.get("primary_diagnosis", {}),
                raw_output=raw_output,
            )
        except Exception as e:
            logger.error(f"Zero-shot prediction failed for {visit.visit_id}: {e}")
            return PredictionOutput(
                visit_id=visit.visit_id,
                top_5_diagnoses=[],
                primary_diagnosis={},
                raw_output=str(e),
            )

    def run_patient(self, patient: PatientRecord) -> List[PredictionOutput]:
        """Run zero-shot prediction for all visits of a patient."""
        predictions = []
        visits = patient.visits
        total = len(visits)
        gender = patient.sex or "Unknown"
        age = "Unknown"
        for i, visit in enumerate(visits):
            self.prediction_llm.set_call_context(
                visit_id=visit.visit_id,
                visit_index=i + 1,
                total_visits=total,
                is_last_visit=(i + 1 == total),
                phase="prediction",
            )
            pred = self.predict(visit, age=age, gender=gender)
            predictions.append(pred)
            logger.info(f"  Zero-shot predicted {visit.visit_id}")
        return predictions
