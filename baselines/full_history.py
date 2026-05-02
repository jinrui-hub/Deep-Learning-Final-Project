"""
Full History Concatenation (FHC) Baseline
Predicts using the current visit plus all prior visits concatenated as context.
Uses the paper's FHC/MemPrompt shared prompt template (Appendix C.3).
"""

import logging
from typing import List, Dict, Any

from models.llm_interface import LLMInterface, safe_json_parse
from models.rln_core import VisitData, PredictionOutput
from data_loader import PatientRecord
from baselines.visit_format import format_current_visit_block, format_prior_visit_block

logger = logging.getLogger(__name__)


class FullHistoryBaseline:
    """FHC: concatenate all prior visits (with known outcomes) into the context for prediction."""

    def __init__(
        self,
        prediction_llm: LLMInterface,
        templates: Dict[str, Any],
        max_history_tokens: int = 4096,
    ):
        self.prediction_llm = prediction_llm
        self.templates = templates
        self.max_history_tokens = max_history_tokens

    def predict(
        self, visit: VisitData, prior_visits: List[VisitData]
    ) -> PredictionOutput:
        """Generate prediction with full history of prior visits."""
        patient_history = self._build_patient_history(prior_visits)
        current_visit = format_current_visit_block(visit)

        prompt = self.templates["history_prompt"].format(
            patient_history=patient_history,
            current_visit=current_visit,
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
            logger.error(f"FHC prediction failed for {visit.visit_id}: {e}")
            return PredictionOutput(
                visit_id=visit.visit_id,
                top_5_diagnoses=[],
                primary_diagnosis={},
                raw_output=str(e),
            )

    def run_patient(self, patient: PatientRecord) -> List[PredictionOutput]:
        """Run FHC prediction for all visits of a patient."""
        predictions = []
        visits = patient.visits
        total = len(visits)
        for i, visit in enumerate(visits):
            self.prediction_llm.set_call_context(
                visit_id=visit.visit_id,
                visit_index=i + 1,
                total_visits=total,
                is_last_visit=(i + 1 == total),
                phase="prediction",
            )
            prior = visits[:i]
            pred = self.predict(visit, prior)
            predictions.append(pred)
            logger.info(f"  FHC predicted {visit.visit_id} (history: {len(prior)} visits)")
        return predictions

    def _build_patient_history(self, prior_visits: List[VisitData]) -> str:
        """Serialize prior visits with known outcomes. Truncates oldest first if over budget."""
        if not prior_visits:
            return "No prior visits recorded."

        blocks = [format_prior_visit_block(v) for v in prior_visits]
        full_history = "\n\n".join(blocks)

        char_budget = self.max_history_tokens * 4  # ~4 chars per token
        while len(full_history) > char_budget and len(blocks) > 1:
            blocks.pop(0)
            full_history = "\n\n".join(blocks)

        return full_history
