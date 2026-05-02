"""
MemPrompt Baseline
Each prior visit is LLM-summarized; summaries are concatenated as context.
Uses the paper's FHC/MemPrompt shared prompt template (Appendix C.3).
"""

import logging
from typing import List, Dict, Any

from models.llm_interface import LLMInterface, safe_json_parse
from models.rln_core import VisitData, PredictionOutput
from data_loader import PatientRecord
from baselines.visit_format import format_current_visit_block

logger = logging.getLogger(__name__)


class MemPromptBaseline:
    """MemPrompt: summarize each prior visit, concatenate summaries + current visit."""

    def __init__(
        self,
        prediction_llm: LLMInterface,
        templates: Dict[str, Any],
        max_history_tokens: int = 4096,
    ):
        self.prediction_llm = prediction_llm
        self.templates = templates
        self.max_history_tokens = max_history_tokens

    def summarize_visit(self, visit: VisitData) -> str:
        """Generate a concise LLM summary of a single visit."""
        summary_template = self.templates.get("visit_summary_prompt", "")
        if not summary_template:
            return self._fallback_summary(visit)

        prompt = summary_template.format(
            chief_complaint=visit.chief_complaint or "Not recorded",
            hospital_course=visit.hospital_course or "Not recorded",
            discharge_diagnosis=", ".join(visit.ground_truth_diagnoses),
            procedures=", ".join(visit.procedures) if visit.procedures else "None",
            pertinent_results=(visit.pertinent_results or "Not available")[:500],
        )

        try:
            summary = self.prediction_llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=256,
            )
            return summary.strip()
        except Exception as e:
            logger.warning(f"MemPrompt summarization failed for {visit.visit_id}: {e}")
            return self._fallback_summary(visit)

    def predict(
        self, visit: VisitData, prior_summaries: List[str]
    ) -> PredictionOutput:
        """Generate prediction with summarized history."""
        patient_history = self._build_patient_history(prior_summaries)
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
            logger.error(f"MemPrompt prediction failed for {visit.visit_id}: {e}")
            return PredictionOutput(
                visit_id=visit.visit_id,
                top_5_diagnoses=[],
                primary_diagnosis={},
                raw_output=str(e),
            )

    def run_patient(self, patient: PatientRecord) -> List[PredictionOutput]:
        """Run MemPrompt for all visits of a patient."""
        predictions = []
        summaries: List[str] = []
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

            pred = self.predict(visit, summaries)
            predictions.append(pred)
            logger.info(
                f"  MemPrompt predicted {visit.visit_id} (summaries: {len(summaries)})"
            )

            self.prediction_llm.set_call_context(phase="memprompt_summarize")
            summary = self.summarize_visit(visit)
            summaries.append(summary)

        return predictions

    def _build_patient_history(self, prior_summaries: List[str]) -> str:
        if not prior_summaries:
            return "No prior visits recorded."

        blocks = [f"Visit {i}: {s}" for i, s in enumerate(prior_summaries, 1)]
        full_history = "\n".join(blocks)

        char_budget = self.max_history_tokens * 4
        while len(full_history) > char_budget and len(blocks) > 1:
            blocks.pop(0)
            full_history = "\n".join(blocks)

        return full_history

    @staticmethod
    def _fallback_summary(visit: VisitData) -> str:
        parts = [f"Chief complaint: {visit.chief_complaint}"]
        if visit.ground_truth_diagnoses:
            parts.append(f"Diagnoses: {', '.join(visit.ground_truth_diagnoses)}")
        if visit.procedures:
            parts.append(f"Procedures: {', '.join(visit.procedures)}")
        return ". ".join(parts)
