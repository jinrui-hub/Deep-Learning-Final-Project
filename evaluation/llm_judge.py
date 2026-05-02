"""
LLM-as-Judge Evaluator
Shared evaluation module used by all methods for consistent Acc@1 / Acc@5 scoring.
"""

import json
import logging
from typing import List, Dict, Any

from models.llm_interface import LLMInterface, safe_json_parse
from models.rln_core import PredictionOutput, EvaluationResult

logger = logging.getLogger(__name__)


class LLMJudge:
    """Evaluates predictions against ground truth using an evaluation LLM."""

    def __init__(self, evaluation_llm: LLMInterface, evaluation_template: str):
        self.evaluation_llm = evaluation_llm
        self.evaluation_template = evaluation_template

    def evaluate(
        self, prediction: PredictionOutput, ground_truth: List[str]
    ) -> EvaluationResult:
        """
        Evaluate a single prediction against ground truth diagnoses.

        Returns an EvaluationResult with diagnosis_evaluation containing
        primary_correct (Acc@1) and any_top5_correct (Acc@5).
        """
        prediction_json = json.dumps(
            {
                "top_5_diagnoses": prediction.top_5_diagnoses,
                "primary_diagnosis": prediction.primary_diagnosis,
            },
            indent=2,
        )

        prompt = self.evaluation_template.format(
            prediction=prediction_json,
            ground_truth_diagnoses=", ".join(ground_truth),
        )

        try:
            raw_output = self.evaluation_llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1536,
            )

            parsed = safe_json_parse(raw_output, default={})

            return EvaluationResult(
                visit_id=prediction.visit_id,
                diagnosis_evaluation=parsed.get("diagnosis_evaluation", {}),
                improvement_suggestions=_normalize_suggestions(
                    parsed.get("improvement_suggestions", "")
                ),
                raw_output=raw_output,
            )

        except Exception as e:
            logger.error(f"Evaluation failed for {prediction.visit_id}: {e}")
            return EvaluationResult(
                visit_id=prediction.visit_id,
                diagnosis_evaluation={},
                improvement_suggestions="",
                raw_output=str(e),
            )

    def batch_evaluate(
        self,
        predictions: List[PredictionOutput],
        ground_truths: List[List[str]],
    ) -> List[EvaluationResult]:
        """Evaluate a batch of predictions sequentially."""
        results = []
        for pred, gt in zip(predictions, ground_truths):
            results.append(self.evaluate(pred, gt))
        return results


def _normalize_suggestions(value: Any) -> str:
    """Normalize improvement_suggestions to a string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(str(v).strip() for v in value if v)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value).strip() if value else ""
