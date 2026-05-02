"""
Metrics Aggregation
Computes Acc@1, Acc@5 and generates summary reports.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from models.rln_core import EvaluationResult, PredictionOutput

logger = logging.getLogger(__name__)


@dataclass
class VisitResult:
    """Stores a single visit's evaluation outcome plus full detail for later analysis."""
    method: str
    patient_id: str
    visit_id: str
    primary_correct: bool
    any_top5_correct: bool
    # Rich detail — populated when available, dumped into per_visit_detailed.
    prediction: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    ground_truth: Optional[List[str]] = None
    improvement_suggestions: str = ""
    evolving_summary_before: Optional[str] = None


class MetricsAggregator:
    """Collects evaluation results and computes aggregate metrics."""

    def __init__(self):
        self.results: List[VisitResult] = []
        self._stream_file = None  # append-only file handle for per-visit rows

    def enable_streaming(self, path: str) -> None:
        """Open `path` in append mode so every add_result() call is flushed.

        Must be called before add_result() for rows that should be streamed.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if self._stream_file is not None:
            try:
                self._stream_file.close()
            except Exception:
                pass
        self._stream_file = open(path, "a")
        logger.info(f"Streaming per-visit rows to {path}")

    def close_stream(self) -> None:
        if self._stream_file is not None:
            try:
                self._stream_file.close()
            finally:
                self._stream_file = None

    def load_from_stream(
        self, path: str, allowed_pairs: Optional[set] = None
    ) -> int:
        """Replay previously-written per-visit rows back into self.results.

        If `allowed_pairs` is given (a set of (method, patient_id) tuples),
        rows from pairs NOT in the set are dropped — this is how we filter out
        orphan rows from crashed runs that never wrote a completion marker.
        Returns the number of rows replayed.
        """
        if not os.path.exists(path):
            return 0
        replayed = 0
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if allowed_pairs is not None:
                    key = (row.get("method"), row.get("patient_id"))
                    if key not in allowed_pairs:
                        continue
                self.results.append(
                    VisitResult(
                        method=row["method"],
                        patient_id=row["patient_id"],
                        visit_id=row["visit_id"],
                        primary_correct=bool(row.get("primary_correct", False)),
                        any_top5_correct=bool(row.get("any_top5_correct", False)),
                        prediction=row.get("prediction"),
                        evaluation=row.get("evaluation"),
                        ground_truth=row.get("ground_truth"),
                        improvement_suggestions=row.get("improvement_suggestions", ""),
                        evolving_summary_before=row.get("evolving_summary_before"),
                    )
                )
                replayed += 1
        return replayed

    def rewrite_stream_file(self, path: str, allowed_pairs: set) -> int:
        """Rewrite `path` in place, keeping only rows whose (method, patient_id)
        is in `allowed_pairs`. Used on startup to drop orphan rows from crashed
        pairs. Writes atomically via temp file + os.replace.
        Returns the number of rows kept.
        """
        if not os.path.exists(path):
            return 0
        tmp_path = path + ".tmp"
        kept = 0
        with open(path, "r") as src, open(tmp_path, "w") as dst:
            for line in src:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                key = (row.get("method"), row.get("patient_id"))
                if key in allowed_pairs:
                    dst.write(stripped + "\n")
                    kept += 1
        os.replace(tmp_path, path)
        return kept

    def add_result(
        self,
        method: str,
        patient_id: str,
        visit_id: str,
        eval_result: EvaluationResult,
        prediction: Optional[PredictionOutput] = None,
        ground_truth: Optional[List[str]] = None,
        evolving_summary_before: Optional[str] = None,
    ):
        """Add a single evaluation result.

        Optional args preserve the full prediction/evaluation/hidden-state for
        qualitative analysis; they're dumped into results.json alongside metrics.
        If streaming is enabled, the row is also appended to disk immediately.
        """
        diag_eval = eval_result.diagnosis_evaluation
        primary_correct = bool(diag_eval.get("primary_correct", False))
        any_top5_correct = bool(diag_eval.get("any_top5_correct", False))

        prediction_payload = None
        if prediction is not None:
            prediction_payload = {
                "primary_diagnosis": prediction.primary_diagnosis,
                "top_5_diagnoses": prediction.top_5_diagnoses,
                "raw_output": prediction.raw_output,
            }

        evaluation_payload = {
            "diagnosis_evaluation": eval_result.diagnosis_evaluation,
            "raw_output": eval_result.raw_output,
        }

        visit_result = VisitResult(
            method=method,
            patient_id=patient_id,
            visit_id=visit_id,
            primary_correct=primary_correct,
            any_top5_correct=any_top5_correct,
            prediction=prediction_payload,
            evaluation=evaluation_payload,
            ground_truth=list(ground_truth) if ground_truth else None,
            improvement_suggestions=eval_result.improvement_suggestions or "",
            evolving_summary_before=evolving_summary_before,
        )
        self.results.append(visit_result)

        if self._stream_file is not None:
            row = {
                "method": visit_result.method,
                "patient_id": visit_result.patient_id,
                "visit_id": visit_result.visit_id,
                "primary_correct": visit_result.primary_correct,
                "any_top5_correct": visit_result.any_top5_correct,
                "ground_truth": visit_result.ground_truth,
                "prediction": visit_result.prediction,
                "evaluation": visit_result.evaluation,
                "improvement_suggestions": visit_result.improvement_suggestions,
                "evolving_summary_before": visit_result.evolving_summary_before,
            }
            try:
                self._stream_file.write(json.dumps(row, default=str) + "\n")
                self._stream_file.flush()
            except Exception as exc:
                logger.warning(f"Failed to stream per-visit row: {exc}")

    def _last_visit_per_patient(self) -> List[VisitResult]:
        """Return the final VisitResult for each (method, patient_id) group.

        Relies on insertion order: visits are appended chronologically by the
        experiment runner, so the last entry per group is the patient's final visit.
        """
        last: Dict[tuple, VisitResult] = {}
        for r in self.results:
            last[(r.method, r.patient_id)] = r
        return list(last.values())

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute overall Acc@1 and Acc@5 per method, scored only on each
        patient's FINAL visit. Earlier visits serve as context (history
        concatenation, hidden-state buildup, etc.) and are not scored.
        """
        last_results = self._last_visit_per_patient()

        methods: Dict[str, List[VisitResult]] = {}
        for r in last_results:
            methods.setdefault(r.method, []).append(r)

        output = {}
        for method, results in sorted(methods.items()):
            n = len(results)
            acc1 = sum(1 for r in results if r.primary_correct) / n if n else 0.0
            acc5 = sum(1 for r in results if r.any_top5_correct) / n if n else 0.0
            output[method] = {
                "acc_at_1": round(acc1, 4),
                "acc_at_5": round(acc5, 4),
                "n_patients": n,
            }
        return output

    def per_patient_metrics(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Per-patient breakdown: each patient contributes a single score per
        method, computed on that patient's LAST visit only.
        """
        last_results = self._last_visit_per_patient()

        output: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for r in last_results:
            output.setdefault(r.patient_id, {})[r.method] = {
                "acc_at_1": 1.0 if r.primary_correct else 0.0,
                "acc_at_5": 1.0 if r.any_top5_correct else 0.0,
                "last_visit_id": r.visit_id,
            }
        return {pid: dict(sorted(m.items())) for pid, m in sorted(output.items())}

    def print_summary(self):
        """Print a formatted summary table (last-visit accuracy)."""
        overall = self.compute_metrics()

        print("\n" + "=" * 64)
        print("EXPERIMENT RESULTS  (scored on last visit per patient)")
        print("=" * 64)
        print(f"{'Method':<20} {'Acc@1':>10} {'Acc@5':>10} {'N Patients':>12}")
        print("-" * 64)
        for method, metrics in overall.items():
            print(
                f"{method:<20} {metrics['acc_at_1']:>10.4f} "
                f"{metrics['acc_at_5']:>10.4f} {metrics['n_patients']:>12}"
            )
        print("=" * 64)

        # Per-patient breakdown available in results.json["per_patient"]

    def save_results(self, output_dir: str):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "overall": self.compute_metrics(),
            "per_patient": self.per_patient_metrics(),
            "per_visit": [
                {
                    "method": r.method,
                    "patient_id": r.patient_id,
                    "visit_id": r.visit_id,
                    "primary_correct": r.primary_correct,
                    "any_top5_correct": r.any_top5_correct,
                }
                for r in self.results
            ],
            "per_visit_detailed": [
                {
                    "method": r.method,
                    "patient_id": r.patient_id,
                    "visit_id": r.visit_id,
                    "primary_correct": r.primary_correct,
                    "any_top5_correct": r.any_top5_correct,
                    "ground_truth": r.ground_truth,
                    "prediction": r.prediction,
                    "evaluation": r.evaluation,
                    "improvement_suggestions": r.improvement_suggestions,
                    "evolving_summary_before": r.evolving_summary_before,
                }
                for r in self.results
            ],
        }

        output_path = os.path.join(output_dir, "results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")
        print(f"\nResults saved to {output_path}")
