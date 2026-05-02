"""
Shared in-memory data containers for the GRPO training pipeline.

These dataclasses flow through the pipeline as Python objects (no JSONL on the
hot path). Only the periodic checkpoint writer (training/io/artifact_writer.py)
serializes them to disk, every N epochs, for audit and resume.

Pipeline stages and the type produced at each stage:
    Step 1 (collect_trajectories)  -> List[TrajectoryStep]
    Step 2 (sample_rollouts)       -> List[RolloutGroup]
    Step 3 (score_rollouts)        -> List[ScoredGroup]
    Step 4 (policy_trainer)        consumes List[ScoredGroup]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models.rln_core import VisitData, PredictionOutput, EvaluationResult


@dataclass
class TrajectoryStep:
    """One memory-update training instance: (h_{t-1}, x_t, y_hat_t, e_t, x_{t+1}, gt_{t+1}).

    Captured AFTER an RLN forward pass so we have everything needed to re-sample
    the memory-update head at this step under the current policy and to score
    each candidate via rubric + downstream prediction at t+1.
    """

    patient_id: str
    visit_index: int                                # 0-based index of x_t within the patient

    # Memory-update prompt and its raw inputs (kept for reproducibility / re-render)
    update_prompt: str                              # rendered, ready to feed to policy LLM
    h_prev_evolving: str                            # h_{t-1} content (the recurrent state)
    h_prev_full_prompt: str                         # full system prompt used at visit t
    h_curr_evolving_ref: str                        # h_t produced by the original (frozen) policy this epoch — reference only

    # Original observations / outcomes
    visit: VisitData                                # x_t
    next_visit: VisitData                           # x_{t+1}
    prediction: PredictionOutput                    # y_hat_t (from the predictor at visit t)
    evaluation: EvaluationResult                    # e_t (judge feedback used to drive the update)

    # Demographics so downstream-acc scoring can re-render system_prompt with a candidate h_t
    age: str = "Unknown"
    gender: str = ""
    primary_conditions: str = "Unknown"


@dataclass
class Rollout:
    """One sampled candidate from the policy at a given TrajectoryStep."""

    text: str                                       # raw policy output
    cumulative_logprob: float                       # sum of per-token logprobs at sample time (old policy)
    per_token_logprobs: List[float]
    parsed_evolving_summary: Optional[str] = None   # extracted h_t string (None if JSON parse failed)


@dataclass
class RolloutGroup:
    """G candidate rollouts sharing the same TrajectoryStep — the GRPO group."""

    step: TrajectoryStep
    candidates: List[Rollout]


@dataclass
class ScoredRollout:
    """A Rollout enriched with rubric judge score and (optionally) downstream Acc_{t+1}."""

    text: str
    cumulative_logprob: float
    per_token_logprobs: List[float]
    parsed_evolving_summary: Optional[str]

    judge_score: float                              # in (0, 1] from masked digit decoding
    downstream_acc: Optional[float] = None          # 0.0 or 1.0 from t+1 prediction check; None if disabled
    reward: float = 0.0                             # composite r = J_psi + epsilon * Acc_{t+1}

    extras: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScoredGroup:
    """G ScoredRollouts that feed into one GRPO group-relative advantage."""

    step: TrajectoryStep
    candidates: List[ScoredRollout]
