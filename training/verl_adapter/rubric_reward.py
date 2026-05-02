"""
Phase C: verl-compatible reward function for LLM-as-RNN memory updates.

verl calls `compute_score(data_source, solution_str, ground_truth, extra_info)`
once per (state, candidate) pair via its RewardManager. We:
  1. Decode the candidate evolving_summary out of `solution_str` (policy output JSON)
  2. json.loads the TrajectoryStep out of `ground_truth` (Phase B dump put it there)
  3. Render the rubric prompt (template from rubric YAML)
  4. HTTP-call a vLLM-served judge model
  5. Parse the last digit 1-9 in the judge response, return digit/9 ∈ (0, 1]

The rubric YAML is the SINGLE SOURCE OF TRUTH for everything:
  - rubric content (dimensions, prompt template)
  - judge model id + endpoint
  - judge sampling params
  - serve config (used by start_judge_server.sh — not by this file)

Swap rubric/judge by EITHER:
  (a) editing the active YAML in place, or
  (b) pointing RUBRIC_YAML_PATH at a different YAML file (default constant
      below; adjust the constant in this file or copy a new yaml).

Plug into verl via:
  +reward.custom_reward_function.path=training/verl_adapter/rubric_reward.py
  +reward.custom_reward_function.name=compute_score
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger("rubric_reward")

# Single source of truth for which rubric YAML to use. Edit this constant (or
# copy/rename the YAML) when iterating across rubric versions.
RUBRIC_YAML_PATH = "training/configs/rubric_v1.yaml"

# verl router tag — must match what dump_trajectories_to_parquet.py writes.
DATA_SOURCE_TAG = "llm_as_rnn_memory_update"

# Reward when the policy's output isn't valid JSON: low but non-zero so gradient
# still flows. GRPO would clip if this matches the group mean exactly.
PARSE_FAIL_REWARD = 0.05
# Reward when the judge returns no digit (rare): neutral midpoint.
JUDGE_NO_DIGIT_REWARD = 0.5

# Parser strategy (in priority order):
#   1. "Score: N" / "Score=N" / "score : N" — strong signal, judge followed instruction
#   2. Short text (< 20 chars) ending with a digit — judge said just "7" or "Score: 7"
#   3. Long verbose text without (1) — judge ignored instruction (e.g., RubricARM doing CoT)
#                                      → JUDGE_NO_DIGIT_REWARD, don't false-positive on
#                                        embedded dimension numbers ("2. Compression...")
SCORE_LINE_RE = re.compile(r"(?i)\bscore\s*[:=]\s*([1-9])\b")
DIGIT_RE = re.compile(r"[1-9]")
SHORT_TEXT_THRESHOLD = 20

# ---- Audit logging ----------------------------------------------------------
# Every reward call (rollout candidate + rendered judge prompt + judge response
# + final reward) is appended to a JSONL under AUDIT_DIR. One file per worker
# PID to avoid concurrent-write races (verl spawns multiple reward workers).
# Tests monkey-patch AUDIT_DIR to a tempdir; production runs default to
# $SCRATCH/llm_as_rnn/rubric_audit/.
AUDIT_DIR: Path = Path(
    os.environ.get("SCRATCH", str(Path.home()))
) / "llm_as_rnn" / "rubric_audit"


# ---- Cached singletons (verl loads this module once per worker) -------------
@lru_cache(maxsize=1)
def _load_rubric() -> dict[str, Any]:
    """Load rubric YAML once per process. Rubric YAML is the source of truth
    for prompt template, dimensions, AND judge config (model, endpoint, etc.)."""
    repo_root = Path(__file__).resolve().parents[2]
    path = (repo_root / RUBRIC_YAML_PATH).resolve()
    with open(path) as f:
        rubric = yaml.safe_load(f)
    judge = rubric.get("judge", {})
    logger.info(f"[rubric_reward] loaded rubric: {path}")
    logger.info(f"[rubric_reward]   dims:     {rubric.get('dimensions', [])}")
    logger.info(f"[rubric_reward]   judge:    {judge.get('model', '?')} @ {judge.get('endpoint', '?')}")
    return rubric


@lru_cache(maxsize=1)
def _judge_client():
    """Lazy import openai so this module loads without it (e.g. on Mac for tests)."""
    from openai import OpenAI
    judge = _load_rubric()["judge"]
    return OpenAI(base_url=judge["endpoint"], api_key="EMPTY")


# ---- Helpers ----------------------------------------------------------------
# def _parse_candidate_h_t(solution_str: str) -> Optional[str]:
#     """Extract evolving_summary from the policy's output.

#     The policy is prompted to return `{"evolving_summary": "..."}`. We accept:
#       - clean JSON
#       - JSON embedded in extra text (find {...} block)
#     Returns None if no valid JSON or no evolving_summary key.
#     """
#     try:
#         data = json.loads(solution_str)
#         s = data.get("evolving_summary")
#         return s if isinstance(s, str) and s.strip() else None
#     except (json.JSONDecodeError, AttributeError):
#         pass
#     # Try embedded JSON (greedy: outermost braces)
#     try:
#         start = solution_str.index("{")
#         end = solution_str.rindex("}") + 1
#         data = json.loads(solution_str[start:end])
#         s = data.get("evolving_summary")
#         return s if isinstance(s, str) and s.strip() else None
#     except (ValueError, json.JSONDecodeError, AttributeError):
#         return None
FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _parse_candidate_h_t(solution_str: str) -> Optional[str]:
    """Extract evolving_summary from the policy output.

    Accepts:
      - {"evolving_summary": "..."}
      - {"evolving_summary": {...}}
      - {"evolving_summary": [...]}
      - JSON inside ```json ... ```
      - JSON embedded in surrounding text

    Returns a string candidate hidden state for the judge.
    """

    def normalize_summary(s: Any) -> Optional[str]:
        # Ideal case: evolving_summary is already a plain string.
        if isinstance(s, str) and s.strip():
            return s.strip()

        # Current rollout often returns a dict/list. Convert it to text
        # so the judge can still evaluate it.
        if isinstance(s, (dict, list)):
            return json.dumps(s, ensure_ascii=False, indent=2)

        return None

    # 1. Direct JSON
    try:
        data = json.loads(solution_str)
        s = normalize_summary(data.get("evolving_summary"))
        if s:
            return s
    except (json.JSONDecodeError, AttributeError):
        pass

    # 2. Markdown fenced JSON: ```json { ... } ```
    m = FENCED_JSON_RE.search(solution_str)
    if m:
        try:
            data = json.loads(m.group(1))
            s = normalize_summary(data.get("evolving_summary"))
            if s:
                return s
        except (json.JSONDecodeError, AttributeError):
            pass

    # 3. Embedded JSON fallback: take text between first "{" and last "}"
    try:
        start = solution_str.index("{")
        end = solution_str.rindex("}") + 1
        data = json.loads(solution_str[start:end])
        s = normalize_summary(data.get("evolving_summary"))
        if s:
            return s
    except (ValueError, json.JSONDecodeError, AttributeError):
        pass

    return None

def _summarize_visit(visit: dict[str, Any]) -> str:
    """Compact text rendering of a VisitData for embedding in the rubric prompt."""
    fields = [
        ("chief_complaint", "chief_complaint"),
        ("history_present_illness", "hpi"),
        ("pertinent_results", "pertinent_results"),
        ("hospital_course", "hospital_course"),
    ]
    parts = []
    for key, label in fields:
        val = visit.get(key, "") or ""
        if val:
            parts.append(f"{label}: {val}")
    return "\n".join(parts)


def _summarize_prediction(pred: dict[str, Any]) -> str:
    return json.dumps(
        {
            "primary": pred.get("primary_diagnosis", {}),
            "top5": pred.get("top_5_diagnoses", []),
        },
        indent=2,
    )


def _summarize_evaluation(eval_: dict[str, Any]) -> str:
    return json.dumps(
        {
            "diagnosis_evaluation": eval_.get("diagnosis_evaluation", {}) or {},
            "suggestions": eval_.get("improvement_suggestions", "") or "",
        },
        indent=2,
    )


def _render_rubric_prompt(step: dict[str, Any], candidate_h_t: str) -> str:
    rubric = _load_rubric()
    return rubric["prompt_template"].format(
        h_prev=step.get("h_prev_evolving", ""),
        x_t=_summarize_visit(step.get("visit", {})),
        y_hat=_summarize_prediction(step.get("prediction", {})),
        e_t=_summarize_evaluation(step.get("evaluation", {})),
        h_t=candidate_h_t,
    )


def _call_judge(prompt: str) -> str:
    """HTTP completion call to vllm-served judge. Returns raw response text."""
    judge = _load_rubric()["judge"]
    resp = _judge_client().completions.create(
        model=judge["model"],
        prompt=prompt,
        max_tokens=judge.get("max_tokens", 8),
        temperature=judge.get("temperature", 0.0),
    )
    return resp.choices[0].text


def _judge_text_to_reward(judge_text: str) -> float:
    """Map judge response → reward in (0, 1]. Defensive against false-positive
    digit grabs (e.g., dimension numbers in CoT-style verbose responses)."""
    text = judge_text.strip()

    # 1. Strong signal: judge wrote "Score: N" (the format our prompt requests).
    m = SCORE_LINE_RE.search(text)
    if m:
        return int(m.group(1)) / 9.0

    # 2. Short text — judge probably just emitted a digit ("7", or "Score: 7\n7").
    if len(text) < SHORT_TEXT_THRESHOLD:
        digits = DIGIT_RE.findall(text)
        if digits:
            return int(digits[-1]) / 9.0

    # 3. Long verbose text without "Score:" — judge ignored the format instruction.
    #    Don't grab embedded digits (would catch "2. Compression efficiency").
    logger.warning(
        f"[rubric_reward] judge response has no 'Score: N' marker and is too long "
        f"to safely extract a digit (len={len(text)}); falling back to "
        f"{JUDGE_NO_DIGIT_REWARD}. text head={text[:200]!r}"
    )
    return JUDGE_NO_DIGIT_REWARD


def _audit_path() -> Path:
    """One JSONL per worker PID, under AUDIT_DIR."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIT_DIR / f"calls_{os.getpid()}.jsonl"


def _audit(record: dict[str, Any]) -> None:
    """Append one reward-call record. Never raises (must not break training)."""
    try:
        with open(_audit_path(), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[rubric_reward] audit write failed: {e}")


# ---- verl entry point -------------------------------------------------------
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> float:
    """verl reward function. Called once per (state, candidate) pair.

    Returns a scalar reward in [0, 1]. Special-case rewards (low but non-zero):
      - solution_str isn't parseable JSON   → PARSE_FAIL_REWARD (0.05)
      - judge returns no digit              → JUDGE_NO_DIGIT_REWARD (0.5)
      - data_source mismatch (defensive)    → 0.0
    """
    extra = extra_info or {}

    if data_source != DATA_SOURCE_TAG:
        logger.warning(f"[rubric_reward] unexpected data_source={data_source}; returning 0.0")
        _audit({
            "ts": time.time(), "branch": "wrong_data_source",
            "patient_id": extra.get("patient_id"), "visit_index": extra.get("visit_index"),
            "data_source": data_source,
            "solution_str": solution_str[:500],
            "reward": 0.0,
        })
        return 0.0

    candidate_h_t = _parse_candidate_h_t(solution_str)
    if candidate_h_t is None:
        _audit({
            "ts": time.time(), "branch": "parse_fail",
            "patient_id": extra.get("patient_id"), "visit_index": extra.get("visit_index"),
            "solution_str": solution_str,                    # full raw, since this is the failure case to debug
            "reward": PARSE_FAIL_REWARD,
        })
        return PARSE_FAIL_REWARD

    try:
        step = json.loads(ground_truth)
    except json.JSONDecodeError as e:
        logger.error(f"[rubric_reward] ground_truth not valid JSON: {e}; returning 0.0")
        _audit({
            "ts": time.time(), "branch": "ground_truth_bad_json",
            "patient_id": extra.get("patient_id"), "visit_index": extra.get("visit_index"),
            "error": str(e), "reward": 0.0,
        })
        return 0.0

    prompt = _render_rubric_prompt(step, candidate_h_t)
    judge_text = _call_judge(prompt)
    reward = _judge_text_to_reward(judge_text)

    _audit({
        "ts": time.time(), "branch": "judge_called",
        "patient_id": extra.get("patient_id"), "visit_index": extra.get("visit_index"),
        "solution_str": solution_str,                        # raw policy output (the rollout)
        "candidate_h_t": candidate_h_t,                      # parsed evolving_summary
        "judge_prompt": prompt,                              # full rendered rubric prompt
        "judge_response": judge_text,                        # raw judge text
        "reward": reward,
    })
    return reward
