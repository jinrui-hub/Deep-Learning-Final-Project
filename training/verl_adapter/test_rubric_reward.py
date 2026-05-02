"""
Sanity tests for rubric_reward.compute_score — runs WITHOUT a live judge server
by monkey-patching `_call_judge` to return canned responses.

Run on Mac (no GPU needed):
    python -m training.verl_adapter.test_rubric_reward

Or via pytest:
    pytest training/verl_adapter/test_rubric_reward.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Make the repo root importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.verl_adapter import rubric_reward as RR

# Redirect audit writes to a tempdir for the duration of the tests so we
# don't pollute $SCRATCH or whatever default the module computed at import.
_TMPDIR = tempfile.mkdtemp(prefix="rubric_reward_test_")
RR.AUDIT_DIR = Path(_TMPDIR)


# Minimal but realistic ground_truth (mirrors what dump_trajectories produces).
SAMPLE_STEP = {
    "patient_id": "10001401",
    "visit_index": 0,
    "h_prev_evolving": "Patient first visit. No history yet.",
    "visit": {
        "chief_complaint": "Fever, cough",
        "history_present_illness": "3-day fever, productive cough",
        "pertinent_results": "WBC 15k, CXR shows RLL infiltrate",
        "hospital_course": "Treated with ceftriaxone, improved",
    },
    "prediction": {
        "primary_diagnosis": {"name": "Pneumonia"},
        "top_5_diagnoses": [{"name": "Pneumonia"}, {"name": "Bronchitis"}],
    },
    "evaluation": {
        "diagnosis_evaluation": {"primary_correct": True, "any_top5_correct": True},
        "improvement_suggestions": "Consider ABG for severity",
    },
}
SAMPLE_GROUND_TRUTH = json.dumps(SAMPLE_STEP)

GOOD_SOLUTION = json.dumps({
    "evolving_summary": "Patient with community-acquired pneumonia, treated successfully. "
                        "Watch for recurrence and antibiotic resistance.",
})
EMBEDDED_JSON_SOLUTION = (
    "Here is my answer: " + GOOD_SOLUTION + " That's my final answer."
)
BAD_SOLUTION_NOT_JSON = "I don't know what to summarize"
EMPTY_SUMMARY_SOLUTION = json.dumps({"evolving_summary": ""})


# ---- Tests ------------------------------------------------------------------

def _run(name, fn):
    try:
        fn()
        print(f"  ✓ {name}")
        return True
    except AssertionError as e:
        print(f"  ✗ {name}: {e}")
        return False
    except Exception as e:
        print(f"  ✗ {name}: unexpected {type(e).__name__}: {e}")
        return False


def test_parse_clean_json():
    s = RR._parse_candidate_h_t(GOOD_SOLUTION)
    assert s is not None and "pneumonia" in s.lower(), f"got {s!r}"


def test_parse_embedded_json():
    s = RR._parse_candidate_h_t(EMBEDDED_JSON_SOLUTION)
    assert s is not None and "pneumonia" in s.lower(), f"got {s!r}"


def test_parse_invalid_json_returns_none():
    assert RR._parse_candidate_h_t(BAD_SOLUTION_NOT_JSON) is None


def test_parse_empty_summary_returns_none():
    assert RR._parse_candidate_h_t(EMPTY_SUMMARY_SOLUTION) is None


def test_judge_text_to_reward_score_marker():
    """Strong signal: 'Score: N' pattern wins over any embedded digits."""
    assert RR._judge_text_to_reward("Score: 7") == 7 / 9
    assert RR._judge_text_to_reward("score: 8") == 8 / 9              # case insensitive
    assert RR._judge_text_to_reward("Score=5") == 5 / 9               # = also accepted
    # Score: N wins even with other digits in the text (the 2026-04-29 bug fix)
    assert RR._judge_text_to_reward("1. xyz\n2. abc\n\nScore: 6") == 6 / 9


def test_judge_text_to_reward_short_digit_only():
    """Short text containing just a digit: accept (judge is terse)."""
    assert RR._judge_text_to_reward("7") == 7 / 9
    assert RR._judge_text_to_reward("8\n") == 8 / 9


def test_judge_text_long_verbose_no_score_marker_falls_back():
    """Long CoT-style text WITHOUT 'Score: N' must NOT grab embedded digits.

    Reproduces the 2026-04-29 RubricARM bug: response was a truncated
    dimension-by-dimension analysis ('1. Faithfulness... 2. Compression...')
    and the old parser grabbed '2' from '2. Compression efficiency'. Now we
    refuse to extract from long verbose text and return JUDGE_NO_DIGIT_REWARD,
    which correctly signals 'judge ignored format instruction'.
    """
    rubric_arm_truncated = (
        "Now, I need to evaluate the candidate hidden state h_t against the "
        "seven dimensions. Let me go through each one systematically.\n\n"
        "1. Factual faithfulness: The candidate h_t does not include any "
        "specific labs or history details from x_t, so it fails to faithfully "
        "represent the actual data.\n2. Compression efficiency"
    )
    r = RR._judge_text_to_reward(rubric_arm_truncated)
    assert r == RR.JUDGE_NO_DIGIT_REWARD, (
        f"long verbose text WITHOUT 'Score:' marker must fall back to "
        f"JUDGE_NO_DIGIT_REWARD (not grab embedded dim numbers); got {r}"
    )


def test_judge_text_no_digit_falls_back():
    assert RR._judge_text_to_reward("undefined") == RR.JUDGE_NO_DIGIT_REWARD
    assert RR._judge_text_to_reward("") == RR.JUDGE_NO_DIGIT_REWARD


def test_render_rubric_prompt_has_all_substitutions():
    prompt = RR._render_rubric_prompt(SAMPLE_STEP, "candidate h_t")
    for needed in ["Patient first visit", "Fever, cough", "Pneumonia",
                   "primary_correct", "candidate h_t"]:
        assert needed in prompt, f"missing {needed!r} in rendered prompt"


def test_compute_score_happy_path_returns_normalized_reward():
    with patch.object(RR, "_call_judge", return_value="6"):
        r = RR.compute_score(
            data_source=RR.DATA_SOURCE_TAG,
            solution_str=GOOD_SOLUTION,
            ground_truth=SAMPLE_GROUND_TRUTH,
        )
    assert abs(r - 6 / 9) < 1e-9, f"expected 6/9, got {r}"


def test_compute_score_bad_json_returns_parse_fail_reward():
    with patch.object(RR, "_call_judge", return_value="9"):  # judge would have given high score
        r = RR.compute_score(
            data_source=RR.DATA_SOURCE_TAG,
            solution_str=BAD_SOLUTION_NOT_JSON,
            ground_truth=SAMPLE_GROUND_TRUTH,
        )
    assert r == RR.PARSE_FAIL_REWARD, f"expected {RR.PARSE_FAIL_REWARD}, got {r}"


def test_compute_score_wrong_data_source_returns_zero():
    with patch.object(RR, "_call_judge", return_value="9"):
        r = RR.compute_score(
            data_source="something_else",
            solution_str=GOOD_SOLUTION,
            ground_truth=SAMPLE_GROUND_TRUTH,
        )
    assert r == 0.0, f"expected 0.0, got {r}"


def test_compute_score_judge_returns_no_digit():
    with patch.object(RR, "_call_judge", return_value="undefined"):
        r = RR.compute_score(
            data_source=RR.DATA_SOURCE_TAG,
            solution_str=GOOD_SOLUTION,
            ground_truth=SAMPLE_GROUND_TRUTH,
        )
    assert r == RR.JUDGE_NO_DIGIT_REWARD, f"expected {RR.JUDGE_NO_DIGIT_REWARD}, got {r}"


def test_audit_log_written_for_judge_called_branch():
    """A successful judge call should produce one audit record with all fields."""
    audit_path = RR._audit_path()
    n_before = sum(1 for _ in open(audit_path)) if audit_path.exists() else 0
    with patch.object(RR, "_call_judge", return_value="Score: 7"):
        RR.compute_score(
            data_source=RR.DATA_SOURCE_TAG,
            solution_str=GOOD_SOLUTION,
            ground_truth=SAMPLE_GROUND_TRUTH,
            extra_info={"patient_id": "P123", "visit_index": 2},
        )
    rows = list(open(audit_path))
    assert len(rows) == n_before + 1, f"expected 1 new audit row, got {len(rows) - n_before}"
    rec = json.loads(rows[-1])
    assert rec["branch"] == "judge_called"
    assert rec["patient_id"] == "P123"
    assert rec["visit_index"] == 2
    assert rec["judge_response"] == "Score: 7"
    assert "judge_prompt" in rec and len(rec["judge_prompt"]) > 100   # full rubric rendered
    assert rec["candidate_h_t"] is not None
    assert rec["solution_str"] == GOOD_SOLUTION                       # full rollout text
    assert abs(rec["reward"] - 7 / 9) < 1e-9


def test_audit_log_written_for_parse_fail_branch():
    """A bad-JSON candidate should produce an audit record marked parse_fail."""
    audit_path = RR._audit_path()
    n_before = sum(1 for _ in open(audit_path)) if audit_path.exists() else 0
    RR.compute_score(
        data_source=RR.DATA_SOURCE_TAG,
        solution_str=BAD_SOLUTION_NOT_JSON,
        ground_truth=SAMPLE_GROUND_TRUTH,
        extra_info={"patient_id": "P999", "visit_index": 0},
    )
    rows = list(open(audit_path))
    assert len(rows) == n_before + 1
    rec = json.loads(rows[-1])
    assert rec["branch"] == "parse_fail"
    assert rec["patient_id"] == "P999"
    assert rec["solution_str"] == BAD_SOLUTION_NOT_JSON
    assert rec["reward"] == RR.PARSE_FAIL_REWARD


def main():
    tests = [
        ("parse_clean_json",                       test_parse_clean_json),
        ("parse_embedded_json",                    test_parse_embedded_json),
        ("parse_invalid_json_returns_none",        test_parse_invalid_json_returns_none),
        ("parse_empty_summary_returns_none",       test_parse_empty_summary_returns_none),
        ("judge_text_score_marker",                test_judge_text_to_reward_score_marker),
        ("judge_text_short_digit_only",            test_judge_text_to_reward_short_digit_only),
        ("judge_text_long_verbose_falls_back",     test_judge_text_long_verbose_no_score_marker_falls_back),
        ("judge_text_no_digit_falls_back",         test_judge_text_no_digit_falls_back),
        ("render_rubric_prompt_has_all_subs",      test_render_rubric_prompt_has_all_substitutions),
        ("compute_score_happy_path",               test_compute_score_happy_path_returns_normalized_reward),
        ("compute_score_bad_json",                 test_compute_score_bad_json_returns_parse_fail_reward),
        ("compute_score_wrong_data_source",        test_compute_score_wrong_data_source_returns_zero),
        ("compute_score_judge_no_digit",           test_compute_score_judge_returns_no_digit),
        ("audit_log_written_for_judge_called",     test_audit_log_written_for_judge_called_branch),
        ("audit_log_written_for_parse_fail",       test_audit_log_written_for_parse_fail_branch),
    ]
    print(f"Running {len(tests)} sanity tests for rubric_reward.py")
    passed = sum(_run(n, f) for n, f in tests)
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
