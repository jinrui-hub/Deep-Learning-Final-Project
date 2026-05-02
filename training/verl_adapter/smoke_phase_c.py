"""
Phase C end-to-end smoke: validate the reward function against a REAL parquet
(from Phase B) + a REAL vLLM-served judge.

Unlike test_rubric_reward.py (which mocks the judge), this hits a live judge
server. Shows you, per row:
  - what prompt got sent to the judge
  - what the judge actually replied
  - what reward score came out

Use to:
  1. Verify the judge server is reachable (network, port, model loaded)
  2. Eyeball a few rubric-prompt → judge-output → reward triplets
  3. Smoke-test before swapping in your real SFT judge

Prereqs:
  - Phase B parquet exists (run dump_trajectories.sh first)
  - Judge server is running (start_judge_server.sh in another tmux pane)
  - The model named in rubric_v1.yaml's judge.model is reachable at judge.endpoint

Usage (inside idev, container or container env):
  apptainer exec --nv $VERL_SIF python -m training.verl_adapter.smoke_phase_c \\
      --parquet $SCRATCH/data/llm_as_rnn/train.parquet \\
      --n 3                          # how many rows to score; default 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo root importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.verl_adapter import rubric_reward as RR


# A plausible model output (good case) and a bad-format one — used to mimic
# what verl's policy would generate during training. We score BOTH against
# the same step so you can see how reward differs.
GOOD_CANDIDATE = json.dumps({
    "evolving_summary": (
        "CLINICAL HISTORY: Patient presented with primary diagnosis. "
        "CURRENT STATUS & FOCUS AREAS: Active condition being managed. "
        "KEY CONSIDERATIONS: Watch for recurrence; monitor key labs."
    )
})
BAD_CANDIDATE_NOT_JSON = "I'm not sure how to summarize this."
EMPTY_CANDIDATE = json.dumps({"evolving_summary": ""})


def _score_one(step_row, candidate_str, label):
    """Run compute_score and pretty-print the result."""
    print(f"\n--- {label} ---")
    print(f"  candidate (first 100 chars): {candidate_str[:100]!r}")
    reward = RR.compute_score(
        data_source=step_row["data_source"],
        solution_str=candidate_str,
        ground_truth=step_row["reward_model"]["ground_truth"],
        extra_info=step_row.get("extra_info"),
    )
    print(f"  → reward: {reward:.4f}")
    return reward


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True, help="Path to a Phase B-output parquet")
    parser.add_argument("--n", type=int, default=3, help="Number of rows to score (default 3)")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    table = pq.read_table(args.parquet)
    rows = table.slice(0, args.n).to_pylist()
    print(f"Loaded {len(rows)} rows from {args.parquet}")

    # Print rubric + judge config (so you can verify yaml was read correctly)
    rubric = RR._load_rubric()
    judge = rubric["judge"]
    print(f"\nRubric YAML : {RR.RUBRIC_YAML_PATH}")
    print(f"  dims      : {rubric.get('dimensions', [])}")
    print(f"  judge     : {judge['model']} @ {judge['endpoint']}")

    summary = []
    for i, row in enumerate(rows):
        print(f"\n========== ROW {i + 1}/{len(rows)} (patient={row['extra_info']['patient_id']}, visit={row['extra_info']['visit_index']}) ==========")
        r_good = _score_one(row, GOOD_CANDIDATE, "GOOD candidate (well-formed JSON)")
        r_bad = _score_one(row, BAD_CANDIDATE_NOT_JSON, "BAD candidate (not JSON)")
        r_empty = _score_one(row, EMPTY_CANDIDATE, "EMPTY candidate (valid JSON, empty summary)")
        summary.append((r_good, r_bad, r_empty))

    print(f"\n========== SUMMARY ==========")
    print(f"{'row':<5} {'good':>8} {'bad':>8} {'empty':>8}")
    for i, (g, b, e) in enumerate(summary):
        print(f"{i+1:<5} {g:>8.4f} {b:>8.4f} {e:>8.4f}")

    # Sanity assertions
    print(f"\n========== HEALTH CHECKS ==========")
    print(f"  bad candidates always = PARSE_FAIL_REWARD ({RR.PARSE_FAIL_REWARD}): "
          f"{all(b == RR.PARSE_FAIL_REWARD for _, b, _ in summary)}")
    print(f"  empty candidates always = PARSE_FAIL_REWARD: "
          f"{all(e == RR.PARSE_FAIL_REWARD for _, _, e in summary)}")
    print(f"  good candidates have variance > 0: "
          f"{len(set(round(g, 3) for g, _, _ in summary)) > 1 if len(summary) > 1 else 'n/a (only 1 row)'}")
    print(f"  good candidates all in [0, 1]: "
          f"{all(0 <= g <= 1 for g, _, _ in summary)}")


if __name__ == "__main__":
    main()
