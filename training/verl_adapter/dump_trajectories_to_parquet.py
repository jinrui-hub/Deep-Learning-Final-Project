"""
Phase B: dump LLM-as-RNN trajectories into a verl-compatible parquet file.

Pipeline:
  1. Load the inference YAML (4-model setup: predictor + memory + evaluator + final).
  2. Run RLN forward on a patient split — produces one TrajectoryStep per
     (patient, intermediate_visit) pair.
  3. Convert each TrajectoryStep to a verl parquet row (chat-format prompt +
     full step serialized into reward_model.ground_truth so the Phase C
     reward function can decode and use it).
  4. Write parquet under $SCRATCH/data/llm_as_rnn/<split>.parquet.

This is a one-time per-epoch pre-processing step. It costs one full RLN forward
pass over the patient set (~3 LLM calls per visit × n visits × n patients).

Verl parquet schema (matches verl/examples/data_preprocess/gsm8k.py):
    prompt        : List[{"role": "user", "content": <update_prompt>}]
    data_source   : "llm_as_rnn_memory_update"   # Phase C reward router uses this
    ability       : "clinical_summary"
    reward_model  : {"style": "rubric",
                     "ground_truth": JSON-serialized TrajectoryStep}
    extra_info    : {"split", "patient_id", "visit_index", "visit_id"}

Usage:
    python -m training.verl_adapter.dump_trajectories_to_parquet \
        --config configs/experiment1_seperateJudge.yaml \
        --data-path data/splits/cleaned_df_train_100.json \
        --output-dir $SCRATCH/data/llm_as_rnn \
        --split train \
        --max-patients 5      # smoke; drop for full run
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Make the repo root importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.llm_interface import create_llm_interface
from models.prompt_manager import PromptManager
from models.rln_core import RecurrentLanguageNetwork

from training.collect.collect_trajectories import collect_trajectories
from training.types import TrajectoryStep

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("dump_trajectories")


DATA_SOURCE_TAG = "llm_as_rnn_memory_update"
ABILITY_TAG = "clinical_summary"


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer for items dataclasses.asdict can't handle."""
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _step_to_ground_truth(step: TrajectoryStep) -> str:
    """Serialize a TrajectoryStep to a JSON string for reward_model.ground_truth.

    Phase C's compute_score will json.loads this back to reconstruct the step
    context (h_prev, x_t, y_hat, e_t) needed to render the rubric prompt.
    """
    return json.dumps(asdict(step), default=_json_default, ensure_ascii=False)


def _step_to_row(step: TrajectoryStep, split: str) -> Dict[str, Any]:
    """Convert one TrajectoryStep to a verl parquet row."""
    return {
        "prompt": [{"role": "user", "content": step.update_prompt}],
        "data_source": DATA_SOURCE_TAG,
        "ability": ABILITY_TAG,
        "reward_model": {
            "style": "rubric",
            "ground_truth": _step_to_ground_truth(step),
        },
        "extra_info": {
            "split": split,
            "patient_id": str(step.patient_id),
            "visit_index": int(step.visit_index),
            "visit_id": str(getattr(step.visit, "visit_id", "")),
        },
    }


def _build_rln_from_config(cfg: Dict[str, Any]) -> RecurrentLanguageNetwork:
    """Instantiate the 3 RLN-side LLMs + PromptManager + RLN, mirroring run_experiment.py."""
    pred_cfg = cfg["prediction_model"]
    eval_cfg = cfg["evaluation_model"]
    mem_cfg = cfg["memory_model"]

    prediction_llm = create_llm_interface(
        backend=pred_cfg["backend"], model_name=pred_cfg["model_name"], config=pred_cfg
    )
    evaluation_llm = create_llm_interface(
        backend=eval_cfg["backend"], model_name=eval_cfg["model_name"], config=eval_cfg
    )
    memory_llm = create_llm_interface(
        backend=mem_cfg["backend"], model_name=mem_cfg["model_name"], config=mem_cfg
    )

    # Templates can either be inline under `templates:` or referenced via
    # `templates_config: <path/to/other.yaml>` to avoid copying ~200 lines of
    # prompt templates between configs (smoke / experiment6 / etc).
    if "templates" in cfg:
        templates = cfg["templates"]
    elif "templates_config" in cfg:
        with open(cfg["templates_config"]) as f:
            templates = yaml.safe_load(f)["templates"]
    else:
        raise KeyError("config must define either `templates:` or `templates_config:`")
    # PromptManager (4-model API): memory_llm drives prompt_update writes,
    # compression_llm summarizes long-running system prompts. Pattern matches
    # run_experiment.py — compression delegated to prediction_llm by convention.
    prompt_manager = PromptManager(
        templates=templates,
        memory_llm=memory_llm,
        compression_llm=prediction_llm,
        max_prompt_length=4096,
        compress_every_n=5,
    )
    rln = RecurrentLanguageNetwork(
        prediction_llm=prediction_llm,
        evaluation_llm=evaluation_llm,
        memory_llm=memory_llm,
        prompt_manager=prompt_manager,
        config={"update_frequency": "every_visit"},
    )
    return rln, prompt_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Inference YAML (4-model)")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Patient split JSON. Default: config.data.input_path",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("SCRATCH", str(Path.home())) + "/data/llm_as_rnn",
        help="Where to write <split>.parquet",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-patients", type=int, default=None, help="Debug cap")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_path = args.data_path or cfg["data"]["input_path"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.parquet"

    logger.info(f"config:      {args.config}")
    logger.info(f"data_path:   {data_path}")
    logger.info(f"output:      {out_path}")
    logger.info(f"max_patients:{args.max_patients}")

    # ---- Step 1: build RLN, run forward over patients ----
    rln, prompt_manager = _build_rln_from_config(cfg)
    logger.info("RLN constructed; running forward to collect trajectories...")

    steps: List[TrajectoryStep] = collect_trajectories(
        data_path=data_path,
        rln=rln,
        prompt_manager=prompt_manager,
        max_patients=args.max_patients,
    )
    if not steps:
        logger.error("No trajectory steps produced; aborting.")
        sys.exit(1)
    logger.info(f"Collected {len(steps)} TrajectoryStep records")

    # ---- Step 2: convert to verl rows + write parquet ----
    rows = [_step_to_row(s, args.split) for s in steps]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(out_path))
    logger.info(f"Wrote {len(rows)} rows -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # ---- Step 3: sanity check — read it back and print one row ----
    rt = pq.read_table(str(out_path))
    logger.info(f"Sanity: read back {rt.num_rows} rows, {rt.num_columns} cols: {rt.column_names}")
    sample = rt.slice(0, 1).to_pylist()[0]
    logger.info("=" * 60)
    logger.info("Sample row preview:")
    logger.info(f"  prompt[0].content    : {sample['prompt'][0]['content'][:200]}...")
    logger.info(f"  data_source          : {sample['data_source']}")
    logger.info(f"  ability              : {sample['ability']}")
    logger.info(f"  reward_model.style   : {sample['reward_model']['style']}")
    logger.info(f"  reward_model.gt[:200]: {sample['reward_model']['ground_truth'][:200]}...")
    logger.info(f"  extra_info           : {sample['extra_info']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
