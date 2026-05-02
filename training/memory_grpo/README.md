# Memory GRPO Skeleton

This folder is a minimal scaffold for training with a prompt-based generative reward model in `verl`.

It is designed for the case where:

- the policy generates a candidate output
- a separate judge LLM reads a scoring prompt
- the judge returns text or JSON
- the reward function extracts a numeric score and returns it to GRPO

## Files

- [reward_fn.py](/Users/jinrui/Desktop/LLMasRNN_all/LLMasRNN/training/memory_grpo/reward_fn.py): custom reward function used by `reward.custom_reward_function`
- [run_memory_grpo.sh](/Users/jinrui/Desktop/LLMasRNN_all/LLMasRNN/training/memory_grpo/run_memory_grpo.sh): example launch script

## Recommended Wiring

Use the hybrid path:

- `reward.custom_reward_function.path=training/memory_grpo/reward_fn.py`
- `reward.reward_model.enable=True`

In that setup, `verl` launches the judge model itself, and `reward_fn.py` talks to the internal reward router via `reward_router_address`.

This is the same pattern used in `verl`'s generative reward model tests and examples.

## KL Strategy

The launch script is wired with two distinct KL mechanisms.

- Reward-side KL: enabled through `algorithm.use_kl_in_reward=True`
- Actor-side KL: enabled through `actor_rollout_ref.actor.use_kl_loss=True`

They play different roles:

- reward-side KL penalizes samples that drift too far from the reference policy before GRPO advantage normalization
- actor-side KL regularizes the policy optimization objective directly

For memory-update training, keeping both can be reasonable if you are explicitly trying to reduce reward hacking. The default script uses a smaller reward-side coefficient than actor-side coefficient so the policy is not over-constrained from the first run.

Relevant environment variables in `run_memory_grpo.sh`:

- `ACTOR_USE_KL`
- `ACTOR_KL_COEF`
- `ACTOR_KL_TYPE`
- `USE_REWARD_KL`
- `REWARD_KL_PENALTY`
- `REWARD_KL_CTRL_TYPE`
- `REWARD_KL_COEF`
- `REWARD_KL_TARGET`
- `REWARD_KL_HORIZON`

## Judge Output Contract

The judge should ideally return strict JSON:

```json
{"score": 7.5, "reason": "Concise justification"}
```

`reward_fn.py` will:

- first try to parse a JSON object with a `score` field
- then fall back to regex extraction
- clamp the raw score into `[MEMORY_REWARD_SCORE_MIN, MEMORY_REWARD_SCORE_MAX]`
- rescale it according to `MEMORY_REWARD_OUTPUT_SCALE`

Supported output scales:

- `raw`
- `zero_one`
- `minus_one_one`

## Prompt Contract

`reward_fn.py` supports two ways to build the judge prompt.

1. Recommended: provide a fully assembled `extra_info["judge_prompt"]` in your dataset pipeline.
2. Fallback: it builds a generic prompt from:
   - `extra_info["task"]` or `extra_info["question"]` or `extra_info["input"]`
   - `ground_truth`
   - `solution_str`

If you are optimizing the memory-update step specifically, your training data should eventually supply a task-specific `judge_prompt` that includes:

- prior memory
- current visit information
- prediction output
- evaluation feedback
- the candidate memory update to be scored

## Environment Variables

Useful knobs:

- `MEMORY_JUDGE_MODEL_NAME`: model name/path sent in OpenAI-compatible requests. Default falls back to the reward model tokenizer path.
- `MEMORY_JUDGE_BASE_URL`: optional external endpoint. Only needed if you disable `reward.reward_model.enable`.
- `MEMORY_REWARD_SCORE_MIN`: default `1`
- `MEMORY_REWARD_SCORE_MAX`: default `10`
- `MEMORY_REWARD_PARSE_FAIL_SCORE`: default `0`
- `MEMORY_REWARD_OUTPUT_SCALE`: one of `raw`, `zero_one`, `minus_one_one`
- `MEMORY_REWARD_TEMPERATURE`: default `0.0`
- `MEMORY_REWARD_MAX_TOKENS`: default `256`
- `MEMORY_REWARD_STRICT_JSON`: set to `1` if your backend supports `response_format={"type":"json_object"}`

## What You Still Need To Customize

Before using this for real training, you still need to decide:

1. What exactly the policy output is.
2. What fields your parquet examples provide in `extra_info`.
3. The final judge prompt template.
4. Whether reward should stay in raw `1-10` scale or be normalized.
5. How to handle parse failures in a stricter way if needed.

## Suggested Next Step

When you are ready, the next concrete step is not tuning `verl`, but defining the reward sample schema for the memory-update task. Once that schema is fixed, `reward_fn.py` can be tightened around the exact fields instead of staying generic.
