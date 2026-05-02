import json
import os
import re
from typing import Any

import aiohttp


DEFAULT_SYSTEM_PROMPT = (
    "You are a strict reward model for reinforcement learning. "
    "Score the candidate output and return JSON only."
)

DEFAULT_USER_TEMPLATE = """Evaluate the candidate output for the task below.

Task:
{task}

Ground truth:
{ground_truth}

Candidate output:
{solution}

Score the candidate from {score_min} to {score_max}.
Return JSON only in this format:
{{"score": <number>, "reason": "<short reason>"}}
""".strip()


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _extract_score_payload(raw_text: str) -> tuple[float | None, str | None]:
    payload = None
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            payload = parsed
    except json.JSONDecodeError:
        payload = _first_json_object(raw_text)

    if payload is not None and "score" in payload:
        try:
            return float(payload["score"]), payload.get("reason")
        except (TypeError, ValueError):
            pass

    score_match = re.search(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)', raw_text)
    if score_match:
        return float(score_match.group(1)), None

    number_match = re.search(r"(-?\d+(?:\.\d+)?)", raw_text)
    if number_match:
        return float(number_match.group(1)), None

    return None, None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _rescale_score(raw_score: float, score_min: float, score_max: float, output_scale: str) -> float:
    if output_scale == "raw":
        return raw_score

    span = score_max - score_min
    if span <= 0:
        raise ValueError("score_max must be larger than score_min")

    normalized = (raw_score - score_min) / span
    if output_scale == "zero_one":
        return normalized
    if output_scale == "minus_one_one":
        return normalized * 2.0 - 1.0

    raise ValueError(f"Unsupported MEMORY_REWARD_OUTPUT_SCALE: {output_scale}")


def _build_messages(solution_str: str, ground_truth: str, extra_info: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = extra_info.get("judge_system_prompt", DEFAULT_SYSTEM_PROMPT)

    if extra_info.get("judge_prompt"):
        user_prompt = str(extra_info["judge_prompt"])
    else:
        task = extra_info.get("task") or extra_info.get("question") or extra_info.get("input") or ""
        score_min = _get_env_float("MEMORY_REWARD_SCORE_MIN", 1.0)
        score_max = _get_env_float("MEMORY_REWARD_SCORE_MAX", 10.0)
        user_prompt = DEFAULT_USER_TEMPLATE.format(
            task=task,
            ground_truth=ground_truth,
            solution=solution_str,
            score_min=score_min,
            score_max=score_max,
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def _chat_complete(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            return await response.json()


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    reward_router_address: str | None = None,
    reward_model_tokenizer: Any | None = None,
):
    del data_source

    score_min = _get_env_float("MEMORY_REWARD_SCORE_MIN", 1.0)
    score_max = _get_env_float("MEMORY_REWARD_SCORE_MAX", 10.0)
    parse_fail_score = _get_env_float("MEMORY_REWARD_PARSE_FAIL_SCORE", 0.0)
    output_scale = os.getenv("MEMORY_REWARD_OUTPUT_SCALE", "raw")
    temperature = _get_env_float("MEMORY_REWARD_TEMPERATURE", 0.0)
    max_tokens = _get_env_int("MEMORY_REWARD_MAX_TOKENS", 256)
    strict_json = _get_env_bool("MEMORY_REWARD_STRICT_JSON", False)

    base_url = os.getenv("MEMORY_JUDGE_BASE_URL")
    if reward_router_address is not None:
        base_url = f"http://{reward_router_address}"
    if base_url is None:
        raise ValueError(
            "No judge endpoint available. Set reward.reward_model.enable=True or MEMORY_JUDGE_BASE_URL."
        )

    model_name = os.getenv("MEMORY_JUDGE_MODEL_NAME")
    if model_name is None and reward_model_tokenizer is not None:
        model_name = getattr(reward_model_tokenizer, "name_or_path", None)
    if model_name is None:
        raise ValueError("Set MEMORY_JUDGE_MODEL_NAME or provide reward.reward_model.model_path.")

    messages = _build_messages(solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if strict_json:
        payload["response_format"] = {"type": "json_object"}

    response = await _chat_complete(base_url=base_url, payload=payload)
    raw_output = response["choices"][0]["message"]["content"]

    raw_score, reason = _extract_score_payload(raw_output)
    parse_ok = raw_score is not None

    if parse_ok:
        clipped_raw_score = _clamp(raw_score, score_min, score_max)
        reward_score = _rescale_score(clipped_raw_score, score_min, score_max, output_scale)
    else:
        clipped_raw_score = None
        reward_score = parse_fail_score

    return {
        "score": reward_score,
        "judge_raw_score": clipped_raw_score,
        "judge_parse_ok": parse_ok,
        "judge_reason": reason,
        "judge_raw_output": raw_output,
        "judge_prompt_used": messages[-1]["content"],
    }
