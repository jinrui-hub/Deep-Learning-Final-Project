"""
LLM Interface Module
Provides unified interface for different LLM backends (vLLM, OpenAI, HuggingFace)
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import os
import time
import httpx

# Pydantic is used only for optional guided decoding hints. Provide a lightweight
# fallback so inference can run in environments without the dependency.
try:
    from pydantic import BaseModel, Field  # type: ignore
except ImportError:  # pragma: no cover - lightweight shim
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def model_json_schema(cls):
            return {}

    def Field(default=None, description: str = ""):  # noqa: N802
        return default

logger = logging.getLogger(__name__)


# Pydantic models for structured outputs
class EvolvingSummaryResponse(BaseModel):
    """Schema for evolving summary updates"""
    evolving_summary: str = Field(
        ...,
        description="Concise unified summary (max 200 words) with three sections: CLINICAL HISTORY, FOCUS AREAS, and FUTURE CONSIDERATIONS"
    )


class LLMInterface(ABC):

    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2048)
        self.top_p = config.get("top_p", 0.9)
        self._usage_stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Full audit trail of every generate() call on this instance.
        # Each entry: {timestamp, model, latency_s, temperature, max_tokens,
        #              system_prompt, prompt, response, error, **context_tags}
        self.call_log: List[Dict[str, Any]] = []
        self._call_context: Dict[str, Any] = {}
        self._call_log_stream = None  # optional append-only file handle
        self._install_generate_logger()

    def _install_generate_logger(self):
        """Wrap self.generate with a logger that captures prompt, response, and context tags.

        Wrapping at instance level (rather than class level) means every concrete
        subclass gets instrumented automatically, without each subclass needing
        to opt in. Subclass generate() methods are resolved via the bound
        reference captured here before we overwrite the instance attribute.
        """
        original_generate = self.generate  # bound method from the subclass

        def wrapped_generate(
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs,
        ) -> str:
            start = time.time()
            record: Dict[str, Any] = {
                "timestamp": start,
                "model": self.model_name,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "response": None,
                "error": None,
                "latency_s": None,
                **self._call_context,
            }
            try:
                response = original_generate(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                record["response"] = response
                return response
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["latency_s"] = round(time.time() - start, 3)
                self.call_log.append(record)
                if self._call_log_stream is not None:
                    try:
                        self._call_log_stream.write(json.dumps(record, default=str) + "\n")
                        self._call_log_stream.flush()
                    except Exception as write_exc:
                        logger.warning(f"Failed to stream call record: {write_exc}")

        self.generate = wrapped_generate  # type: ignore[method-assign]

    def set_call_context(self, **tags: Any) -> None:
        """Merge tags into the context attached to subsequent generate() calls.

        Use to stamp method/patient/visit identifiers onto log records. Call
        again to overwrite specific keys; unrelated keys are preserved.
        """
        self._call_context.update(tags)

    def clear_call_context(self) -> None:
        """Drop all context tags."""
        self._call_context = {}

    def save_call_log(self, path: str) -> None:
        """Write the full call log to a JSONL file, one record per line.

        No-op if streaming is already enabled for this instance (records are
        already on disk). Use `start_call_log_stream()` for crash-safe logging.
        """
        if self._call_log_stream is not None:
            logger.info(f"Call log already streamed to disk; skipping save_call_log({path})")
            return
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            for record in self.call_log:
                f.write(json.dumps(record, default=str) + "\n")
        logger.info(f"Wrote {len(self.call_log)} call records to {path}")

    def start_call_log_stream(self, path: str) -> None:
        """Open `path` in append mode so each generate() call is flushed immediately.

        Must be called before any generate() calls that should be streamed.
        The file is opened in append mode — existing records (from an earlier,
        interrupted run) are preserved.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if self._call_log_stream is not None:
            try:
                self._call_log_stream.close()
            except Exception:
                pass
        self._call_log_stream = open(path, "a")
        logger.info(f"Streaming call log to {path}")

    def close_call_log_stream(self) -> None:
        if self._call_log_stream is not None:
            try:
                self._call_log_stream.close()
            finally:
                self._call_log_stream = None

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        guided_json_schema: Optional[type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate text for multiple prompts"""
        pass

    def reset_usage_stats(self):
        """Reset cumulative generation usage counters."""
        self._usage_stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def get_usage_stats(self) -> Dict[str, int]:
        """Get cumulative generation usage counters."""
        return self._usage_stats.copy()

    def _record_usage(
        self,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ):
        """Record usage for one model call."""
        self._usage_stats["calls"] += 1

        p = int(prompt_tokens or 0)
        c = int(completion_tokens or 0)
        t = int(total_tokens) if total_tokens is not None else (p + c)

        self._usage_stats["prompt_tokens"] += p
        self._usage_stats["completion_tokens"] += c
        self._usage_stats["total_tokens"] += t


class AnthropicInterface(LLMInterface):
    """Interface for Anthropic Claude API."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        try:
            import anthropic  # type: ignore
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
        self._anthropic = anthropic
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set api_key or ANTHROPIC_API_KEY.")
        self.base_url = config.get("base_url", "https://api.anthropic.com")
        self.max_retries = config.get("max_retries", 8)
        self.retry_delay = config.get("retry_delay", 5.0)
        self.top_p = config.get("top_p")  # only used if temperature is None
        self.client = self._anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=config.get("timeout", 120.0),
            max_retries=self.max_retries,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        guided_json_schema: Optional[type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        message_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Use temperature by default; only fall back to top_p if temperature is not provided
        if temperature is not None:
            message_kwargs["temperature"] = temperature
        elif self.temperature is not None:
            message_kwargs["temperature"] = self.temperature
        elif self.top_p is not None:
            message_kwargs["top_p"] = self.top_p

        if system_prompt:
            message_kwargs["system"] = system_prompt

        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(**message_kwargs)
                content_blocks = getattr(resp, "content", []) or []
                texts = []
                for block in content_blocks:
                    text = getattr(block, "text", None)
                    if text:
                        texts.append(text)
                return "\n".join(texts).strip()
            except self._anthropic.APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                if status in {429, 500, 502, 503, 504, 529} and attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise
            except (self._anthropic.APIConnectionError, httpx.RequestError):
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise

    def batch_generate(
        self,
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        results = []
        system_prompts = system_prompts or [None] * len(prompts)
        for p, s in zip(prompts, system_prompts):
            results.append(self.generate(p, s, temperature, max_tokens))
        return results

class PortkeyInterface(LLMInterface):
    """Interface for Portkey AI gateway (supports Anthropic, OpenAI, and other providers)."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        try:
            from portkey_ai import Portkey  # type: ignore
        except ImportError:
            raise ImportError("Portkey package not installed. Install with: pip install portkey-ai")
        self.api_key = config.get("api_key") or os.getenv("PORTKEY_API_KEY")
        if not self.api_key:
            raise ValueError("Portkey API key not provided. Set api_key or PORTKEY_API_KEY.")
        self.client = Portkey(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        guided_json_schema: Optional[type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )

        usage = getattr(response, "usage", None)
        if usage is not None:
            self._record_usage(
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
                getattr(usage, "total_tokens", None),
            )
        else:
            self._record_usage()

        return response.choices[0].message.content.strip()

    def batch_generate(
        self,
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        return [
            self.generate(p, s, temperature, max_tokens)
            for p, s in zip(prompts, system_prompts)
        ]


class VLLMInterface(LLMInterface):
    """Interface for vLLM backend"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self._initialize_vllm()

    def _initialize_vllm(self):
        """Initialize vLLM model"""
        try:
            from vllm import LLM, SamplingParams

            logger.info(f"Loading vLLM model: {self.model_name}")

            # Pin this model to specific GPU(s) by setting CUDA_VISIBLE_DEVICES
            # before LLM() spawns its engine subprocess. The subprocess inherits
            # this env var, so each model can be placed on a different card.
            saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            cvd = self.config.get("cuda_visible_devices")
            if cvd is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)
                logger.info(f"  pinning to CUDA_VISIBLE_DEVICES={cvd}")

            vllm_kwargs = {
                "model": self.model_name,
                "gpu_memory_utilization": self.config.get("gpu_memory_utilization", 0.9),
                "tensor_parallel_size": self.config.get("tensor_parallel_size", 1),
                "trust_remote_code": True,
            }

            # Add max_model_len if specified in config
            if "max_model_len" in self.config:
                vllm_kwargs["max_model_len"] = self.config["max_model_len"]

            try:
                self.llm = LLM(**vllm_kwargs)
            finally:
                # Restore so the next model's pinning starts from a clean slate.
                if saved_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd

            self.sampling_params_class = SamplingParams
            logger.info(f"vLLM model loaded successfully: {self.model_name}")

        except ImportError:
            logger.error("vLLM not installed. Install with: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise

    def _create_sampling_params(
        self, temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        guided_json_schema: Optional[type[BaseModel]] = None
    ):
        """Create sampling parameters with optional guided JSON decoding"""
        params = {
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "top_p": self.top_p,
        }
        return self.sampling_params_class(**params)

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt with system prompt for chat models"""
        if system_prompt:
            # Llama-3 chat format
            formatted = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            return formatted
        else:
            return prompt

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        guided_json_schema: Optional[type[BaseModel]] = None,
    ) -> str:
        """Generate text from prompt with optional guided JSON decoding"""
        formatted_prompt = self._format_prompt(prompt, system_prompt)
        sampling_params = self._create_sampling_params(temperature, max_tokens, guided_json_schema)

        outputs = self.llm.generate([formatted_prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def batch_generate(
        self,
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate text for multiple prompts"""
        if system_prompts is None:
            system_prompts = [None] * len(prompts)

        formatted_prompts = [
            self._format_prompt(p, sp) for p, sp in zip(prompts, system_prompts)
        ]
        sampling_params = self._create_sampling_params(temperature, max_tokens)

        outputs = self.llm.generate(formatted_prompts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]


class OpenAIInterface(LLMInterface):
    """Interface for OpenAI API"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.client = None
        self._use_httpx_fallback = False
        self._api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY") or ""
        self._base_url = (config.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self._timeout = float(config.get("timeout", 120.0))
        self._initialize_openai()

    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI

            # Initialize client with custom base_url if provided
            self.client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            logger.info(
                f"OpenAI-compatible client initialized for model: {self.model_name} at {self._base_url}"
            )

        except ImportError:
            # Fallback path for OpenAI-compatible endpoints (e.g., local vLLM)
            self._use_httpx_fallback = True
            logger.warning(
                "OpenAI package unavailable; using direct HTTP fallback for OpenAI-compatible API."
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        guided_json_schema: Optional[type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        """Generate text from prompt"""
        if self._use_httpx_fallback:
            return self._generate_via_httpx(prompt, system_prompt, temperature, max_tokens)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            top_p=self.top_p,
        )

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_toks = getattr(usage, "prompt_tokens", None)
            completion_toks = getattr(usage, "completion_tokens", None)
            total_toks = getattr(usage, "total_tokens", None)
            self._record_usage(prompt_toks, completion_toks, total_toks)
        else:
            self._record_usage()

        return response.choices[0].message.content.strip()

    def _generate_via_httpx(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Fallback generation via direct HTTP call to /chat/completions."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "top_p": self.top_p,
        }

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/chat/completions"
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        self._record_usage(
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise ValueError(f"Unexpected OpenAI-compatible response format: {data}") from exc

    def batch_generate(
        self,
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate text for multiple prompts (sequential for OpenAI)"""
        if system_prompts is None:
            system_prompts = [None] * len(prompts)

        results = []
        for prompt, sys_prompt in zip(prompts, system_prompts):
            result = self.generate(prompt, sys_prompt, temperature, max_tokens)
            results.append(result)

        return results


class PortkeyInterface(LLMInterface):
    """Interface for Portkey AI Gateway (OpenAI-compatible)."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        try:
            from portkey_ai import Portkey
        except ImportError:
            raise ImportError("portkey-ai not installed. Install with: pip install portkey-ai")
        api_key = config.get("api_key") or os.getenv("PORTKEY_API_KEY")
        if not api_key:
            raise ValueError("Portkey API key not provided. Set api_key or PORTKEY_API_KEY.")
        self.client = Portkey(api_key=api_key)
        self.max_retries = config.get("max_retries", 8)
        self.retry_delay = config.get("retry_delay", 5.0)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        guided_json_schema: Optional[type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self._record_usage(
                        getattr(usage, "prompt_tokens", None),
                        getattr(usage, "completion_tokens", None),
                        getattr(usage, "total_tokens", None),
                    )
                else:
                    self._record_usage()
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise

    def batch_generate(
        self,
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        return [self.generate(p, s, temperature, max_tokens) for p, s in zip(prompts, system_prompts)]


class HuggingFaceInterface(LLMInterface):
    """Interface for HuggingFace Transformers"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self._initialize_huggingface()

    def _initialize_huggingface(self):
        """Initialize HuggingFace model"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info(f"Loading HuggingFace model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            logger.info(f"HuggingFace model loaded successfully: {self.model_name}")

        except ImportError:
            logger.error(
                "Transformers not installed. Install with: pip install transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        guided_json_schema: Optional[type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        """Generate text from prompt"""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from output
        generated_text = generated_text[len(full_prompt) :].strip()

        return generated_text

    def batch_generate(
        self,
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate text for multiple prompts"""
        if system_prompts is None:
            system_prompts = [None] * len(prompts)

        results = []
        for prompt, sys_prompt in zip(prompts, system_prompts):
            result = self.generate(prompt, sys_prompt, temperature, max_tokens)
            results.append(result)

        return results


def create_llm_interface(
    backend: str, model_name: str, config: Dict[str, Any]
) -> LLMInterface:
    """Factory function to create LLM interface based on backend"""
    backend = backend.lower()

    # Allow forcing stub backend via env for quick offline runs
    if backend == "vllm":
        return VLLMInterface(model_name, config)
    elif backend == "openai":
        return OpenAIInterface(model_name, config)
    elif backend == "huggingface":
        return HuggingFaceInterface(model_name, config)
    elif backend == "anthropic":
        return AnthropicInterface(model_name, config)
    elif backend == "portkey":
        return PortkeyInterface(model_name, config)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported: vllm, openai, huggingface, anthropic, portkey"
        )


# Utility functions for JSON parsing
def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract JSON from LLM output that might contain markdown or extra text
    """
    import re

    # Remove chat template artifacts
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'user<\|eot_id\|>', '', text)

    # Try to find JSON block in markdown code blocks
    # First, try to extract the content between ``` markers
    markdown_pattern = r"```(?:json)?\s*(.*?)\s*```"
    markdown_matches = re.finditer(markdown_pattern, text, re.DOTALL)
    for markdown_match in markdown_matches:
        content = markdown_match.group(1).strip()
        # Now find balanced JSON within the markdown block
        def find_balanced_json_simple(s):
            """Find a balanced JSON object in string"""
            stack = []
            start = None
            for i, c in enumerate(s):
                if c == '{':
                    if start is None:
                        start = i
                    stack.append(c)
                elif c == '}':
                    if stack:
                        stack.pop()
                        if not stack and start is not None:
                            try:
                                return json.loads(s[start:i+1])
                            except json.JSONDecodeError:
                                start = None
            return None

        result = find_balanced_json_simple(content)
        if result:
            return result

    # Try to find complete JSON objects by matching balanced braces
    # Look for the largest balanced JSON object
    def find_balanced_json(text):
        """Find a balanced JSON object in text"""
        stack = []
        start_idx = None

        for i, char in enumerate(text):
            if char == '{':
                if start_idx is None:
                    start_idx = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        # Found balanced JSON
                        try:
                            candidate = text[start_idx:i+1]
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            # Continue searching
                            start_idx = None
        return None

    result = find_balanced_json(text)
    if result:
        return result

    # Try parsing the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from text (len={len(text)}): First 300 chars: {text[:300]}... Last 200 chars: ...{text[-200:]}")
        logger.warning(f"JSON decode error: {e}")
        return None


def safe_json_parse(text: str, default: Optional[Dict] = None) -> Dict:
    """
    Safely parse JSON from text, return default if parsing fails
    """
    result = extract_json_from_text(text)
    if result is None:
        logger.warning("JSON parsing failed, returning default")
        return default if default is not None else {}
    return result
