"""One judge engine per backend: ``infer(text, image=None, **overrides) -> str`` (HF takes only ``max_tokens`` and
``temperature``; other overrides raise TypeError), optional ``infer_many(texts, image)`` and ``first_token(question,
image) -> (answer, [(token, probability)])``; ``image`` is a path. ``TERRAVIS_REMOTE_VLLM_ENDPOINTS`` maps HF id ->
base URL (``{"google/gemma-4-31B-it": "http://host:8000/v1"}``); a server must serve the HF id as its model name."""

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Any

import torch
from transformers import GenerationConfig

from terravis.scores.base import load_image, order_chat_content
from terravis.scores.vlm_chat import build_chat_messages, to_openai_chat_kwargs, to_vllm_sampling_params

# Instruct mode on every backend; Qwen's chat template thinks by default.
_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def _resolve_device(device: Any) -> torch.device:
    if device == "balanced":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class HFJudge:
    infer_many = None  # sequential: keeps the per-question progress bar

    def __init__(self, model: torch.nn.Module, processor: Any, device: Any, dtype: torch.dtype,
                 sampling_params: dict, model_name: str):
        self.model, self.processor, self.model_name = model, processor, model_name
        self.device, self.dtype = _resolve_device(device), dtype
        self.sampling_params = sampling_params
        self.tokenizer = getattr(processor, "tokenizer", processor)

    @cached_property
    def _yes_ids(self) -> list[int]:
        ids = [tid for tid in range(self.tokenizer.vocab_size) if self.tokenizer.decode([tid]).strip().lower() == "yes"]
        if not ids:
            raise ValueError(f"{self.model_name}: no vocabulary token decodes to 'yes'.")
        return ids

    def _generate(self, text: str, image: str | None, max_tokens: int | None = None, **generate_kwargs):
        content = order_chat_content({"type": "text", "text": text},
                                     None if image is None else {"type": "image"}, self.model_name)
        chat_text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, **_CHAT_TEMPLATE_KWARGS,
        )
        if image is None:
            inputs = self.processor(text=chat_text, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(load_image(image), chat_text, return_tensors="pt").to(self.dtype).to(self.device)
        params = {"do_sample": False, **self.sampling_params, **generate_kwargs}
        if max_tokens:
            params["max_new_tokens"] = max_tokens
        out = self.model.generate(**inputs, generation_config=GenerationConfig(**params))
        sequences = getattr(out, "sequences", out)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, sequences)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip(), out

    @torch.inference_mode()
    def infer(self, text: str, image: str | None = None, max_tokens: int | None = None,
              temperature: float | None = None) -> str:
        # As on vLLM: a temperature override samples with no top_p / top_k.
        sampling = ({} if temperature is None
                    else {"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0})
        return self._generate(text, image, max_tokens, **sampling)[0]

    @torch.inference_mode()
    def first_token(self, question: str, image: str) -> tuple[str, list[tuple[str, float]]]:
        answer, out = self._generate(question, image, output_scores=True, return_dict_in_generate=True)
        probs = torch.softmax(out.scores[0], dim=-1)
        return answer, [("yes", probs[0, tid].item()) for tid in self._yes_ids]


class VLLMJudge:
    def __init__(self, llm: Any, model_name: str, sampling_params: dict):
        self.llm, self.model_name, self.sampling_params = llm, model_name, sampling_params
        self._default = to_vllm_sampling_params(sampling_params)

    def _chat(self, texts: list[str], image: str | None, sampling: Any) -> list[Any]:
        pil_image = None if image is None else load_image(image)
        outputs = self.llm.chat(
            [build_chat_messages(text, pil_image, self.model_name) for text in texts],
            sampling_params=sampling,
            chat_template_kwargs=_CHAT_TEMPLATE_KWARGS,
            use_tqdm=False,
        )
        return [output.outputs[0] for output in outputs]

    def infer(self, text: str, image: str | None = None, **overrides: Any) -> str:
        sampling = to_vllm_sampling_params(self.sampling_params, **overrides) if overrides else self._default
        return self._chat([text], image, sampling)[0].text.strip()

    def infer_many(self, texts: list[str], image: str) -> list[str]:
        return [output.text.strip() for output in self._chat(texts, image, self._default)]

    def first_token(self, question: str, image: str) -> tuple[str, list[tuple[str, float]]]:
        # Greedy, like the other backends; vLLM logprobs are pre-temperature by default (logprobs_mode=raw_logprobs).
        params = {**self.sampling_params, "do_sample": False}
        params.setdefault("max_new_tokens", 8)
        output = self._chat([question], image, to_vllm_sampling_params(params, logprobs=20))[0]
        first = output.logprobs[0] if output.logprobs else {}
        return output.text.strip(), [(lp.decoded_token or "", math.exp(lp.logprob)) for lp in first.values()]


class RemoteVLLMJudge:
    def __init__(self, model_name: str, base_url: str, sampling_params: dict):
        from openai import OpenAI

        self.model_name = model_name
        self._chat_kwargs = to_openai_chat_kwargs(sampling_params)
        self._chat_kwargs.setdefault("extra_body", {})["chat_template_kwargs"] = _CHAT_TEMPLATE_KWARGS
        self.client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=600, max_retries=3)

    def _create(self, text: str, pil_image: Any, **kwargs: Any) -> Any:
        return self.client.chat.completions.create(
            model=self.model_name, messages=build_chat_messages(text, pil_image, self.model_name), **kwargs
        )

    @staticmethod
    def _answer(resp: Any) -> str:
        return (resp.choices[0].message.content or "").strip()

    def infer(self, text: str, image: str | None = None, **overrides: Any) -> str:
        pil_image = None if image is None else load_image(image)
        return self._answer(self._create(text, pil_image, **{**self._chat_kwargs, **overrides}))

    def infer_many(self, texts: list[str], image: str) -> list[str]:
        if not texts:  # ThreadPoolExecutor rejects max_workers=0
            return []
        pil_image = load_image(image)
        with ThreadPoolExecutor(max_workers=len(texts)) as pool:
            return list(pool.map(lambda text: self._answer(self._create(text, pil_image, **self._chat_kwargs)), texts))

    def first_token(self, question: str, image: str) -> tuple[str, list[tuple[str, float]]]:
        kwargs = {"max_tokens": 8, **self._chat_kwargs, "temperature": 0.0}
        choice = self._create(question, load_image(image), logprobs=True, top_logprobs=20, **kwargs).choices[0]
        tokens = choice.logprobs.content if (choice.logprobs and choice.logprobs.content) else []
        top = tokens[0].top_logprobs if tokens else []
        return (choice.message.content or "").strip(), [(t.token, math.exp(t.logprob)) for t in top]


ENDPOINTS_ENV = "TERRAVIS_REMOTE_VLLM_ENDPOINTS"


def remote_target(model_name: str) -> dict[str, str] | None:
    """``{"model_name", "base_url"}`` of the judge; ``None`` when the env var is unset."""
    raw = os.environ.get(ENDPOINTS_ENV, "")
    if not raw:
        return None
    try:
        endpoints = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse {ENDPOINTS_ENV} as JSON: {e}\nGot: {raw!r}") from e
    if not isinstance(endpoints, dict) or not endpoints:
        raise RuntimeError(f"{ENDPOINTS_ENV} must be a non-empty JSON object.")
    if model_name not in endpoints:
        raise RuntimeError(
            f"No remote vLLM endpoint configured for model {model_name!r}. "
            f"Known endpoints: {sorted(endpoints.keys())}"
        )
    return {"model_name": model_name, "base_url": endpoints[model_name]}


def build_judge(ctx: Any, model_name: str, sampling_params: dict):
    """Remote vLLM if ``TERRAVIS_REMOTE_VLLM_ENDPOINTS`` is set (it must list the judge), else local vLLM, else HF."""
    target = remote_target(model_name)
    if target is not None:
        return RemoteVLLMJudge(**target, sampling_params=sampling_params)
    if ctx.vllm_tp_size > 0:
        llm = ctx.model_manager.get_vllm_model(model_name, tensor_parallel_size=ctx.vllm_tp_size)["llm"]
        return VLLMJudge(llm, model_name, sampling_params)
    components = ctx.model_manager.get_vqa_model(model_name)
    return HFJudge(components["model"], components["processor"], ctx.device, ctx.dtype, sampling_params, model_name)
