"""Shared by the in-process and remote vLLM backends, which must feed a judge identically."""

import base64
import io
from typing import Any

from PIL import Image

from terravis.scores.base import order_chat_content


def image_to_data_url(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_chat_messages(
    text: str, pil_image: Image.Image | None, model_name: str | None,
) -> list[dict[str, Any]]:
    """One user turn; ``pil_image=None`` is text-only. Order per ``order_chat_content``, or a served
    VLM may reply "please provide the image"."""
    image_block = (
        None if pil_image is None
        else {"type": "image_url", "image_url": {"url": image_to_data_url(pil_image)}}
    )
    content = order_chat_content({"type": "text", "text": text}, image_block, model_name)
    return [{"role": "user", "content": content}]


def normalize_sampling_params(hf_params: dict) -> dict[str, Any]:
    """``do_sample`` False drops the sampling knobs and pins temperature to 0."""
    params = hf_params or {}
    canonical = {"max_tokens": params["max_new_tokens"]} if "max_new_tokens" in params else {}
    if params.get("do_sample"):
        canonical.update({k: params[k] for k in ("temperature", "top_p", "top_k", "min_p") if k in params})
    else:
        canonical["temperature"] = 0.0
    canonical.update({k: params[k] for k in ("presence_penalty", "frequency_penalty", "repetition_penalty")
                      if k in params})
    return canonical


def to_vllm_sampling_params(hf_params: dict, **overrides: Any) -> Any:
    from vllm import SamplingParams

    return SamplingParams(**{**normalize_sampling_params(hf_params), **overrides})


def to_openai_chat_kwargs(hf_params: dict) -> dict[str, Any]:
    """vLLM-only params (top_k, min_p, repetition_penalty) go in ``extra_body``."""
    kwargs = normalize_sampling_params(hf_params)
    extra_body = {k: kwargs.pop(k) for k in ("top_k", "min_p", "repetition_penalty") if k in kwargs}
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs
