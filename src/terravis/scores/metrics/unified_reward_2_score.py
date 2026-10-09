"""UnifiedReward-2.0 pointwise T2I scoring, reimplemented from CodeGoat24/UnifiedReward
``inference_qwen/UnifiedReward-2.0-inference/point_score_ACS_image_generation.py`` (arXiv:2503.05236)."""

import os
import re
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from terravis.scores.base import ScoreModel
from terravis.scores.registry import register_score_model
from terravis.utils import use_flash_attn

_UNIFIED_REWARD_2_DESCRIPTION = (
    "Pointwise scores predicted by UnifiedReward-2.0 (Qwen3.5-9B) for a prompt-image pair: "
    "'score' is the mean of the alignment, coherence and style scores, reported alongside "
    "each. Range: [1, 5], higher is better."
)

UNIFIED_REWARD_2_REPO_ID = "CodeGoat24/UnifiedReward-2.0-qwen35-9b"

# Verbatim official prompt (UnifiedReward, MIT, Copyright (c) 2025 Yibin Wang (王逸彬)); the image precedes it
# in the chat content.
_PROMPT = (
    "You are presented with a generated image and its associated text caption. "
    "Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n"
    "Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
    "- Alignment Score: How well the image matches the caption in terms of content.\n"
    "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
    "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
    "Output your evaluation using the format below:\n\n"
    "Alignment Score (1-5): X\n"
    "Coherence Score (1-5): Y\n"
    "Style Score (1-5): Z\n\n"
    "Your task is provided as follows:\n"
    "Text Caption: [{prompt}]"
)

_AXES = ("Alignment", "Coherence", "Style")
_SCORE_LINE = re.compile(r"(Alignment|Coherence|Style) Score \(1-5\):\s*([-+]?\d+(?:\.\d+)?)")


def parse_acs_scores(output: str) -> dict[str, float] | None:
    """{'alignment': 2.69, 'coherence': ..., 'style': ...}; None unless all three lines are present."""
    found = {axis: float(value) for axis, value in _SCORE_LINE.findall(output)}
    if any(axis not in found for axis in _AXES):
        return None
    return {axis.lower(): found[axis] for axis in _AXES}


class UnifiedReward2ScoreModel(ScoreModel):
    def __init__(self, model_weight_root: str, device: str | torch.device, dtype: torch.dtype):
        weights_dir = os.path.join(model_weight_root, "UnifiedReward")
        processor = AutoProcessor.from_pretrained(UNIFIED_REWARD_2_REPO_ID, cache_dir=weights_dir)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            UNIFIED_REWARD_2_REPO_ID, dtype=dtype, device_map=device, cache_dir=weights_dir,
            attn_implementation="flash_attention_2" if use_flash_attn(dtype) else "sdpa",
        )
        model.eval()
        super().__init__(model=model, processor=processor, device=device, dtype=dtype)

    @torch.inference_mode()
    def score(self, image: str, text: str) -> dict[str, Any]:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": _PROMPT.format(prompt=text)},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        # Greedy: the official client sends do_sample=False, which vLLM drops (its server samples).
        gen_ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        output = self.processor.batch_decode(
            gen_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )[0].strip()
        scores = parse_acs_scores(output)
        if scores is None:
            return {
                "score": "n/a",
                "score_entries": {"raw_output": output, "score_description": _UNIFIED_REWARD_2_DESCRIPTION},
            }
        return {
            "score": sum(scores.values()) / len(scores),
            **scores,
            "score_description": _UNIFIED_REWARD_2_DESCRIPTION,
        }


def build_unified_reward_2_score(ctx) -> UnifiedReward2ScoreModel:
    return UnifiedReward2ScoreModel(
        model_weight_root=ctx.model_weight_root, device=ctx.device, dtype=ctx.dtype,
    )


register_score_model(
    name="unified_reward_2_score", category="prompt-conditioned",
    description=_UNIFIED_REWARD_2_DESCRIPTION, builder=build_unified_reward_2_score,
    supports_multi_device=True,
    judge_label="unifiedreward-2.0-qwen35-9b",
)
