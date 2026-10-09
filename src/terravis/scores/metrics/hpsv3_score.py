import os
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor

from terravis.scores.base import ScoreModel
from terravis.scores.registry import register_score_model
from terravis.third_party.hpsv3.HPSV3Qwen2VL import (
    HPSV3_INSTRUCTION,
    HPSV3_PROMPT_WITH_SPECIAL_TOKEN,
    Qwen2VLRewardModelBT,
    download_hpsv3_checkpoint,
    load_and_resize_image,
    move_to_device,
)
from terravis.utils import use_flash_attn

_HPSV3_DESCRIPTION = (
    "A human preference score predicted by HPSv3 for a prompt-image pair. "
    "No fixed range (typically around [0, 15]), higher is better."
)


def _remap_hpsv3_state_dict(sd):
    """transformers<5 Qwen2-VL keys -> 5.x ``model.visual.*`` / ``model.language_model.*``; load_state_dict
    bypasses from_pretrained's own renaming."""
    def remap(k):
        if k.startswith("visual."):
            return "model." + k
        if k.startswith("model."):
            return "model.language_model." + k[len("model."):]
        return k
    return {remap(k): v for k, v in sd.items()}


_BASE_MODEL_REPO_ID = "Qwen/Qwen2-VL-7B-Instruct"
_PIXELS = 256 * 28 * 28


class HPSv3ScoreModel(ScoreModel):
    """Paper: https://arxiv.org/abs/2508.03789"""

    def __init__(self, model_weight_root: str, device: str | torch.device, dtype: torch.dtype):
        hpsv3_dir = os.path.join(model_weight_root, "HPSv3")
        checkpoint_path = download_hpsv3_checkpoint(save_dir=hpsv3_dir)

        processor = AutoProcessor.from_pretrained(
            _BASE_MODEL_REPO_ID, padding_side="right", cache_dir=hpsv3_dir,
        )
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        model = Qwen2VLRewardModelBT.from_pretrained(
            _BASE_MODEL_REPO_ID,
            output_dim=2,
            reward_token="special",
            special_token_ids=processor.tokenizer.convert_tokens_to_ids(special_tokens),
            torch_dtype=dtype,
            attn_implementation="flash_attention_2" if use_flash_attn(dtype) else "sdpa",
            cache_dir=hpsv3_dir,
            rm_head_type="ranknet",
        )
        model.resize_token_embeddings(len(processor.tokenizer))

        state_dict = load_file(checkpoint_path, device="cpu")
        model.load_state_dict(_remap_hpsv3_state_dict(state_dict), strict=True)
        model.eval()
        model.to(device)

        super().__init__(model=model, processor=processor, device=device, dtype=dtype)

    def prepare_inputs(self, image: str, text: str) -> dict[str, Any]:
        user_text = HPSV3_INSTRUCTION.format(text_prompt=text) + HPSV3_PROMPT_WITH_SPECIAL_TOKEN
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_text}],
        }]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_obj = load_and_resize_image(image=image, min_pixels=_PIXELS, max_pixels=_PIXELS)
        batch = self.processor(
            text=[prompt], images=[image_obj], padding=True,
            return_tensors="pt", return_mm_token_type_ids=True,
        )
        return move_to_device(batch, self.device)

    @torch.inference_mode()
    def score(self, image: str, text: str) -> dict[str, Any]:
        # The head outputs [mu, sigma]; mu is the official HPSv3 score.
        outputs = self.model(return_dict=True, **self.prepare_inputs(image=image, text=text))["logits"]
        mu, sigma = outputs[0].tolist()
        return {
            "score": mu,
            "uncertainty": sigma,
            "score_entries": {
                "raw_output": [mu, sigma],
                "score_description": _HPSV3_DESCRIPTION,
            },
        }


def build_hpsv3_score(ctx) -> HPSv3ScoreModel:
    return HPSv3ScoreModel(
        model_weight_root=ctx.model_weight_root, device=ctx.device, dtype=ctx.dtype,
    )


register_score_model(
    name="hpsv3_score", category="prompt-conditioned",
    description=_HPSV3_DESCRIPTION, builder=build_hpsv3_score,
    judge_label="qwen2-vl-7b-instruct",
)
