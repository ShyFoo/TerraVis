# SPDX-License-Identifier: MIT AND Apache-2.0
# Copyright (c) 2024 HPSv3 Team
"""
A minimal local inference implementation of HPSv3, reproducing the
official model architecture and loading weights from the official released checkpoint.

This file includes code adapted from:
    https://github.com/MizzenAI/HPSv3  (MIT)
The image-resizing helpers come from qwen-vl-utils (https://github.com/QwenLM/Qwen2-VL, Apache-2.0)
by way of HPSv3.
"""

import os
import math
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from typing import Union, Optional, List, Any
from transformers import Qwen2VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature


def download_hpsv3_checkpoint(
    save_dir: str,
    repo_id: str = "MizzenAI/HPSv3",
    filename: str = "HPSv3.safetensors",
) -> str:
    """
    Download the official HPSv3 reward checkpoint into ``save_dir`` (the score's
    dedicated ``<model_weight_root>/HPSv3`` folder).
    """
    os.makedirs(save_dir, exist_ok=True)

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=save_dir,
    )

# ------------------------------------------------------------------------------
# Official HPSv3 prompt pieces extracted from the repo and simplified for inference
# ------------------------------------------------------------------------------

HPSV3_INSTRUCTION = """You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference.
Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best.

**Visual Quality:**
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

**Text Alignment:**
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance:** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency:** Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity:** Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence:** Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.

Textual prompt - {text_prompt}
"""

HPSV3_PROMPT_WITH_SPECIAL_TOKEN = " Please provide the overall ratings of this image: <|Reward|> END "

# ------------------------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------------------------

def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 256 * 28 * 28,
    max_ratio: int = 200,
) -> tuple[int, int]:
    """
    Resize logic adapted from the official HPSv3 image preprocessing path.
    """
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"Absolute aspect ratio must be smaller than {max_ratio}, "
            f"got {max(height, width) / min(height, width)}."
        )

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def load_and_resize_image(
    image: Union[str, Image.Image],
    min_pixels: int,
    max_pixels: int,
) -> Image.Image:
    """
    Load an image from a local path or a PIL image and resize it similarly
    to the official HPSv3 preprocessing.
    """
    if isinstance(image, str):
        image_obj = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image_obj = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    width, height = image_obj.size
    resized_height, resized_width = _smart_resize(
        height=height,
        width=width,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return image_obj.resize((resized_width, resized_height), Image.BICUBIC)


def move_to_device(obj: Any, device: Union[str, torch.device]) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, BatchFeature):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj

# ------------------------------------------------------------------------------
# Minimal local reward model extracted from the official HPSv3 implementation
# ------------------------------------------------------------------------------

class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
    """
    Minimal reward-model wrapper extracted from HPSv3:
    Qwen2-VL hidden states + reward head.
    """

    def __init__(
        self,
        config,
        output_dim: int = 2,
        reward_token: str = "special",
        special_token_ids: Optional[List[int]] = None,
        rm_head_type: str = "ranknet",
    ):
        super().__init__(config)

        self.output_dim = output_dim
        self.reward_token = reward_token
        self.special_token_ids = special_token_ids

        # Qwen2VLConfig is a composite config.
        # The language-model hidden size is stored in config.text_config.hidden_size.
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
                hidden_size = config.text_config.hidden_size
            else:
                raise AttributeError(
                    "Cannot find hidden_size in config. Expected config.hidden_size "
                    "or config.text_config.hidden_size."
                )

        if rm_head_type == "ranknet":
            self.rm_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.05),
                torch.nn.Linear(1024, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, output_dim),
            )
        else:
            self.rm_head = torch.nn.Linear(hidden_size, output_dim, bias=False)

        # Official code keeps reward head in fp32.
        self.rm_head.to(torch.float32)

        if self.special_token_ids is not None:
            self.reward_token = "special"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # 1. Core fix: directly call unified self.model, completely resolving the 3D mRoPE loss issue
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mm_token_type_ids=mm_token_type_ids,
            return_dict=True,
        )

        hidden_states = outputs[0]  # [B, L, D]
        logits = self.rm_head(hidden_states.to(torch.float32))  # [B, L, 2]

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        # 2. Extract special Reward Token logits
        if self.reward_token == "special":
            if input_ids is None:
                raise ValueError("input_ids must be provided when reward_token='special'.")

            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask |= (input_ids == special_token_id)

            pooled_logits = logits[special_token_mask].view(batch_size, -1)
        else:
            sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1) % input_ids.shape[-1]
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        return {"logits": pooled_logits}
