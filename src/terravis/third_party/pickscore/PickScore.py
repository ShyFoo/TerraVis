# SPDX-License-Identifier: MIT
# Copyright (c) 2021
# ==============================================================================
# Source Notice
#
# This file includes code adapted from:
#   PickScore
#   https://github.com/yuvalkirstain/PickScore  (MIT)
# ==============================================================================

import os
import torch
from PIL import Image
from typing import Tuple, Union
from transformers import AutoModel, AutoProcessor

# The only released PickScore checkpoint: a CLIP ViT-H/14 fine-tuned on the
# Pick-a-Pic v1 dataset. "pickapic_v1"/"pickapic_v2" are versions of the
# *dataset*; no PickScore_v2 model exists.
PICKSCORE_MODEL_REPO_ID = "yuvalkirstain/PickScore_v1"
# PickScore_v1 ships no processor config; the official code loads the
# processor of the CLIP-H base model it was fine-tuned from.
PICKSCORE_PROCESSOR_REPO_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def load_model_and_processor(
    model_weight_root: str,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> Tuple[torch.nn.Module, "AutoProcessor"]:
    """PickScore_v1 and its CLIP-H processor, both cached under ``<model_weight_root>/PickScore``."""
    pickscore_dir = os.path.join(model_weight_root, "PickScore")
    processor = AutoProcessor.from_pretrained(
        PICKSCORE_PROCESSOR_REPO_ID, cache_dir=pickscore_dir,
    )
    model = AutoModel.from_pretrained(
        PICKSCORE_MODEL_REPO_ID, torch_dtype=dtype, cache_dir=pickscore_dir,
        adapter_kwargs={"cache_dir": pickscore_dir},   # Auto*'s PEFT probe ignores cache_dir
    )
    model.to(device)
    model.eval()
    return model, processor


def _as_embedding(features) -> torch.Tensor:
    """transformers>=5 returns a ModelOutput whose ``pooler_output`` holds the
    projected embeddings; 4.x (the official code's target) returned the tensor."""
    return features if isinstance(features, torch.Tensor) else features.pooler_output


@torch.no_grad()
def compute_pickscore(
    model: torch.nn.Module,
    processor,
    prompt: str,
    image: Image.Image,
    device: Union[str, torch.device],
) -> float:
    """Official single-pair inference (the README's ``calc_probs`` before the
    multi-image softmax): scaled cosine similarity between the CLIP text and
    image embeddings, ``logit_scale.exp() * cos(text_emb, image_emb)``."""
    image_inputs = processor(
        images=[image], padding=True, truncation=True, max_length=77, return_tensors="pt",
    ).to(device)
    image_inputs["pixel_values"] = image_inputs["pixel_values"].to(model.dtype)
    text_inputs = processor(
        text=[prompt], padding=True, truncation=True, max_length=77, return_tensors="pt",
    ).to(device)

    image_embs = _as_embedding(model.get_image_features(**image_inputs))
    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    text_embs = _as_embedding(model.get_text_features(**text_inputs))
    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

    scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
    return float(scores[0])
