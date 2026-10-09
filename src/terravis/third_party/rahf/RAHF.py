# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# ==============================================================================
# Source Notice
#
# This file includes code adapted from:
#   RichHF (the first author's implementation of the CVPR'24 paper
#   "Rich Human Feedback for Text-to-Image Generation")
#   https://github.com/youweiliang/RichHF
# Noncommercial use only: PolyForm Noncommercial License 1.0.0,
#   https://polyformproject.org/licenses/noncommercial/1.0.0; the checkpoint is CC BY-NC 4.0.
# Required Notice: Copyright 2025 Youwei Liang
# ==============================================================================

import os
import re
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    T5ForConditionalGeneration,
    ViTModel,
)

RAHF_VIT_MODEL_ID = "google/vit-large-patch16-384"
RAHF_T5_MODEL_ID = "google-t5/t5-base"
# The released multi-head RAHF checkpoint (CC BY-NC 4.0) from the RichHF repo.
RAHF_CHECKPOINT_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1-jKfmpyGtJ0UAgEQ23zylRsmQ82qigzB&export=download&confirm=t"
)
RAHF_CHECKPOINT_FILENAME = "rahf_model.pt"

# RichHF-18K strips punctuation from prompts before tokenization (upstream
# get_dataset.py builds ``clean_prompt`` exactly this way), so the checkpoint
# only ever saw cleaned prompts and inference must apply the same normalization.
_PROMPT_DELIMITERS = re.compile("|".join(map(re.escape, ',.?!":; ')))


def clean_prompt(prompt: str) -> str:
    return " ".join(token for token in _PROMPT_DELIMITERS.split(prompt) if token)


class LayerNorm(nn.Module):
    """T5-style LayerNorm over the channel dimension (no bias, no mean subtraction)."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a feature map of shape: batch_size x n_channels x h x w
        var = x.square().mean(dim=1, keepdim=True)
        return x * (var + 1e-8).rsqrt() * self.scale


class HeatmapPredictor(nn.Module):
    """Conv decoder that upsamples the 24x24 feature map to a [0, 1] heatmap."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 768, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            LayerNorm(768),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            LayerNorm(384),
        )

        self.deconv_layers = nn.ModuleList()
        self.conv_layers2 = nn.ModuleList()
        in_channels = 384
        for out_channels in [768, 384, 384, 192]:
            self.deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels, out_channels,
                        kernel_size=3, stride=2, padding=1, output_padding=1,
                    ),
                    LayerNorm(out_channels),
                    nn.ReLU(),
                )
            )
            self.conv_layers2.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
                    LayerNorm(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
                    LayerNorm(out_channels),
                )
            )
            in_channels = out_channels

        self.relu = nn.ReLU()
        self.last_conv1 = nn.Conv2d(in_channels, 192, kernel_size=3, stride=1, padding="same")
        self.last_conv2 = nn.Conv2d(192, 1, kernel_size=3, stride=1, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        for deconv, conv in zip(self.deconv_layers, self.conv_layers2):
            x = deconv(x)
            identity = x
            x = conv(x)
            x = x + identity
            x = self.relu(x)

        x = self.last_conv1(x)
        x = self.relu(x)
        x = self.last_conv2(x)
        x = self.sigmoid(x)  # (batch_size, 1, height, width)
        return x.squeeze(1)


class ScorePredictor(nn.Module):
    """Conv + MLP head that maps the 24x24 feature map to a [0, 1] score."""

    def __init__(self, n_channels: int, n_patches: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 2),
            nn.ReLU(),
            nn.Conv2d(n_channels // 2, n_channels // 4, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 4),
            nn.ReLU(),
            nn.Conv2d(n_channels // 4, n_channels // 8, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 8),
            nn.ReLU(),
            nn.Conv2d(n_channels // 8, n_channels // 16, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 16),
            nn.ReLU(),
            nn.Conv2d(n_channels // 16, n_channels // 64, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 64),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(n_channels // 64 * n_patches, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_output = self.conv_layers(x).flatten(1)
        return self.linear_layers(conv_output)


class RAHF(nn.Module):
    """Multi-head RAHF fixed to the released checkpoint: ViT-large patch tokens
    are projected to the T5 width, concatenated with the tokenized prompt and
    re-encoded by the T5 encoder; one ScorePredictor per feedback type reads
    the re-encoded visual tokens. Forward returns only the four [0, 1] scores;
    the heatmap heads and the T5 decoder (misaligned-token generation) are kept
    as modules so the checkpoint loads with strict=True, but are never run.
    """

    def __init__(
        self,
        model_weight_root: str,
        score_types: Tuple[str, ...] = ("plausibility", "alignment", "aesthetics", "overall"),
        heatmap_types: Tuple[str, ...] = ("implausibility", "misalignment"),
        patch_size: int = 16,
        image_size: int = 384,
    ):
        super().__init__()
        self.n_patches = image_size // patch_size

        # Backbones are built from config only: the checkpoint holds every weight.
        vit_config = AutoConfig.from_pretrained(RAHF_VIT_MODEL_ID, cache_dir=model_weight_root)
        t5_config = AutoConfig.from_pretrained(RAHF_T5_MODEL_ID, cache_dir=model_weight_root)
        self.vit = ViTModel(vit_config)
        self.t5 = T5ForConditionalGeneration(t5_config)
        self.tokenizer = AutoTokenizer.from_pretrained(RAHF_T5_MODEL_ID, cache_dir=model_weight_root)

        self.visual_token_projection = nn.Linear(vit_config.hidden_size, t5_config.d_model)

        n_channels = t5_config.d_model
        self.heatmap_predictor = nn.ModuleDict(
            {hm: HeatmapPredictor(n_channels) for hm in heatmap_types}
        )
        self.score_predictor = nn.ModuleDict(
            {sc: ScorePredictor(n_channels, self.n_patches ** 2) for sc in score_types}
        )

    def forward(self, image: torch.Tensor, caption: List[str]) -> Dict[str, torch.Tensor]:
        visual_tokens = self.vit(pixel_values=image).last_hidden_state
        batch_size = visual_tokens.shape[0]
        visual_tokens = self.visual_token_projection(visual_tokens)

        input_ids = self.tokenizer(
            caption, return_tensors="pt", padding=True, truncation=True,
        ).input_ids.to(visual_tokens.device)
        textual_embeddings = self.t5.encoder.embed_tokens(input_ids)
        concatenated_tokens = torch.cat([visual_tokens, textual_embeddings], dim=1)
        encoded = self.t5.encoder(inputs_embeds=concatenated_tokens).last_hidden_state

        # Re-encoded visual tokens without the ViT CLS token, as a 2D feature map.
        feature_map = (
            encoded[:, 1:visual_tokens.size(1), :]
            .transpose(1, 2)
            .view(batch_size, -1, self.n_patches, self.n_patches)
        )
        return {sc: predictor(feature_map).flatten() for sc, predictor in self.score_predictor.items()}


def load_model_and_processor(
    model_weight_root: str,
    device: Union[str, torch.device],
    dtype: torch.dtype,
):
    """Build RAHF, load the released checkpoint, and return (model, ViT image
    processor). Every RAHF weight — the ViT/T5 backbones and the released
    multi-head checkpoint — is kept under the score's dedicated
    ``<model_weight_root>/RAHF`` folder."""
    rahf_dir = os.path.join(model_weight_root, "RAHF")
    processor = AutoImageProcessor.from_pretrained(RAHF_VIT_MODEL_ID, cache_dir=rahf_dir)
    model = RAHF(model_weight_root=rahf_dir)

    state_dict = torch.hub.load_state_dict_from_url(
        RAHF_CHECKPOINT_URL,
        model_dir=rahf_dir,
        map_location="cpu",
        file_name=RAHF_CHECKPOINT_FILENAME,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)

    model.to(device=device, dtype=dtype)
    model.eval()
    return model, processor
