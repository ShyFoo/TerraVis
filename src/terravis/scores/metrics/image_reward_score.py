import os
from typing import Any

import torch

from terravis.scores.base import ScoreModel
from terravis.scores.registry import register_score_model

_IMAGE_REWARD_DESCRIPTION = (
    "A human preference score predicted by ImageReward for a prompt-image pair. "
    "No fixed range (typically around [-2, 2]), higher is better."
)


class ImageRewardScoreModel(ScoreModel):
    """Paper: https://arxiv.org/abs/2304.05977"""

    def __init__(self, model_weight_root: str, device: str | torch.device, dtype: torch.dtype):
        from terravis.third_party.imagereward.ImageReward import ImageReward, download_imagereward_file

        super().__init__(device=device, dtype=dtype)
        image_reward_dir = os.path.join(model_weight_root, "ImageReward")
        ckpt_path = download_imagereward_file("ImageReward.pt", image_reward_dir)
        med_config_path = download_imagereward_file("med_config.json", image_reward_dir)

        model = ImageReward(med_config=med_config_path, device=device, cache_dir=image_reward_dir).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        model.eval()
        self.model = model

    @torch.inference_mode()
    def score(self, image: str, text: str) -> dict[str, Any]:
        return {
            "score": float(self.model.score(text, image)),
            "score_description": _IMAGE_REWARD_DESCRIPTION,
        }


def build_image_reward_score(ctx) -> ImageRewardScoreModel:
    return ImageRewardScoreModel(
        model_weight_root=ctx.model_weight_root, device=ctx.device, dtype=ctx.dtype,
    )


register_score_model(
    name="image_reward_score", category="prompt-conditioned",
    description=_IMAGE_REWARD_DESCRIPTION, builder=build_image_reward_score,
    judge_label="imagereward-v1.0",
)
