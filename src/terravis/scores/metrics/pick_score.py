from typing import Any

import torch

from terravis.scores.base import ScoreModel
from terravis.scores.registry import register_score_model
from terravis.third_party.pickscore.PickScore import compute_pickscore, load_model_and_processor

_PICKSCORE_DESCRIPTION = (
    "A human preference score predicted by PickScore (CLIP-H trained on Pick-a-Pic v1) "
    "for a prompt-image pair. No fixed range (typically around [20, 30]), higher is better."
)


class PickScoreModel(ScoreModel):
    """Paper: https://arxiv.org/abs/2305.01569"""

    def __init__(self, model_weight_root: str, device: str | torch.device, dtype: torch.dtype):
        model, processor = load_model_and_processor(
            model_weight_root=model_weight_root, device=device, dtype=dtype,
        )
        super().__init__(model=model, processor=processor, device=device, dtype=dtype)

    @torch.inference_mode()
    def score(self, image: str, text: str) -> dict[str, Any]:
        score = compute_pickscore(
            self.model, self.processor,
            prompt=text, image=self.load_image(image), device=self.device,
        )
        return {"score": score, "score_description": _PICKSCORE_DESCRIPTION}


def build_pick_score(ctx) -> PickScoreModel:
    return PickScoreModel(
        model_weight_root=ctx.model_weight_root, device=ctx.device, dtype=ctx.dtype,
    )


register_score_model(
    name="pick_score", category="prompt-conditioned",
    description=_PICKSCORE_DESCRIPTION, builder=build_pick_score,
    judge_label="pickscore-v1",
)
