from typing import Any

import torch

from terravis.scores.base import ScoreModel
from terravis.scores.registry import register_score_model

_RAHF_DESCRIPTION = (
    "Rich human-feedback scores predicted by RAHF (multi-head) for a prompt-image pair: "
    "'score' is the overall head, reported alongside the plausibility, alignment and "
    "aesthetics heads. Range: [0, 1], higher is better."
)


class RAHFScoreModel(ScoreModel):
    """Paper: https://arxiv.org/abs/2312.10240"""

    def __init__(self, model_weight_root: str, device: str | torch.device, dtype: torch.dtype):
        from terravis.third_party.rahf.RAHF import load_model_and_processor

        model, processor = load_model_and_processor(
            model_weight_root=model_weight_root, device=device, dtype=dtype,
        )
        super().__init__(model=model, processor=processor, device=device, dtype=dtype)

    @torch.inference_mode()
    def score(self, image: str, text: str) -> dict[str, Any]:
        from terravis.third_party.rahf.RAHF import clean_prompt

        pixel_values = self.processor(self.load_image(image), return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        scores = {
            name: float(value)
            for name, value in self.model(pixel_values, [clean_prompt(text)]).items()
        }
        return {
            "score": scores.pop("overall"),
            **scores,
            "score_description": _RAHF_DESCRIPTION,
        }


def build_rahf_score(ctx) -> RAHFScoreModel:
    return RAHFScoreModel(
        model_weight_root=ctx.model_weight_root, device=ctx.device, dtype=ctx.dtype,
    )


register_score_model(
    name="rahf_score", category="prompt-conditioned",
    description=_RAHF_DESCRIPTION, builder=build_rahf_score,
    judge_label="rahf-multihead",
)
