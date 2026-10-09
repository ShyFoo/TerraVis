import os
from typing import Any

import torch

from terravis.scores.base import ScoreModel
from terravis.scores.registry import register_score_model

_MUSIQ_DESCRIPTION = (
    "A technical image quality score predicted by MUSIQ. "
    "Range: [0, 100], higher is better."
)


class MUSIQScoreModel(ScoreModel):
    def __init__(self, model_weight_root: str, device: str | torch.device, dtype: torch.dtype):
        from terravis.third_party.musiq.MUSIQ import build_musiq_model

        super().__init__(device=device, dtype=dtype)
        self.model = build_musiq_model(
            model_weight_root=os.path.join(model_weight_root, "MUSIQ"),
            device=self.device, dtype=dtype,
        )

    @torch.inference_mode()
    def score(self, image: str, text: str | None = None) -> dict[str, Any]:
        from terravis.third_party.musiq.MUSIQ import load_image_as_tensor

        score = self.model(load_image_as_tensor(image=image, device=self.device, dtype=self.dtype)).item()
        return {"score": score, "score_description": _MUSIQ_DESCRIPTION}


def build_musiq_score(ctx) -> MUSIQScoreModel:
    return MUSIQScoreModel(
        model_weight_root=ctx.model_weight_root, device=ctx.device, dtype=ctx.dtype,
    )


register_score_model(
    name="musiq_score", category="prompt-independent",
    description=_MUSIQ_DESCRIPTION, builder=build_musiq_score,
    judge_label="musiq-koniq",
)
