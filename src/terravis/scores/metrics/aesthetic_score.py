import os
from typing import Any

import torch

from terravis.scores.base import ScoreModel
from terravis.scores.registry import register_score_model

_AESTHETIC_DESCRIPTION = (
    "An aesthetic score predicted by SigLIP + MLP head. "
    "Range: [1, 10], higher is better."
)

AESTHETIC_HEAD_URL = (
    "https://github.com/discus0434/aesthetic-predictor-v2-5/raw/main/"
    "models/aesthetic_predictor_v2_5.pth"
)


class SigLIPLAIONAestheticScoreModel(ScoreModel):
    """Aesthetic Predictor V2.5: https://github.com/discus0434/aesthetic-predictor-v2-5"""

    def __init__(self, model_weight_root: str, device: str | torch.device, dtype: torch.dtype):
        try:
            from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        except ImportError as exc:
            raise ImportError(
                "laion_aesthetic_score needs the optional AGPL-3.0 package: "
                'pip install "aesthetic-predictor-v2-5>=2024.12.18.1" (or pip install -e ".[aesthetic]" from the repo)'
            ) from exc
        aesthetic_dir = os.path.join(model_weight_root, "Aesthetic")
        head_path = os.path.join(aesthetic_dir, "aesthetic_predictor_v2_5.pth")
        if not os.path.exists(head_path):  # upstream would fetch a missing head into ~/.cache/torch instead
            os.makedirs(aesthetic_dir, exist_ok=True)
            torch.hub.download_url_to_file(AESTHETIC_HEAD_URL, head_path)
        model, processor = convert_v2_5_from_siglip(head_path, cache_dir=aesthetic_dir)
        super().__init__(model=model.to(device=device, dtype=dtype).eval(), processor=processor,
                         device=device, dtype=dtype)

    @torch.inference_mode()
    def score(self, image: str, text: str | None = None) -> dict[str, Any]:
        pixel_values = self.processor(images=self.load_image(image), return_tensors="pt")["pixel_values"].to(
            device=self.device, dtype=self.dtype,
        )
        score = self.model(pixel_values=pixel_values).logits.item()
        return {"score": score, "score_description": _AESTHETIC_DESCRIPTION}


def build_laion_aesthetic_score(ctx) -> SigLIPLAIONAestheticScoreModel:
    return SigLIPLAIONAestheticScoreModel(
        model_weight_root=ctx.model_weight_root, device=ctx.device, dtype=ctx.dtype,
    )


register_score_model(
    name="laion_aesthetic_score", category="prompt-independent",
    description=_AESTHETIC_DESCRIPTION, builder=build_laion_aesthetic_score,
    judge_label="siglip-so400m-patch14-384",
)
