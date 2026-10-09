from typing import Any

import torch

from terravis.scores.base import ScoreModel
from terravis.scores.registry import register_score_model

_CLIP_DESCRIPTION = (
    "A simple alignment score between the text prompt and image by computing their cosine similarity. "
    "Range: [0, 1], higher is better."
)


class CLIPScoreModel(ScoreModel):
    def __init__(self, model, processor, tokenizer, device, dtype):
        super().__init__(model=model, processor=processor, device=device, dtype=dtype)
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def score(self, image: str, text: str) -> dict[str, Any]:
        image_tensor = self.processor(self.load_image(image)).to(self.dtype).to(self.device).unsqueeze(0)
        img_feats = torch.nn.functional.normalize(self.model.encode_image(image_tensor), dim=-1)
        txt_feats = torch.nn.functional.normalize(self.model.encode_text(self.tokenizer(text).to(self.device)), dim=-1)
        sim = torch.sum(img_feats * txt_feats, dim=-1).item()
        return {"score": sim, "score_description": _CLIP_DESCRIPTION}


def build_clip_score(ctx) -> CLIPScoreModel:
    components = ctx.model_manager.get_clip_model(ctx.clip_model_name)
    return CLIPScoreModel(
        model=components["model"],
        processor=components["processor"],
        tokenizer=components["tokenizer"],
        device=ctx.device,
        dtype=ctx.dtype,
    )


register_score_model(
    name="clip_score", category="prompt-conditioned",
    description=_CLIP_DESCRIPTION, builder=build_clip_score,
)
