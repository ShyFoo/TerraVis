from typing import Any

import torch

from terravis.models.model_manager import ModelManager
from terravis.scores.registry import BuildContext, get_score_model
from terravis.utils import torch_dtype


class T2IScoreEvaluator:
    def __init__(
        self,
        score_mode: str,
        device: str | torch.device,
        model_weight_root: str,
        dtype: str,
        clip_model_name: str = "",
        vqa_model_names: str = "",
        api_provider: str = "",
        api_model_name: str = "",
        sampling_params: dict | None = None,
        vllm_tp_size: int = 0,
        model_manager: ModelManager | None = None,   # pass one to share a judge across evaluators
    ):
        self.score_mode = score_mode
        self.dtype = torch_dtype(dtype)
        self.spec = get_score_model(score_mode)
        if model_manager is None:
            model_manager = ModelManager(device=device, model_weight_root=model_weight_root, dtype=self.dtype)
        elif (str(model_manager.device), model_manager.dtype) != (str(device), self.dtype):
            raise ValueError(
                f"model_manager loads on {model_manager.device} in {model_manager.dtype}, this evaluator on "
                f"{device} in {self.dtype}; share one only between evaluators with the same device and dtype."
            )

        ctx = BuildContext(
            device=device,
            dtype=self.dtype,
            model_weight_root=model_weight_root,
            model_manager=model_manager,
            clip_model_name=clip_model_name,
            vqa_model_names=vqa_model_names,
            api_provider=api_provider,
            api_model_name=api_model_name,
            sampling_params=sampling_params or {},
            vllm_tp_size=vllm_tp_size,
        )
        self.scorer = self.spec.builder(ctx)

    @torch.inference_mode()
    def score(self, image: str, text: str | None = None) -> dict[str, dict[str, Any]]:
        if self.spec.category == "prompt-independent":
            return {self.score_mode: self.scorer(image=image)}
        if text is None:
            raise ValueError(f"score_mode '{self.score_mode}' is prompt-conditioned and requires a text prompt.")
        return {self.score_mode: self.scorer(image=image, text=text)}

    __call__ = score
