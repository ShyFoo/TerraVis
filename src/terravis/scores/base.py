from abc import ABC, abstractmethod
from typing import Any

import torch
from PIL import Image

TEXT_FIRST_JUDGES = ("gemma-4-31b",)
"""Text-before-image judges (A/B-measured); a lowercased substring match that excludes gemma-4-26B-A4B."""


def order_chat_content(text_block: Any, image_block: Any, model_name: str | None) -> list[Any]:
    """Block order for the open-weight judges; unknown names are image-first."""
    if image_block is None:
        return [text_block]
    lowered = (model_name or "").lower()
    if any(pattern in lowered for pattern in TEXT_FIRST_JUDGES):
        return [text_block, image_block]
    return [image_block, text_block]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class ScoreModel(ABC):
    def __init__(
        self,
        model: torch.nn.Module | None = None,
        processor: Any = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype

    load_image = staticmethod(load_image)

    @abstractmethod
    def score(self, image: str, text: str | None) -> Any: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.score(*args, **kwargs)
