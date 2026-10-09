import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    import torch

    from terravis.models.model_manager import ModelManager
    from terravis.scores.base import ScoreModel

logger = logging.getLogger(__name__)

Category = Literal["prompt-conditioned", "prompt-independent"]


@dataclass
class BuildContext:
    """Inputs for a score-model builder."""

    device: "str | torch.device"
    dtype: "torch.dtype"
    model_weight_root: str
    model_manager: "ModelManager"
    clip_model_name: str = ""
    vqa_model_names: str = ""
    api_provider: str = ""
    api_model_name: str = ""
    sampling_params: dict[str, Any] = field(default_factory=dict)
    vllm_tp_size: int = 0

    @property
    def use_api(self) -> bool:
        return bool(self.api_provider and self.api_model_name)


Builder = Callable[[BuildContext], "ScoreModel"]


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    category: Category
    description: str
    builder: Builder
    supports_api: bool = False
    supports_batch_api: bool = False
    supports_multi_device: bool = False
    judge_label: str = ""   # built-in weights: the judge= label in result file names, no model argument needed


_REGISTRY: dict[str, ScoreSpec] = {}


def register_score_model(
    name: str,
    *,
    category: Category,
    description: str,
    builder: Builder,
    supports_api: bool = False,
    supports_batch_api: bool = False,
    supports_multi_device: bool = False,
    judge_label: str = "",
) -> ScoreSpec:
    if category not in get_args(Category):
        raise ValueError(f"{name}: category must be one of {get_args(Category)}, got {category!r}.")
    if name in _REGISTRY:
        logger.warning("Overwriting existing score_model registration for '%s'", name)
    spec = ScoreSpec(
        name=name,
        category=category,
        description=description,
        builder=builder,
        supports_api=supports_api,
        supports_batch_api=supports_batch_api,
        supports_multi_device=supports_multi_device,
        judge_label=judge_label,
    )
    _REGISTRY[name] = spec
    return spec


_plugins_lock = threading.RLock()  # re-entrant: a plugin may look up the registry while it loads
_plugins_loaded = False


def _load_plugins() -> None:
    """On first lookup, not at import: a plugin's ``from terravis import ...`` would hit a half-initialised package."""
    global _plugins_loaded
    with _plugins_lock:
        if _plugins_loaded:
            return
        _plugins_loaded = True
        for ep in entry_points(group="terravis.score_models"):
            try:
                ep.load()
            except Exception as exc:
                logger.warning("Failed to load terravis score plugin %r from %r: %s", ep.name, ep.value, exc)


def get_score_model(name: str) -> ScoreSpec:
    _load_plugins()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown score_mode '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_score_models() -> list[str]:
    _load_plugins()
    return sorted(_REGISTRY)
