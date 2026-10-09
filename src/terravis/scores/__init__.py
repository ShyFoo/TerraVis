"""Import registers every module under metrics/; ``terravis.score_models`` plugins load on the first registry lookup."""

import pkgutil
from importlib import import_module

from terravis.scores import metrics
from terravis.scores.base import ScoreModel
from terravis.scores.registry import (
    BuildContext,
    ScoreSpec,
    get_score_model,
    list_score_models,
    register_score_model,
)

for module in pkgutil.iter_modules(metrics.__path__):
    import_module(f"{metrics.__name__}.{module.name}")


__all__ = [
    "BuildContext",
    "ScoreModel",
    "ScoreSpec",
    "get_score_model",
    "list_score_models",
    "register_score_model",
]
