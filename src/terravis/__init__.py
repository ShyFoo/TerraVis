"""TerraVis: evaluating world-grounded visual consistency in T2I generated images."""

import importlib

__version__ = "1.0.0"

# Imported on first use: `import terravis.cli.*` must reach terravis.cli, which loads the root paths,
# before anything imports huggingface_hub.
_EXPORTS = {
    "BuildContext": "terravis.scores.registry",
    "ScoreModel": "terravis.scores.base",
    "ScoreSpec": "terravis.scores.registry",
    "get_score_model": "terravis.scores.registry",
    "list_score_models": "terravis.scores.registry",
    "register_score_model": "terravis.scores.registry",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'terravis' has no attribute {name!r}")
