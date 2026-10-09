"""Model registry backed by models.yaml. short_name names files; display_name goes in stored
metadata and must invert back to short_name (frozen)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml


def short_name(model_id: str) -> str:
    """'google/gemma-4-31B-it' -> 'gemma-4-31b-it'; API names just lowercase."""
    return model_id.rsplit("/", 1)[-1].lower()


_ALL_CAPS_BRANDS = {"gpt", "fp8"}


def display_name(model_id: str) -> str:
    """'black-forest-labs/FLUX.2-dev' -> 'FLUX.2 Dev'; 'gpt-image-1.5' -> 'GPT Image 1.5'."""
    tokens = model_id.rsplit("/", 1)[-1].split("-")
    return " ".join(t.upper() if t.lower() in _ALL_CAPS_BRANDS else t[:1].upper() + t[1:]
                    for t in tokens)


@dataclass(frozen=True)
class JudgeModel:
    id: str
    provider: str = ""   # "" = open weights; "openai" / "google" = API
    sampling: dict[str, Any] = field(default_factory=dict)

    @property
    def is_api(self) -> bool:
        return bool(self.provider)

    @property
    def short_name(self) -> str:
        return short_name(self.id)


@dataclass(frozen=True)
class T2IModel:
    id: str
    provider: str = ""            # "" = local weights; "openai" / "google" = API
    loader: str = "diffusers"
    parallel_mode: str = "data_parallel"
    dtype: str = "bf16"
    batch_size: int = 1
    num_workers: int = 0
    use_batch_api: bool = False
    height: int = 0
    width: int = 0
    # 0 = one replica over all GPUs.
    gpus_per_replica: int = 0
    torch_compile: bool = False
    inference_kwargs: dict[str, Any] = field(default_factory=dict)
    modes: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Active mode; set by the registry, never in models.yaml.
    mode: str = ""

    def __post_init__(self):
        if bool(self.height) != bool(self.width):
            raise ValueError(f"{self.id}: set both height and width, or neither "
                             f"(got height={self.height}, width={self.width}).")

    @property
    def is_api(self) -> bool:
        return bool(self.provider)

    @property
    def input_id(self) -> str:
        """Mode-suffixed id; short_name and display_name derive from it."""
        return f"{self.id}-{self.mode}" if self.mode else self.id

    @property
    def short_name(self) -> str:
        return short_name(self.input_id)

    @property
    def display_name(self) -> str:
        return display_name(self.input_id)

    def for_mode(self, mode: str) -> T2IModel:
        return replace(self, mode=mode,
                       inference_kwargs={**self.inference_kwargs, **self.modes[mode]})

    @property
    def generation_kwargs(self) -> dict[str, Any]:
        size = {"height": self.height, "width": self.width} if self.height else {}
        return {**size, **self.inference_kwargs}


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, Any]:
    with resources.files("terravis.configs").joinpath("models.yaml").open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    default_sampling = (raw.get("judge_defaults") or {}).get("sampling", {})

    judges: dict[str, JudgeModel] = {}
    for entry in raw.get("judges", []):
        spec = JudgeModel(sampling={**default_sampling, **entry.pop("sampling", {})}, **entry)
        for key in (spec.id.lower(), spec.short_name):
            judges[key] = spec

    t2i: dict[str, T2IModel] = {}
    for entry in raw.get("t2i", []):
        spec = T2IModel(**{**entry, "id": entry["id"].rstrip("/")})
        for resolved in (spec, *(spec.for_mode(mode) for mode in spec.modes)):
            for key in (resolved.input_id.lower(), resolved.short_name):
                if key in t2i and t2i[key].input_id != resolved.input_id:
                    raise ValueError(f"models.yaml: {resolved.input_id} and {t2i[key].input_id} share the name "
                                     f"{key!r}, so their results would overwrite each other.")
                t2i[key] = resolved

    return {"judges": judges, "t2i": t2i, "default_sampling": default_sampling}


def find_judge(name: str, api_provider: str = "") -> JudgeModel:
    """Unlisted names need a '/' or api_provider, else ValueError."""
    registry = _load_registry()
    spec = registry["judges"].get(name.strip().lower())
    if spec is not None:
        if api_provider and api_provider != spec.provider:
            return replace(spec, provider=api_provider)
        return spec

    if "/" in name or api_provider:
        return JudgeModel(id=name.strip(), provider=api_provider, sampling=dict(registry["default_sampling"]))

    known = sorted({s.id for s in registry["judges"].values()})
    raise ValueError(
        f"Unknown judge model '{name}'. Pass a HuggingFace repo id (e.g. 'google/gemma-4-31B-it'), "
        f"an API model name with --api_provider, or one of: {known}"
    )


def find_t2i(name: str) -> T2IModel:
    """An unlisted HF repo id or local checkpoint directory gets the default local-diffusers spec."""
    name = name.strip().rstrip("/")   # a tab-completed directory keeps its trailing '/'
    registry = _load_registry()
    spec = registry["t2i"].get(name.lower())
    if spec is not None:
        return spec
    if "/" in name:
        spec = T2IModel(id=name)
        if spec.short_name in registry["t2i"]:
            raise ValueError(f"'{name}' has the same short name as {registry['t2i'][spec.short_name].input_id}, "
                             f"so their results would overwrite each other; rename the checkpoint directory.")
        return spec

    known = sorted({s.input_id for s in registry["t2i"].values()})
    raise ValueError(f"Unknown T2I model '{name}'. Pass a HuggingFace repo id, a local checkpoint path "
                     f"(with a '/'), or one of: {known}")
