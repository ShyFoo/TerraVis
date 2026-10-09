import importlib.util
import os
import random

import numpy as np
import torch

TORCH_DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16, "auto": None}
_VLLM_DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}


def torch_dtype(name: str) -> torch.dtype | None:
    """'auto' -> None. transformers reads None as the checkpoint dtype; diffusers reads it as fp32."""
    if name not in TORCH_DTYPES:
        raise ValueError(f"Unsupported dtype '{name}'. Expected one of {list(TORCH_DTYPES)}.")
    return TORCH_DTYPES[name]


def vllm_dtype(dtype: torch.dtype) -> str:
    return _VLLM_DTYPE_NAMES[dtype]


def use_flash_attn(dtype: torch.dtype | None, model_name: str = "") -> bool:
    """flash-attn is installed and usable: fp16/bf16 on Ampere+, head_dim <= 256 (gemma-4 global heads are 512)."""
    return bool(importlib.util.find_spec("flash_attn") and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 8 and dtype in (torch.float16, torch.bfloat16)
                and "gemma-4" not in model_name.lower())


def hf_cache(model_weight_root: str) -> str:
    """Hugging Face cache for judges, T2I models and CLIP; scripts/set_root_paths.sh points HF_HUB_CACHE here."""
    return os.path.join(model_weight_root, "hf_cache")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
