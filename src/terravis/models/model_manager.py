import os
from typing import Any

import open_clip
import torch
from huggingface_hub import is_offline_mode, snapshot_download
from open_clip.tokenizer import DEFAULT_CONTEXT_LENGTH, HFTokenizer
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

from terravis.utils import hf_cache, use_flash_attn, vllm_dtype


def vllm_attention_backend(model_name: str) -> str | None:
    """TERRAVIS_VLLM_ATTENTION_BACKEND overrides; gemma-4 gets TRITON_ATTN: FA4 corrupts logits on its
    mixed 256/512 head dims."""
    override = os.environ.get("TERRAVIS_VLLM_ATTENTION_BACKEND")
    return override or ("TRITON_ATTN" if "gemma-4" in model_name.lower() else None)


def _compile_judge() -> bool:
    """TERRAVIS_COMPILE_JUDGE=1 (default off): pays back only on a long-running judge server; vLLM ignores it."""
    return os.environ.get("TERRAVIS_COMPILE_JUDGE", "").strip().lower() in {"1", "true", "yes", "on"}


def load_vllm_llm(model_name: str, tensor_parallel_size: int, dtype: str, download_dir: str):
    """max_model_len 8192 = unigenbench_score's 4096-token budget plus prompt."""
    from vllm import LLM

    extra = {}
    backend = vllm_attention_backend(model_name)
    if backend is not None:
        extra["attention_backend"] = backend
    if is_offline_mode() and not os.path.isdir(model_name):
        # Offline, vLLM looks a repo id up in HF_HUB_CACHE, not download_dir.
        model_name = snapshot_download(model_name, cache_dir=download_dir, local_files_only=True)

    return LLM(
        model=model_name, tensor_parallel_size=tensor_parallel_size, dtype=dtype,
        download_dir=download_dir, trust_remote_code=True,
        max_model_len=8192, limit_mm_per_prompt={"image": 1}, **extra,
    )


def load_clip_tokenizer(arch: str, cache_dir: str):
    """open_clip.get_tokenizer. Offline, an HF tokenizer loads from its cached snapshot instead: AutoTokenizer also
    looks up the repo's config.json, which timm's SigLIP repos lack, and offline that lookup fails even when cached."""
    text_cfg = (open_clip.get_model_config(arch) or {}).get("text_cfg", {})
    repo = text_cfg.get("hf_tokenizer_name")
    if not (repo and is_offline_mode()):
        return open_clip.get_tokenizer(arch, cache_dir=cache_dir)
    return HFTokenizer(
        snapshot_download(repo, cache_dir=cache_dir, local_files_only=True),
        context_length=text_cfg.get("context_length", DEFAULT_CONTEXT_LENGTH),
        tokenizer_mode=text_cfg.get("tokenizer_mode"), **text_cfg.get("tokenizer_kwargs", {}),
    )


def fetch_clip(model_name: str, cache_dir: str) -> None:
    """Download what get_clip_model reads; with HF_HUB_OFFLINE=1, raise if any of it is not cached."""
    pretrained, _, arch = model_name.partition(':')
    cfg = open_clip.get_pretrained_cfg(arch, pretrained)
    if not cfg:
        raise ValueError(f"Unknown OpenCLIP model {model_name!r}; expected 'pretrained:arch'.")
    open_clip.download_pretrained(cfg, cache_dir=cache_dir)
    load_clip_tokenizer(arch, cache_dir)
    if hf_text_model := open_clip.get_model_config(arch)["text_cfg"].get("hf_model_name"):
        AutoConfig.from_pretrained(hf_text_model)   # open_clip's HF text tower reads it from HF_HUB_CACHE


class ModelManager:
    def __init__(self, device: str | torch.device, model_weight_root: str, dtype: torch.dtype):
        self.device = device
        self.hf_cache = hf_cache(model_weight_root)
        self.dtype = dtype
        self._model_cache = {}

    def get_vqa_model(self, model_name: str) -> dict[str, Any]:
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        print(f"Loading VQA model: {model_name}...")
        kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self.device,
            "cache_dir": self.hf_cache,
            "adapter_kwargs": {"cache_dir": self.hf_cache},   # Auto*'s PEFT probe ignores cache_dir
            "trust_remote_code": True,
        }
        if use_flash_attn(self.dtype, model_name):
            kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)
        if _compile_judge():
            print("torch.compile (max-autotune) on the judge; the first prompts pay for it.")
            # A compiled wrapper's generate() runs the original forward; compile forward itself.
            model.forward = torch.compile(model.forward, mode="max-autotune")
        model.eval()

        processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=self.hf_cache,
            use_fast=True, trust_remote_code=True,
        )

        self._model_cache[model_name] = {"model": model, "processor": processor}
        return self._model_cache[model_name]

    def get_vllm_model(self, model_name: str, tensor_parallel_size: int = 1) -> dict[str, Any]:
        cache_key = f"vllm_{model_name}_tp{tensor_parallel_size}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        print(f"Loading VQA model via vLLM: {model_name} (tp={tensor_parallel_size})...")
        llm = load_vllm_llm(model_name, tensor_parallel_size,
                            dtype=vllm_dtype(self.dtype), download_dir=self.hf_cache)
        self._model_cache[cache_key] = {"llm": llm}
        return self._model_cache[cache_key]

    def get_clip_model(self, model_name: str) -> dict[str, Any]:
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        print(f"Loading CLIP model: {model_name}...")
        pretrained, arch = model_name.split(':')
        model, _, processor = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=self.device, cache_dir=self.hf_cache,
        )
        tokenizer = load_clip_tokenizer(arch, self.hf_cache)
        model.to(self.dtype).eval()

        self._model_cache[model_name] = {"model": model, "processor": processor, "tokenizer": tokenizer}
        return self._model_cache[model_name]
