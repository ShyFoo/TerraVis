"""T2I generation pipelines: local checkpoints and provider APIs. Text2ImageModelPipeline
dispatches on the registry loader field; a loader returns a diffusers pipeline or any object
with generate_image(prompt=[...], seeds=[...]) -> [PIL.Image | None]."""

import base64
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable
from typing import Any

import requests
import torch
from diffusers import DiffusionPipeline
from google import genai
from openai import APIStatusError, OpenAI
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor

from terravis.configs import T2IModel, find_t2i, short_name
from terravis.third_party.ideogram.caption_verifier import CaptionVerifier
from terravis.third_party.ideogram.magic_prompt import Ideogram4MagicPromptV1, aspect_ratio_from_size
from terravis.utils import hf_cache, torch_dtype


def compile_denoiser(pipeline) -> None:
    """Regional compile for denoisers with _repeated_blocks (Flux.2, Qwen-Image), full otherwise.
    Compiled output is not bit-exact with eager; do not mix the two in one benchmark."""
    denoiser = getattr(pipeline, "transformer", None) or getattr(pipeline, "unet", None)
    if denoiser is None:
        raise ValueError(
            f"torch.compile found no denoiser on {type(pipeline).__name__}: no `transformer` or "
            f"`unet`. Give the wrapper a denoiser property, or list its loader in NO_COMPILE."
        )

    if any(hasattr(module, "_hf_hook") for module in denoiser.modules()):
        raise ValueError(
            "torch.compile cannot be combined with a sharded (device_map) placement: dynamo raises "
            "on accelerate's cross-device hooks. Use data_parallel or drop the compile."
        )

    # Both from the diffusers performance guide.
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True

    mode = "regional" if getattr(denoiser, "_repeated_blocks", None) else "full"
    if mode == "regional":
        denoiser.compile_repeated_blocks(fullgraph=True)
    else:
        # No fullgraph: a denoiser forward may graph-break.
        denoiser.compile()
    print(f"torch.compile enabled ({mode}) on {type(denoiser).__name__}.")


# --- Loaders, one per registry loader value ---

def load_diffusers(spec: T2IModel, weight_root: str, parallel_mode: str, dtype: str):
    torch_dt = torch_dtype(dtype)
    if torch_dt is None:
        raise ValueError(f"{spec.id}: diffusers has no dtype 'auto' (omitted torch_dtype loads fp32); name a dtype.")
    kwargs: dict[str, Any] = {"cache_dir": hf_cache(weight_root), "torch_dtype": torch_dt}
    if parallel_mode == "model_sharding":
        kwargs["device_map"] = "balanced"

    pipeline = DiffusionPipeline.from_pretrained(spec.id, **kwargs)
    # Unlisted repos may ship a safety checker (SD 1.x); the registered pipelines have none.
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    return pipeline


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

#: Thirds of the official fused qkv projection's output rows, in unbind order.
_IDEOGRAM_QKV = ("to_q", "to_k", "to_v")


#: Separate file: diffusers drops weight_scale as an unexpected key, and unscaled fp8 renders noise.
_IDEOGRAM_SCALES_FILE = "weight_scales.safetensors"


class Fp8ScaledLinear(torch.nn.Linear):
    """Weight stays fp8, dequantized per forward as in the official runtime (26G resident;
    folding the scale in would be 53G). A class swap survives deepcopy, a hook closure does not."""

    weight_scale: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype)[:, None]
        return torch.nn.functional.linear(x, weight, self.bias)


def _ideogram_unfuse(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, tensor in state_dict.items():   # suffix: "weight" | "weight_scale"
        if ".attention.qkv." in key:
            stem, _, suffix = key.partition(".qkv.")
            for name, part in zip(_IDEOGRAM_QKV, tensor.chunk(3, dim=0)):
                out[f"{stem}.{name}.{suffix}"] = part.contiguous()
        elif ".attention.o." in key:
            stem, _, suffix = key.partition(".o.")
            out[f"{stem}.to_out.0.{suffix}"] = tensor
        else:
            out[key] = tensor
    return out


#: Each carries a scales file, even an empty one, so a partial snapshot fails by name.
_IDEOGRAM_QUANTIZED = ("transformer", "unconditional_transformer", "text_encoder")


def _apply_fp8_scales(component: torch.nn.Module, scales: dict[str, torch.Tensor]) -> int:
    """Returns the number of Linears rewired to Fp8ScaledLinear."""
    modules = dict(component.named_modules())
    for key, scale in scales.items():
        name = key[: -len(".weight_scale")]
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear) or tuple(scale.shape) != (module.out_features,):
            raise RuntimeError(
                f"{name}: weight_scale {tuple(scale.shape)} does not fit "
                f"{type(module).__name__}: the snapshot was built for another class layout; delete it to rebuild.")
        if module.weight.dtype not in _FP8_DTYPES:
            # bf16 round-trips fp8 exactly; the repo is all e4m3.
            module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
        module.__class__ = Fp8ScaledLinear
        module.register_buffer("weight_scale", scale.to(module.weight.device))
    if scales:
        # Ideogram4Pipeline casts activations to component.dtype, which would now read fp8.
        cls = type(component)
        component.__class__ = type(
            cls.__name__, (cls,), {"dtype": property(lambda self: torch.bfloat16)})
    return len(scales)


def _ideogram_fp8_snapshot(spec: T2IModel, weight_root: str) -> str:
    """Builds <weight_root>/<short name>-diffusers (~26G) on first use: the official fused
    Linears and fp8 scales renamed for the diffusers Ideogram4 classes, weights still fp8."""
    from filelock import FileLock

    dst = os.path.join(weight_root, f"{short_name(spec.id)}-diffusers")
    marker = os.path.join(dst, ".converted")
    if os.path.exists(marker):   # built: no lock file, so a read-only weight root still loads
        return dst
    # data_parallel ranks (and concurrent jobs) share dst: one builds it, the rest wait.
    with FileLock(f"{dst}.lock"):
        if not os.path.exists(marker):
            _build_ideogram_fp8_snapshot(spec, weight_root, dst)
            open(marker, "w").close()
    return dst


def _build_ideogram_fp8_snapshot(spec: T2IModel, weight_root: str, dst: str) -> None:
    from accelerate import init_empty_weights
    from diffusers import Ideogram4Transformer2DModel
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, Qwen3VLModel

    if os.path.exists(dst):
        shutil.rmtree(dst)
    src = spec.id if os.path.isdir(spec.id) else snapshot_download(spec.id, cache_dir=hf_cache(weight_root))
    os.makedirs(dst)
    shutil.copy(os.path.join(src, "model_index.json"), dst)
    for name in ("scheduler", "tokenizer", "vae"):
        shutil.copytree(os.path.join(src, name), os.path.join(dst, name))

    def convert(component: str, build_empty, weights_name: str) -> None:
        comp_src = os.path.join(src, component)
        comp_dst = os.path.join(dst, component)
        os.makedirs(comp_dst)
        shutil.copy(os.path.join(comp_src, "config.json"), comp_dst)

        state: dict[str, torch.Tensor] = {}
        for fname in sorted(os.listdir(comp_src)):
            if fname.endswith(".safetensors"):
                state.update(load_file(os.path.join(comp_src, fname)))
        state = _ideogram_unfuse(state)
        scales = {k: state.pop(k) for k in list(state) if k.endswith(".weight_scale")}
        missing = [k for k, v in state.items() if v.dtype in _FP8_DTYPES and f"{k}_scale" not in scales]
        if missing:
            raise RuntimeError(f"{component}: fp8 weights with no weight_scale: {missing[:5]}")

        with init_empty_weights():
            expected = build_empty(comp_src).state_dict()
        if {k: tuple(v.shape) for k, v in state.items()} != {k: tuple(v.shape) for k, v in expected.items()}:
            raise RuntimeError(
                f"{component}: converted keys do not match the class: "
                f"missing {sorted(expected.keys() - state.keys())[:5]}, "
                f"unexpected {sorted(state.keys() - expected.keys())[:5]}"
            )
        print(f"Converted {component}: {len(state)} tensors ({len(scales)} fp8 scales) -> {comp_dst}")
        save_file(state, os.path.join(comp_dst, weights_name))
        save_file(scales, os.path.join(comp_dst, _IDEOGRAM_SCALES_FILE))

    def transformer_empty(path: str):
        return Ideogram4Transformer2DModel.from_config(Ideogram4Transformer2DModel.load_config(path))

    convert("transformer", transformer_empty, "diffusion_pytorch_model.safetensors")
    convert("unconditional_transformer", transformer_empty, "diffusion_pytorch_model.safetensors")
    convert("text_encoder", lambda path: Qwen3VLModel(AutoConfig.from_pretrained(path)),
            "model.safetensors")


def _expand_magic_prompt(magic: Ideogram4MagicPromptV1, prompt: str, aspect_ratio: str,
                         max_retries: int = 10) -> str | None:
    """None after transient failures (one failure-marked image); a non-retryable 4xx raises."""
    for attempt in range(max_retries):
        try:
            caption = magic.expand(prompt, aspect_ratio=aspect_ratio)
            # Upstream aborts on verifier findings; here they are warnings only.
            for warning in CaptionVerifier().verify_raw(caption):
                print(f"[magic prompt] caption warning for {prompt[:60]!r}: {warning}")
            return caption
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            print(f"[magic prompt attempt {attempt + 1}/{max_retries}] {e}")
        # Network errors and malformed 200 responses.
        except (requests.RequestException, RuntimeError, AttributeError, TypeError, ValueError) as e:
            print(f"[magic prompt attempt {attempt + 1}/{max_retries}] {e}")
        if attempt < max_retries - 1:
            time.sleep(2)

    print(f"Failed to expand prompt after {max_retries} attempts: {prompt[:60]!r}")
    return None


class IdeogramMagicPromptPipeline:
    """ideogram-4 behind the hosted magic-prompt expansion; test_data.json keeps the original
    prompt. The expansion is unseeded, so a rerun at the same seed renders different images."""

    def __init__(self, pipeline, magic: Ideogram4MagicPromptV1):
        self.pipeline = pipeline
        self.magic = magic

    @property
    def transformer(self):
        return self.pipeline.transformer

    def to(self, device) -> "IdeogramMagicPromptPipeline":
        self.pipeline.to(device)
        return self

    def generate_image(self, prompt: list[str], seeds: list[int], **kwargs) -> list[Image.Image | None]:
        aspect_ratio = aspect_ratio_from_size(kwargs["width"], kwargs["height"])
        captions = [_expand_magic_prompt(self.magic, p, aspect_ratio) for p in prompt]

        images: list[Image.Image | None] = [None] * len(prompt)
        rendered = [i for i, caption in enumerate(captions) if caption is not None]
        if rendered:
            generators = [torch.Generator().manual_seed(seeds[i]) for i in rendered]
            batch = self.pipeline(prompt=[captions[i] for i in rendered], output_type="pil",
                                  generator=generators, **kwargs).images
            for i, image in zip(rendered, batch):
                images[i] = image
        return images


def load_ideogram(spec: T2IModel, weight_root: str, parallel_mode: str, dtype: str):
    api_key = os.environ.get("IDEOGRAM_API_KEY", "")
    if not api_key:
        raise ValueError(
            f"{spec.id} needs IDEOGRAM_API_KEY (https://developer.ideogram.ai/) for the hosted "
            "magic-prompt rewrite; on bare text the model degenerates."
        )
    if torch_dtype(dtype) is not None:
        raise ValueError(
            f"{spec.id} runs fp8-resident (registry `dtype: auto`); dtype {dtype!r} would double "
            f"it to ~53G. See Fp8ScaledLinear.")

    # Probe once: a bad key must fail before the 26G pipeline loads.
    magic = Ideogram4MagicPromptV1(api_key=api_key)
    if _expand_magic_prompt(magic, "a cat", "1:1") is None:
        raise RuntimeError(f"{spec.id}: the magic-prompt API is unreachable (see above); "
                           "generation would fail on every prompt.")

    # Without torch_dtype diffusers materializes fp32; bf16 round-trips fp8 exactly.
    kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if parallel_mode == "model_sharding":
        kwargs["device_map"] = "balanced"

    snapshot = _ideogram_fp8_snapshot(spec, weight_root)
    pipeline = DiffusionPipeline.from_pretrained(snapshot, **kwargs)

    rewired = {}
    for name in _IDEOGRAM_QUANTIZED:
        scales = load_file(os.path.join(snapshot, name, _IDEOGRAM_SCALES_FILE))
        rewired[name] = _apply_fp8_scales(getattr(pipeline, name), scales)
    print(f"fp8 weights kept resident (dequantized per forward): {rewired}")
    return IdeogramMagicPromptPipeline(pipeline, magic)


class Cosmos3ImagePipeline:
    """One prompt per call; the image is a 1-frame clip under output="videos". NVIDIA's
    recommended prompt upsampling is skipped: it would score a different prompt."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @property
    def transformer(self):
        return self.pipeline.transformer

    def generate_image(self, prompt: list[str], seeds: list[int], **kwargs) -> list[Image.Image | None]:
        images = []
        for p, seed in zip(prompt, seeds):
            frames = self.pipeline(prompt=p, num_frames=1, output="videos",
                                   generator=torch.Generator().manual_seed(seed), **kwargs)
            images.append(frames[0] if len(frames) else None)
        return images


# Cosmos3OmniTransformer modules outside `layers`; audio_modality_embed is a bare nn.Parameter.
_COSMOS3_TOP_LEVEL = (
    "embed_tokens", "norm", "norm_moe_gen", "rotary_emb", "lm_head",
    "proj_in", "proj_out", "time_proj", "time_embedder",
    "audio_proj_in", "audio_proj_out", "audio_modality_embed",
)
_COSMOS3_NUM_LAYERS = 64


def cosmos3_device_map(num_gpus: int) -> dict:
    """device_map="balanced" raises "Expected all tensors to be on the same device" here. Top-level
    modules pin to cuda:0, which takes two layers fewer (embed_tokens and lm_head are ~1.6G each)."""
    num_gpus = max(1, num_gpus)
    device_map = {name: 0 for name in _COSMOS3_TOP_LEVEL}
    base, remainder = divmod(_COSMOS3_NUM_LAYERS, num_gpus)
    counts = [base] * num_gpus
    for i in range(remainder):
        counts[-(i + 1)] += 1
    if num_gpus > 1:
        moved = min(2, counts[0] - 1)
        counts[0] -= moved
        for i in range(moved):
            counts[1 + (i % (num_gpus - 1))] += 1
    layer = 0
    for device, count in enumerate(counts):
        for _ in range(count):
            device_map[f"layers.{layer}"] = device
            layer += 1
    return device_map


def load_cosmos3(spec: T2IModel, weight_root: str, parallel_mode: str, dtype: str):
    if parallel_mode != "model_sharding":
        raise ValueError(
            f"{spec.id} needs model_sharding: ~122G of bf16 weights do not fit one card "
            f"(got parallel_mode={parallel_mode!r})."
        )
    from diffusers import Cosmos3DistilledModularPipeline

    torch_dt = torch_dtype(dtype)
    if torch_dt is None:
        raise ValueError(f"{spec.id}: dtype 'auto' loads the ~122G bf16 weights as fp32; name a dtype.")
    # Diffusers' sharded-checkpoint resolver needs local_files_only explicitly; HF_HUB_OFFLINE alone is not enough.
    local_files_only = os.getenv("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}
    pipeline = Cosmos3DistilledModularPipeline.from_pretrained(
        spec.id, dtype=torch_dt, cache_dir=hf_cache(weight_root), local_files_only=local_files_only,
    )
    pipeline.load_components(
        dtype=torch_dt, cache_dir=hf_cache(weight_root), local_files_only=local_files_only,
        device_map={"transformer": cosmos3_device_map(torch.cuda.device_count()),
                    "vae": "cuda", "sound_tokenizer": "cuda"},
    )
    # load_components only logs a component failure and leaves the attribute None.
    if pipeline.transformer is None:
        raise RuntimeError(f"{spec.id}: the transformer failed to load (see the traceback above).")
    # Otherwise requires_safety_checker inherits enable_safety_checker=True and generate() raises.
    pipeline.disable_safety_checker()
    return Cosmos3ImagePipeline(pipeline)


class HiDreamImagePipeline:
    """A qwen3_vl model whose forward is a pixel-space denoiser, driven by the loop under
    third_party/hidream (snap_to_predefined=False: see its __init__.py)."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def to(self, device) -> "HiDreamImagePipeline":
        self.model.to(device)
        return self

    def generate_image(self, prompt: list[str], seeds: list[int], **kwargs) -> list[Image.Image | None]:
        from terravis.third_party.hidream.pipeline import generate_image

        return [
            generate_image(model=self.model, processor=self.processor, prompt=p, seed=seed,
                           snap_to_predefined=False, **kwargs)
            for p, seed in zip(prompt, seeds)
        ]


def load_hidream(spec: T2IModel, weight_root: str, parallel_mode: str, dtype: str):
    from terravis.third_party.hidream.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    if parallel_mode != "data_parallel":
        raise ValueError(
            f"{spec.id} needs data_parallel: 17G in bf16 fits one card, nothing to shard "
            f"(got parallel_mode={parallel_mode!r})."
        )

    torch_dt = torch_dtype(dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        spec.id,
        # bf16 (the card's recipe) halves the 33G fp32 checkpoint on the way in.
        torch_dtype=torch_dt,
        cache_dir=hf_cache(weight_root),
        # No attn_implementation: flash_attention_2 rejects the 4D float mask.
    ).eval()

    processor = AutoProcessor.from_pretrained(spec.id, cache_dir=hf_cache(weight_root))
    return HiDreamImagePipeline(model, processor)


LOADERS: dict[str, Callable[[T2IModel, str, str, str], Any]] = {
    "diffusers": load_diffusers,
    "ideogram": load_ideogram,
    "cosmos3": load_cosmos3,
    "hidream": load_hidream,
}


def resolve_loader(name: str) -> Callable[[T2IModel, str, str, str], Any]:
    """A LOADERS key, or your own function as 'path/to/file.py:function' or 'package.module:function'."""
    if name in LOADERS:
        return LOADERS[name]
    target, sep, function = name.rpartition(":")
    if not sep:
        raise ValueError(f"Unknown loader {name!r}. Registered: {sorted(LOADERS)}, or 'path/to/file.py:function'.")
    if target.endswith(".py"):
        sys.path.append(os.path.dirname(os.path.abspath(target)))   # for model code beside the file
        # A fixed name cannot shadow a stdlib module; registering it is what dataclasses need.
        module_spec = importlib.util.spec_from_file_location("terravis_user_loader", target)
        module = sys.modules[module_spec.name] = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, function)


NO_COMPILE = {
    "hidream": "its flash path reassigns each decoder layer's attention per call, so dynamo "
               "would recompile every step",
}


class Text2ImageModelPipeline:
    @classmethod
    def load(cls, model_name: str, model_weight_root: str,
             parallel_mode: str = "data_parallel", dtype: str | None = None,
             torch_compile: bool = False):
        """dtype None takes the registry's."""
        spec = find_t2i(model_name)
        if spec.is_api:
            raise ValueError(f"{spec.id!r} is an API model; use Text2ImageAPIPipeline.")
        loader = resolve_loader(spec.loader)
        if torch_compile and spec.loader in NO_COMPILE:
            raise ValueError(
                f"torch.compile is not available for {spec.id}: {NO_COMPILE[spec.loader]}. "
                f"Drop the compile."
            )

        pipeline = loader(spec, model_weight_root, parallel_mode, dtype or spec.dtype)
        print(f"Pipeline loaded: {spec.short_name} ({spec.id})")
        if torch_compile:
            compile_denoiser(pipeline)
        return pipeline


# --- Provider APIs ---

def _poll_until_done(label: str, fetch, state_of, terminal: set, poll_interval: float):
    while True:
        job = fetch()
        state = state_of(job)
        if state in terminal:
            print(f"[{label}] Final state: {state}")
            return job
        print(f"[{label}] State: {state} - waiting {poll_interval}s...")
        time.sleep(poll_interval)


_GEMINI_ASPECT_RATIOS = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}


def gemini_image_config(width: int, height: int) -> dict[str, str]:
    """Registry pixel size as Gemini spells it: aspect ratio plus resolution tier."""
    divisor = math.gcd(width, height)
    aspect_ratio = f"{width // divisor}:{height // divisor}"
    if aspect_ratio not in _GEMINI_ASPECT_RATIOS:
        raise ValueError(f"Gemini does not offer aspect ratio {aspect_ratio} ({width}x{height}); "
                         f"pick a size in {sorted(_GEMINI_ASPECT_RATIOS)}.")
    longest = max(width, height)
    tier = next((t for limit, t in ((1024, "1K"), (2048, "2K"), (4096, "4K")) if longest <= limit), None)
    if tier is None:
        raise ValueError(f"Gemini renders up to 4K; {width}x{height} is larger.")
    return {"aspect_ratio": aspect_ratio, "image_size": tier}


class Text2ImageAPIPipeline:
    def __init__(self, model_name: str):
        spec = find_t2i(model_name)
        if not spec.is_api:
            raise ValueError(f"{spec.id!r} is a local model; use Text2ImageModelPipeline.")
        self.model_id = spec.id
        self.provider = spec.provider
        if self.provider == "openai":
            self.image_size = f"{spec.width}x{spec.height}" if spec.width else "auto"
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            self.image_config = gemini_image_config(spec.width, spec.height) if spec.width else {}
            self.client = genai.Client(
                api_key=os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"])

    def generate(self, prompt: str) -> list[Image.Image | None]:
        if self.provider == "openai":
            return [self._openai_generate(prompt)]
        return [self._gemini_generate(prompt)]

    def generate_batch(self, prompts: list[str],
                       poll_interval: int = 20) -> list[Image.Image | None]:
        if self.provider == "openai":
            return self._openai_generate_batch(prompts, poll_interval)
        return self._gemini_generate_batch(prompts, poll_interval)

    # -- OpenAI ------------------------------------------------------------

    def _openai_generate(self, prompt: str, max_retries: int = 10) -> Image.Image | None:
        """None after retries or on a prompt-specific 4xx; 401/403/404 raise."""
        for attempt in range(max_retries):
            try:
                result = self.client.images.generate(
                    model=self.model_id, prompt=prompt, size=self.image_size)
                return _image_from_b64(result.data[0].b64_json)
            except APIStatusError as e:
                if e.status_code in (401, 403, 404):
                    raise
                if 400 <= e.status_code < 500 and e.status_code != 429:
                    print(f"[OpenAI] non-retryable {e.status_code} for {prompt[:60]!r}: {e}")
                    return None
                print(f"[OpenAI attempt {attempt + 1}/{max_retries}] {e}")
            except Exception as e:  # connection errors, malformed responses
                print(f"[OpenAI attempt {attempt + 1}/{max_retries}] {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

        print(f"Failed to generate image after {max_retries} attempts.")
        return None

    def _openai_generate_batch(self, prompts: list[str],
                               poll_interval: float) -> list[Image.Image | None]:
        data = "".join(json.dumps({
            "custom_id": f"img-{i}",
            "method": "POST",
            "url": "/v1/images/generations",
            "body": {"model": self.model_id, "prompt": prompt, "size": self.image_size},
        }, ensure_ascii=False) + "\n" for i, prompt in enumerate(prompts)).encode("utf-8")
        input_file = self.client.files.create(file=("batch.jsonl", data), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/images/generations",
            completion_window="24h",
        )
        print(f"[OpenAI Batch] Job created: {batch.id}")

        batch = _poll_until_done(
            "OpenAI Batch", lambda: self.client.batches.retrieve(batch.id),
            lambda b: b.status, {"completed", "failed", "expired", "cancelled"}, poll_interval)
        if batch.status != "completed":
            # Job-level failure: raise, so no prompt gets a failure marker.
            raise RuntimeError(f"[OpenAI Batch] {batch.id} finished with status={batch.status}.")
        if not batch.output_file_id:
            # "completed" scopes the job only; all-failed leaves no output file.
            print("[OpenAI Batch] Completed with no output file: every request failed.")
            return [None] * len(prompts)

        results: list[Image.Image | None] = [None] * len(prompts)
        for line in self.client.files.content(batch.output_file_id).text.strip().splitlines():
            row = json.loads(line)
            data = row.get("response", {}).get("body", {}).get("data", [])
            if data:
                results[int(row["custom_id"].split("-")[1])] = _image_from_b64(data[0]["b64_json"])
        return results

    # -- Gemini ------------------------------------------------------------

    def _gemini_generate(self, prompt: str, max_retries: int = 10) -> Image.Image | None:
        config = genai.types.GenerateContentConfig(
            image_config=genai.types.ImageConfig(**self.image_config))
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id, contents=prompt, config=config)
                for part in response.parts or []:
                    if part.inline_data:
                        # Not part.as_image(): the SDK's Image type, not PIL.
                        return _image_from_bytes(part.inline_data.data)
            except Exception as e:
                if getattr(e, "code", None) in (401, 403, 404) or "API key not valid" in str(e):
                    raise
                print(f"[Gemini attempt {attempt + 1}/{max_retries}] {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

        print(f"Failed to generate image after {max_retries} attempts.")
        return None

    def _gemini_generate_batch(self, prompts: list[str],
                               poll_interval: float) -> list[Image.Image | None]:
        job = self.client.batches.create(
            model=self.model_id,
            src=[{
                "contents": [{"parts": [{"text": prompt}]}],
                "config": {"responseModalities": ["TEXT", "IMAGE"],
                           "image_config": self.image_config},
            } for prompt in prompts],
            config={"display_name": f"batch-{uuid.uuid4().hex[:8]}"},
        )
        print(f"[Gemini Batch] Job created: {job.name}")

        job = _poll_until_done(
            "Gemini Batch", lambda: self.client.batches.get(name=job.name),
            lambda j: j.state.name,
            {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"},
            poll_interval)
        if job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"[Gemini Batch] {job.name} ended in {job.state.name}: {job.error}")

        results: list[Image.Image | None] = [None] * len(prompts)
        for i, inline_response in enumerate(job.dest.inlined_responses):  # order matches the input
            response = inline_response.response
            # A safety-stopped request in a SUCCEEDED job has candidates/content/parts None.
            for part in (response.parts if response else None) or []:
                if part.inline_data:
                    results[i] = _image_from_bytes(part.inline_data.data)
                    break
            if results[i] is None:
                print(f"[Gemini Batch] response[{i}] has no image (blocked?)")
        return results


def _image_from_b64(b64: str) -> Image.Image:
    return _image_from_bytes(base64.b64decode(b64))


def _image_from_bytes(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw)).convert("RGB")
