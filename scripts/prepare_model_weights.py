#!/usr/bin/env python3
"""Stage each score's weights (MANIFEST, or --list) into per-score folders under model_weight_root
for offline scoring (HF_HUB_OFFLINE=1). --judge / --t2i / --clip stage into <root>/hf_cache instead
(add --only to stage scores too). Needs only huggingface_hub; --clip needs the terravis package.

    python scripts/prepare_model_weights.py [--only rahf pickscore|--list|--verify-only]
    python scripts/prepare_model_weights.py --judge google/gemma-4-31B-it --t2i black-forest-labs/FLUX.2-klein-9B
    python scripts/prepare_model_weights.py --clip webli:ViT-gopt-16-SigLIP2-384
"""

import argparse
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field

# Exports the roots from scripts/set_root_paths.sh before huggingface_hub loads; stdlib only, so no install needed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))
import terravis.cli  # noqa: F401

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    sys.exit("error: huggingface_hub is required (pip install huggingface_hub).")

# --- Manifest: folder names and repo ids must match the loaders in src/terravis/{scores,third_party}/. ---

# mirrors third_party/rahf/RAHF.py :: RAHF_CHECKPOINT_URL
RAHF_CHECKPOINT_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1-jKfmpyGtJ0UAgEQ23zylRsmQ82qigzB&export=download&confirm=t"
)
# mirrors scores/metrics/aesthetic_score.py :: AESTHETIC_HEAD_URL
AESTHETIC_HEAD_URL = (
    "https://github.com/discus0434/aesthetic-predictor-v2-5/raw/main/"
    "models/aesthetic_predictor_v2_5.pth"
)

_IGNORE = ["*.h5", "*.ot", "*.msgpack", "*.onnx", "*.onnx_data",
           "tf_model*", "flax_model*", "*.tflite"]
# Files the loaders never read (~25G): repos used only for config/tokenizer/processor, and duplicate weight formats.
_UNUSED = {
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": ["*.bin", "*.safetensors"],   # PickScore: processor only
    "yuvalkirstain/PickScore_v1": ["*.bin"],                               # model.safetensors is loaded
    "google/vit-large-patch16-384": ["*.bin", "*.safetensors"],            # RAHF: config + image processor,
    "google-t5/t5-base": ["*.bin", "*.safetensors"],                       # config + tokenizer; weights in rahf_model.pt
    "bert-base-uncased": ["*.bin", "*.safetensors", "coreml/*"],           # BLIP: tokenizer only
    "sentence-transformers/all-mpnet-base-v2": ["*.bin", "openvino/*"],    # model.safetensors is loaded
}


@dataclass
class Asset:
    key: str
    folder: str
    snapshots: list[str] = field(default_factory=list)
    files: list[tuple[str, str]] = field(default_factory=list)  # (repo, filename)
    urls: list[tuple[str, str]] = field(default_factory=list)   # (url, filename)
    note: str = ""


MANIFEST: list[Asset] = [
    Asset("hpsv3", "HPSv3",
          snapshots=["Qwen/Qwen2-VL-7B-Instruct"],
          files=[("MizzenAI/HPSv3", "HPSv3.safetensors")],
          note="HPSv3 reward head on a Qwen2-VL-7B backbone"),
    Asset("imagereward", "ImageReward",
          snapshots=["bert-base-uncased"],
          files=[("zai-org/ImageReward", "ImageReward.pt"), ("zai-org/ImageReward", "med_config.json")],
          note="ImageReward + BLIP (bert-base-uncased tokenizer)"),
    Asset("musiq", "MUSIQ",
          files=[("chaofengc/IQA-PyTorch-Weights", "musiq_koniq_ckpt-e95806b9.pth")],
          note="MUSIQ KonIQ checkpoint (self-contained)"),
    Asset("rahf", "RAHF",
          snapshots=["google/vit-large-patch16-384", "google-t5/t5-base"],
          urls=[(RAHF_CHECKPOINT_URL, "rahf_model.pt")],
          note="RAHF multi-head + ViT-large-384 / T5-base backbones"),
    Asset("aesthetic", "Aesthetic",
          snapshots=["google/siglip-so400m-patch14-384"],
          urls=[(AESTHETIC_HEAD_URL, "aesthetic_predictor_v2_5.pth")],
          note="Aesthetic Predictor V2.5 head + SigLIP backbone (scoring needs terravis[aesthetic])"),
    Asset("pickscore", "PickScore",
          snapshots=["yuvalkirstain/PickScore_v1",
                     "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"],
          note="PickScore_v1 + CLIP-H processor"),
    Asset("tifa", "TIFA",
          snapshots=["sentence-transformers/all-mpnet-base-v2"],
          note="TIFA SBERT matcher (the judge LLM is shared, staged separately)"),
    Asset("geneval2", "GenEval2",
          snapshots=["Qwen/Qwen3-VL-8B-Instruct"],
          note="GenEval 2 Soft-TIFA judge, pinned by the official protocol"),
    Asset("unifiedreward", "UnifiedReward",
          snapshots=["CodeGoat24/UnifiedReward-2.0-qwen35-9b"],
          note="UnifiedReward-2.0 on a Qwen3.5-9B backbone (19G bf16)"),
]


# --- Download helpers ---

def _download_url(url: str, dest: str) -> None:
    name = os.path.basename(dest)
    if os.path.exists(dest):
        print(f"    cached   {name}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "terravis-stage/1.0"})
    with urllib.request.urlopen(req) as resp:
        if "text/html" in resp.headers.get("Content-Type", ""):
            raise RuntimeError(f"got an HTML page, not a file, from {url} (download interstitial / expired link?)")
        total = int(resp.headers.get("Content-Length", 0)) >> 20
        with open(dest + ".part", "wb") as f:
            for done, chunk in enumerate(iter(lambda: resp.read(1 << 20), b""), start=1):
                f.write(chunk)
                print(f"\r    {name}: {done}/{total or '?'} MB", end="", flush=True)
        print()
    os.replace(dest + ".part", dest)
    print(f"    done     {name}")


def stage_asset(asset: Asset, root: str) -> None:
    folder = os.path.join(root, asset.folder)
    print(f"\n[{asset.key}] -> {asset.folder}/   ({asset.note})")
    for repo in asset.snapshots:
        print(f"    snapshot {repo} ...")
        snapshot_download(repo_id=repo, cache_dir=folder, ignore_patterns=_IGNORE + _UNUSED.get(repo, []))
    for repo, filename in asset.files:
        if os.path.exists(os.path.join(folder, filename)):
            print(f"    cached   {filename}")
            continue
        hf_hub_download(repo_id=repo, filename=filename, local_dir=folder)
        print(f"    done     {filename}  (from {repo})")
    for url, filename in asset.urls:
        _download_url(url, os.path.join(folder, filename))


def verify_snapshot(repo: str, cache_dir: str, ignore: list[str] = _IGNORE) -> bool:
    try:
        snapshot_download(repo_id=repo, cache_dir=cache_dir, local_files_only=True, ignore_patterns=ignore)
    except Exception:
        print(f"    MISS snapshot {repo}")
        return False
    print(f"    ok   snapshot {repo}")
    return True


def stage_clip(name: str, root: str) -> None:
    from terravis.models.model_manager import fetch_clip  # heavy import, only for --clip

    print(f"\n[hf_cache] clip {name}")
    fetch_clip(name, root)
    print(f"    done     {name}")


def verify_clip(name: str, root: str) -> bool:
    """Fetch again with HF_HUB_OFFLINE=1, as an offline node would: passes only if everything is cached."""
    code = "import sys; from terravis.models.model_manager import fetch_clip; fetch_clip(*sys.argv[1:])"
    proc = subprocess.run([sys.executable, "-c", code, name, root], capture_output=True, text=True,
                          env={**os.environ, "HF_HUB_OFFLINE": "1"})
    ok, reason = proc.returncode == 0, proc.stderr.strip().rpartition("\n")[2]
    print(f"    {'ok  ' if ok else 'MISS'} clip     {name}" + ("" if ok else f"  ({reason})"))
    return ok


def verify_asset(asset: Asset, root: str) -> bool:
    folder = os.path.join(root, asset.folder)
    ok = all([verify_snapshot(repo, folder, _IGNORE + _UNUSED.get(repo, []))   # a list: print every repo
              for repo in asset.snapshots])
    for _, filename in [*asset.files, *asset.urls]:
        present = os.path.exists(os.path.join(folder, filename))
        ok = ok and present
        print(f"    {'ok  ' if present else 'MISS'} file     {filename}")
    return ok


# --- CLI ---

def main() -> None:
    keys = [a.key for a in MANIFEST]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-weight-root", default=os.environ.get("MODEL_WEIGHT_ROOT"),
                   required=not os.environ.get("MODEL_WEIGHT_ROOT"),
                   help="Root dir for weights. Default: MODEL_WEIGHT_ROOT from scripts/set_root_paths.sh.")
    p.add_argument("--only", nargs="*", metavar="KEY", choices=keys,
                   help="Stage only these scores (default: all; none with --judge/--t2i/--clip). Choices: "
                        + ", ".join(keys))
    p.add_argument("--judge", action="append", default=[], metavar="HF_REPO",
                   help="Snapshot this judge/VLM repo into <root>/hf_cache (repeatable).")
    p.add_argument("--t2i", action="append", default=[], metavar="HF_REPO",
                   help="Snapshot this T2I repo into <root>/hf_cache (repeatable).")
    p.add_argument("--clip", action="append", default=[], metavar="PRETRAINED:ARCH",
                   help="Stage this clip_score OpenCLIP model into <root>/hf_cache (repeatable).")
    p.add_argument("--list", action="store_true", help="Print the manifest and exit.")
    p.add_argument("--verify-only", action="store_true", help="Only check what is already present; download nothing.")
    args = p.parse_args()

    root = os.path.abspath(os.path.expanduser(args.model_weight_root))
    hub = os.path.join(root, "hf_cache")   # terravis.utils.hf_cache
    root_repos = args.judge + args.t2i
    selected = [a for a in MANIFEST if a.key in args.only] if args.only else (
        [] if root_repos or args.clip else MANIFEST)
    print(f"model_weight_root = {root}")

    if args.list:
        for a in selected:
            print(f"\n[{a.key}] -> {a.folder}/  ({a.note})")
            for repo in a.snapshots:
                print(f"    snapshot {repo}")
            for _, filename in [*a.files, *a.urls]:
                print(f"    file     {filename}")
        for repo in root_repos:
            print(f"\n[hf_cache] snapshot {repo}")
        for name in args.clip:
            print(f"\n[hf_cache] clip     {name}")
        return

    failures = []
    if not args.verify_only:
        os.makedirs(root, exist_ok=True)
        for a in selected:
            try:
                stage_asset(a, root)
            except Exception as exc:
                failures.append(a.key)
                print(f"    ERROR staging {a.key}: {exc}", file=sys.stderr)
        for repo in root_repos:
            print(f"\n[hf_cache] {repo}")
            try:
                snapshot_download(repo_id=repo, cache_dir=hub, ignore_patterns=_IGNORE)
                print(f"    done     {repo}")
            except Exception as exc:
                failures.append(repo)
                print(f"    ERROR staging {repo}: {exc}", file=sys.stderr)
        for name in args.clip:
            try:
                stage_clip(name, hub)
            except Exception as exc:
                failures.append(name)
                print(f"    ERROR staging {name}: {exc}", file=sys.stderr)

    print("\n== verifying ==")
    missing = [a.key for a in selected if not verify_asset(a, root)]
    missing += [repo for repo in root_repos if not verify_snapshot(repo, hub)]
    missing += [name for name in args.clip if not verify_clip(name, hub)]
    if failures or missing:
        sys.exit(f"\n== FAILED (errors: {failures}; missing: {missing}) ==")
    print("\n== all staged ✓ ==")


if __name__ == "__main__":
    main()
