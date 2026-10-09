"""CLI: generate benchmark images with a T2I model (`terravis-generate`)."""

import argparse
import glob
import hashlib
import json
import os
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial
from typing import Any

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from terravis.configs import T2IModel, find_t2i
from terravis.data.datasets import benchmark_key, load_gen_prompts
from terravis.models.text2image import Text2ImageAPIPipeline, Text2ImageModelPipeline
from terravis.utils import seed_all

PARALLEL_MODES = ("data_parallel", "model_sharding", "api")


class ImageWriter:
    """Threaded PNG encoding: ~0.27 s per 1024x1024 PNG, and PIL releases the GIL."""

    def __init__(self, max_workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="png-save")
        self._pending = deque()
        self._max_pending = 4 * max_workers

    def save(self, image: Image.Image, path: str) -> None:
        self._pending.append(self._pool.submit(self._write, image, path))
        while len(self._pending) > self._max_pending:
            self._pending.popleft().result()

    @staticmethod
    def _write(image: Image.Image, path: str) -> None:
        # pid in the temp name: DistributedSampler's padding renders one prompt on two ranks.
        tmp = f"{path}.{os.getpid()}.partial"
        image.save(tmp, format="PNG")
        os.replace(tmp, path)

    def __enter__(self) -> "ImageWriter":
        return self

    def __exit__(self, exc_type, *_) -> None:
        self._pool.shutdown(wait=True)
        if exc_type is None:
            for future in self._pending:
                future.result()


def prompt_seeds(image_ids: list[str], seed: int) -> list[int]:
    """Per-prompt seeds from (seed, image_id) alone, independent of batch size and GPU count."""
    return [
        # 63 bits: torch seeds stop at 2**64 - 1 and HiDream seeds with seed + 1.
        int.from_bytes(hashlib.blake2b(f"{seed}:{image_id}".encode(), digest_size=8).digest(), "big") >> 1
        for image_id in image_ids
    ]


def meta_row(prompt_data: dict[str, Any], t2i_spec: T2IModel, key: str,
             total_num_images: int, seed: int, save_path: str) -> dict:
    # benchmark_key() (coco_t2i) prefixes ids; data_source ("COCO-T2I") is metadata.
    image_id = f"{key}-{t2i_spec.short_name}-seed={seed}-{prompt_data['image_id']}"
    return {
        "benchmark": prompt_data["benchmark"],
        "num_images": total_num_images,
        "t2i_model": t2i_spec.display_name,
        "image_id": image_id,
        "prompt": prompt_data["prompt"],
        "seed": seed,
        "gen_img_path": os.path.join(save_path, f"{image_id}.png"),
    }


def resume_pending(dataset, make_row: Callable[[dict], dict], save_path: str,
                   announce: bool = True):
    """Returns (Subset of prompts not yet on disk, metadata rows of those that are)."""
    on_disk = set(os.listdir(save_path))
    done_meta, pending = [], []
    for idx in range(len(dataset)):
        row = make_row(dataset[idx])
        if f"{row['image_id']}.png" in on_disk or f"{row['image_id']}.txt" in on_disk:
            done_meta.append(row)
        else:
            pending.append(idx)
    if announce:
        print(f"Resume: {len(done_meta)}/{len(dataset)} of this run's prompts are already on disk, "
              f"rendering the remaining {len(pending)}.")
    return Subset(dataset, pending), done_meta


def save_results(batch_prompts_data: dict[str, str | list],
                 batch_gen_img_results: list[Image.Image | None],
                 writer: ImageWriter, make_row: Callable[[dict], dict]) -> list[dict]:
    batch_gen_img_meta = []
    for i, image in enumerate(batch_gen_img_results):
        row = make_row({k: batch_prompts_data[k][i] for k in ("image_id", "prompt", "benchmark")})

        if image is not None:
            writer.save(image, row["gen_img_path"])
        else:
            with open(os.path.splitext(row["gen_img_path"])[0] + ".txt", "w", encoding="utf-8") as f:
                f.write("Generation failed or returned None.\n")

        batch_gen_img_meta.append(row)
    return batch_gen_img_meta


def write_metadata(img_save_path: str, gen_img_meta: list[dict], shard: str = "") -> None:
    """``shard`` (``"<i>-of-<n>"``) names a per-replica partial file."""
    name = f"test_data-shard{shard}.json" if shard else "test_data.json"
    gen_img_meta = sorted(gen_img_meta, key=lambda m: int(m["image_id"].rsplit("-", 1)[-1]))   # prompt order
    with open(os.path.join(img_save_path, name), "w", encoding="utf-8") as f:
        json.dump(gen_img_meta, f, ensure_ascii=False, indent=2)


def merge_shard_metadata(img_save_path: str) -> None:
    parts = sorted(glob.glob(os.path.join(img_save_path, "test_data-shard*-of-*.json")))
    if not parts:
        raise FileNotFoundError(
            f"No shard metadata under {img_save_path}. Either the replicas never ran, "
            f"or they were launched without --num_shards."
        )

    merged, seen = [], set()
    for part in parts:
        with open(part, "r", encoding="utf-8") as f:
            for row in json.load(f):
                if row["image_id"] not in seen:
                    seen.add(row["image_id"])
                    merged.append(row)

    expected = merged[0]["num_images"] if merged else 0
    if len(merged) != expected:
        print(f"WARNING: merged {len(merged)} rows from {len(parts)} shards, "
              f"expected {expected}; a replica is missing or died early.")

    write_metadata(img_save_path, merged)
    for part in parts:
        os.remove(part)
    print(f"Merged {len(parts)} shards into {os.path.join(img_save_path, 'test_data.json')} "
          f"({len(merged)} images).")


def local_generate(pipeline, prompts_data: dict[str, list], seed: int,
                   **call_kwargs) -> list[Image.Image | None]:
    prompts = prompts_data["prompt"]
    seeds = prompt_seeds(prompts_data["image_id"], seed)
    if hasattr(pipeline, "generate_image"):
        return pipeline.generate_image(prompt=prompts, seeds=seeds, **call_kwargs)
    generators = [torch.Generator().manual_seed(s) for s in seeds]   # CPU: the same noise on any GPU
    return pipeline(prompt=prompts, output_type="pil", generator=generators, **call_kwargs).images


def generate_and_save(dataloader: DataLoader,
                      generate_fn: Callable[[dict[str, list]], list],
                      save_batch: Callable[..., list[dict]],
                      done_meta: list[dict]) -> list[dict]:
    """Loop shared by all parallel modes; no rank sync, each writes its own files."""
    meta = list(done_meta)
    with ImageWriter() as writer:
        for prompts_data in tqdm(dataloader, desc="Generating images..."):
            gen_images = generate_fn(prompts_data)
            meta.extend(save_batch(batch_prompts_data=prompts_data,
                                   batch_gen_img_results=gen_images, writer=writer))
    return meta


def run_data_parallel(pipeline, dataset, batch_size: int, num_workers: int, seed: int,
                      accelerator: Accelerator, save_batch, img_save_path: str,
                      call_kwargs: dict[str, Any], done_meta: list[dict]) -> None:
    sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes,
                                 rank=accelerator.process_index, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    pipeline.to(accelerator.device)

    generate_fn = partial(local_generate, pipeline, seed=seed, **call_kwargs)
    per_gpu_meta = generate_and_save(dataloader, generate_fn, save_batch,
                                     done_meta if accelerator.is_main_process else [])
    all_meta = accelerator.gather_for_metrics(per_gpu_meta, use_gather_object=True)

    if accelerator.is_main_process:
        # Unprepared dataloader: gather_for_metrics keeps DistributedSampler's padding duplicates.
        all_meta = list({m["image_id"]: m for m in all_meta}.values())
        write_metadata(img_save_path, all_meta)


def _run(args):
    spec = find_t2i(args.model_name)
    key = benchmark_key(args.benchmark)
    img_save_path = os.path.join(args.save_root, key, spec.short_name, f"seed={args.seed}")

    if args.merge_shards:
        merge_shard_metadata(img_save_path)
        return

    seed_all(args.seed)
    # Disables the 10 min NCCL watchdog.
    timeout = InitProcessGroupKwargs(timeout=timedelta(hours=100000))
    accelerator = Accelerator(kwargs_handlers=[timeout])

    parallel_mode = args.parallel_mode or spec.parallel_mode
    dtype = args.dtype or spec.dtype
    batch_size = args.eval_batch_size if args.eval_batch_size is not None else spec.batch_size
    num_workers = args.num_workers if args.num_workers is not None else spec.num_workers
    use_batch_api = args.use_batch_api if args.use_batch_api is not None else spec.use_batch_api
    torch_compile = args.torch_compile or spec.torch_compile

    if parallel_mode not in PARALLEL_MODES:
        raise ValueError(f"Unknown parallel mode: {parallel_mode}, supported modes: {PARALLEL_MODES}.")
    if args.num_shards > 1 and parallel_mode != "model_sharding":
        raise ValueError("--num_shards only applies to model_sharding.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard_index must be in [0, {args.num_shards}), got {args.shard_index}.")

    dataset = load_gen_prompts(args.benchmark)
    os.makedirs(img_save_path, exist_ok=True)
    total_num_images = len(dataset)  # the benchmark total, not this run's share
    make_row = partial(meta_row, t2i_spec=spec, key=key,
                       total_num_images=total_num_images, seed=args.seed, save_path=img_save_path)
    save_batch = partial(save_results, make_row=make_row)
    load_local = partial(Text2ImageModelPipeline.load, model_name=args.model_name,
                         model_weight_root=args.model_weight_root, parallel_mode=parallel_mode,
                         dtype=dtype, torch_compile=torch_compile)

    done_meta = []
    if parallel_mode == "data_parallel":
        if args.resume:
            dataset, done_meta = resume_pending(dataset, make_row, img_save_path,
                                                announce=accelerator.is_main_process)
            # Every rank must finish its scan before any writes, or the samplers see different lengths.
            accelerator.wait_for_everyone()
        run_data_parallel(load_local(), dataset, batch_size, num_workers, args.seed,
                          accelerator, save_batch, img_save_path, spec.generation_kwargs, done_meta)
        return

    if not accelerator.is_main_process:
        return

    shard = f"{args.shard_index}-of-{args.num_shards}" if args.num_shards > 1 else ""
    dataset = Subset(dataset, range(args.shard_index, len(dataset), args.num_shards))
    if args.resume:
        # After the shard split, so a replica keeps its original slice.
        dataset, done_meta = resume_pending(dataset, make_row, img_save_path)

    if parallel_mode == "model_sharding":
        if batch_size != 1:
            raise ValueError("model_sharding generation runs with batch_size=1.")
        generate_fn = partial(local_generate, load_local(), seed=args.seed, **spec.generation_kwargs)
    else:
        if use_batch_api and batch_size == 1:
            raise ValueError("--use_batch_api with batch_size 1 submits one Batch job per prompt; "
                             "raise --eval_batch_size (or the registry batch_size).")
        if not use_batch_api:
            batch_size = 1   # one request per prompt: each image is saved as soon as it arrives

        pipeline = Text2ImageAPIPipeline(model_name=args.model_name)

        def generate_fn(prompts_data):
            prompts = prompts_data["prompt"]
            return pipeline.generate_batch(prompts) if use_batch_api else pipeline.generate(prompts[0])

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    write_metadata(img_save_path, generate_and_save(dataloader, generate_fn, save_batch, done_meta), shard)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate benchmark images with a T2I model. "
                    "Unset flags fall back to the model registry defaults."
    )
    parser.add_argument('--model_name', type=str, required=True,
                        help="HF repo id, API model name, registry short name, or local checkpoint path.")
    parser.add_argument('--benchmark', type=str, default="coco_t2i",
                        help="Bundled benchmark name, or your own prompts: a .json list of strings.")
    parser.add_argument('--seed', type=int, default=1111)

    parser.add_argument('--model_weight_root', type=str, default=os.environ.get("MODEL_WEIGHT_ROOT") or None,
                        required=not os.environ.get("MODEL_WEIGHT_ROOT"),
                        help="Default: MODEL_WEIGHT_ROOT from scripts/set_root_paths.sh.")
    parser.add_argument('--save_root', type=str, default=os.environ.get("SAVE_ROOT") or None,
                        required=not os.environ.get("SAVE_ROOT"),
                        help="Default: SAVE_ROOT from scripts/set_root_paths.sh.")

    parser.add_argument('--parallel_mode', type=str, default=None, choices=PARALLEL_MODES)
    parser.add_argument('--dtype', type=str, default=None, choices=["fp16", "bf16", "fp32", "auto"],
                        help="'auto' keeps checkpoint dtypes; ideogram requires it, diffusers and cosmos3 reject it.")
    parser.add_argument('--eval_batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--use_batch_api', action=argparse.BooleanOptionalAction, default=None,
                        help="Provider Batch API; --no-use_batch_api overrides a registry default.")
    parser.add_argument('--resume', action="store_true",
                        help="Skip prompts with a .png or .txt marker on disk; delete a marker to retry.")
    parser.add_argument('--torch_compile', action="store_true",
                        help="torch.compile the denoiser; not bit-exact, not for model_sharding.")

    # Replica sharding; scripts/t2i_generation.sh sets these.
    parser.add_argument('--num_shards', type=int, default=1,
                        help="Number of model_sharding replicas splitting this benchmark.")
    parser.add_argument('--shard_index', type=int, default=0,
                        help="Which slice (0-based) this replica renders.")
    parser.add_argument('--merge_shards', action="store_true",
                        help="Merge replica metadata into test_data.json and exit, after all replicas finish.")
    return parser


def main() -> None:
    _run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
