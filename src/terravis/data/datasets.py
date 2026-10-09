"""Bundled prompt pools, t2i_prompts-<benchmark>.json: image_id / prompt / data_source plus
the benchmark's own per-prompt annotations where it ships them."""

import json
import os
from importlib import resources


def list_prompt_pools() -> list[str]:
    return sorted(
        p.name.removeprefix("t2i_prompts-").removesuffix(".json")
        for p in resources.files("terravis.data").iterdir()
        if p.name.startswith("t2i_prompts-") and p.name.endswith(".json")
    )


def load_prompt_pool(benchmark: str) -> list[dict]:
    resource = resources.files("terravis.data").joinpath(f"t2i_prompts-{benchmark}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown benchmark '{benchmark}'. Bundled benchmarks: {list_prompt_pools()}, "
                         f"or a path to your own prompts .json.")
    with resource.open("r", encoding="utf-8") as f:
        return json.load(f)


def prompt_index(benchmark: str) -> dict[str, dict]:
    return {record["prompt"]: record for record in load_prompt_pool(benchmark)}


def benchmark_key(benchmark: str) -> str:
    """Directory and image-id prefix: the bundled name, or your prompts file's name."""
    if not benchmark.endswith(".json"):
        return benchmark
    key = os.path.splitext(os.path.basename(benchmark))[0]
    # Any spelling ("COCO-T2I", "Coco_T2I") would share the bundled run's images or score files.
    bundled = {n.lower().replace("-", "_") for b in list_prompt_pools()
               for n in (b, load_prompt_pool(b)[0]["data_source"])}
    if key.lower().replace("-", "_") in bundled:
        raise ValueError(f"Rename {benchmark}: '{key}' is a bundled benchmark.")
    return key


def load_gen_prompts(benchmark: str) -> list[dict]:
    """DataLoader rows: image_id / prompt / benchmark (the pool's data_source, or your file's name)."""
    if benchmark.endswith(".json"):
        with open(benchmark, "r", encoding="utf-8-sig") as f:
            prompts = json.load(f)
        if not (isinstance(prompts, list) and prompts and all(isinstance(p, str) and p.strip() for p in prompts)):
            raise ValueError(f"{benchmark} must be a JSON list of non-empty prompt strings.")
        key = benchmark_key(benchmark)
        # Fixed width: appending prompts never renames (and so reseeds) the earlier ones.
        return [{"image_id": f"{i:05d}", "prompt": p, "benchmark": key} for i, p in enumerate(prompts)]
    return [{"image_id": r["image_id"], "prompt": r["prompt"], "benchmark": r["data_source"]}
            for r in load_prompt_pool(benchmark)]
