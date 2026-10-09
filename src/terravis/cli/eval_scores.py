"""CLI: score generated images with a T2I metric (`terravis-score`)."""

import argparse
import json
import os
from typing import Any

import torch
from tqdm import tqdm

from terravis.cli.summarize import mean_score
from terravis.configs import JudgeModel, find_judge
from terravis.scores.evaluator import T2IScoreEvaluator
from terravis.scores.metrics.terravis_score import has_empty_answer
from terravis.scores.registry import ScoreSpec, get_score_model

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")   # what the API judges accept


def resolve_judge(args) -> JudgeModel:
    if args.api_model_name and args.vqa_model_names.strip():
        raise ValueError("Pass one judge: --vqa_model_names (open-weight) or --api_model_name (API), not both.")
    if args.api_model_name:
        judge = find_judge(args.api_model_name, api_provider=args.api_provider)
        if not judge.is_api:
            raise ValueError(
                f"{judge.id!r} is an open-weight model; pass it via --vqa_model_names, "
                f"or add --api_provider to route it through an API."
            )
        return judge

    name = args.vqa_model_names.strip()
    if "," in name:
        raise ValueError(f"--vqa_model_names takes one judge, got {name!r}; run one job per judge.")
    judge = find_judge(name)
    if judge.is_api:
        raise ValueError(f"{judge.id!r} is an API model; pass it via --api_model_name instead.")
    return judge


def judge_sampling_params(judge: JudgeModel) -> dict:
    return {**judge.sampling, "do_sample": False}


def get_judge_label(spec: ScoreSpec, judge_name: str, clip_model_name: str) -> str:
    """Filename-safe judge label, e.g. 'gemma-4-31b-it' (frozen contract)."""
    if spec.judge_label:
        return spec.judge_label
    if spec.name == "clip_score":
        if not clip_model_name:
            raise ValueError("clip_score requires --clip_model_name.")
        return clip_model_name.split(":")[-1].lower()
    if not judge_name:
        raise ValueError(f"score_mode={spec.name!r} requires --vqa_model_names or --api_model_name.")
    return judge_name


def _inject_judge_model(results: list[dict[str, Any]], score_mode: str, judge_name: str) -> None:
    if score_mode == "terravis_score":
        for gd in results:
            gd[score_mode]["score_entries"]["judge_model"] = judge_name


def validate_api_args(args, spec, judge: JudgeModel | None) -> None:
    if args.api_model_name and not spec.supports_api:
        raise ValueError(f"--api_model_name is not supported for score_mode={args.score_mode!r}.")
    if not args.use_batch_api:
        return
    if not spec.supports_batch_api:
        raise ValueError(f"--use_batch_api is not supported for score_mode={args.score_mode!r}.")
    if not args.api_model_name:
        raise ValueError("--use_batch_api requires --api_model_name.")
    if judge.provider != "openai":
        raise ValueError("--use_batch_api supports only OpenAI.")


def _run_tag(benchmark_name, score_mode, judge, t2i_model_name, gen_data_seed) -> str:
    """The naming tag shared by result files and batch jobs (frozen contract)."""
    t2i = t2i_model_name.lower().replace(" ", "-")   # display name "FLUX.2 Dev" -> short name "flux.2-dev"
    return f"{benchmark_name}__{score_mode}__judge={judge}__@{t2i}__gen_seed={gen_data_seed}"


def _unanswered(scores: dict[str, Any]) -> bool:
    """A terravis_score n/a left by a failed or blocked judge request."""
    entries = scores.get("terravis_score", {}).get("score_entries", {})
    return "eligibility" in entries and has_empty_answer(entries)


def _load_partial(path: str) -> dict[str, Any]:
    """image_id -> scores cached by an earlier run; images whose judge requests failed are left out, to be retried."""
    if not os.path.exists(path):
        return {}
    cached, dropped = {}, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1   # a killed run leaves its last line half-written
                continue
            if not _unanswered(row["scores"]):
                cached[row["image_id"]] = row["scores"]
    if dropped:
        print(f"WARNING: dropped {dropped} unparseable line(s) from {path}; those images are rescored.")
    return cached


def _load_gen_data(gen_data_json_path: str):
    # A relative path not found here is read under SAVE_ROOT: genai_bench/my_model/seed=1111/test_data.json.
    if not os.path.isabs(gen_data_json_path) and not os.path.exists(gen_data_json_path) and os.environ.get("SAVE_ROOT"):
        gen_data_json_path = os.path.join(os.environ["SAVE_ROOT"], gen_data_json_path)
    with open(gen_data_json_path, "r", encoding="utf-8") as f:
        gen_data = json.load(f)
    # Step 1 writes test_data.json beside its images, so a moved or copied run is found there.
    run_dir = os.path.dirname(os.path.abspath(gen_data_json_path))
    for gd in gen_data:
        if not os.path.isfile(gd["gen_img_path"]):
            gd["gen_img_path"] = os.path.join(run_dir, os.path.basename(gd["gen_img_path"]))
    # A failed generation leaves a .txt marker where metadata records a .png.
    missing = [gd["gen_img_path"] for gd in gen_data if not os.path.isfile(gd["gen_img_path"])]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(gen_data)} generated images are missing "
            f"(first: {missing[0]}). Regenerate them before scoring."
        )
    return gen_data, gen_data[0]["benchmark"], gen_data[0]["t2i_model"], gen_data[0]["seed"]


def _load_image_dir(image_dir: str, prompts_json: str) -> list[dict[str, Any]]:
    """The images listed in prompts_json, or else every image in image_dir."""
    if prompts_json:
        with open(prompts_json, "r", encoding="utf-8-sig") as f:   # tolerate a BOM from Windows editors
            rows = json.load(f)
    else:
        # Step 1 folders are all named seed=<seed>, so their result files would overwrite each other.
        if os.path.isfile(os.path.join(image_dir, "test_data.json")):
            raise ValueError(f"{image_dir} is a generation run; pass its test_data.json as --gen_data_json_path.")
        rows = [{"image_id": name} for name in sorted(os.listdir(image_dir))
                if name.lower().endswith(IMAGE_EXTENSIONS) and not name.startswith(".")]   # "._x.png": macOS
        if not rows:
            raise FileNotFoundError(f"No {'/'.join(IMAGE_EXTENSIONS)} images in {image_dir}.")
    for row in rows:
        row["gen_img_path"] = os.path.join(image_dir, row["image_id"])
    missing = [row["image_id"] for row in rows if not os.path.isfile(row["gen_img_path"])]
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{len(rows)} images are missing from {image_dir} "
                                f"(first: {missing[0]}).")
    return rows


def _collect_scores(args, evaluator, gen_data, run_name, desc,
                    partial_path: str, batch_name: str) -> list[dict[str, Any]]:
    images = [gd["gen_img_path"] for gd in gen_data]

    if args.use_batch_api:
        batch_output_dir = args.batch_output_dir or os.path.join(args.save_path, run_name, "batch_api")
        results_by_idx = evaluator.scorer.score_batch(
            images=images,
            output_dir=batch_output_dir,
            batch_name=batch_name,
            completion_window=args.batch_completion_window,
            poll_interval=args.batch_poll_interval,
            verbose=not args.batch_silent,
            keep_ineligible_details=args.keep_ineligible_details,
            resume=args.resume,
        )
        print(f"Batch API intermediate files written under {batch_output_dir}.")
        return [results_by_idx[i] for i in range(len(images))]

    cached = _load_partial(partial_path) if args.resume else {}
    if cached:
        print(f"Resume: {len(cached)}/{len(gen_data)} images already scored in {partial_path}.")

    scores, failed_in_a_row = [], 0
    # "w", not "a": a half-written last line would swallow the next appended row.
    # Cached rows are written first, so a crash keeps them.
    with open(partial_path, "w", encoding="utf-8") as cache:
        cache.writelines(json.dumps({"image_id": i, "scores": s}, ensure_ascii=False) + "\n" for i, s in cached.items())
        cache.flush()
        for gd in tqdm(gen_data, desc=desc):
            row = cached.get(gd["image_id"])
            if row is None:
                row = evaluator(image=gd["gen_img_path"], text=gd.get("prompt"))
                cache.write(json.dumps({"image_id": gd["image_id"], "scores": row}, ensure_ascii=False) + "\n")
                cache.flush()
            failed_in_a_row = failed_in_a_row + 1 if _unanswered(row) else 0   # cached rows are answered
            scores.append(row)
            if failed_in_a_row == 5:
                raise RuntimeError("Judge requests failed for 5 images in a row. Check the API key, quota, "
                                   "network or judge server, then re-run with --resume.")
    return scores


def _run(args):
    spec = get_score_model(args.score_mode)
    judge = resolve_judge(args) if args.vqa_model_names or args.api_model_name else None
    judge_name = judge.short_name if judge else ""
    judge_label = get_judge_label(spec, judge_name, args.clip_model_name)
    validate_api_args(args, spec, judge)

    device = ("balanced" if spec.supports_multi_device
              else torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    sampling_params = judge_sampling_params(judge) if judge and not judge.is_api else None
    vqa_model_names = judge.id if judge and not judge.is_api else ""

    if args.prompts_json and args.image_dir is None:
        raise ValueError("--prompts_json goes with --image_dir.")
    if args.image_dir is not None:
        gen_data = _load_image_dir(args.image_dir, args.prompts_json)
        unprompted = [gd["image_id"] for gd in gen_data if not gd.get("prompt")]
        if spec.category == "prompt-conditioned" and unprompted:
            raise ValueError(
                f"score_mode={args.score_mode!r} needs each image's prompt via --prompts_json; "
                f"{len(unprompted)}/{len(gen_data)} images have none (first: {unprompted[0]})."
            )
        run_name = os.path.basename(os.path.abspath(args.image_dir))
        tag = f"{run_name}__{args.score_mode}__judge={judge_label}"
        desc = f"Computing '{args.score_mode}' on '{run_name}'"
    else:
        gen_data, run_name, t2i_model_name, gen_data_seed = _load_gen_data(args.gen_data_json_path)
        tag = _run_tag(run_name, args.score_mode, judge_label, t2i_model_name, gen_data_seed)
        desc = f"Computing '{args.score_mode}' for '{t2i_model_name}' on '{run_name}'"
    raw_dir = os.path.join(args.save_path, run_name, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    partial_path = os.path.join(raw_dir, f"t2i_scoring__partial__{tag}.jsonl")

    evaluator = T2IScoreEvaluator(
        score_mode=args.score_mode, device=device, model_weight_root=args.model_weight_root,
        dtype=args.dtype, clip_model_name=args.clip_model_name,
        vqa_model_names=vqa_model_names,
        api_provider=judge.provider if judge and judge.is_api else "",
        api_model_name=judge.id if judge and judge.is_api else "",
        sampling_params=sampling_params,
        vllm_tp_size=args.vllm_tp_size,
    )

    per_image_scores = _collect_scores(args, evaluator, gen_data, run_name, desc,
                                       partial_path, f"t2i_scoring__batch__{tag}")
    failed = [gd["image_id"] for gd, scores in zip(gen_data, per_image_scores) if _unanswered(scores)]

    for gd, scores in zip(gen_data, per_image_scores):
        gd.update(scores)
        gd.pop("gen_img_path")

    _inject_judge_model(gen_data, args.score_mode, judge_name)
    results_path = os.path.join(raw_dir, f"t2i_scoring__raw__{tag}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(gen_data, f, ensure_ascii=False, indent=2)
    print(f"All results written to {results_path}.")
    mean, n = mean_score(gen_data, args.score_mode)
    print(f"Mean {args.score_mode}: {mean} over {n}/{len(gen_data)} images (n/a excluded). "
          f"All runs: terravis-summarize {args.save_path or '.'}")
    if failed:
        print(f"WARNING: {len(failed)}/{len(gen_data)} images score n/a because a judge request failed or was "
              f"blocked (first: {', '.join(failed[:5])})."
              + ("" if args.use_batch_api else " Re-run with --resume to retry only those."))
    elif os.path.exists(partial_path):
        os.remove(partial_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score generated images with a T2I metric.")

    parser.add_argument("--score_mode", type=str, default="", help="Metric, e.g. terravis_score or pick_score.")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--gen_data_json_path", type=str,
                      help="test_data.json from terravis-generate; absolute, or relative to SAVE_ROOT.")
    data.add_argument("--image_dir", type=str,
                      help="Or a folder of images: all of them, or only those listed in --prompts_json.")
    parser.add_argument("--prompts_json", type=str, default="",
                        help='With --image_dir: [{"image_id": "cat.png", "prompt": "a cat"}, ...]; '
                             'needed by metrics that read the prompt.')
    parser.add_argument("--save_path", type=str, default="")

    parser.add_argument("--clip_model_name", type=str, default="",
                        help="OpenCLIP model for clip_score, format 'pretrained:arch'.")
    parser.add_argument("--vqa_model_names", type=str, default="",
                        help="Open-weight judge: HF repo id or registered short name.")

    parser.add_argument("--api_model_name", type=str, default="",
                        help="API judge model name, e.g. 'gpt-5.5' or 'gemini-3.8-flash'.")
    parser.add_argument("--api_provider", type=str, default="",
                        help="'openai' or 'google'; only needed for API models not in the registry.")

    parser.add_argument("--resume", action="store_true",
                        help="Reuse an interrupted run's cached scores or Batch API parts; same flags.")
    parser.add_argument("--use_batch_api", action="store_true")
    parser.add_argument("--batch_output_dir", type=str, default="")
    parser.add_argument("--batch_completion_window", type=str, default="24h")
    parser.add_argument("--batch_poll_interval", type=float, default=60.0)
    parser.add_argument("--batch_silent", action="store_true")
    parser.add_argument("--keep_ineligible_details", action="store_true",
                        help="Batch API only: keep per-question details for ineligible images.")

    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--model_weight_root", type=str, default=os.environ.get("MODEL_WEIGHT_ROOT") or None,
                        required=not os.environ.get("MODEL_WEIGHT_ROOT"),
                        help="Default: MODEL_WEIGHT_ROOT from scripts/set_root_paths.sh.")
    parser.add_argument("--vllm_tp_size", type=int, default=0,
                        help="vLLM tensor parallel size (0 = transformers).")
    return parser


def main() -> None:
    _run(_build_parser().parse_args())


if __name__ == '__main__':
    main()
