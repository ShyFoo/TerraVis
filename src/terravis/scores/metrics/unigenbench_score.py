"""UniGenBench++ official English protocol, reimplemented from CodeGoat24/UniGenBench
``eval/src/eval_common.py`` (arXiv:2510.18701); defined only for the unigenbench++ prompts."""

from statistics import mean
from typing import Any

import torch

from terravis.data.datasets import prompt_index
from terravis.scores.base import ScoreModel
from terravis.scores.judges import build_judge
from terravis.scores.registry import register_score_model
from terravis.third_party.unigenbench.UniGenBench import (
    UNIGENBENCH_EVAL_MAX_TOKENS,
    build_system_prompt,
    parse_evaluation_response,
)

_UNIGENBENCH_DESCRIPTION = (
    "UniGenBench++ official protocol: a judge VLM checks each of the prompt's "
    "testpoints on the image (binary, with a written analysis per testpoint) and "
    "the image's score is the fraction of testpoints satisfied. Range: [0, 1], "
    "higher is better. The official leaderboards (one each for short and long prompts) "
    "pool each dimension's testpoints and average the primary dimensions for "
    "Overall; terravis-summarize prints them from score_entries."
)

DEFAULT_UNIGENBENCH_JUDGE = "google/gemma-4-31B-it"


class UniGenBenchScoreModel(ScoreModel):
    def __init__(self, judge: Any):
        super().__init__()
        self.judge = judge
        self._records = prompt_index("unigenbench++")

    @torch.inference_mode()
    def score(self, image: str, text: str) -> dict[str, Any]:
        rec = self._records.get(text)
        if rec is None:
            raise ValueError(
                "unigenbench_score is defined only for the UniGenBench++ benchmark's own "
                f"prompts (t2i_prompts-unigenbench++.json); got unknown prompt: {text!r}"
            )
        testpoints = rec["testpoints"]
        judge_prompt = build_system_prompt(text, testpoints, rec["testpoint_descriptions"])
        output = self.judge.infer(judge_prompt, image)

        parsed = parse_evaluation_response(output, testpoints)
        if parsed is None:
            return {
                "score": "n/a",
                "score_entries": {
                    "testpoints": testpoints,
                    "prompt_type": rec["prompt_type"],
                    "raw_output": output,
                    "score_description": _UNIGENBENCH_DESCRIPTION,
                },
            }

        analysis, scores = parsed
        # Upstream counts a testpoint satisfied iff its entry == 1.
        per_testpoint = [float(s == 1) for s in scores]
        return {
            "score": float(mean(per_testpoint)),
            "score_entries": {
                "testpoints": testpoints,
                "per_testpoint": per_testpoint,
                # literal_eval can yield lone surrogates, which break the results JSON write.
                "analysis": [str(a).encode("utf-8", "replace").decode("utf-8")
                             for a in analysis],
                "prompt_type": rec["prompt_type"],
                "score_description": _UNIGENBENCH_DESCRIPTION,
            },
        }


def build_unigenbench_score(ctx) -> UniGenBenchScoreModel:
    model_name = ctx.vqa_model_names or DEFAULT_UNIGENBENCH_JUDGE
    # The protocol's budget overrides the judge's max_new_tokens.
    return UniGenBenchScoreModel(
        build_judge(ctx, model_name, {**ctx.sampling_params, "max_new_tokens": UNIGENBENCH_EVAL_MAX_TOKENS}))


register_score_model(
    name="unigenbench_score", category="prompt-conditioned",
    description=_UNIGENBENCH_DESCRIPTION, builder=build_unigenbench_score,
    supports_multi_device=True,
)
