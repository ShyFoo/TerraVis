from collections.abc import Iterable
from typing import Any

from terravis.scores.judges import build_judge
from terravis.scores.registry import register_score_model

VQA_SCORE_DESCRIPTION = (
    "A text-image alignment score predicted by VQAScore. "
    "The score is defined as the probability of answering 'Yes' to "
    "a binary question ('Does this figure show \"{prompt}\"? Please answer yes or no.'). "
    "Range: [0, 1], higher is better."
)


def build_vqa_question(text: str) -> str:
    """Official VQAScore binary question (linzhiqiu/t2v_metrics, Apache-2.0); every backend must ask exactly this."""
    return f'Does this figure show "{text}"? Please answer yes or no.'


def yes_probability(first_token_candidates: Iterable[tuple[str, float]]) -> float:
    """Max probability among the first generated token's ``(token, probability)`` candidates that decode to "yes"."""
    yes = [p for token, p in first_token_candidates if token.strip().lower() == "yes"]
    return max(yes) if yes else 0.0


class VQAScoreModel:
    """Paper: https://arxiv.org/abs/2404.01291"""

    def __init__(self, judge: Any):
        self.judge = judge

    def score(self, image: str, text: str) -> dict[str, Any]:
        raw_answer, candidates = self.judge.first_token(build_vqa_question(text), image)
        return {
            "score": yes_probability(candidates),
            "score_entries": {"raw_answers": raw_answer, "score_description": VQA_SCORE_DESCRIPTION},
        }

    __call__ = score


def build_vqa_score(ctx) -> VQAScoreModel:
    # Greedy: the score is the first token's yes-probability.
    return VQAScoreModel(build_judge(ctx, ctx.vqa_model_names, {**ctx.sampling_params, "do_sample": False}))


register_score_model(
    name="vqa_score", category="prompt-conditioned",
    description=VQA_SCORE_DESCRIPTION,
    builder=build_vqa_score, supports_multi_device=True,
)
