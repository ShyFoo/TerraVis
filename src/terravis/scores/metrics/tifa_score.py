from statistics import mean
from typing import Any

import torch

from terravis.scores.base import ScoreModel
from terravis.scores.judges import build_judge
from terravis.scores.registry import register_score_model
from terravis.third_party.tifa.TIFA import (
    QUESTION_GENERATION_MAX_TOKENS,
    SBERTModel,
    build_multiple_choice_question,
    build_question_generation_prompt,
    multiple_choice_from_free_form,
    parse_question_generation_response,
)

_TIFA_DESCRIPTION = (
    "A text-image faithfulness score predicted by TIFA: an LLM generates multiple-choice "
    "question-answer pairs from the prompt and a VQA judge answers them on the image; the "
    "score is the fraction of questions answered correctly. Range: [0, 1], higher is better."
)

DEFAULT_TIFA_JUDGE = "google/gemma-4-31B-it"


class TIFAScoreModel(ScoreModel):
    """Paper: https://arxiv.org/abs/2303.11897"""

    def __init__(self, judge: Any, model_weight_root: str):
        super().__init__()
        self.judge = judge
        # Judge may own every GPU; pin tiny SBERT to cuda:0 (fp32, as upstream).
        self._sbert = SBERTModel(
            model_weight_root=model_weight_root,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        self._qa_cache: dict[str, list[dict]] = {}

    def generate_qa_pairs(self, text: str) -> list[dict]:
        if text not in self._qa_cache:
            response = self.judge.infer(build_question_generation_prompt(text),
                                        max_tokens=QUESTION_GENERATION_MAX_TOKENS)
            self._qa_cache[text] = parse_question_generation_response(text, response)
        return self._qa_cache[text]

    @torch.inference_mode()
    def score(self, image: str, text: str) -> dict[str, Any]:
        qa_pairs = self.generate_qa_pairs(text)
        questions = [build_multiple_choice_question(qa["question"], qa["choices"]) for qa in qa_pairs]
        if self.judge.infer_many is not None and questions:
            free_form_answers = self.judge.infer_many(questions, image)
        else:
            free_form_answers = [self.judge.infer(question, image) for question in questions]

        question_details: dict[str, Any] = {}
        question_scores: list[int] = []
        for qa, free_form in zip(qa_pairs, free_form_answers):
            mc_answer = multiple_choice_from_free_form(self._sbert, free_form, qa["choices"])
            question_score = int(mc_answer == qa["answer"])
            question_scores.append(question_score)
            question_details[qa["question"]] = {
                **qa, "free_form_vqa": free_form,
                "multiple_choice_vqa": mc_answer, "scores": question_score,
            }

        score = float(mean(question_scores)) if question_scores else "n/a"
        return {
            "score": score,
            "score_entries": {
                "question_details": question_details,
                "score_description": _TIFA_DESCRIPTION,
            },
        }


def build_tifa_score(ctx) -> TIFAScoreModel:
    model_name = ctx.vqa_model_names or DEFAULT_TIFA_JUDGE
    judge = build_judge(ctx, model_name, {"max_new_tokens": 256, **ctx.sampling_params})
    return TIFAScoreModel(judge, model_weight_root=ctx.model_weight_root)


register_score_model(
    name="tifa_score", category="prompt-conditioned",
    description=_TIFA_DESCRIPTION, builder=build_tifa_score,
    supports_multi_device=True,
)
