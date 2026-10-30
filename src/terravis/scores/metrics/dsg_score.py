from statistics import mean
from typing import Any

import torch

from terravis.scores.base import ScoreModel
from terravis.scores.judges import build_judge
from terravis.scores.registry import register_score_model
from terravis.third_party.dsg.DSG import (
    DSG_GENERATION_MAX_TOKENS,
    binary_verdict,
    build_binary_question,
    build_dependency_prompt,
    build_question_prompt,
    build_tuple_prompt,
    dependency_filtered_scores,
    parse_dependency_response,
    parse_question_response,
    parse_tuple_response,
    strip_generation_echoes,
)

_DSG_DESCRIPTION = (
    "A text-image faithfulness score predicted by DSG (Davidsonian Scene Graph): an LLM "
    "extracts atomic semantic tuples from the prompt, rewrites each as a yes/no question "
    "with dependencies between the questions, and a VQA judge answers them on the image; "
    "a question whose parent question failed is zeroed out, and the score is the fraction "
    "of questions answered 'yes'. Range: [0, 1], higher is better."
)

DEFAULT_DSG_JUDGE = "google/gemma-4-31B-it"

QuestionGraph = tuple[dict[int, str], dict[int, str], dict[int, list[int]]]


class DSGScoreModel(ScoreModel):
    """Paper: https://arxiv.org/abs/2310.18235"""

    def __init__(self, judge: Any):
        super().__init__()
        self.judge = judge
        self._graph_cache: dict[str, QuestionGraph] = {}

    def generate_question_graph(self, text: str) -> QuestionGraph:
        if text not in self._graph_cache:
            budget = {"max_tokens": DSG_GENERATION_MAX_TOKENS}
            tuple_body = strip_generation_echoes(self.judge.infer(build_tuple_prompt(text), **budget))
            question_response = self.judge.infer(build_question_prompt(text, tuple_body), **budget)
            dependency_response = self.judge.infer(build_dependency_prompt(text, tuple_body), **budget)
            self._graph_cache[text] = (
                parse_tuple_response(tuple_body),
                parse_question_response(question_response),
                parse_dependency_response(dependency_response),
            )
        return self._graph_cache[text]

    @torch.inference_mode()
    def score(self, image: str, text: str) -> dict[str, Any]:
        qid2tuple, qid2question, qid2dependency = self.generate_question_graph(text)
        qids = sorted(qid2question)
        questions = [build_binary_question(qid2question[qid]) for qid in qids]
        if self.judge.infer_many is not None and questions:
            free_form_answers = self.judge.infer_many(questions, image)
        else:
            free_form_answers = [self.judge.infer(question, image) for question in questions]

        qid2answer = {qid: binary_verdict(answer) for qid, answer in zip(qids, free_form_answers)}
        qid2scores = {qid: float(answer == "yes") for qid, answer in qid2answer.items()}
        filtered_scores, qid2validity = dependency_filtered_scores(qid2scores, qid2dependency)

        question_details: dict[str, Any] = {}
        for qid, free_form in zip(qids, free_form_answers):
            question_details[str(qid)] = {
                "tuple": qid2tuple.get(qid, ""),
                "question": qid2question[qid],
                "dependencies": qid2dependency.get(qid, [0]),
                "free_form_vqa": free_form,
                "vqa_answer": qid2answer[qid],
                "valid": qid2validity[qid],
                "scores": filtered_scores[qid],
            }

        score = float(mean(filtered_scores.values())) if filtered_scores else "n/a"
        return {
            "score": score,
            "score_entries": {
                "question_details": question_details,
                "average_score_without_dependency":
                    float(mean(qid2scores.values())) if qid2scores else "n/a",
                "score_description": _DSG_DESCRIPTION,
            },
        }


def build_dsg_score(ctx) -> DSGScoreModel:
    model_name = ctx.vqa_model_names or DEFAULT_DSG_JUDGE
    return DSGScoreModel(build_judge(ctx, model_name, {"max_new_tokens": 256, **ctx.sampling_params}))


register_score_model(
    name="dsg_score", category="prompt-conditioned",
    description=_DSG_DESCRIPTION, builder=build_dsg_score,
    supports_multi_device=True,
)
