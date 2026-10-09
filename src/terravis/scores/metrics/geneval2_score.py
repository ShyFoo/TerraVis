"""GenEval 2 Soft-TIFA, reimplemented from facebookresearch/GenEval2 ``evaluation.py``
(arXiv:2512.16853); defined only for the geneval2 prompts. The answer-matching rules are GenEval 2's
(CC BY-NC 4.0, Copyright (c) 2024 Meta Platforms, Inc. and affiliates): noncommercial use only."""

import os
from typing import Any

import torch
from scipy.stats import gmean
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from terravis.data.datasets import prompt_index
from terravis.scores.base import ScoreModel
from terravis.scores.registry import register_score_model

_GENEVAL2_DESCRIPTION = (
    "GenEval 2 Soft-TIFA GM: per-question first-token probability mass on the official "
    "answer variants, geometric-mean over the prompt's questions. In [0, 1], higher is "
    "better; 100 x the mean over all 800 prompts is the official benchmark number."
)

# Pinned by the official protocol.
GENEVAL2_JUDGE_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Upstream return_numeric_string.
_NUMERALS = {w: str(i) for i, w in enumerate(
    ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"], start=1)}


def _answer_variants(question: str, answer: str) -> list[str]:
    """Upstream's list, including the literal "How many" test."""
    if question.startswith("How many"):
        numeric = _NUMERALS.get(answer, "other")
        return [answer, answer.capitalize(), " " + answer, " " + answer.capitalize(),
                numeric, " " + numeric]
    return ["Yes", "yes", " yes", " Yes"]


class GenEval2ScoreModel(ScoreModel):
    def __init__(self, model_weight_root: str, device: str | torch.device):
        weights_dir = os.path.join(model_weight_root, "GenEval2")
        processor = AutoProcessor.from_pretrained(GENEVAL2_JUDGE_REPO_ID, cache_dir=weights_dir)
        # dtype="auto" (checkpoint bf16) is pinned; another dtype shifts the numbers.
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            GENEVAL2_JUDGE_REPO_ID, dtype="auto", cache_dir=weights_dir,
        )
        model.eval()
        model.to(device)
        super().__init__(model=model, processor=processor, device=device, dtype=model.dtype)
        self._records = prompt_index("geneval2")

    def _first_token_probs(self, text: str, image: str) -> torch.Tensor:
        # Upstream's call shape: image-first content, greedy single-token decode.
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=1, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
        return torch.nn.functional.softmax(out.scores[0], dim=-1)

    @torch.inference_mode()
    def score(self, image: str, text: str) -> dict[str, Any]:
        rec = self._records.get(text)
        if rec is None:
            raise ValueError(
                "geneval2_score is defined only for the GenEval2 benchmark's own prompts "
                f"(t2i_prompts-geneval2.json); got unknown prompt: {text!r}"
            )
        per_question = []
        for question, answer in rec["vqa_list"]:
            probs = self._first_token_probs(f"{question} Answer in one word.", image)
            per_question.append(sum(
                probs[0, self.processor.tokenizer.encode(v)[0]].item()
                for v in _answer_variants(question, answer)
            ))
        return {
            "score": float(gmean(per_question)),
            "score_entries": {
                "soft_tifa_am": sum(per_question) / len(per_question),
                "per_question": per_question,
                "skills": rec["skills"],
                "atom_count": rec["atom_count"],
                "score_description": _GENEVAL2_DESCRIPTION,
            },
        }


def build_geneval2_score(ctx) -> GenEval2ScoreModel:
    return GenEval2ScoreModel(model_weight_root=ctx.model_weight_root, device=ctx.device)


register_score_model(
    name="geneval2_score", category="prompt-conditioned",
    description=_GENEVAL2_DESCRIPTION, builder=build_geneval2_score,
    judge_label="qwen3-vl-8b-instruct",
)
