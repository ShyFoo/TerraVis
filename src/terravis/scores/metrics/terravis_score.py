"""Eligibility -> violation detection -> severity -> exponential penalty, over any judge with ``infer``."""

import math
import re
from collections.abc import Callable
from typing import Any

from tqdm import tqdm

from terravis.scores.judges import build_judge
from terravis.scores.metrics.terravis_prompts import TERRAVIS_PROMPT
from terravis.scores.registry import register_score_model

# Answer parsing. Off-format: eligibility -> n/a, detection -> no violation, severity -> minor.

_FIELD_WRAP = r"\s>*_`\-•#\"'"      # junk before the label / around the colon
# Same, minus newline: the gap after the colon must not swallow the next line.
_FIELD_WRAP_INLINE = r" \t>*_`\-•#\"'"
_VALUE_STRIP = " \t\r\n.。,:;*_`\"'()[]{}<>"

# A "Field:" label line, which the value-on-next-line fallback must not swallow.
_LABEL_LINE_RE = re.compile(rf"^[{_FIELD_WRAP}]*[A-Za-z][\w /()-]*[:：]")


def _field_candidates(text: str, field_name: str):
    """Candidate values for one field, best first: line-start labels, later occurrences, then value-on-next-line."""
    text = str(text)
    label = re.escape(field_name)
    line_start = rf"(?im)^[{_FIELD_WRAP}]*{label}[{_FIELD_WRAP}]*[:：][{_FIELD_WRAP_INLINE}]*(.*?)\s*$"
    mid_line = rf"(?im)(?<![a-z0-9]){label}[{_FIELD_WRAP}]*[:：][{_FIELD_WRAP_INLINE}]*(.*?)\s*$"
    for pattern in (line_start, mid_line):
        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):
            value = match.group(1).strip(_VALUE_STRIP).strip()
            if value:
                yield value
        if matches:
            tail = text[matches[-1].end():].lstrip()
            next_line = tail.splitlines()[0].strip() if tail else ""
            if next_line and not _LABEL_LINE_RE.match(next_line):
                value = next_line.strip(_VALUE_STRIP).strip()
                if value:
                    yield value


def extract_output_field(text: str, field_name: str) -> str:
    if text is None:
        return ""
    return next(_field_candidates(text, field_name), "")


def normalize_label(text: str) -> str:
    """Lowercase without wrappers; keeps "/" so "n/a" survives."""
    return str(text).strip().strip(_VALUE_STRIP).lower().strip(_VALUE_STRIP)


# Word-bounded ("Noticeable" must not read as "No"); rejects "Yes or No" echoes.
def _label_is(answer: str, alternatives: str) -> bool:
    return re.match(rf"(?:{alternatives})\b(?!\s+(?:or|and)\b)", answer) is not None


def parse_yes_no_answer(raw_answer: str) -> bool | None:
    for value in _field_candidates(raw_answer or "", "Answer"):
        answer = normalize_label(value)
        if _label_is(answer, "yes"):
            return True
        if _label_is(answer, "no(?:t|ne|pe)?"):
            return False
    return None


def parse_eligibility_answer(raw_answer: str) -> bool | None:
    for value in _field_candidates(raw_answer or "", "Answer"):
        answer = normalize_label(value)
        if _label_is(answer, "not applicable|not eligible|ineligible|n/?a"):
            return False
        if _label_is(answer, "eligible"):
            return True
        # Off-format but unambiguous: the prompt asks if the image is eligible.
        if _label_is(answer, "yes"):
            return True
        if _label_is(answer, "no"):
            return False
    return None


def parse_severity_answer(raw_answer: str) -> str | None:
    for value in _field_candidates(raw_answer or "", "Severity"):
        severity = normalize_label(value)
        if _label_is(severity, "major"):
            return "major"
        if _label_is(severity, "minor"):
            return "minor"
    return None


# Prompt building; every backend asks the same question.

def fill_prompt_template(template: str, values: dict[str, str]) -> str:
    # Literal replacement, not str.format: prompt text may contain braces.
    text = template
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def is_terravis_workflow_prompt(prompts: Any) -> bool:
    if not isinstance(prompts, dict):
        return False
    required_keys = {"eligibility_prompt", "violation_prompt_template",
                     "severity_prompt_template", "violation_questions"}
    return str(prompts.get("method", "")).lower() == "terravis" or required_keys.issubset(prompts.keys())


def get_violation_question_fields(question_item: dict) -> dict[str, str]:
    return {
        "violation_type": question_item.get("violation_type", ""),
        "domain": question_item.get("domain", ""),
        "question": question_item.get("question", ""),
    }


def build_detection_prompt(prompts: dict, fields: dict[str, str]) -> str:
    return fill_prompt_template(prompts["violation_prompt_template"],
                                {"QUESTION": fields["question"]})


def build_severity_prompt(prompts: dict, fields: dict[str, str], detection_raw: str) -> str:
    """Embeds fields from the detection output; only buildable after a yes."""
    return fill_prompt_template(prompts["severity_prompt_template"], {
        "QUESTION": fields["question"],
        "VIOLATION_TYPE": fields["violation_type"],
        "AFFECTED_ASPECT": extract_output_field(detection_raw, "Affected aspect"),
        "EVIDENCE": extract_output_field(detection_raw, "Evidence"),
    })


# Raw-answer collection.

ELIGIBILITY_PARSE_RETRIES = 3


def infer_eligibility(prompts: dict, infer_fn: Callable[..., str]):
    """``(raw, eligible)``; retries only an unparseable answer, at a higher temperature."""
    eligibility_prompt = prompts["eligibility_prompt"]
    raw, eligible = "", None
    for attempt in range(ELIGIBILITY_PARSE_RETRIES):
        if attempt == 0:
            raw = infer_fn(eligibility_prompt)
        else:
            temperature = min(0.4 + 0.2 * (attempt - 1), 0.9)
            try:
                raw = infer_fn(eligibility_prompt, temperature=temperature)
            except TypeError:                                   # no override support
                raw = infer_fn(eligibility_prompt)
        eligible = parse_eligibility_answer(raw)
        if eligible is not None or not raw:   # "" is a failed request, already retried by its backend
            break
    return raw, eligible


def collect_workflow_raw_for_image(
    infer_fn: Callable[..., str],
    prompts: dict,
    infer_many_fn: Callable[[list[str]], list[str]] | None = None,
) -> dict:
    """Raw answers keyed by question index: severity only for detected violations, none for an ineligible image."""
    eligibility_raw, eligible = infer_eligibility(prompts, infer_fn)

    detection_raw: dict[int, str] = {}
    severity_raw: dict[int, str] = {}
    if eligible is True:
        fields = [get_violation_question_fields(q) for q in prompts["violation_questions"]]
        if infer_many_fn is not None and fields:
            detection_raw = dict(enumerate(
                infer_many_fn([build_detection_prompt(prompts, f) for f in fields])))
            detected = [q_idx for q_idx in range(len(fields))
                        if parse_yes_no_answer(detection_raw[q_idx]) is True]
            if detected:
                severity_raw = dict(zip(detected, infer_many_fn(
                    [build_severity_prompt(prompts, fields[q_idx], detection_raw[q_idx])
                     for q_idx in detected])))
        else:
            for q_idx, question_fields in enumerate(tqdm(fields, desc="TerraVis violation detection")):
                det_raw = infer_fn(build_detection_prompt(prompts, question_fields))
                detection_raw[q_idx] = det_raw
                if parse_yes_no_answer(det_raw) is True:
                    severity_raw[q_idx] = infer_fn(
                        build_severity_prompt(prompts, question_fields, det_raw))

    return {"eligibility_raw": eligibility_raw,
            "detection_raw": detection_raw,
            "severity_raw": severity_raw}


def build_terravis_workflow_result_from_raw_outputs(
    prompts: dict,
    eligibility_raw: str,
    detection_raw_by_question_idx: dict[int, str],
    severity_raw_by_question_idx: dict[int, str] | None = None,
    keep_ineligible_details: bool = False,
) -> dict[str, Any]:
    severity_raw_by_question_idx = severity_raw_by_question_idx or {}
    lambda_, alpha = float(prompts.get("lambda", 1.0)), float(prompts.get("alpha", 0.5))

    eligible = parse_eligibility_answer(eligibility_raw)
    score_entries: dict[str, Any] = {
        "method": prompts.get("name", prompts.get("method", "TerraVis")),
        "eligibility": {
            "raw_answer": eligibility_raw,
            "is_eligible": eligible,
            "reason": extract_output_field(eligibility_raw, "Reason"),
            "valid_output": eligible is not None,
        },
        "violation_detections": [],
        "num_major_violations": 0,
        "num_minor_violations": 0,
        "lambda": lambda_,
        "alpha": alpha,
        "score_description": (
            "TerraVis score computed by eligibility checking, taxonomy-level violation detection, "
            "severity classification, and exponential penalty."
        ),
    }

    if eligible is not True and not keep_ineligible_details:
        return {"score": "n/a", "score_entries": score_entries}

    n_major = n_minor = 0
    for q_idx, question_item in enumerate(prompts["violation_questions"]):
        fields = get_violation_question_fields(question_item)
        detection_raw = detection_raw_by_question_idx.get(q_idx, "")
        detected = parse_yes_no_answer(detection_raw)

        entry = {
            "domain": fields["domain"],
            "violation_type": fields["violation_type"],
            "question": fields["question"],
            "raw_detection_answer": detection_raw,
            "detected": detected,
            "affected_aspect": extract_output_field(detection_raw, "Affected aspect"),
            "evidence": extract_output_field(detection_raw, "Evidence"),
            "valid_detection_output": detected is not None,
        }
        if detected is True:
            severity_raw = severity_raw_by_question_idx.get(q_idx, "")
            parsed = parse_severity_answer(severity_raw)
            severity = parsed if parsed == "major" else "minor"
            if severity == "major":
                n_major += 1
            else:
                n_minor += 1
            entry.update({
                "raw_severity_answer": severity_raw,
                "severity": severity,
                "severity_reason": extract_output_field(severity_raw, "Reason"),
                "valid_severity_output": parsed is not None,
            })
        else:
            entry.update({
                "raw_severity_answer": "",
                "severity": "",
                "severity_reason": "",
                "valid_severity_output": None,
            })
        score_entries["violation_detections"].append(entry)

    score_entries["num_major_violations"] = n_major
    score_entries["num_minor_violations"] = n_minor
    scorable = eligible is True and not has_empty_answer(score_entries)
    score = math.exp(-lambda_ * (n_major + alpha * n_minor)) if scorable else "n/a"
    return {"score": score, "score_entries": score_entries}


def has_empty_answer(score_entries: dict) -> bool:
    """An empty answer is a failed or blocked request, not "no violation": the image scores n/a."""
    return not score_entries["eligibility"]["raw_answer"] or any(
        not d["raw_detection_answer"] or (d["detected"] and not d["raw_severity_answer"])
        for d in score_entries["violation_detections"])


def compute_terravis_workflow_score(
    prompts: dict,
    infer_fn: Callable[..., str],
    infer_many_fn: Callable[[list[str]], list[str]] | None = None,
) -> dict[str, Any]:
    raw = collect_workflow_raw_for_image(infer_fn, prompts, infer_many_fn=infer_many_fn)
    return build_terravis_workflow_result_from_raw_outputs(
        prompts, raw["eligibility_raw"], raw["detection_raw"], raw["severity_raw"],
    )


class TerraVisJudgeBase:
    """``score(image)`` over ``infer(text, image, **overrides)`` and optional ``infer_many(texts, image)``."""

    prompts: dict
    infer_many = None

    def score(self, image: str) -> dict[str, Any]:
        return compute_terravis_workflow_score(
            prompts=self.prompts,
            infer_fn=lambda text, **kw: self.infer(text, image, **kw),
            infer_many_fn=None if self.infer_many is None else (lambda texts: self.infer_many(texts, image)),
        )

    __call__ = score


class TerraVisScoreModel(TerraVisJudgeBase):
    """Any open-weight judge engine from ``build_judge``."""

    def __init__(self, prompts: dict, judge: Any):
        self.prompts = prompts
        self.infer, self.infer_many = judge.infer, judge.infer_many


def _build_terravis(ctx) -> Any:
    from terravis.scores.metrics.terravis_api import TerraVisScoreModelAPI

    if ctx.use_api:
        return TerraVisScoreModelAPI(provider=ctx.api_provider, model_name=ctx.api_model_name, prompts=TERRAVIS_PROMPT)
    return TerraVisScoreModel(TERRAVIS_PROMPT, build_judge(ctx, ctx.vqa_model_names, ctx.sampling_params))


register_score_model(
    name="terravis_score",
    category="prompt-independent",
    description="TerraVis score (workflow) computed with a VQA judge.",
    builder=_build_terravis,
    supports_api=True,
    supports_batch_api=True,
    supports_multi_device=True,
)
