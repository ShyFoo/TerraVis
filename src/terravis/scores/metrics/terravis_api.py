"""API judge backend (OpenAI Responses / Gemini). The OpenAI-only Batch path runs two stages
(severity prompts embed detection output) and deviates: eligibility neither gates detection
nor gets a parse-retry, so an off-format answer scores n/a."""

import base64
import json
import mimetypes
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from terravis.scores.metrics.terravis_score import (
    TerraVisJudgeBase,
    build_detection_prompt,
    build_severity_prompt,
    build_terravis_workflow_result_from_raw_outputs,
    get_violation_question_fields,
    is_terravis_workflow_prompt,
    parse_yes_no_answer,
)
from terravis.scores.openai_batch import OpenAIBatchRunner


def openai_image_part(image: str) -> dict[str, Any]:
    with open(image, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    mime_type = mimetypes.guess_type(image)[0] or "image/jpeg"
    return {"type": "input_image", "image_url": f"data:{mime_type};base64,{image_b64}"}


def openai_image_message(image_part: dict[str, Any], text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "input_text", "text": text}, image_part]}]


def _is_fatal(e: Exception) -> bool:
    """Errors every request would hit (bad key, no access, unknown model, no quota): stop the run."""
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    return status in (401, 403, 404) or any(s in str(e) for s in ("insufficient_quota", "API key not valid"))


def call_with_retries(call: Callable[[], Any], max_retries: int, retry_delay: float) -> Any:
    """Exponential backoff; a request that still fails returns "", so its image scores n/a and --resume retries it."""
    for attempt in range(max_retries):
        try:
            return call()
        except Exception as e:
            if _is_fatal(e):
                raise
            if attempt == max_retries - 1:
                print(f"WARNING: API call failed after {max_retries} attempts: {e}")
                return ""
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
            time.sleep(min(retry_delay * 2 ** attempt, 60))


_MAX_RETRIES, _RETRY_DELAY, _TIMEOUT, _MAX_OUTPUT_TOKENS = 10, 2.0, 120, 8192   # ~5 min of retries per request


class TerraVisScoreModelAPI(TerraVisJudgeBase):
    def __init__(self, provider: str, model_name: str, prompts: dict):
        self.provider = provider.lower()
        self.model_name = model_name
        self.prompts = prompts

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=_TIMEOUT)
        elif self.provider == "google":
            from google import genai
            key = os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"]
            self.client = genai.Client(api_key=key)
        else:
            raise ValueError(f"Unsupported provider: {provider!r}. Expected 'openai' or 'google'.")

    def prepare_inputs(self, image: str, text: str) -> list[Any]:
        if self.provider == "openai":
            return openai_image_message(openai_image_part(image), text)
        from google import genai
        mime_type = mimetypes.guess_type(image)[0] or "image/jpeg"
        with open(image, "rb") as f:
            image_part = genai.types.Part.from_bytes(data=f.read(), mime_type=mime_type)
        return [image_part, text]

    def call_api(self, contents: list[Any]) -> str:
        # .lower() is a frozen output contract: stored answers and the Batch path are lowercase.
        def _once() -> str:
            if self.provider == "openai":
                resp = self.client.responses.create(
                    model=self.model_name, input=contents, max_output_tokens=_MAX_OUTPUT_TOKENS,
                )
                return resp.output_text.strip().lower()
            from google import genai
            resp = self.client.models.generate_content(
                model=self.model_name, contents=contents,
                config=genai.types.GenerateContentConfig(max_output_tokens=_MAX_OUTPUT_TOKENS),
            )
            if not resp.text:
                if resp.prompt_feedback and resp.prompt_feedback.block_reason:
                    return ""   # blocked inputs stay blocked on retry; an empty answer scores n/a
                reason = resp.candidates[0].finish_reason if resp.candidates else None
                raise RuntimeError(f"Gemini returned empty text (finish_reason={reason})")
            return resp.text.strip().lower()

        return call_with_retries(_once, _MAX_RETRIES, _RETRY_DELAY)

    def infer(self, text: str, image: str) -> str:
        return self.call_api(self.prepare_inputs(image=image, text=text))

    # ---- OpenAI Batch API path ----

    @staticmethod
    def _custom_id(image_idx: int, task_type: str, question_idx: int | None = None) -> str:
        base = f"img{image_idx:08d}__{task_type}"
        return base if question_idx is None else f"{base}__q{question_idx:04d}"

    def _build_batch_request(self, custom_id: str, image_part: dict[str, Any],
                             text: str) -> dict[str, Any]:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": self.model_name,
                "input": openai_image_message(image_part, text),
                "max_output_tokens": _MAX_OUTPUT_TOKENS,
            },
        }

    def build_score_workflow_stage1_batch_requests(self, images: list[str]) -> list[dict[str, Any]]:
        """Eligibility + every detection question for every image."""
        if not is_terravis_workflow_prompt(self.prompts):
            raise ValueError("score_batch requires a TerraVis workflow prompt dict.")

        requests = []
        for image_idx, image in enumerate(images):
            image_part = openai_image_part(image)
            requests.append(self._build_batch_request(
                custom_id=self._custom_id(image_idx, "eligibility"),
                image_part=image_part,
                text=self.prompts["eligibility_prompt"],
            ))
            for question_idx, question_item in enumerate(self.prompts["violation_questions"]):
                fields = get_violation_question_fields(question_item)
                requests.append(self._build_batch_request(
                    custom_id=self._custom_id(image_idx, "detection", question_idx),
                    image_part=image_part,
                    text=build_detection_prompt(self.prompts, fields),
                ))
        return requests

    def build_score_workflow_stage2_batch_requests(
        self, images: list[str], stage1_responses: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Severity questions for stage-1 detections that answered yes."""
        requests = []
        for image_idx, image in enumerate(images):
            image_part = None
            for question_idx, question_item in enumerate(self.prompts["violation_questions"]):
                detection_raw = stage1_responses.get(self._custom_id(image_idx, "detection", question_idx), "")
                if parse_yes_no_answer(detection_raw) is not True:
                    continue
                if image_part is None:
                    image_part = openai_image_part(image)
                fields = get_violation_question_fields(question_item)
                requests.append(self._build_batch_request(
                    custom_id=self._custom_id(image_idx, "severity", question_idx),
                    image_part=image_part,
                    text=build_severity_prompt(self.prompts, fields, detection_raw),
                ))
        return requests

    def reconstruct_score_workflow_batch_results(
        self,
        images: list[str],
        stage1_responses: dict[str, str],
        stage2_responses: dict[str, str] | None = None,
        keep_ineligible_details: bool = False,
    ) -> dict[int, dict[str, Any]]:
        stage2_responses = stage2_responses or {}

        results = {}
        for image_idx in range(len(images)):
            eligibility_raw = stage1_responses.get(self._custom_id(image_idx, "eligibility"), "")
            detection_raw = {
                q: stage1_responses.get(self._custom_id(image_idx, "detection", q), "")
                for q in range(len(self.prompts["violation_questions"]))
            }
            severity_raw = {
                q: stage2_responses.get(self._custom_id(image_idx, "severity", q), "")
                for q in range(len(self.prompts["violation_questions"]))
            }
            results[image_idx] = {
                "terravis_score": build_terravis_workflow_result_from_raw_outputs(
                    prompts=self.prompts,
                    eligibility_raw=eligibility_raw,
                    detection_raw_by_question_idx=detection_raw,
                    severity_raw_by_question_idx=severity_raw,
                    keep_ineligible_details=keep_ineligible_details,
                )
            }
        return results

    def _run_workflow_batch(self, runner: OpenAIBatchRunner, images: list[str],
                            output_dir: str, batch_name: str,
                            keep_ineligible_details: bool) -> dict[int, dict[str, Any]]:
        stage1_responses = runner.run(
            self.build_score_workflow_stage1_batch_requests(images),
            output_dir, f"{batch_name}__stage1_eligibility_detection",
        )
        stage2_requests = self.build_score_workflow_stage2_batch_requests(images, stage1_responses)
        stage2_responses = (
            runner.run(stage2_requests, output_dir, f"{batch_name}__stage2_severity")
            if stage2_requests else {}
        )
        return self.reconstruct_score_workflow_batch_results(
            images=images,
            stage1_responses=stage1_responses,
            stage2_responses=stage2_responses,
            keep_ineligible_details=keep_ineligible_details,
        )

    def score_batch(
        self,
        images: list[str],
        output_dir: str = "./openai_batch_outputs",
        batch_name: str = "terravis",
        completion_window: str = "24h",
        poll_interval: float = 60.0,
        verbose: bool = True,
        keep_ineligible_details: bool = False,
        resume: bool = False,
    ) -> dict[int, dict[str, Any]]:
        """Batch ``score()``; deviations in the module docstring. ``resume`` reuses the
        parts an interrupted run left under the same ``output_dir`` / ``batch_name``."""
        if not images:
            return {}
        runner = OpenAIBatchRunner(
            self.client, completion_window=completion_window,
            poll_interval=poll_interval, verbose=verbose, resume=resume,
        )
        results = self._run_workflow_batch(runner, images, output_dir,
                                           batch_name, keep_ineligible_details)
        path = Path(output_dir) / f"{batch_name}__final_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return results
