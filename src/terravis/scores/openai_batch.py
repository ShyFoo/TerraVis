"""OpenAI Batch API transport: split, upload, poll, download, parse. Each body carries a unique ``custom_id``."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

# OpenAI allows 200 MB per input file; 150 MB leaves room for proxy body-size limits.
_MAX_FILE_BYTES = 150 * 1024 * 1024
_MAX_REQUESTS_PER_BATCH = 50000


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_output_text(body: Any) -> str:
    """``Response.output_text`` over a raw /v1/responses body, lowercased to match ``call_api``."""
    texts = [
        content["text"]
        for item in (body or {}).get("output") or [] if item.get("type") == "message"
        for content in item.get("content") or [] if content.get("type") == "output_text"
    ]
    return "".join(texts).strip().lower()


class OpenAIBatchRunner:
    def __init__(self, client: Any, *, completion_window: str = "24h", poll_interval: float = 60.0,
                 verbose: bool = True, resume: bool = False):
        self.client = client
        self.completion_window = completion_window
        self.poll_interval = poll_interval
        self.verbose = verbose
        self.resume = resume

    def run(self, requests: list[dict[str, Any]], output_dir: str, batch_name: str) -> dict[str, str]:
        """-> ``{custom_id: answer text}`` ('' for failed rows); writes ``{batch_name}__manifest.json``."""
        chunks = self._split(requests)
        if not chunks:
            raise ValueError("Cannot create an empty OpenAI batch.")
        if self.verbose and len(chunks) > 1:
            print(f"Splitting {len(requests)} OpenAI Batch requests into {len(chunks)} parts.")

        responses: dict[str, str] = {}
        for part_idx, part_requests in enumerate(chunks):
            part_name = (batch_name if len(chunks) == 1
                         else f"{batch_name}__part{part_idx + 1:04d}_of_{len(chunks):04d}")
            responses.update(self._run_part(part_requests, output_dir, part_name))

        failed = sorted(cid for cid, text in responses.items() if not text)
        missing = sorted(r["custom_id"] for r in requests if r["custom_id"] not in responses)
        if failed or missing:
            print(f"WARNING: OpenAI batch {batch_name}: {len(failed)} failed/empty + {len(missing)} missing of "
                  f"{len(requests)} requests score n/a. First: {(failed + missing)[:5]}")

        manifest = {
            "batch_name": batch_name,
            "num_total_requests": len(requests),
            "num_parts": len(chunks),
            "num_failed_or_empty": len(failed),
            "num_missing": len(missing),
            "failed_or_empty_custom_ids": failed[:1000],
            "missing_custom_ids": missing[:1000],
        }
        with open(Path(output_dir) / f"{batch_name}__manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return responses

    @staticmethod
    def _split(requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_bytes = 0
        for request in requests:
            row_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8")) + 1
            if row_bytes > _MAX_FILE_BYTES:
                raise ValueError(
                    f"Batch request line too large: custom_id={request.get('custom_id')}, "
                    f"size={row_bytes / 1024 / 1024:.2f} MB > {_MAX_FILE_BYTES / 1024 / 1024:.2f} MB."
                )
            if current and (current_bytes + row_bytes > _MAX_FILE_BYTES
                            or len(current) >= _MAX_REQUESTS_PER_BATCH):
                chunks.append(current)
                current, current_bytes = [], 0
            current.append(request)
            current_bytes += row_bytes
        if current:
            chunks.append(current)
        return chunks

    def _run_part(self, requests: list[dict[str, Any]], output_dir: str, part_name: str) -> dict[str, str]:
        output_path = str(Path(output_dir) / f"{part_name}__output.jsonl")
        error_path = str(Path(output_dir) / f"{part_name}__errors.jsonl")
        batch_id = self._resume_state(requests, output_dir, part_name) if self.resume else None

        if batch_id and Path(output_path).exists():
            # Already downloaded: re-submitting would bill again.
            if self.verbose:
                print(f"Reusing OpenAI batch part {part_name} from {output_path}.")
        else:
            if batch_id:
                # Submitted before the crash: billed already, so collect it.
                print(f"Re-attaching to OpenAI batch {batch_id} for part {part_name}.")
            else:
                batch_id = self._submit(requests, output_dir, part_name)
            batch = self._wait(batch_id)
            # Download before checking status: an expired job still holds its billed answers.
            self._download(batch, output_path, error_path)
            if batch.status != "completed" or not batch.output_file_id:
                raise RuntimeError(
                    f"OpenAI batch {batch_id} finished with status={batch.status}. "
                    + (f"Whatever it did answer is in {output_path}; re-run with --resume to score those "
                       f"and take the rest as n/a." if batch.output_file_id
                       else f"No output file; delete {part_name}__state.json under {output_dir} to resubmit.")
                )

        responses = {
            row.get("custom_id"): extract_output_text((row.get("response") or {}).get("body"))
            for row in read_jsonl(output_path)
        }
        # Failed rows land in the error file, not the output file.
        if Path(error_path).exists():
            for row in read_jsonl(error_path):
                responses.setdefault(row.get("custom_id"), "")
        return responses

    def _submit(self, requests: list[dict[str, Any]], output_dir: str, part_name: str) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests).encode("utf-8")
        input_file = self.client.files.create(file=(f"{part_name}__input.jsonl", data), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/responses",
            completion_window=self.completion_window,
        )
        # Persist the id before polling; --resume needs it.
        with open(self._state_path(output_dir, part_name), "w", encoding="utf-8") as f:
            json.dump({"batch_id": batch.id, "input_file_id": input_file.id,
                       "requests_digest": self._digest(requests)}, f, indent=2)
        return batch.id

    @staticmethod
    def _state_path(output_dir: str, part_name: str) -> Path:
        return Path(output_dir) / f"{part_name}__state.json"

    @staticmethod
    def _digest(requests: list[dict[str, Any]]) -> str:
        h = hashlib.sha256()
        for request in requests:
            h.update(json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def _resume_state(self, requests: list[dict[str, Any]], output_dir: str, part_name: str) -> str | None:
        """An earlier run's batch id, or ``None``; raises if the part's requests changed."""
        path = self._state_path(output_dir, part_name)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("requests_digest") != self._digest(requests):
            raise RuntimeError(
                f"{path} was written for different requests (images, prompts or split sizes changed). "
                f"Delete {part_name}__* under {output_dir} to re-run this part from scratch (billed again)."
            )
        return state["batch_id"]

    def _wait(self, batch_id: str) -> Any:
        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if self.verbose:
                print(f"OpenAI batch {batch_id} status={batch.status}, "
                      f"request_counts={batch.request_counts}")
            if batch.status in terminal:
                return batch
            time.sleep(self.poll_interval)

    def _download(self, batch: Any, output_path: str, error_path: str) -> None:
        if batch.output_file_id:
            self._write_file(batch.output_file_id, output_path)
        if batch.error_file_id:
            self._write_file(batch.error_file_id, error_path)

    def _write_file(self, file_id: str, path: str) -> None:
        # Rename after download so --resume never sees a half-written file.
        self.client.files.content(file_id).write_to_file(f"{path}.partial")
        os.replace(f"{path}.partial", path)
