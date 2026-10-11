"""Remote vLLM judge calls: the image cache every call goes through, and (``TERRAVIS_JUDGE_PROCS=N``) worker
processes. A judge request is mostly a ~2 MB data URL that base64 and JSON copy under the GIL, so a client making
hundreds of calls a second tops out at one interpreter; N processes, each with its own threads, lift that ceiling.

Kept outside ``terravis.scores`` (whose import loads every metric): a worker imports only PIL and openai."""

import atexit
import base64
import io
import itertools
import json
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from PIL import Image

PROCS_ENV = "TERRAVIS_JUDGE_PROCS"
THREADS_PER_PROC = 64       # in-flight requests per worker; a request is ~all network wait
IMAGE_REF = "terravis-path:"   # an image_url that names a file; resolved to a data URL where the request is sent


@lru_cache(maxsize=32)      # ~2-3 MB each; a miss costs a file read and a base64, not a PNG encode
def _data_url(path: str, mtime_ns: int, size: int) -> str:
    with Image.open(path) as im:
        if im.format == "PNG" and im.mode == "RGB" and "transparency" not in im.info:
            with open(path, "rb") as fh:       # already the pixels load_image would give: no decode/re-encode
                return "data:image/png;base64," + base64.b64encode(fh.read()).decode()
        rgb = im.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image_url(path: str) -> str:
    """``load_image(path)``'s pixels as a PNG data URL, built once per file version."""
    st = os.stat(path)
    return _data_url(path, st.st_mtime_ns, st.st_size)


def complete(client: Any, request: dict) -> tuple[str, list[tuple[str, float]]]:
    """(answer, the first token's (token, probability) alternatives when the request asks for logprobs)."""
    for message in request["messages"]:
        for block in message["content"]:
            url = block.get("image_url", {}).get("url", "") if isinstance(block, dict) else ""
            if url.startswith(IMAGE_REF):
                block["image_url"]["url"] = image_url(url[len(IMAGE_REF):])
    choice = client.chat.completions.create(**request).choices[0]
    tokens = choice.logprobs.content if (choice.logprobs and choice.logprobs.content) else []
    top = [(t.token, math.exp(t.logprob)) for t in tokens[0].top_logprobs] if tokens else []
    return (choice.message.content or "").strip(), top


def new_client(base_url: str) -> Any:
    from openai import OpenAI

    return OpenAI(api_key="EMPTY", base_url=base_url, timeout=600, max_retries=3)


def _serve() -> None:
    """Worker main (``python -m terravis.judge_pool``): JSON lines ``[id, base_url, request]`` in on stdin,
    ``[id, ok, (answer, top) | (error type, message, status)]`` out on stdout."""
    clients: dict[str, Any] = {}
    lock, out = threading.Lock(), sys.stdout

    def client(base_url: str) -> Any:
        with lock:
            if base_url not in clients:
                clients[base_url] = new_client(base_url)
            return clients[base_url]

    def handle(line: str) -> None:
        rid, base_url, request = json.loads(line)
        try:
            reply = [rid, True, complete(client(base_url), request)]
        except Exception as e:  # every failure goes back to the caller
            status = getattr(e, "status_code", None)
            reply = [rid, False, [type(e).__name__, str(e), status if isinstance(status, int) else None]]
        with lock:
            out.write(json.dumps(reply) + "\n")
            out.flush()

    with ThreadPoolExecutor(max_workers=THREADS_PER_PROC) as threads:
        for line in sys.stdin:
            threads.submit(handle, line)


_ERROR_TYPES: dict[str, type] = {}


def _error(name: str, message: str, status: int | None) -> Exception:
    """The worker's exception under its own type name (callers match retryable errors by name and status_code)."""
    err = _ERROR_TYPES.setdefault(name, type(name, (RuntimeError,), {}))(message)
    err.status_code = status
    return err


class _Worker:
    def __init__(self, pool: "JudgePool", idx: int):
        # A fresh interpreter, not multiprocessing: spawn would import the caller's __main__ (torch and all).
        self.proc = subprocess.Popen([sys.executable, "-m", "terravis.judge_pool"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock, self.pending, self.answered = threading.Lock(), {}, False
        threading.Thread(target=pool._read, args=(self, idx), daemon=True).start()


class JudgePool:
    def __init__(self, procs: int):
        self.lock, self.ids, self.broken = threading.Lock(), itertools.count(), False
        self.workers = [_Worker(self, i) for i in range(procs)]
        self.startup_fails = [0] * procs      # exits in a row before a first answer, per slot

    def complete(self, base_url: str, request: dict) -> tuple[str, list[tuple[str, float]]]:
        fut: Future = Future()
        with self.lock:
            if self.broken:
                raise _error("ConnectionError", "judge worker processes exit before answering (python -m "
                             "terravis.judge_pool < /dev/null shows why)", None)
            rid = next(self.ids)
            worker = min(self.workers, key=lambda w: len(w.pending))     # least in flight
            worker.pending[rid] = fut
        line = json.dumps([rid, base_url, request]) + "\n"
        try:
            with worker.lock:
                worker.proc.stdin.write(line)
                worker.proc.stdin.flush()
        except OSError:
            pass                        # it exited: _read fails this request with the rest of its own
        answer, top = fut.result()
        return answer, [tuple(t) for t in top]

    def _read(self, worker: _Worker, idx: int) -> None:
        for line in worker.proc.stdout:
            rid, ok, value = json.loads(line)
            with self.lock:
                fut = worker.pending.pop(rid)
                worker.answered, self.startup_fails[idx] = True, 0
            if ok:
                fut.set_result(value)
            else:
                fut.set_exception(_error(*value))
        worker.proc.wait()
        with self.lock:         # it exited: its requests never come back; one that never answers 3 times breaks the pool
            pending, worker.pending = worker.pending, {}
            self.startup_fails[idx] += not worker.answered
            if self.startup_fails[idx] >= 3:
                self.broken = True
            else:
                self.workers[idx] = _Worker(self, idx)
        for fut in pending.values():
            fut.set_exception(_error("ConnectionError", f"judge worker exited with {worker.proc.returncode}", None))

    def close(self) -> None:
        for w in self.workers:
            w.proc.terminate()


_pool: JudgePool | None = None
_pool_lock = threading.Lock()


def pool() -> JudgePool | None:
    """The process-wide pool when ``TERRAVIS_JUDGE_PROCS`` > 0, started on first use; else None (in-process calls)."""
    global _pool
    procs = int(os.environ.get(PROCS_ENV) or 0)
    if procs <= 0:
        return None
    with _pool_lock:
        if _pool is None:
            _pool = JudgePool(procs)
            atexit.register(_pool.close)
        return _pool


if __name__ == "__main__":
    _serve()
