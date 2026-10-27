"""Remote vLLM judge calls and the per-file image cache they read. A request is mostly a ~2 MB data URL that JSON
copies under the GIL, so ``TERRAVIS_JUDGE_PROCS=N`` sends them from N worker processes. Outside ``terravis.scores``
(whose import loads every metric): a worker imports only PIL and openai."""

import base64
import io
import itertools
import json
import math
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from PIL import Image

PROCS_ENV = "TERRAVIS_JUDGE_PROCS"
THREADS_PER_PROC = 64       # in-flight requests per worker; a request is ~all network wait


@lru_cache(maxsize=32)      # ~2-3 MB each
def _data_url(path: str, mtime_ns: int, size: int) -> str:
    with Image.open(path) as im:
        oriented = {"exif", "Raw profile type exif", "XML:com.adobe.xmp"} & im.info.keys()   # vLLM rotates by any
        if im.format == "PNG" and im.mode == "RGB" and not oriented:
            with open(path, "rb") as fh:       # load_image's pixels as they are
                return "data:image/png;base64," + base64.b64encode(fh.read()).decode()
        rgb = im.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG", icc_profile=None)   # as vlm_chat.image_to_data_url
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


_url_locks = [threading.Lock() for _ in range(256)]   # striped by path: other images build in parallel


def image_url(path: str) -> str:
    """``load_image(path)``'s pixels as a PNG data URL, built once per file version."""
    st = os.stat(path)
    with _url_locks[hash(path) % 256]:     # an image's questions arrive together: one build, not one per thread
        return _data_url(path, st.st_mtime_ns, st.st_size)


def complete(client: Any, request: dict) -> tuple[str, list[tuple[str, float]]]:
    """(answer, the first token's (token, probability) alternatives when the request asks for logprobs)."""
    choice = client.chat.completions.create(**request).choices[0]
    tokens = choice.logprobs.content if (choice.logprobs and choice.logprobs.content) else []
    top = [(t.token, math.exp(t.logprob)) for t in tokens[0].top_logprobs] if tokens else []
    return (choice.message.content or "").strip(), top


def new_client(base_url: str) -> Any:
    from openai import OpenAI

    return OpenAI(api_key="EMPTY", base_url=base_url, timeout=600, max_retries=3)


def _serve() -> None:
    """Worker main: JSON lines [id, base_url, request] in, [id, ok, (answer, top) | (type, message, status)] out."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # Ctrl-C is the caller's: a worker ends when its stdin does
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
            for block in request["messages"][0]["content"]:
                if block["type"] == "image_url":
                    block["image_url"]["url"] = image_url(block["image_url"]["url"])
            reply = [rid, True, complete(client(base_url), request)]
        except Exception as e:
            status = getattr(e, "status_code", None)
            reply = [rid, False, [type(e).__name__, str(e), status if isinstance(status, int) else None]]
        with lock:
            out.write(json.dumps(reply) + "\n")
            out.flush()

    threads = ThreadPoolExecutor(max_workers=THREADS_PER_PROC)
    for line in sys.stdin:
        threads.submit(handle, line)
    os._exit(0)         # the caller is gone: don't wait for in-flight requests


class _Worker:
    def __init__(self, pool: "JudgePool", idx: int):
        # A fresh interpreter, not multiprocessing: spawn would import the caller's __main__ (torch and all).
        self.proc = subprocess.Popen([sys.executable, "-m", "terravis.judge_pool"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock, self.pending = threading.Lock(), {}
        threading.Thread(target=pool._read, args=(self, idx), daemon=True).start()


class JudgePool:
    def __init__(self, procs: int):
        self.lock, self.ids, self.broken = threading.Lock(), itertools.count(), ""
        self.workers = [_Worker(self, i) for i in range(procs)]

    def complete(self, base_url: str, request: dict) -> tuple[str, list[tuple[str, float]]]:
        fut: Future = Future()
        rid = next(self.ids)
        line = json.dumps([rid, base_url, request]) + "\n"      # first: one that cannot encode is never pending
        with self.lock:
            if self.broken:
                raise RuntimeError(self.broken)
            worker = min(self.workers, key=lambda w: len(w.pending))
            worker.pending[rid] = fut
        try:
            with worker.lock:
                worker.proc.stdin.write(line)
                worker.proc.stdin.flush()
        except OSError:
            pass                        # it exited: _read fails this request with the rest of its own
        answer, top = fut.result()
        return answer, [tuple(t) for t in top]

    def _read(self, worker: _Worker, idx: int) -> None:
        answered = False
        for line in worker.proc.stdout:
            try:
                rid, ok, value = json.loads(line)
            except ValueError:      # a reply cut short: the worker died mid-write
                worker.proc.kill()
                break
            with self.lock:
                fut = worker.pending.pop(rid)
            answered = True
            if ok:
                fut.set_result(value)
            else:       # the worker's type name and status_code: what callers' retry rules read
                name, message, status = value
                fut.set_exception(type(name, (RuntimeError,), {"status_code": status})(message))
        worker.proc.wait()
        with self.lock:         # its requests never come back; one that never answered would only exit again
            pending, worker.pending = worker.pending, {}
            if not answered:
                self.broken = (f"judge workers exit before answering (exit {worker.proc.returncode}; "
                               "python -m terravis.judge_pool </dev/null shows why)")
            else:
                try:
                    self.workers[idx] = _Worker(self, idx)
                except (OSError, RuntimeError) as e:     # out of fds, processes or threads
                    self.broken = f"cannot start a judge worker ({e})"
        for fut in pending.values():
            fut.set_exception(ConnectionError(f"judge worker exited with {worker.proc.returncode}"))


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
        return _pool


if __name__ == "__main__":
    _serve()
