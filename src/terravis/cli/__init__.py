import os
import subprocess
import sys
from pathlib import Path

_ROOT_PATHS = Path(__file__).resolve().parents[3] / "scripts" / "set_root_paths.sh"
_PRINT_ROOTS = 'source "$0" >/dev/null && printf "%s\\0%s\\0%s" "$MODEL_WEIGHT_ROOT" "$SAVE_ROOT" "$HF_HUB_CACHE"'


def _load_root_paths() -> None:
    """Fill MODEL_WEIGHT_ROOT, SAVE_ROOT and HF_HUB_CACHE from scripts/set_root_paths.sh before anything reads
    HF_HUB_CACHE, so the CLIs need no `source`; values already set win. A no-op outside a source checkout."""
    if not _ROOT_PATHS.is_file():
        return
    try:
        out = subprocess.run(["bash", "-c", _PRINT_ROOTS, str(_ROOT_PATHS)], capture_output=True, text=True)
    except OSError:   # no bash: exported values and the CLI flags still work
        return
    if out.returncode:
        print(f"warning: could not read {_ROOT_PATHS}: {out.stderr.strip()}", file=sys.stderr)
        return
    for key, value in zip(("MODEL_WEIGHT_ROOT", "SAVE_ROOT", "HF_HUB_CACHE"), out.stdout.split("\0")):
        os.environ.setdefault(key, value)


_load_root_paths()
