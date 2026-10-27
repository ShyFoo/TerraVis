#!/usr/bin/env bash
# Usage: bash scripts/serve_vllm_judge.sh <judge> [tp_size] [port]
#   judge: HF repo id or registered short name, or CodeGoat24/UnifiedReward-2.0-qwen35-9b for unified_reward_2_score
# Env: MODEL_WEIGHT_ROOT (scripts/set_root_paths.sh), CUDA_VISIBLE_DEVICES, HF_TOKEN (gated repos),
#      TERRAVIS_VLLM_ATTENTION_BACKEND (override; gemma-4 defaults to TRITON_ATTN)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/set_root_paths.sh"

judge=${1:?usage: bash scripts/serve_vllm_judge.sh <judge> [tp_size=1] [port=8000]}
tp_size=${2:-1}
port=${3:-8000}

command -v vllm >/dev/null \
  || { echo "!!! vllm not on PATH: pip install -e '.[vllm]' --no-build-isolation" >&2; exit 1; }

# Serve under the HF repo id (RemoteVLLMJudge sends it as the model name), with its flags as in load_vllm_llm.
resolved=$(python - "$judge" <<'PY'
import sys
from terravis.configs import find_judge
from terravis.models.model_manager import vllm_attention_backend, vllm_prefix_caching
judge = find_judge(sys.argv[1])
if judge.is_api:
    sys.exit(f"!!! {judge.id!r} is an API judge; nothing to serve.")
backend = vllm_attention_backend(judge.id)
print(judge.id, *(["--attention-backend", backend] if backend else []),
      "--enable-prefix-caching" if vllm_prefix_caching(judge.id) else "--no-enable-prefix-caching")
PY
)
read -r -a served <<<"$resolved"   # repo id, then its flags
model_id=${served[0]}
extra_args=("${served[@]:1}")
# UnifiedReward-2.0 reads the folder unified_reward_2_score loads from, not hf_cache.
case $model_id in *UnifiedReward*) export HF_HUB_CACHE="$MODEL_WEIGHT_ROOT/UnifiedReward" ;; esac

export CUDA_DEVICE_ORDER=PCI_BUS_ID   # GPU ids as in nvidia-smi

url="http://localhost:$port/v1"
echo ">>> vllm serve $model_id  tp=$tp_size  $url"
echo ">>> submit_local_eval_jobs.sh: vllm_server_url=\"$url\""
echo ">>> compute_t2i_scores.sh:     export TERRAVIS_REMOTE_VLLM_ENDPOINTS='{\"$model_id\": \"$url\"}'"

# max-model-len / max-num-seqs / limit-mm-per-prompt as in load_vllm_llm; --max-logprobs 20 for vqa_score.
exec vllm serve "$model_id" \
  --tensor-parallel-size "$tp_size" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --max-logprobs 20 \
  --gpu-memory-utilization 0.95 \
  --limit-mm-per-prompt '{"image": 1}' \
  --trust-remote-code \
  --download-dir "$HF_HUB_CACHE" \
  --port "$port" \
  "${extra_args[@]}"
