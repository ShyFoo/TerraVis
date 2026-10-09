#!/usr/bin/env bash
# Edit the arrays below, then: bash scripts/submit_local_eval_jobs.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/set_root_paths.sh"

score_modes=(
#  "clip_score"
#  "vqa_score"
#  "tifa_score"
#  "dsg_score"
#  "unigenbench_score"   # only on unigenbench++-generated images
#  "geneval2_score"      # only on geneval2-generated images
#  "image_reward_score"
#  "hpsv3_score"
#  "unified_reward_2_score"
#  "pick_score"
#  "rahf_score"
#  "musiq_score"
#  "laion_aesthetic_score"
  "terravis_score"
)

gen_data_json_paths=(
  "genai_bench/stable-diffusion-3.5-large/seed=1111/test_data.json"   # relative to SAVE_ROOT, or absolute
)

save_path="$(dirname "$SAVE_ROOT")/t2i_scores"
clip_model_name="webli:ViT-gopt-16-SigLIP2-384"

# Judge: an open-weight VLM (HF repo id or registered short name), or leave it empty and set an API model.
vqa_model_name="google/gemma-4-31B-it"   # or "Qwen/Qwen3.8-27B"
api_model_name=""                        # e.g. "gpt-5.5", "gemini-3.8-flash"

vllm_tp_size=2   # open-weight judge only; 0 = transformers, N = in-process vLLM on N GPUs

# Or judge on a vLLM server from `bash scripts/serve_vllm_judge.sh <judge> [tp_size] [port]`:
# the judge loads once for the whole sweep and this process needs no GPU.
vllm_server_url=""   # e.g. "http://localhost:8000/v1"

if [ -n "$vllm_server_url" ]; then
  # Keyed by the served model id, which vqa_model_name must resolve to.
  TERRAVIS_REMOTE_VLLM_ENDPOINTS=$(python - "$vllm_server_url" <<'PY'
import json, sys, urllib.request
url = sys.argv[1].rstrip("/")
try:
    with urllib.request.urlopen(f"{url}/models", timeout=10) as r:
        served = [m["id"] for m in json.load(r)["data"]]
except Exception as e:
    sys.exit(f"!!! no vLLM server answering at {url}/models: {e}")
print(json.dumps({m: url for m in served}))
PY
  )
  export TERRAVIS_REMOTE_VLLM_ENDPOINTS
  vllm_tp_size=0
  echo ">>> judging on $TERRAVIS_REMOTE_VLLM_ENDPOINTS"
fi

for score_mode in "${score_modes[@]}"; do
  for gen_data_json_path in "${gen_data_json_paths[@]}"; do
    echo ">>> $score_mode | judge=${vqa_model_name:-$api_model_name} | $gen_data_json_path"
    bash ./scripts/compute_t2i_scores.sh \
      "$score_mode" \
      "$gen_data_json_path" \
      "$save_path" \
      "$clip_model_name" \
      "$vqa_model_name" \
      "$api_model_name" \
      "$vllm_tp_size"

    sleep 5
  done
done
