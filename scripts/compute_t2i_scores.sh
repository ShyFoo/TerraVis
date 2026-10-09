#!/usr/bin/env bash
# Usage: bash scripts/compute_t2i_scores.sh <score_mode> <gen_data_json_path> <save_path> \
#          <clip_model_name> <vqa_model_names> <api_model_name> [vllm_tp_size]
# Env: MODEL_WEIGHT_ROOT  weights root (scripts/set_root_paths.sh)
#      API_PROVIDER       openai | google, for an API judge with no models.yaml entry
#      RESUME             1 = reuse an interrupted run's cached per-image scores
#      TERRAVIS_REMOTE_VLLM_ENDPOINTS  judge on a vLLM server (scripts/serve_vllm_judge.sh); vllm_tp_size is ignored
#      OPENAI_API_KEY, GOOGLE_API_KEY or GEMINI_API_KEY  for an API judge

source "$(dirname "${BASH_SOURCE[0]}")/set_root_paths.sh"

score_mode=$1              # e.g. clip_score, terravis_score
gen_data_json_path=$2
save_path=$3
clip_model_name=$4         # OpenCLIP, format 'pretrained:arch'
vqa_model_names=$5         # open-weight judge: HF repo id or registered short name
api_model_name=$6          # e.g. 'gpt-5.5', 'gemini-3.8-flash'
vllm_tp_size=${7:-0}       # 0 = transformers, N = in-process vLLM on N GPUs

if [ "$vllm_tp_size" -ge 1 ]; then
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
fi

extra_args=()
if [ -n "${API_PROVIDER:-}" ]; then
    extra_args+=(--api_provider "$API_PROVIDER")
fi
if [ "${RESUME:-0}" = "1" ]; then
    extra_args+=(--resume)
fi

terravis-score \
  --score_mode "$score_mode" \
  --gen_data_json_path "$gen_data_json_path" \
  --save_path "$save_path" \
  --clip_model_name "$clip_model_name" \
  --vqa_model_names "$vqa_model_names" \
  --api_model_name "$api_model_name" \
  --vllm_tp_size "$vllm_tp_size" \
  --model_weight_root "$MODEL_WEIGHT_ROOT" \
  "${extra_args[@]}"
