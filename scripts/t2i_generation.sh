#!/usr/bin/env bash
# Usage: bash scripts/t2i_generation.sh <benchmark> <model_name> <num_gpus>
#   benchmark: bundled benchmark name, or your own prompts .json (a list of strings; output goes under its file name)
#   model_name: HF repo id, API model name, registry short name (src/terravis/configs/models.yaml), or local checkpoint path
# Env: HF_TOKEN (gated repos), OPENAI_API_KEY, GOOGLE_API_KEY or GEMINI_API_KEY, IDEOGRAM_API_KEY (ideogram-4-fp8)
#   SAVE_ROOT / MODEL_WEIGHT_ROOT  scripts/set_root_paths.sh; writes $SAVE_ROOT/<benchmark>/<model>/seed=<seed>/
#   RESUME             1 = skip prompts already rendered (own prompts: by list position, so only append); a .txt failure marker counts, delete it to retry
#   GPUS_PER_REPLICA   model_sharding only: GPUs per copy, overriding models.yaml (sized for 80 GB cards)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/set_root_paths.sh"

PORT=$(shuf -i 2000-65000 -n 1)

seeds=(1111)

benchmark=$1
model_name=$2
num_gpus=$3                      # 0 = CPU

extra_args=()
if [ "${RESUME:-0}" = "1" ]; then
  extra_args+=(--resume)
fi

if ! read -r parallel_mode gpus_per_replica < <(
  MODEL_NAME="$model_name" python -c "
import os
from terravis.configs import find_t2i
spec = find_t2i(os.environ['MODEL_NAME'])
print(spec.parallel_mode, spec.gpus_per_replica)
"); then
  echo "ERROR: could not resolve '${model_name}' in the model registry (see the traceback above)." >&2
  exit 1
fi
gpus_per_replica=${GPUS_PER_REPLICA:-$gpus_per_replica}
if ! [[ "$gpus_per_replica" =~ ^(0|[1-9][0-9]{0,2})$ ]]; then
  echo "ERROR: GPUS_PER_REPLICA must be a whole number, got '${gpus_per_replica}'." >&2
  exit 1
fi
[ -n "${GPUS_PER_REPLICA:-}" ] && [ "$parallel_mode" != "model_sharding" ] \
  && echo "NOTE: GPUS_PER_REPLICA only applies to model_sharding models; ignored for ${model_name}."

replicas=1
if [ "$parallel_mode" = "model_sharding" ] && [ "$gpus_per_replica" -gt 0 ]; then
  if [ "$num_gpus" -lt "$gpus_per_replica" ]; then
    echo "WARNING: $model_name wants ${gpus_per_replica} GPUs per copy but only ${num_gpus} given;" \
         "running one replica across all of them."
  else
    replicas=$(( num_gpus / gpus_per_replica ))
    leftover=$(( num_gpus - replicas * gpus_per_replica ))
    [ "$leftover" -gt 0 ] && [ "$replicas" -gt 1 ] && echo "NOTE: ${leftover} GPU(s) unused (${num_gpus} is not a multiple of ${gpus_per_replica})."
  fi
fi

run_replicas() {
  local seed=$1 log_dir idx devices gpu_ids rc=0
  local pids=()
  log_dir=$(mktemp -d)
  # Slice the caller's CUDA_VISIBLE_DEVICES, if any, rather than claiming GPUs 0..n-1.
  IFS=, read -ra gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(( num_gpus - 1 )))}"
  if (( ${#gpu_ids[@]} < replicas * gpus_per_replica )); then
    echo "ERROR: CUDA_VISIBLE_DEVICES lists ${#gpu_ids[@]} GPU(s); ${replicas} replica(s) x ${gpus_per_replica} need more." >&2
    return 1
  fi
  echo "model_sharding: ${replicas} replica(s) x ${gpus_per_replica} GPU(s); per-replica logs in ${log_dir}"

  for (( idx = 0; idx < replicas; idx++ )); do
    devices=$(IFS=,; echo "${gpu_ids[*]:idx * gpus_per_replica:gpus_per_replica}")
    CUDA_VISIBLE_DEVICES="$devices" accelerate launch --num_processes 1 --mixed_precision bf16 \
      --main_process_port="$(( PORT + idx ))" -m terravis.cli.generate \
      --model_name "$model_name" \
      --benchmark "$benchmark" \
      --seed "$seed" \
      --save_root "$SAVE_ROOT" \
      --model_weight_root "$MODEL_WEIGHT_ROOT" \
      --num_shards "$replicas" \
      --shard_index "$idx" \
      "${extra_args[@]}" \
      > "${log_dir}/shard${idx}.log" 2>&1 &
    pids+=("$!")
  done

  for idx in "${!pids[@]}"; do
    if ! wait "${pids[idx]}"; then
      echo "=== replica ${idx} FAILED (${log_dir}/shard${idx}.log) ==="
      tail -40 "${log_dir}/shard${idx}.log"
      rc=1
    fi
  done
  # Merging after a lost replica writes a short test_data.json.
  [ "$rc" -ne 0 ] && return "$rc"

  python -m terravis.cli.generate --merge_shards \
    --model_name "$model_name" --benchmark "$benchmark" --seed "$seed" --save_root "$SAVE_ROOT"
}

# device_map spreads model_sharding over every visible GPU: show it only the first num_gpus.
if [ "$parallel_mode" = "model_sharding" ] && [ "$num_gpus" -ge 1 ]; then
  IFS=, read -ra visible <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(( num_gpus - 1 )))}"
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${visible[*]:0:num_gpus}")
fi

if [ "$num_gpus" -ge 2 ]; then
  launch_args=(--multi_gpu --num_processes "$num_gpus" --mixed_precision bf16 --enable_cpu_affinity)
elif [ "$num_gpus" -eq 1 ]; then
  launch_args=(--num_processes 1 --mixed_precision bf16)
else
  launch_args=(--cpu --num_processes 1)
fi

for seed in "${seeds[@]}"; do
  if [ "$replicas" -gt 1 ]; then
    run_replicas "$seed"
    continue
  fi
  accelerate launch "${launch_args[@]}" --main_process_port="$PORT" -m terravis.cli.generate \
    --model_name "$model_name" \
    --benchmark "$benchmark" \
    --seed "$seed" \
    --save_root "$SAVE_ROOT" \
    --model_weight_root "$MODEL_WEIGHT_ROOT" \
    "${extra_args[@]}"
done
