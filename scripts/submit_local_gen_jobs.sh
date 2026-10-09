#!/usr/bin/env bash
# Edit the arrays below, then: bash scripts/submit_local_gen_jobs.sh

set -euo pipefail

num_gpus=1

benchmarks=(
  "coco_t2i"
  "genai_bench"
# "/path/to/my_prompts.json"   # your own: a JSON list of prompt strings
)

# VRAM and GPU counts are in models.yaml.
# For model_sharding make num_gpus a multiple of gpus_per_replica.
model_names=(
  "stabilityai/stable-diffusion-3.5-large"
  "black-forest-labs/FLUX.2-dev"
  "black-forest-labs/FLUX.2-klein-9B"
  "Qwen/Qwen-Image-2512"
  "ideogram-ai/ideogram-4-fp8"
  "nvidia/Cosmos3-Super-Text2Image-4Step"
  "HiDream-ai/HiDream-O1-Image"
  "HiDream-ai/HiDream-O1-Image-Dev-2604"
# "/path/to/my-finetuned-model"   # your own: a local checkpoint or a models.yaml entry
)

for benchmark in "${benchmarks[@]}"; do
  for model_name in "${model_names[@]}"; do
    bash ./scripts/t2i_generation.sh "$benchmark" "$model_name" "$num_gpus"
    sleep 5
  done
done
