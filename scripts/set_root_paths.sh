# Edit the two paths for your machine (absolute; write $HOME, not ~); every script and CLI reads them from here.
# "${VAR:-path}": an already-exported VAR wins, otherwise path.
export MODEL_WEIGHT_ROOT="${MODEL_WEIGHT_ROOT:-/data/model_weights}"      # all model weights
export SAVE_ROOT="${SAVE_ROOT:-/data/TerraVis-results/gen_images}"        # Step 1 output, Step 2 input
export HF_HUB_CACHE="$MODEL_WEIGHT_ROOT/hf_cache"                         # Hugging Face repos: judges, T2I models, CLIP
