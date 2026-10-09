# SPDX-License-Identifier: MIT
# Copyright (c) 2026 HiDream.ai
# ==============================================================================
# Source Notice
#
# Vendored from HiDream-O1-Image (arXiv:2605.11061), the official implementation:
#   https://github.com/HiDream-ai/HiDream-O1-Image  @ 2c2d29f  (MIT)
# The HF repo ships weights only and the GitHub repo is a bare script tree; the
# checkpoint is a transformers qwen3_vl model whose forward is a pixel-space
# denoiser, so stock Qwen3VLForConditionalGeneration cannot run it.
#
# Upstream's models/ package, copied verbatim from that commit:
#   qwen3_vl_transformers.py  the extended Qwen3-VL model
#   pipeline.py               generate_image(): sampling loop, patchify/de-patchify
#   utils.py                  resolution snapping, reference-image layout
#   flash_scheduler.py        the distilled ("dev") checkpoint's scheduler
#   fm_solvers_unipc.py       FlowUniPCMultistepScheduler, the "full" default
#
# Four edits, each marked "TerraVis deviation" at its site:
#   pipeline.py               use_flash_attn follows what is installed instead of
#                             a literal True.
#   pipeline.py               resolution snapping (aspect ratio only; every square
#                             request rendered 2048x2048) is behind snap_to_predefined,
#                             default True; the loader passes False.
#   qwen3_vl_transformers.py  Qwen3VLTextRotaryEmbedding, two edits for transformers 5
#                             (ROPE_INIT_FUNCTIONS lost "default", rope_theta moved into
#                             config.rope_parameters, meta-device init); inv_freq
#                             verified bit-identical to 4.57.1.
#
# Only the text-to-image path is used; ref_image_paths is not fixed for transformers 5
# (the vision tower's rotary inv_freq loads uninitialized). Importing prints eight cosmetic
# "[ERROR] `<arg>` is part of ...'s signature, but not documented" lines from
# transformers' @auto_docstring.
# ==============================================================================
