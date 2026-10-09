# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Source Notice
#
# Vendored from the Ideogram 4 inference stack:
#   https://github.com/ideogram-oss/ideogram4  @ 990fe1c  (Apache-2.0; the model
#   weights carry Ideogram's separate non-commercial licence and are not bundled)
# ideogram4 is git-only and pulls in bitsandbytes and its own pins; TerraVis
# needs only these two modules.
#
# From that commit's src/ideogram4/:
#   caption_verifier.py  verbatim
#   magic_prompt.py      the hosted magic-prompt path (Ideogram4MagicPromptV1,
#                        "ideogram-4-v1", and aspect_ratio_from_size), verbatim
#                        apart from two edits: the CaptionVerifier import is repointed
#                        into this package (marked "TerraVis deviation") and the
#                        OpenRouter configurations and their helpers are dropped.
# ==============================================================================
