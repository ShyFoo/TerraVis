# SPDX-License-Identifier: CC-BY-4.0
# Copyright (C) 2025 Tencent. All rights reserved.
# ==============================================================================
# Source Notice
#
# Adapted from UniGenBench++: https://github.com/CodeGoat24/UniGenBench  (CC BY 4.0)
# The checkpoint-explanation dictionary, judge prompt template and prompt assembly
# are eval/src/eval_common.py's English protocol, verbatim (including the
# "Checkpoints Defination" spelling); the Chinese protocol is not ported.
# Adaptations for one in-process chat VLM instead of Gemini / finetuned-Qwen servers:
#   * parse_evaluation_response returns None where the source prints a warning or
#     crashes (non-list or wrong-length literal, literal_eval overflow).
#   * No parse retry: the local judge decodes greedily, so a failed parse scores
#     'n/a' (scores/metrics/unigenbench_score.py).
# ==============================================================================

import ast
import re
from typing import List, Optional, Tuple

# The source's max_tokens per evaluation call (same on both its backends).
UNIGENBENCH_EVAL_MAX_TOKENS = 4096

EXPLANATION_DICT_EN = {
    "Relationship - Comparison": "Comparison of attributes between two entities",
    "Relationship - Composition": "An entity is composed of one or more other entities",
    "Relationship - Inclusion": "A container contains an entity; the container can also be a plane, e.g., a snake in a painting on a wall",
    "Relationship - Similarity": "Existence of similarities between different entities",

    "Compound - Imagination": "Things that are impossible in real life",
    "Compound - Feature Matching": "Different entities possess different types of attribute features",

    "Attribute - Size": "Assessment of the subject's size, height, length, thickness, width, or tallness/shortness",
    "Attribute - Expression": "Distinguishing expressions from facial actions; expressions must convey a clear emotion",
    "Attribute - Quantity": "Focuses on the challenge of depicting three or more items accurately",
    "Attribute - Material": "Evaluation of different material types and textures",
    "Attribute - Color": "Assessment of different colors",
    "Attribute - Shape": "Assessment of different shapes",

    "Entity Layout - Two-Dimensional Space": "Arrangement and positioning of entities in two-dimensional space",
    "Entity Layout - Three-Dimensional Space": "Arrangement and positioning of entities in three-dimensional space",

    "Action - Full-body (Character/Anthropomorphic)": "Full-body actions by characters or anthropomorphized entities, such as running, diving, breakdancing, swinging, or hanging upside down",
    "Action - Hand (Character/Anthropomorphic)": "Focuses on hand structure—checking if fingers are missing, broken, or distorted",
    "Action - Animal": "Actions performed by animals",
    "Action - Contact Interaction": "Physical interactions between entities",
    "Action - Non-contact Interaction": "For example, two people making eye contact—testing if the model can accurately depict such interactions",
    "Action - State": "A sustained state of an entity, typically expressed with a verb",

    "Grammar - Negation": "Tests the model's understanding of negation grammar",
    "Grammar - Pronoun Reference": "Tests if the model can resolve ambiguous pronoun references correctly",
    "Grammar - Consistency": "Evaluation of shared attributes among entities",

    "World Knowledge": "Covers knowledge of celebrities, architecture, basic domain knowledge, and internet slang. Celebrities with modern copyright risk should be avoided",

    "Style": "Art, painting, photography, design styles, and corresponding artist names",
    "Text Generation": "The text content model needed to accurately generate without any omissions or extra words",

    "Logical Reasoning": "Requires the model to deeply understand the intent and perform reasoning",
}

SYSTEM_PROMPT_EN_TEMPLATE = '''You are a precise and objective English-language image description system. I will provide you with a prompt for image generation, as well as the corresponding generated image. You will be given a set of evaluation criteria (checkpoints) and their explanations that define the relevance between the prompt and the image. You must evaluate whether the generated image fulfills the requirements implied by each checkpoint in the prompt.

            For each image, follow the steps below in order:

            1. The prompt for the generated image is: 「{prompt}」. You are to analyze the image content in detail from the angles specified in {testpoint}. Detailed definitions of these checkpoints are provided here: {explanation}. The specific description of each checkpoint in the context of the prompt is: {test_explanation}. You must analyze whether the image meets the requirements for each checkpoint individually.

            2. Based on the above analysis, determine whether the generated image satisfies each checkpoint in terms of its visual alignment with the prompt. If the image meets the requirements of a checkpoint, assign a score of 1 to that checkpoint; otherwise, assign a score of 0.

            Constraints:
            - Only describe content that is directly visible; do not interpret, speculate, or infer any background story.
            - Focus solely on visually verifiable details.
            - Omit any uncertain or ambiguous elements.
            - Even if mentioned in the input, do not describe abstract entities, emotions, or speculative ideas.

            Please strictly follow the output format below:

            <description>
                <prompt>{prompt}</prompt>
                <checkpoint>{testpoint}</checkpoint>
                <analysis>A list using square brackets `[]`, where each element is a string of detailed analysis corresponding to one checkpoint, as required in Step 1. **Ensure the list length matches the number of checkpoints**. Each element should be a string representing the analysis for that specific checkpoint.</analysis>
                <score>A list using square brackets `[]`, where each element is a binary score (0 or 1) corresponding to a checkpoint, as required in Step 2. **Ensure the list length matches the number of checkpoints**. Each element should be either 0 or 1, indicating whether the checkpoint was satisfied.</score>
            </description>
            '''


def build_system_prompt(prompt: str, testpoints: List[str], test_descs: List[str]) -> str:
    """The source's ``build_system_prompt(..., lang="en")``, byte-identical:
    one judge prompt embedding the T2I prompt, the checkpoint list (its Python
    list repr, as the source's str.format renders it), the benchmark-wide
    checkpoint definitions, and this prompt's per-checkpoint descriptions."""
    explanation = "Checkpoints Defination:「"  # sic — the source's spelling
    for point in testpoints:
        if point not in EXPLANATION_DICT_EN:
            raise ValueError(f"Checkpoint '{point}' not found in en explanation dict")
        explanation += f"\n{point}: {EXPLANATION_DICT_EN[point]}"
    explanation += "\n」"

    test_explanation = "Checkpoints Description:「"
    for idx, point in enumerate(testpoints):
        test_explanation += f"\n{point}: {test_descs[idx]}"
    test_explanation += "\n」"

    return SYSTEM_PROMPT_EN_TEMPLATE.format(
        prompt=prompt, testpoint=testpoints,
        explanation=explanation, test_explanation=test_explanation,
    )


def parse_evaluation_response(text: Optional[str], testpoints: List[str]) -> Optional[Tuple[list, list]]:
    """The source's ``parse_evaluation_response``: the first ``<analysis>`` and
    ``<score>`` tag pair, each body read as a Python literal, both required to
    have exactly one entry per checkpoint. None on a missing tag, an
    unparsable literal, or a length mismatch."""
    if text is None:
        return None
    analysis_match = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL)
    score_match = re.search(r"<score>(.*?)</score>", text, re.DOTALL)
    if analysis_match is None or score_match is None:
        return None
    try:
        analysis = ast.literal_eval(analysis_match.group(1).strip())
        score = ast.literal_eval(score_match.group(1).strip())
    except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
        # Every failure mode literal_eval documents; the last three fire only on
        # pathological input (e.g. a several-thousand-char unary chain).
        return None
    # Lists only: a same-length string like "010" passes len() and silently scores every checkpoint 0.
    if not (isinstance(analysis, list) and isinstance(score, list)):
        return None
    if len(testpoints) != len(analysis) or len(testpoints) != len(score):
        return None
    return analysis, score
