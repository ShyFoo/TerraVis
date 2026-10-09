# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Source Notice
#
# Adapted from TIFA (ICCV'23): https://github.com/Yushi-Hu/tifa  (Apache-2.0)
# The question-generation prompt, question categories and multiple-choice machinery
# are verbatim. Adaptations for one chat VLM serving both roles (question generation
# and VQA) instead of GPT-3.5 completions plus a per-model VQA zoo:
#   * The parser reacts only to "About/Q/Choices/A" lines and cuts at a continued
#     "Description:" example (chat models put blank lines inside one answer).
#     Question de-duplication happens here; the UnifiedQA filter is not ported.
#   * A free-form VQA answer outside the choices maps to the nearest choice with the
#     official SBERT model, but a choice stated in the reply ("Answer: yes") is read
#     out first (resolve_stated_choice): SBERT similarity inverts it.
# ==============================================================================

import os
import re
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

SBERT_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"

# The source's openai_completion token budget for question generation.
QUESTION_GENERATION_MAX_TOKENS = 700

QUESTION_CATEGORIES = ["object", "human", "animal", "food", "activity", "attribute", "counting",
                       "color", "material", "spatial", "location", "shape", "other"]

# The official in-context question-generation prompt, verbatim.
QUESTION_GENERATION_PROMPT = """
Given a image descriptions, generate one or two multiple-choice questions that verifies if the image description is correct.
Classify each concept into a type (object, human, animal, food, activity, attribute, counting, color, material, spatial, location, shape, other), and then generate a question for each type.

Description: A man posing for a selfie in a jacket and bow tie.
Entities: man, selfie, jacket, bow tie
Activities: posing
Colors:
Counting:
Other attributes:
Questions and answers are below:
About man (human):
Q: is this a man?
Choices: yes, no
A: yes
Q: who is posing for a selfie?
Choices: man, woman, boy, girl
A: man
About selfie (activity):
Q: is the man taking a selfie?
Choices: yes, no
A: yes
Q: what type of photo is the person taking?
Choices: selfie, landscape, sports, portrait
A: selfie
About jacket (object):
Q: is the man wearing a jacket?
Choices: yes, no
A: yes
Q: what is the man wearing?
Choices:jacket, t-shirt, tuxedo, swearter
A: jacket
About bow tie (object):
Q: is the man wearing a bow tie?
Choices: yes, no
A: yes
Q: is the man wearing a bow tie or a neck tie?
Choices: bow tie, neck tie, cravat, bolo tie
A: bow tie
About posing (activity):
Q: is the man posing for the selfie?
Choices: yes, no
A: yes
Q: what is the man doing besides taking the selfie?
Choices: posing, waving, nothing, shaking
A: posing

Description: A horse and several cows feed on hay.
Entities: horse, cows, hay
Activities: feed on
Colors:
Counting: several
Other attributes:
Questions and answers are below:
About horse (animal):
Q: is there a horse?
Choices: yes, no
A: yes
About cows (animal):
Q: are there cows?
Choices: yes, no
A: yes
About hay (object):
Q: is there hay?
Choices: yes, no
A: yes
Q: what is the horse and cows feeding on?
Choices: hay, grass, leaves, twigs
A: hay
About feed on (activity):
Q: are the horse and cows feeding on hay?
Choices: yes, no
A: yes
About several (counting):
Q: are there several cows?
Choices: yes, no
A: yes

Description: A red colored dog.
Entities: dog
Activities:
Colors: red
Counting:
Other attributes:
Questions and answers are below:
About dog (animal):
Q: is this a dog?
Choices: yes, no
A: yes
Q: what animal is in the picture?
Choices: dog, cat, bird, fish
A: dog
About red (color):
Q: is the dog red?
Choices: yes, no
A: yes
Q: what color is the dog?
Choices: red, black, white, yellow
A: red

Description: A busy intersection with an ice cream truck driving by.
Entities: intersection, ice cream truck
Activities: driving by
Colors:
Counting:
Other attributes: busy
Questions and answers are below:
About intersection (location):
Q: is this an intersection?
Choices: yes, no
A: yes
Q: is this a intersection or a straight road?
Choices: intersection, straight road, parking lot, school
A: intersection
About ice cream truck (object):
Q: is this an ice cream truck?
Choices: yes, no
A: yes
Q: what type of truck is driving by?
Choices: ice cream truck, car, pickup truck, dumper truck
A: ice cream truck
About driving by (activity):
Q: is the ice cream truck driving by?
Choices: yes, no
A: yes
About busy (attribute):
Q: is this a busy intersection?
Choices: yes, no
A: yes
Q: is this a busy or a quiet intersection?
Choices: busy, quiet, silent, noisy
A: busy

Description: Portrait of a gecko wearing a train conductor's hat and holding a flag that has a yin-yang symbol on it. Woodcut.
Entities: gecko, train conductor's hat, flag, yin-yang symbol
Activities: wearing, holding
Colors:
Counting:
Other attributes: portrait, woodcut
Questions and answers are below:
About gecko (animal):
Q: is this a gecko?
Choices: yes, no
A: yes
Q: what animal is in the photo?
Choices: gecko, human, snake, frog
A: gecko
About train conductor's hat (object):
Q: is the gecko wearing a train conductor's hat?
Choices: yes, no
A: yes
About flag (object):
Q: is the gecko holding a flag?
Choices: yes, no
A: yes
Q: what is the gecko holding?
Choices: flag, sign, banner, poster
A: flag
About yin-yang symbol (object):
Q: is there a yin-yang symbol on the flag?
Choices: yes, no
A: yes
Q: what is on the flag?
Choices: yin-yang symbol, star, heart, cross
A: yin-yang symbol
About wearing (activity):
Q: is the gecko wearing a train conductor's hat?
Choices: yes, no
A: yes
About holding (activity):
Q: is the gecko holding a flag?
Choices: yes, no
A: yes
About portrait (attribute):
Q: is this a portrait?
Choices: yes, no
A: yes
About woodcut (attribute):
Q: is this a woodcut?
Choices: yes, no
A: yes
Q: is this a woodcut, a painting, or a photograph?
Choices: woodcut, painting, drawing, photograph
A: woodcut

Description: A woman is showing a watermelon slice to a woman on a scooter.
Entities: woman, watermelon slice, woman, scooter
Activities: showing
Colors:
Counting:
Other attributes: on
Questions and answers are below:
About woman (human):
Q: is this a woman?
Choices: yes, no
A: yes
Q: who is showing a watermelon slice?
Choices: woman, man, boy, girl
A: woman
About watermelon slice (food):
Q: is there a watermelon slice?
Choices: yes, no
A: yes
Q: what is the woman showing to a woman on a scooter?
Choices: watermelon slice, apple, orange, banana
A: watermelon slice
About woman (human):
Q: is there a woman on a scooter?
Choices: yes, no
A: yes
Q: who is on a scooter?
Choices: woman, man, boy, girl
A: woman
About scooter (object):
Q: is there a scooter?
Choices: yes, no
A: yes
Q: what vehicle is one of the woman on?
Choices: scooter, bicycle, motorcycle, car
A: scooter
About showing (activity):
Q: is the woman showing a watermelon slice to another woman?
Choices: yes, no
A: yes
About on (spatial):
Q: is one of the woman on a scooter?
Choices: yes, no
A: yes
Q: is the woman on the scooter or standing next to it?
Choices: on, next to, behind, in front of
A: on

Description: A photo of three dogs.
Entities: dogs
Activities:
Colors:
Counting: three
Other attributes: photo
Questions and answers are below:
About dogs (animal):
Q: are there dogs?
Choices: yes, no
A: yes
Q: what animals are in the photo?
Choices: dogs, cats, birds, fish
A: dogs
About three (counting):
Q: are there three dogs?
Choices: yes, no
A: yes
Q: how many dogs are in the photo?
Choices: 1, 2, 3, 4
A: 3
About photo (attribute):
Q: is this a photo?
Choices: yes, no
A: yes
Q: is this a photo, a painting, or a comic?
Choices: photo, painting, comic, sculpture
A: photo

Description: A white milk truck with a license plate that reads \"pro milk.\"
Entities: milk truck, license plate
Activities:
Colors: white
Counting:
Other attributes: pro milk
Questions and answers are below:
About milk truck (object):
Q: is this a milk truck?
Choices: yes, no
A: yes
About license plate (object):
Q: is there a license plate on the vehicle?
Choices: yes, no
A: yes
About white (color):
Q: is the truck white?
Choices: yes, no
A: yes
Q: what color is the truck?
Choices: white, black, red, blue
A: white
About pro milk (other):
Q: does the license plate read \"pro milk\"?
Choices: yes, no
A: yes

Description: A person sitting on a horse in air over gate in grass with people and trees in background.
Entities: person, horse, gate, grass, people, trees
Activities: sitting
Colors:
Counting:
Other attributes: over, in air, background
Questions and answers are below:
About person (human):
Q: is there a person on a horse?
Choices: yes, no
A: yes
Q: who is sitting on a horse?
Choices: person, animal, robot, alien
A: person
About horse (animal):
Q: is this a horse?
Choices: yes, no
A: yes
Q: what animal is in the picture?
Choices: horse, cow, sheep, goat
A: horse
About gate (object):
Q: is there a gate?
Choices: yes, no
A: yes
About grass (object):
Q: is there grass?
Choices: yes, no
A: yes
About people (human):
Q: are there people in the background?
Choices: yes, no
A: yes
About trees (object):
Q: are there trees in the background?
Choices: yes, no
A: yes
About sitting (activity):
Q: is the person sitting on the horse?
Choices: yes, no
A: yes
About over (spatial):
Q: is the horse over the gate?
Choices: yes, no
A: yes
Q: is the horse over or next to the gate?
Choices: over, under, next to, behind
A: over
About in air (attribute):
Q: is the horse in air?
Choices: yes, no
A: yes
Q: is the horse in air or on the ground?
Choices: in air, on the ground, in the water, in the sky
A: in air
About background (attribute):
Q: are there people and trees in the background?
Choices: yes, no
A: yes
Q: are there people and trees in the background or in the foreground?
Choices: background, foreground, middle ground, sky
A: background

Description: a red blue and yellow train and some people on a platform
Entities: train, people, platform
Activities:
Colors: red blue and yellow
Counting: some
Other attributes:
Questions and answers are below:
About train (object):
Q: is this a train?
Choices: yes, no
A: yes
Q: what type of vehicle is this?
Choices: train, car, motorcycle, bus
A: train
About people (human):
Q: are there people on the platform?
Choices: yes, no
A: yes
About platform (location):
Q: is there a platform?
Choices: yes, no
A: yes
Q: what type of place is this?
Choices: platform, parking lot, airport, bus stop
A: platform
About red blue and yellow (color):
Q: is the train red blue and yellow?
Choices: yes, no
A: yes
Q: what color is the train?
Choices: red blue and yellow, pure blue, blue and yellow, red and blue
A: red blue and yellow
About some (counting):
Q: are there some people?
Choices: yes, no
A: yes

Description: square blue apples on a tree with circular yellow leaves
Entities: apples, tree, leaves
Activities:
Colors: blue, yellow
Counting:
Other attributes: square, circular
Questions and answers are below:
About apples (food):
Q: are there apples on the tree?
Choices: yes, no
A: yes
Q: what fruit is on the tree?
Choices: apples, oranges, bananas, pears
A: apples
About tree (object):
Q: is this a tree?
Choices: yes, no
A: yes
Q: what type of plant is this?
Choices: tree, flower, bush, grass
A: tree
About leaves (object):
Q: are there leaves on the tree?
Choices: yes, no
A: yes
Q: what is on the tree?
Choices: leaves, branches, roots, flowers
A: leaves
About blue (color):
Q: are the apples blue?
Choices: yes, no
A: yes
Q: what color are the apples?
Choices: blue, red, yellow, green
A: blue
About yellow (color):
Q: are the leaves yellow?
Choices: yes, no
A: yes
Q: what color are the leaves?
Choices: yellow, red, blue, green
A: yellow
About square (shape):
Q: are the apples square?
Choices: yes, no
A: yes
Q: what shape are the apples?
Choices: square, round, oval, triangle
A: square
About circular (shape):
Q: are the leaves circular?
Choices: yes, no
A: yes
Q: what shape are the leaves?
Choices: circular, square, oval, triangle
A: circular

Description: Overview of a pot of vegetable soup with wooden spoon on a stainless steel stove top.
Entities: pot, vegetable soup, spoon, stove top
Activities:
Colors:
Counting:
Other attributes: overview, wooden, stainless steel
Questions and answers are below:
About pot (object):
Q: is there a pot of vegetable soup?
Choices: yes, no
A: yes
Q: what type of container is this?
Choices: pot, bowl, cup, glass
A: pot
About vegetable soup (food):
Q: is there vegetable soup?
Choices: yes, no
A: yes
Q: what type of food is this?
Choices: vegetable soup, chicken soup, beef soup, fish soup
A: vegetable soup
About spoon (object):
Q: is there a spoon?
Choices: yes, no
A: yes
Q: what type of utensil is this?
Choices: spoon, fork, knife, chopsticks
A: spoon
About stove top (object):
Q: is there a stove top?
Choices: yes, no
A: yes
Q: what type of appliance is this?
Choices: stove top, oven, microwave, refrigerator
A: stove top
About overview (attribute):
Q: is this an overview?
Choices: yes, no
A: yes
About wooden (material):
Q: is the spoon wooden?
Choices: yes, no
A: yes
Q: what is the material of the spoon?
Choices: wooden, metal, plastic, glass
A: wooden
About stainless steel (material):
Q: is the stove top stainless steel?
Choices: yes, no
A: yes
Q: what is the material of the stove top?
Choices: stainless steel, iron, aluminum, copper
A: stainless steel

Description: """


def build_question_generation_prompt(caption: str) -> str:
    """The official prompt plus the caption, cued to continue from an
    "Entities" line (the source's ``get_question_and_answers``)."""
    return QUESTION_GENERATION_PROMPT + caption + "\nEntities"


def parse_question_generation_response(caption: str, response: str) -> List[Dict[str, Any]]:
    """Parse "About <element> (<type>):" blocks of Q/Choices/A lines into TIFA
    QA dicts — the source's ``parse_resp`` plus the ``get_question_and_answers``
    post-processing (category filter, animal/human merge) and the filter step's
    question de-duplication."""
    # A continued "Description:" block is the model hallucinating the next
    # in-context example; its questions are not about this caption.
    response = str(response or "").split("\nDescription:")[0]

    qa_pairs: List[Dict[str, Any]] = []
    seen_questions = set()
    element = element_type = question = choices = None
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("About "):
            # A header drops pending Q/Choices (a block truncated before "A:" would leak them onto the
            # next answer); one without "(type)" also resets the element, so its Q/A count for nothing
            # (the source's parse_resp IndexErrors on it).
            question = choices = None
            if " (" in line:
                element = line[len("About "):].split(" (")[0]
                element_type = line.split(" (")[1].split(")")[0]
            else:
                element = element_type = None
        elif line.startswith("Q: "):
            question = line[3:]
        elif line.startswith("Choices:"):
            # No space needed after the colon: the prompt's own example writes "Choices:jacket, t-shirt, ...".
            choices = [c.strip() for c in line[len("Choices:"):].split(",")]
        elif line.startswith("A: "):
            answer = line[3:]
            if (element and question and choices and question not in seen_questions
                    and element_type in QUESTION_CATEGORIES):
                seen_questions.add(question)
                qa_pairs.append({
                    "caption": caption,
                    "element": element,
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "element_type": ("animal/human" if element_type in ("animal", "human")
                                     else element_type),
                })
            question = choices = None
    return qa_pairs


def build_multiple_choice_question(question: str, choices: List[str]) -> str:
    """The choice-aware VQA prompt (the source's BLIP-2 template)."""
    return (f"Answer the multiple-choice question. "
            f"Question: {question} Choices: {', '.join(choices)} Answer:")


class SBERTModel:
    """Sentence-BERT nearest-choice matcher (the source's ``mc_sbert.py``)."""

    def __init__(self, model_weight_root: str, device: Union[str, torch.device] = "cpu"):
        tifa_dir = os.path.join(model_weight_root, "TIFA")
        self.tokenizer = AutoTokenizer.from_pretrained(SBERT_MODEL_ID, cache_dir=tifa_dir)
        # Auto*'s PEFT probe ignores cache_dir and would leave an empty repo dir in HF_HUB_CACHE.
        self.model = AutoModel.from_pretrained(SBERT_MODEL_ID, cache_dir=tifa_dir,
                                               adapter_kwargs={"cache_dir": tifa_dir})
        self.model.to(device).eval()

    def mean_pooling(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def embed_sentences(self, sentences: List[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        model_output = self.model(**encoded_input.to(self.model.device))
        sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
        return F.normalize(sentence_embeddings, p=2, dim=1).detach().cpu()

    def multiple_choice(self, answer: str, choices: List[str]) -> str:
        answer_embedding = self.embed_sentences([answer])
        choices_embedding = self.embed_sentences(choices)
        top_choice_index = torch.argmax(torch.matmul(choices_embedding, answer_embedding.T)).item()
        return choices[top_choice_index]


_ANSWER_PREFIX = re.compile(
    r"^(?:the\s+)?(?:correct\s+|final\s+|short\s+|best\s+|most\s+likely\s+)?"
    r"answer(?:\s+to\s+(?:the|this)\s+question)?(?:\s+is)?\s*[:\-–]?\s*", re.I)
_ANSWER_MARKER = re.compile(
    r"(?:the\s+)?(?:correct\s+|final\s+|short\s+|best\s+)?answer(?:\s+is)?\s*[:\-–]", re.I)
_EMPHASIS = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
# Punctuation and quoting a chat model wraps its verdict in. Choices are trimmed
# with the same set, so a choice that is itself quoted still compares equal.
_TRIM = " \t.!,;:\"'*`‘’“”"


def _canonical(text: str) -> str:
    return text.strip().strip(_TRIM).strip().lower()


def _answer_spans(free_form_answer: str) -> List[str]:
    """Every span of a chat answer that could be the verdict stated outright."""
    answer = (free_form_answer or "").strip()
    spans = [answer, _ANSWER_PREFIX.sub("", answer, count=1)]

    lines = [line for line in answer.splitlines() if line.strip()]
    if lines:  # a model that reasons first puts the verdict on the last line
        spans.append(_ANSWER_PREFIX.sub("", lines[-1].strip(), count=1))

    markers = list(_ANSWER_MARKER.finditer(answer))
    if markers:  # ... and often labels it, so read past the LAST label
        spans.append(_ANSWER_PREFIX.sub("", answer[markers[-1].end():].strip(), count=1))

    spans.extend(g for m in _EMPHASIS.finditer(answer) for g in m.groups() if g)
    return spans


def resolve_stated_choice(free_form_answer: str, choices: List[str]) -> Optional[str]:
    """The choice a chat judge stated outright, or None if it must be inferred.

    Returns a choice only when every span that matches one matches the *same*
    one, so genuinely free-form or multi-answer replies fall through to SBERT
    rather than being resolved by span precedence.
    """
    keys = [_canonical(c) for c in choices]
    if not all(keys) or len(set(keys)) != len(set(choices)):
        return None  # choices collapse under canonicalisation: no safe comparison
    by_key = dict(zip(keys, choices))
    stated = {by_key[k] for k in map(_canonical, _answer_spans(free_form_answer)) if k in by_key}
    return stated.pop() if len(stated) == 1 else None


def multiple_choice_from_free_form(sbert_model: SBERTModel, free_form_answer: str,
                                   choices: List[str]) -> str:
    """Limit a free-form VQA answer to the choices (the source's ``VQAModel.multiple_choice_vqa``).
    A stated choice is read out before SBERT: all-mpnet-base-v2 puts "Answer: yes" 0.0002 closer
    to "no" than to "yes".
    """
    if free_form_answer in choices:
        return free_form_answer
    stated = resolve_stated_choice(free_form_answer, choices)
    if stated is not None:
        return stated
    return sbert_model.multiple_choice(free_form_answer, choices)
