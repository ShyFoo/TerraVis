TERRAVIS_ELIGIBILITY_PROMPT = """
### Task
You are an image eligibility classifier. Determine whether the given image is eligible for world-consistency evaluation.

### Rules

Mark as "Eligible" if the image is presented as a single-frame representational image depicting identifiable macroscopic objects or scenes. Photorealistic and stylized images both count.

Mark as "N/A" if the image's primary content belongs to any of the following categories:
- Abstract content: (e.g., color blocks, geometric patterns, noise textures, abstract shapes, abstract illustrations)
- Symbolic content: (e.g., logos, icons, emblems, flags, emojis, QR codes)
- Informational content: (e.g., charts, diagrams, maps, tables, flowcharts, infographics, timelines, mind maps, menus)
- Interface content: (e.g., UI mockups, app/website screenshots, software interfaces, operating-system windows, video game menus, control panels)
- Layout-based compositions: (e.g., posters, advertisements, book covers, storyboards, collages, multi-panel layouts)
- Microscopic content: (e.g., cell structures, material surfaces, mineral textures, crystal structures)

### Priority Rules
- If the image is clearly composed of multiple independent panels, frames, or views, mark it as "N/A".
- If the image shows a physical object that happens to contain text, a map, a menu, a cover design, or other printed information, mark it as "Eligible" unless the image is only a flat graphic/layout with no physical-object presentation.

### Output Format
Output strictly in the following format, with no additional text:

Answer: <Eligible or N/A>
Reason: <One sentence describing the reason.>
"""

TERRAVIS_VIOLATION_DETECTION_PROMPT_TEMPLATE = """
### Task
You are an image issue detector. Determine whether the following issue is present in the image.

### Rules
- Do not penalize intentional reduced photorealism in stylized images (e.g., simplified texture, color, lighting, motion depiction, medium-response details).
- If the issue is present, report the single most salient instance only.
- Focus only on visible evidence.
- Do not rely on imagination.
- Use a strict standard, as some violations may be subtle.

### Question:
{QUESTION}

### Output Format
Output strictly in the following format, with no additional text:

Answer: <Yes or No>.
Affected aspect: <name of the affected object, interaction, or scene element; or empty if No>
Evidence: <one sentence describing the visible evidence; or empty if No>
"""

TERRAVIS_SEVERITY_PROMPT_TEMPLATE = """
### Task
You are an image issue severity classifier. Determine whether the detected issue is a major or minor world-consistency violation.

### Rules
- Major violation: the violation affects the main subject, a foreground object, the depicted action/event, or another semantically central element in the image, even if the visible anomaly is small.
- Minor violation: the violation affects a background, peripheral, or semantically non-central element.
- Use only the detected issue and visible evidence. Do not introduce new issues.

### Detected Issue
Violation type: {VIOLATION_TYPE}
Question: {QUESTION}
Affected aspect: {AFFECTED_ASPECT}
Evidence: {EVIDENCE}

### Output Format
Output strictly in the following format, with no additional text:

Severity: <Major or Minor>.
Reason: <One sentence explaining whether the affected aspect is semantically central or peripheral.>
"""

TERRAVIS_VIOLATION_QUESTIONS = [
    {
        "domain": "object-level",
        "violation_type": "Object identity",
        "question": "Ignoring embedded depictions such as murals, paintings, screens, or printed images, does the image contain any physically present entity that is presented as a real object or living being but has impossible identity traits, such as a living dragon or an impossible animal/object hybrid?",
    },
    {
        "domain": "object-level",
        "violation_type": "Structural distortion",
        "question": "Does any object appear visibly malformed due to generation failure, for example melted, broken in shape, or geometrically distorted in an implausible way?",
    },
    {
        "domain": "object-level",
        "violation_type": "Biological anatomy",
        "question": "Does any human or animal body show anatomically impossible features, such as an impossible limb orientation, or eyes looking in incompatible directions?",
    },
    {
        "domain": "object-level",
        "violation_type": "Non-biological structure",
        "question": "Does any non-living object have an impossible structure or part arrangement, such as disconnected wheels, misplaced parts, physically invalid construction, or parts whose geometry or alignment is inconsistent under the apparent viewpoint?",
    },
    {
        "domain": "object-level",
        "violation_type": "Texture/surface",
        "question": "Does any object show a physically implausible material or surface pattern, such as skin that looks like wood or stone without scene justification?",
    },
    {
        "domain": "object-level",
        "violation_type": "Symbolic content",
        "question": "Does any visible text, sign, symbol, logo, or marking appear broken, distorted, or unrecognizable in a way that violates real-world plausibility?",
    },
    {
        "domain": "object-level",
        "violation_type": "Object-context",
        "question": "Is there any object or symbolic content that is clearly inappropriate for the scene context according to common real-world knowledge, such as a dog working in an office?",
    },
    {
        "domain": "interaction-level",
        "violation_type": "Contact state",
        "question": "Do any two objects appear to pass through each other or interpenetrate when they should be in normal physical contact?",
    },
    {
        "domain": "interaction-level",
        "violation_type": "Support and stability",
        "question": "Does any person or object appear to float, hover, or remain stably at rest without adequate support, balance, or friction, and without plausible motion context?",
    },
    {
        "domain": "interaction-level",
        "violation_type": "Dynamic response",
        "question": "Is there any missing or implausible physical response to force, motion, impact, or turning, such as no splash, no deformation, or contradictory motion cues?",
    },
    {
        "domain": "interaction-level",
        "violation_type": "Medium interaction",
        "question": "Is there any implausible interaction with water, air, snow, sand, or similar media, such as incorrect buoyancy, immersion depth, wake, or air-resistance cues?",
    },
    {
        "domain": "interaction-level",
        "violation_type": "Optical effect",
        "question": "Is there any physically implausible shadow, reflection, refraction, or transmission effect?",
    },
    {
        "domain": "interaction-level",
        "violation_type": "Energy source",
        "question": "Is there visible light, heat, glow, or other energy output without a plausible source or supporting cue?",
    },
    {
        "domain": "interaction-level",
        "violation_type": "Thermal response",
        "question": "Is there any implausible response to heat or cold, such as missing or inappropriate melting, burning, boiling, steam emission, freezing, or condensation?",
    },
    {
        "domain": "interaction-level",
        "violation_type": "Behavior-event",
        "question": "Does any person or animal behave in a way that contradicts the depicted event or action?",
    },
    {
        "domain": "scene-level",
        "violation_type": "Relative scale",
        "question": "Do objects or subjects appear to have unrealistic relative sizes compared to each other or to the scene?",
    },
    {
        "domain": "scene-level",
        "violation_type": "Depth-order/occlusion",
        "question": "Are there any implausible occlusion relationships or depth-order inconsistencies, such as overlapping subjects causing parts to appear implausibly missing, duplicated, or incorrectly occluded?",
    },
    {
        "domain": "scene-level",
        "violation_type": "Region continuity",
        "question": "Does the image show unnatural transitions between adjacent regions, such as visible seams, inconsistent lighting across regions, mismatched perspective, or spatial discontinuities at region boundaries?",
    },
]

TERRAVIS_PROMPT = {
    "name": "TerraVis",
    "method": "terravis",
    "lambda": 1.0,
    "alpha": 0.5,
    "eligibility_prompt": TERRAVIS_ELIGIBILITY_PROMPT,
    "violation_prompt_template": TERRAVIS_VIOLATION_DETECTION_PROMPT_TEMPLATE,
    "severity_prompt_template": TERRAVIS_SEVERITY_PROMPT_TEMPLATE,
    "violation_questions": TERRAVIS_VIOLATION_QUESTIONS,
}