# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Source Notice
#
# Adapted from DSG (ICLR'24): https://github.com/j-min/DSG  (Apache-2.0)
# The three-stage question generation (tuples -> questions -> dependencies) and the
# dependency-aware scoring are verbatim; the three in-context prompts are
# query_utils.make_prompt's output over the 23 TIFA160 train examples, captured as
# literals. Adaptations for one chat VLM serving both roles instead of GPT-3.5 plus
# a separate VQA model:
#   * Parsers skip a malformed "id | value" line instead of raising; a dependency
#     entry with no numeric ids parses as [0] (no parents); the "output:"/"input:"
#     cuts also apply to the tuple text fed into the later prompts.
#   * The yes/no verdict is the first word left after dropping a label and wrappers
#     (binary_verdict): a chat judge decorates ("**Yes**", "Answer: yes").
#   * Dependency filtering ignores unknown question and parent ids instead of raising.
# ==============================================================================

import re
from typing import Dict, List, Tuple

# The source's openai_completion token budget, shared by all three
# generation stages.
DSG_GENERATION_MAX_TOKENS = 500

# The official in-context prompts (preamble + the 23 TIFA160 train examples),
# verbatim up to the test input; a prompt is the prefix plus the test input
# plus TEST_OUTPUT_CUE. Stage inputs chain exactly as in the source's
# ``generate_dsg``: the tuple stage reads the caption alone, the question and
# dependency stages read the caption plus the tuple stage's generated lines.
TEST_OUTPUT_CUE = "\noutput: "

TUPLE_PROMPT_PREFIX = """Task: given input prompts, describe each scene with skill-specific tuples.
Do not generate same tuples again. Do not generate tuples that are not explicitly described in the prompts.
output format: id | tuple

input: A male skateboarder is trying to pull off a trick on the ramp.
output: 1 | entity - whole (skateboarder)
2 | entity - whole (ramp)
3 | attribute - type (skateboarder, male)
4 | action - (skateboarder, pull off a trick)
5 | relation - spatial (skateboarder, ramp, on)

input: A car playing soccer, digital art.
output: 1 | entity - whole (car)
2 | global - (digital art)
3 | action - (car, soccer, play)

input: A set of 2x2 emoji icons with happy, angry, surprised and sobbing faces. The emoji icons look like pigs. All of the pigs are wearing crowns.
output: 1 | entity - whole (emoji icons)
2 | other - count (emoji icons, ==4)
3 | attribute - state (emoji icons, 2x2 grid)
4 | attribute - type (emoji icons, pig)
5 | attribute - state (emoji_1, happy)
6 | attribute - state (emoji_2, angry)
7 | attribute - state (emoji_3, surprised)
8 | attribute - state (emoji_4, sobbing face)
9 | entity - part (pig's crown)

input: a photo of bear and dining table; dining table is below bear
output: 1 | global - (photo)
2 | entity - whole (bear)
3 | entity - whole (dining table)
4 | relation - spatial (dining table, bear, below)

input: A group of children sitting in the grass with two of them holding a Frisbee .
output: 1 | entity - whole (children)
2 | entity - whole (grass)
3 | entity - whole (frisbee)
4 | attribute - state (children, sit)
5 | relation - spatial (children, grass, in)
6 | entity - part (two of the children)
7 | action - (two of the children, frisbee, hold)

input: the word 'START' written in chalk on a sidewalk
output: 1 | entity - whole (word)
2 | entity - whole (sidewalk)
3 | other - text rendering (word, "START")
4 | attribute - texture (word, chalk)
5 | relation - spatial (word, sidewalk, on)

input: A pear, orange, and two bananas in a wooden bowl.
output: 1 | entity - whole (pear)
2 | entity - whole (orange)
3 | entity - whole (bananas)
4 | other - count (bananas, ==2)
5 | entity - whole (bowl)
6 | attribute - material (bowl, wood)
7 | relation - spatial (pear, bowl, in)
8 | relation - spatial (orange, bowl, in)
9 | relation - spatial (bananas, bowl, in)

input: Closeup picture of the front of a clean motorcycle.
output: 1 | entity - whole (motorcycle)
2 | global - (closeup)
3 | global - (picture)
4 | attribute - state (motorcycle, clean)
5 | entity - part (front of the motorcycle)

input: a sad man with green hair
output: 1 | entity - whole (man)
2 | entity - part (man's hair)
3 | attribute - state (man, sad)
4 | attribute - color (man's hair, green)

input: A commercial airplane with propellers flying through the air.
output: 1 | entity - whole (airplane)
2 | entity - part (airplane's propellers)
3 | action - (airplane, air, fly through)

input: A little boy grips a soccer ball in his arms surrounded by other youth soccer players.
output: 1 | entity - whole (boy)
2 | entity - whole (ball)
3 | entity - whole (soccer players)
4 | entity - part (boy's arms)
5 | entity - scale (boy, little)
6 | attribute - type (ball, soccer)
7 | attribute - state (soccer players, youth)
8 | relation - spatial (boy, ball, grip in his arms)
9 | relation - spatial (boy, soccer players, surrounded by)

input: A traffic light and a signpost at a crossroads intersection near a waterway.
output: 1 | entity - whole (traffic light)
2 | entity - whole (signpost)
3 | entity - whole (crossroads intersection)
4 | entity - whole (waterway)
5 | relation - spatial (traffic light, crossroads intersection, at)
6 | relation - spatial (signpost, crossroads intersection, at)
7 | relation - spatial (traffic light, waterway, near)
8 | relation - spatial (signpost, waterway, near)
9 | relation - spatial (crossroads intersection, waterway, near)

input: a photo of dining table and traffic light; traffic light is below dining table
output: 1 | global - (photo)
2 | entity - whole (dining table)
3 | entity - whole (traffic light)
4 | relation - spatial (traffic light, dining table, below)

input: A realistic photo of a Pomeranian dressed up like a 1980s professional wrestler with neon green and neon orange face paint and bright green wrestling tights with bright orange boots.
output: 1 | global - (photo)
2 | entity - whole (Pomeranian)
3 | global - (realistic)
4 | entity - part (Pomeranian's costume)
5 | attribute - type (Pomeranian's costume, 1980s professional wrestler)
6 | entity - part (Pomeranian's costume's wrestling tights)
7 | entity - part (Pomeranian's costume's wrestling tights' boots)
8 | entity - part (Pomeranian's facepaint)
9 | attribute - color (Pomeranian's facepaint, neon green)
10 | attribute - color (Pomeranian's facepaint, neon orange)
11 | attribute - color (Pomeranian's costume's wrestling tights, bright green)
12 | attribute - color (Pomeranian's costume's wrestling tights' boots, bright orange)

input: a four-piece band on a stage in front of a small crowd
output: 1 | entity - whole (band)
2 | entity - whole (stage)
3 | entity - whole (crowd)
4 | other - count (band members, ==4)
5 | attribute - shape (crowd, small)
6 | relation - spatial (band, stage, on)
7 | relation - spatial (band, crowd, in front of)
8 | relation - spatial (stage, crowd, in front of)

input: two laptops, a mouse cord, and a monitor
output: 1 | entity - whole (laptops)
2 | other - count (laptops, ==2)
3 | entity - whole (mouse coord)
4 | entity - whole (monitor)

input: A red motorcycle parked by paint chipped doors.
output: 1 | entity - whole (motorcycle)
2 | entity - whole (doors)
3 | attribute - color (motorcycle, red)
4 | attribute - state (door, paint chipped)
5 | relation - spatial (motorcycle, door, next to)
6 | attribute - state (motorcycle, parked)

input: A cube made of denim. A cube with the texture of denim.
output: 1 | entity - whole (cube)
2 | attribute - material (cube, denim)
3 | attribute - texture (cube, denim)

input: an espresso machine that makes coffee from human souls
output: 1 | entity - whole (espresso machine)
2 | entity - whole (coffee)
3 | entity - whole (human souls)
4 | action - (espresso machine, coffee, make)
5 | attribute - material (coffee, human souls)

input: Three people standing next to an elephant along a river.
output: 1 | entity - whole (people)
2 | other - count (people, ==3)
3 | entity - whole (elephant)
4 | entity - whole (river)
5 | attribute - state (people, stand)
6 | relation - spatial (people, elephant, next to)
7 | relation - spatial (people, river, next to)
8 | relation - spatial (elephant, river, next to)

input: Aerial view of downtown Manhattan, but with Millennium Wheel next to the Statue of Liberty. The Great Pyramid is on a sandy island near the buildings.
output: 1 | entity - (downtown Manhattan)
2 | entity - (Millennium Wheel)
3 | entity - (the Statue of the Liberty)
4 | entity - (the Great Pyramid)
5 | entity - (island)
6 | entity - (buildings)
7 | global - (aerial view)
8 | attribute - texture (island, sandy)
9 | relation - spatial (Millennium Wheel, the Statue of Liberty, next to)
10 | relation - spatial (the Great Pyramid, island, on)
11 | relation - spatial (the Great Pyramid, buildings, near)

input: A laptop with external keyboard, mouse, phone and photo on a desk.
output: 1 | entity - whole (laptop)
2 | entity - whole (keyboard)
3 | entity - whole (mouse)
4 | entity - whole (phone)
5 | entity - whole (photo)
6 | entity - whole (desk)
7 | attribute - type (keyboard, external)
8 | relation - spatial (laptop, desk, on)
9 | relation - spatial (keyboard, desk, on)
10 | relation - spatial (mouse, desk, on)
11 | relation - spatial (phone, desk, on)
12 | relation - spatial (photo, desk, on)

input: A white slope covers the background, while the foreground features a grassy slope with several rams grazing and one measly and underdeveloped evergreen in the foreground.
output: 1 | entity - whole (slopes)
2 | other - count (slopes, ==2)
3 | entity - whole (rams)
4 | entity - whole (evergreen)
5 | attribute - color (slope_1, white)
6 | attribute - texture (slope_2, grassy)
7 | attribute - state (evergreen, measly and underdeveloped)
8 | relation - spatial (slope_1, background, in)
9 | relation - spatial (slope_2, foreground, in)
10 | relation - spatial (rams, slope_2, on)
11 | attribute - state (rams, graze)

input: """

QUESTION_PROMPT_PREFIX = """Task: given input prompts and skill-specific tuples, re-write tuple each in natural language question.
output format: id | question

input: A male skateboarder is trying to pull off a trick on the ramp. 
1 | entity - whole (skateboarder)
2 | entity - whole (ramp)
3 | attribute - type (skateboarder, male)
4 | action - (skateboarder, pull off a trick)
5 | relation - spatial (skateboarder, ramp, on)
output: 1 | Is there a skateboarder?
2 | Is there a ramp?
3 | Is the skateboarder male?
4 | Is the skateboarder pulling off a trick?
5 | Is the skateboarder on the ramp?

input: A car playing soccer, digital art.
1 | entity - whole (car)
2 | global - (digital art)
3 | action - (car, soccer, play)
output: 1 | Is there a car?
2 | Is this digital art?
3 | Is the car playing soccer?

input: A set of 2x2 emoji icons with happy, angry, surprised and sobbing faces. The emoji icons look like pigs. All of the pigs are wearing crowns.
1 | entity - whole (emoji icons)
2 | other - count (emoji icons, ==4)
3 | attribute - state (emoji icons, 2x2 grid)
4 | attribute - type (emoji icons, pig)
5 | attribute - state (emoji_1, happy)
6 | attribute - state (emoji_2, angry)
7 | attribute - state (emoji_3, surprised)
8 | attribute - state (emoji_4, sobbing face)
9 | entity - part (pig's crown)
output: 1 |
2 | Is there a total of four emoji icons?
3 | Are the emojis in a 2x2 grid?
4 | Do emojis look like pigs?
5 | Does one emoji look happy?
6 | Does one emoji look angry?
7 | Does one emoji look surprised?
8 | Does the emoji have a sobbing face?
9 | Are all the emoji wearing crowns?

input: a photo of bear and dining table; dining table is below bear
1 | global - (photo)
2 | entity - whole (bear)
3 | entity - whole (dining table)
4 | relation - spatial (dining table, bear, below)
output: 1 | Is this a photo?
2 | Is there a bear?
3 | Is there a dining table?
4 | Is the dining table below the bear?

input: A group of children sitting in the grass with two of them holding a Frisbee .
1 | entity - whole (children)
2 | entity - whole (grass)
3 | entity - whole (frisbee)
4 | attribute - state (children, sit)
5 | relation - spatial (children, grass, in)
6 | entity - part (two of the children)
7 | action - (two of the children, frisbee, hold)
output: 1 | Are there a group of children?
2 | Is there grass?
3 | Is there a frisbee?
4 | Are the children sitting?
5 | Are the children in the grass?
6 | Are there two of the children?
7 | Are two of the children holding a frisbee?

input: the word 'START' written in chalk on a sidewalk
1 | entity - whole (word)
2 | entity - whole (sidewalk)
3 | other - text rendering (word, "START")
4 | attribute - texture (word, chalk)
5 | relation - spatial (word, sidewalk, on)
output: 1 | Is there a word?
2 | Is there a sidewalk?
3 | Does the word say "START"?
4 | Is the word written in chalk?
5 | Is the word on the sidewalk?

input: A pear, orange, and two bananas in a wooden bowl.
1 | entity - whole (pear)
2 | entity - whole (orange)
3 | entity - whole (bananas)
4 | other - count (bananas, ==2)
5 | entity - whole (bowl)
6 | attribute - material (bowl, wood)
7 | relation - spatial (pear, bowl, in)
8 | relation - spatial (orange, bowl, in)
9 | relation - spatial (bananas, bowl, in)
output: 1 | Is there a pear?
2 | Is there an orange?
3 | Are there bananas?
4 | Are there two bananas?
5 | Is there a bowl?
6 | Is the bowl made of wood?
7 | Is the pear in the bowl?
8 | Is the orange in the bowl?
9 | Are the bananas in the bowl?

input: Closeup picture of the front of a clean motorcycle.
1 | entity - whole (motorcycle)
2 | global - (closeup)
3 | global - (picture)
4 | attribute - state (motorcycle, clean)
5 | entity - part (front of the motorcycle)
output: 1 | Is there a motorcycle?
2 | Is this a closeup image?
3 | Is this a picture?
4 | Is the motorcycle dirty?
5 | Is the picture in the front of the motorcycle?

input: a sad man with green hair
1 | entity - whole (man)
2 | entity - part (man's hair)
3 | attribute - state (man, sad)
4 | attribute - color (man's hair, green)
output: 1 | Is there a man?
2 | Is there hair?
3 | Is the man sad?
4 | Is the hair green?

input: A commercial airplane with propellers flying through the air.
1 | entity - whole (airplane)
2 | entity - part (airplane's propellers)
3 | action - (airplane, air, fly through)
output: 1 | Is there an airplane?
2 | Does the airplane have propellers?
3 | Is the airplane flying through the air?

input: A little boy grips a soccer ball in his arms surrounded by other youth soccer players.
1 | entity - whole (boy)
2 | entity - whole (ball)
3 | entity - whole (soccer players)
4 | entity - part (boy's arms)
5 | entity - scale (boy, little)
6 | attribute - type (ball, soccer)
7 | attribute - state (soccer players, youth)
8 | relation - spatial (boy, ball, grip in his arms)
9 | relation - spatial (boy, soccer players, surrounded by)
output: 1 | Is there a boy?
2 | Is there a ball?
3 | Are there other soccer players?
4 | Does the boy have arms?
5 | Is the boy little?
6 | Is the ball a soccer ball?
7 | Are the other soccer players young?
8 | Is the boy gripping the ball in his arms?
9 | Is the boy surrounded by the other soccer players?

input: A traffic light and a signpost at a crossroads intersection near a waterway.
1 | entity - whole (traffic light)
2 | entity - whole (signpost)
3 | entity - whole (crossroads intersection)
4 | entity - whole (waterway)
5 | relation - spatial (traffic light, crossroads intersection, at)
6 | relation - spatial (signpost, crossroads intersection, at)
7 | relation - spatial (traffic light, waterway, near)
8 | relation - spatial (signpost, waterway, near)
9 | relation - spatial (crossroads intersection, waterway, near)
output: 1 | Is there a light?
2 | Is there a signpost?
3 | Is there an intersection?
4 | Is there a waterway?
5 | Is the light a traffic light?
6 | Is the intersection a crossroads intersection?
7 | Is the traffic light at the crossroads intersection?
8 | Is the signpost at the crossroads intersection?
9 | Is the intersection near the waterway?

input: a photo of dining table and traffic light; traffic light is below dining table
1 | global - (photo)
2 | entity - whole (dining table)
3 | entity - whole (traffic light)
4 | relation - spatial (traffic light, dining table, below)
output: 1 | Is this a photo?
2 | Is there a dining table?
3 | Is there a traffic light?
4 | Is the traffice light below the dining table?

input: A realistic photo of a Pomeranian dressed up like a 1980s professional wrestler with neon green and neon orange face paint and bright green wrestling tights with bright orange boots.
1 | global - (photo)
2 | entity - whole (Pomeranian)
3 | global - (realistic)
4 | entity - part (Pomeranian's costume)
5 | attribute - type (Pomeranian's costume, 1980s professional wrestler)
6 | entity - part (Pomeranian's costume's wrestling tights)
7 | entity - part (Pomeranian's costume's wrestling tights' boots)
8 | entity - part (Pomeranian's facepaint)
9 | attribute - color (Pomeranian's facepaint, neon green)
10 | attribute - color (Pomeranian's facepaint, neon orange)
11 | attribute - color (Pomeranian's costume's wrestling tights, bright green)
12 | attribute - color (Pomeranian's costume's wrestling tights' boots, bright orange)
output: 1 | Is this a photo?
2 | Is there a Pomeranian?
3 | Is the photo realistic?
4 | Is the Pomeranian dressed up?
5 | Is the costume of a 1980s professional wrestler?
6 | Are wrestling tights included in the costume?
7 | Does the costume come with boots?
8 | Does the Pomeranian has a facepaint?
9 | Is the facepaint neon green?
10 | Is the facepaint neon orange?
11 | Are the wrestling tights bright green?
12 | Are the boots bright orange?

input: a four-piece band on a stage in front of a small crowd
1 | entity - whole (band)
2 | entity - whole (stage)
3 | entity - whole (crowd)
4 | other - count (band members, ==4)
5 | attribute - shape (crowd, small)
6 | relation - spatial (band, stage, on)
7 | relation - spatial (band, crowd, in front of)
8 | relation - spatial (stage, crowd, in front of)
output: 1 | Is there a band?
2 | Is there a stage?
3 | Is there a crowd?
4 | Is the band a fourpiece band?
5 | Is the crowd small?
6 | Are the band on the stage?
7 | Is the band in front of the crowd?
8 | Is the stage in front of the crowd?

input: two laptops, a mouse cord, and a monitor 
1 | entity - whole (laptops)
2 | other - count (laptops, ==2)
3 | entity - whole (mouse coord)
4 | entity - whole (monitor)
output: 1 | Are there laptops?
2 | Are there two laptops?
3 | Is there a cord?
4 | Is there a monitor?

input: A red motorcycle parked by paint chipped doors.
1 | entity - whole (motorcycle)
2 | entity - whole (doors)
3 | attribute - color (motorcycle, red)
4 | attribute - state (door, paint chipped)
5 | relation - spatial (motorcycle, door, next to)
6 | attribute - state (motorcycle, parked)
output: 1 | Is there a motorcycle?
2 | Are there any doors?
3 | Are the doors painted?
4 | Is the paint chipped?
5 | Is the motorcycle next to doors?
6 | Is the motorcycle parked?

input: A cube made of denim. A cube with the texture of denim.
1 | entity - whole (cube)
2 | attribute - material (cube, denim)
3 | attribute - texture (cube, denim)
output: 1 | Is there a cube?
2 | Is the cube made of denim?
3 | Does the cube have texture of denim?

input: an espresso machine that makes coffee from human souls
1 | entity - whole (espresso machine)
2 | entity - whole (coffee)
3 | entity - whole (human souls)
4 | action - (espresso machine, coffee, make)
5 | attribute - material (coffee, human souls)
output: 1 | Do we have an espresso machine?
2 | Do we have coffee?
3 | Do human beings have souls?
4 | Is the espresso machine making coffee?
5 | Is the expersso made of human souls?

input: Three people standing next to an elephant along a river.
1 | entity - whole (people)
2 | other - count (people, ==3)
3 | entity - whole (elephant)
4 | entity - whole (river)
5 | attribute - state (people, stand)
6 | relation - spatial (people, elephant, next to)
7 | relation - spatial (people, river, next to)
8 | relation - spatial (elephant, river, next to)
output: 1 | Are there people?
2 | Are there three people?
3 | Is there an elephant?
4 | Is there a river?
5 | Are the people standing?
6 | Are the people next to the elephant?
7 | Are the people next to the river?
8 | Is the elephant next to the river?

input: Aerial view of downtown Manhattan, but with Millennium Wheel next to the Statue of Liberty. The Great Pyramid is on a sandy island near the buildings.
1 | entity - (downtown Manhattan)
2 | entity - (Millennium Wheel)
3 | entity - (the Statue of the Liberty)
4 | entity - (the Great Pyramid)
5 | entity - (island)
6 | entity - (buildings)
7 | global - (aerial view)
8 | attribute - texture (island, sandy)
9 | relation - spatial (Millennium Wheel, the Statue of Liberty, next to)
10 | relation - spatial (the Great Pyramid, island, on)
11 | relation - spatial (the Great Pyramid, buildings, near)
output: 1 | Is downtown Manhattan there?
2 | Is Millennium Wheel there?
3 | Is the Statue of Liberty there?
4 | Is the Great Pyramid there?
5 | Is there an island?
6 | Are there buildings?
7 | Is this an aerial view?
8 | Is the island sandy?
9 | Is the Millennium Wheel next to the Statue of Liberty?
10 | Is the Great Pyramid on the sandy island?
11 | Is the Great Pyramid near the buildings?

input: A laptop with external keyboard, mouse, phone and photo on a desk.
1 | entity - whole (laptop)
2 | entity - whole (keyboard)
3 | entity - whole (mouse)
4 | entity - whole (phone)
5 | entity - whole (photo)
6 | entity - whole (desk)
7 | attribute - type (keyboard, external)
8 | relation - spatial (laptop, desk, on)
9 | relation - spatial (keyboard, desk, on)
10 | relation - spatial (mouse, desk, on)
11 | relation - spatial (phone, desk, on)
12 | relation - spatial (photo, desk, on)
output: 1 | Is there a laptop?
2 | Is there a keyboard?
3 | Is there a mouse?
4 | Is there a phone?
5 | Is there a photo?
6 | Is there a desk?
7 | Is the keyboard external?
8 | Is the laptop on the desk?
9 | Is the keyboard on the desk?
10 | Is the mouse on the desk?
11 | Is the phone on the desk?
12 | Is the photo on the desk?

input: A white slope covers the background, while the foreground features a grassy slope with several rams grazing and one measly and underdeveloped evergreen in the foreground.  
1 | entity - whole (slopes)
2 | other - count (slopes, ==2)
3 | entity - whole (rams)
4 | entity - whole (evergreen)
5 | attribute - color (slope_1, white)
6 | attribute - texture (slope_2, grassy)
7 | attribute - state (evergreen, measly and underdeveloped)
8 | relation - spatial (slope_1, background, in)
9 | relation - spatial (slope_2, foreground, in)
10 | relation - spatial (rams, slope_2, on)
11 | attribute - state (rams, graze)
output: 1 | Are there slopes?
2 | Are there two slopes?
3 | Are there rams?
4 | Is there evergreen?
5 | Is one slope white?
6 | Is one slope grassy?
7 | Is the evergreen measly and underdeveloped?
8 | Is the white slope in the background?
9 | Is the grassy slope in the foreground?
10 | Are the rams on the grassy slope?
11 | Are the rams grazing on grass?

input: """

DEPENDENCY_PROMPT_PREFIX = """Task: given input prompts and tuples, describe the parent tuples of each tuple.
output format: id | dependencies (comma separated)

input: A male skateboarder is trying to pull off a trick on the ramp. 
1 | entity - whole (skateboarder)
2 | entity - whole (ramp)
3 | attribute - type (skateboarder, male)
4 | action - (skateboarder, pull off a trick)
5 | relation - spatial (skateboarder, ramp, on)
output: 1 | 0
2 | 0
3 | 1
4 | 1
5 | 1,2

input: A car playing soccer, digital art.
1 | entity - whole (car)
2 | global - (digital art)
3 | action - (car, soccer, play)
output: 1 | 0
2 | 0
3 | 1

input: A set of 2x2 emoji icons with happy, angry, surprised and sobbing faces. The emoji icons look like pigs. All of the pigs are wearing crowns.
1 | entity - whole (emoji icons)
2 | other - count (emoji icons, ==4)
3 | attribute - state (emoji icons, 2x2 grid)
4 | attribute - type (emoji icons, pig)
5 | attribute - state (emoji_1, happy)
6 | attribute - state (emoji_2, angry)
7 | attribute - state (emoji_3, surprised)
8 | attribute - state (emoji_4, sobbing face)
9 | entity - part (pig's crown)
output: 1 | 0
2 | 1
3 | 1
4 | 1
5 | 1,2
6 | 1,2
7 | 1,2
8 | 1,2
9 | 1,4

input: a photo of bear and dining table; dining table is below bear
1 | global - (photo)
2 | entity - whole (bear)
3 | entity - whole (dining table)
4 | relation - spatial (dining table, bear, below)
output: 1 | 0
2 | 0
3 | 0
4 | 2,3

input: A group of children sitting in the grass with two of them holding a Frisbee .
1 | entity - whole (children)
2 | entity - whole (grass)
3 | entity - whole (frisbee)
4 | attribute - state (children, sit)
5 | relation - spatial (children, grass, in)
6 | entity - part (two of the children)
7 | action - (two of the children, frisbee, hold)
output: 1 | 0
2 | 0
3 | 0
4 | 1
5 | 1,2
6 | 1
7 | 3,6

input: the word 'START' written in chalk on a sidewalk
1 | entity - whole (word)
2 | entity - whole (sidewalk)
3 | other - text rendering (word, "START")
4 | attribute - texture (word, chalk)
5 | relation - spatial (word, sidewalk, on)
output: 1 | 0
2 | 0
3 | 1
4 | 1
5 | 1,2

input: A pear, orange, and two bananas in a wooden bowl.
1 | entity - whole (pear)
2 | entity - whole (orange)
3 | entity - whole (bananas)
4 | other - count (bananas, ==2)
5 | entity - whole (bowl)
6 | attribute - material (bowl, wood)
7 | relation - spatial (pear, bowl, in)
8 | relation - spatial (orange, bowl, in)
9 | relation - spatial (bananas, bowl, in)
output: 1 | 0
2 | 0
3 | 0
4 | 0
5 | 0
6 | 5
7 | 1,5
8 | 2,5
9 | 3,5

input: Closeup picture of the front of a clean motorcycle.
1 | entity - whole (motorcycle)
2 | global - (closeup)
3 | global - (picture)
4 | attribute - state (motorcycle, clean)
5 | entity - part (front of the motorcycle)
output: 1 | 0
2 | 0
3 | 0
4 | 1
5 | 1

input: a sad man with green hair
1 | entity - whole (man)
2 | entity - part (man's hair)
3 | attribute - state (man, sad)
4 | attribute - color (man's hair, green)
output: 1 | 0
2 | 1
3 | 1
4 | 2

input: A commercial airplane with propellers flying through the air.
1 | entity - whole (airplane)
2 | entity - part (airplane's propellers)
3 | action - (airplane, air, fly through)
output: 1 | 0
2 | 1
3 | 1

input: A little boy grips a soccer ball in his arms surrounded by other youth soccer players.
1 | entity - whole (boy)
2 | entity - whole (ball)
3 | entity - whole (soccer players)
4 | entity - part (boy's arms)
5 | entity - scale (boy, little)
6 | attribute - type (ball, soccer)
7 | attribute - state (soccer players, youth)
8 | relation - spatial (boy, ball, grip in his arms)
9 | relation - spatial (boy, soccer players, surrounded by)
output: 1 | 0
2 | 0
3 | 0
4 | 1
5 | 0
6 | 2
7 | 3
8 | 1,2, 4
9 | 1,3

input: A traffic light and a signpost at a crossroads intersection near a waterway.
1 | entity - whole (traffic light)
2 | entity - whole (signpost)
3 | entity - whole (crossroads intersection)
4 | entity - whole (waterway)
5 | relation - spatial (traffic light, crossroads intersection, at)
6 | relation - spatial (signpost, crossroads intersection, at)
7 | relation - spatial (traffic light, waterway, near)
8 | relation - spatial (signpost, waterway, near)
9 | relation - spatial (crossroads intersection, waterway, near)
output: 1 | 0
2 | 0
3 | 0
4 | 0
5 | 1,3
6 | 2,3
7 | 1,4
8 | 2,4
9 | 3,4

input: a photo of dining table and traffic light; traffic light is below dining table
1 | global - (photo)
2 | entity - whole (dining table)
3 | entity - whole (traffic light)
4 | relation - spatial (traffic light, dining table, below)
output: 1 | 0
2 | 0
3 | 0
4 | 2,3

input: A realistic photo of a Pomeranian dressed up like a 1980s professional wrestler with neon green and neon orange face paint and bright green wrestling tights with bright orange boots.
1 | global - (photo)
2 | entity - whole (Pomeranian)
3 | global - (realistic)
4 | entity - part (Pomeranian's costume)
5 | attribute - type (Pomeranian's costume, 1980s professional wrestler)
6 | entity - part (Pomeranian's costume's wrestling tights)
7 | entity - part (Pomeranian's costume's wrestling tights' boots)
8 | entity - part (Pomeranian's facepaint)
9 | attribute - color (Pomeranian's facepaint, neon green)
10 | attribute - color (Pomeranian's facepaint, neon orange)
11 | attribute - color (Pomeranian's costume's wrestling tights, bright green)
12 | attribute - color (Pomeranian's costume's wrestling tights' boots, bright orange)
output: 1 | 0
2 | 0
3 | 0
4 | 2
5 | 4
6 | 4
7 | 4
8 | 2
9 | 8
10 | 8
11 | 6
12 | 7

input: a four-piece band on a stage in front of a small crowd
1 | entity - whole (band)
2 | entity - whole (stage)
3 | entity - whole (crowd)
4 | other - count (band members, ==4)
5 | attribute - shape (crowd, small)
6 | relation - spatial (band, stage, on)
7 | relation - spatial (band, crowd, in front of)
8 | relation - spatial (stage, crowd, in front of)
output: 1 | 0
2 | 0
3 | 0
4 | 1
5 | 3
6 | 1,2
7 | 1,3
8 | 2,3

input: two laptops, a mouse cord, and a monitor 
1 | entity - whole (laptops)
2 | other - count (laptops, ==2)
3 | entity - whole (mouse coord)
4 | entity - whole (monitor)
output: 1 | 0
2 | 1
3 | 0
4 | 0

input: A red motorcycle parked by paint chipped doors.
1 | entity - whole (motorcycle)
2 | entity - whole (doors)
3 | attribute - color (motorcycle, red)
4 | attribute - state (door, paint chipped)
5 | relation - spatial (motorcycle, door, next to)
6 | attribute - state (motorcycle, parked)
output: 1 | 0
2 | 0
3 | 1
4 | 2
5 | 1,2
6 | 1

input: A cube made of denim. A cube with the texture of denim.
1 | entity - whole (cube)
2 | attribute - material (cube, denim)
3 | attribute - texture (cube, denim)
output: 1 | 0
2 | 1
3 | 1

input: an espresso machine that makes coffee from human souls
1 | entity - whole (espresso machine)
2 | entity - whole (coffee)
3 | entity - whole (human souls)
4 | action - (espresso machine, coffee, make)
5 | attribute - material (coffee, human souls)
output: 1 | 0
2 | 0
3 | 0
4 | 1,2
5 | 2,3

input: Three people standing next to an elephant along a river.
1 | entity - whole (people)
2 | other - count (people, ==3)
3 | entity - whole (elephant)
4 | entity - whole (river)
5 | attribute - state (people, stand)
6 | relation - spatial (people, elephant, next to)
7 | relation - spatial (people, river, next to)
8 | relation - spatial (elephant, river, next to)
output: 1 | 0
2 | 1
3 | 0
4 | 0
5 | 1
6 | 1,3
7 | 1,4
8 | 3,4

input: Aerial view of downtown Manhattan, but with Millennium Wheel next to the Statue of Liberty. The Great Pyramid is on a sandy island near the buildings.
1 | entity - (downtown Manhattan)
2 | entity - (Millennium Wheel)
3 | entity - (the Statue of the Liberty)
4 | entity - (the Great Pyramid)
5 | entity - (island)
6 | entity - (buildings)
7 | global - (aerial view)
8 | attribute - texture (island, sandy)
9 | relation - spatial (Millennium Wheel, the Statue of Liberty, next to)
10 | relation - spatial (the Great Pyramid, island, on)
11 | relation - spatial (the Great Pyramid, buildings, near)
output: 1 | 0
2 | 0
3 | 0
4 | 0
5 | 0
6 | 0
7 | 0
8 | 5
9 | 2,3
10 | 4,5
11 | 4,6

input: A laptop with external keyboard, mouse, phone and photo on a desk.
1 | entity - whole (laptop)
2 | entity - whole (keyboard)
3 | entity - whole (mouse)
4 | entity - whole (phone)
5 | entity - whole (photo)
6 | entity - whole (desk)
7 | attribute - type (keyboard, external)
8 | relation - spatial (laptop, desk, on)
9 | relation - spatial (keyboard, desk, on)
10 | relation - spatial (mouse, desk, on)
11 | relation - spatial (phone, desk, on)
12 | relation - spatial (photo, desk, on)
output: 1 | 0
2 | 0
3 | 0
4 | 0
5 | 0
6 | 0
7 | 2
8 | 1,6
9 | 2,6
10 | 3,6
11 | 4,6
12 | 5,6

input: A white slope covers the background, while the foreground features a grassy slope with several rams grazing and one measly and underdeveloped evergreen in the foreground.  
1 | entity - whole (slopes)
2 | other - count (slopes, ==2)
3 | entity - whole (rams)
4 | entity - whole (evergreen)
5 | attribute - color (slope_1, white)
6 | attribute - texture (slope_2, grassy)
7 | attribute - state (evergreen, measly and underdeveloped)
8 | relation - spatial (slope_1, background, in)
9 | relation - spatial (slope_2, foreground, in)
10 | relation - spatial (rams, slope_2, on)
11 | attribute - state (rams, graze)
output: 1 | 0
2 | 1
3 | 0
4 | 0
5 | 1
6 | 1
7 | 5
8 | 1
9 | 1
10 | 1,3
11 | 3

input: """


def build_tuple_prompt(caption: str) -> str:
    return TUPLE_PROMPT_PREFIX + caption + TEST_OUTPUT_CUE


def build_question_prompt(caption: str, tuple_body: str) -> str:
    """``tuple_body`` is the tuple stage's generated lines, cleaned by
    :func:`strip_generation_echoes` (the source feeds the raw completion cut at
    "input:"; a chat judge may also echo "output:", cut for the same reason)."""
    return QUESTION_PROMPT_PREFIX + caption + "\n" + tuple_body + TEST_OUTPUT_CUE


def build_dependency_prompt(caption: str, tuple_body: str) -> str:
    return DEPENDENCY_PROMPT_PREFIX + caption + "\n" + tuple_body + TEST_OUTPUT_CUE


def strip_generation_echoes(response: str) -> str:
    """The bare "id | value" lines of a generation-stage reply.

    A continued "input:" example is the model hallucinating the next in-context
    block (the source's ``parse_with_input_name``); an "output:" echo is it
    restating the cue it was meant to continue (the source's parsers make the
    same cut). Everything before the echo and after the continuation goes."""
    text = str(response or "").split("input:")[0]
    if "output:" in text:
        text = text[text.index("output:") + len("output:"):]
    return text.strip()


def _iter_id_value_lines(response: str):
    """``(id, value)`` per well-formed "id | value" line. The source raises on a pipeless line,
    two pipes or a non-numeric id; a chat model wraps the lines in prose, so these are skipped."""
    for line in strip_generation_echoes(response).split("\n"):
        parts = line.split("|")
        if len(parts) != 2:
            continue
        entry_id, value = parts[0].strip(), parts[1].strip()
        if entry_id.isnumeric():
            yield int(entry_id), value


def parse_tuple_response(response: str) -> Dict[int, str]:
    """Id -> skill category, e.g. ``{1: "entity - whole"}`` (the source's
    ``parse_tuple_output``: ``clean_tuple_str`` keeps the text before the
    parenthesized arguments)."""
    return {entry_id: value.split("(")[0].strip()
            for entry_id, value in _iter_id_value_lines(response)}


def parse_question_response(response: str) -> Dict[int, str]:
    """Id -> natural-language question (the source's ``parse_question_output``)."""
    return dict(_iter_id_value_lines(response))


def parse_dependency_response(response: str) -> Dict[int, List[int]]:
    """Id -> parent ids (the source's ``parse_dependency_output``): comma-split,
    non-numeric entries dropped, a lone "0" means no parents and is dropped
    when other parents are present. An entry left without numeric ids parses
    as ``[0]`` instead of the source's crash on ``int('-')``."""
    id2deps = {}
    for entry_id, value in _iter_id_value_lines(response):
        parent_ids = [p.strip() for p in value.split(",")]
        parent_ids = [p for p in parent_ids if p.isnumeric()]
        if len(parent_ids) > 1:
            parent_ids = [p for p in parent_ids if p != "0"]
        id2deps[entry_id] = [int(p) for p in parent_ids] or [0]
    return id2deps


def build_binary_question(question: str) -> str:
    """The chat-model VQA prompt (the source's ``GPT4o.vqa`` template; its
    mPLUG/InstructBLIP models instead take the bare question)."""
    return (f"Answer only with 'yes' or 'no'. Do not give other outputs "
            f"or punctuation marks. Question: {question}")


def normalize_binary_answer(answer: str) -> str:
    """The source's reply normalization (``GPT4o.vqa``): lowercase, strip,
    drop the punctuation marks the prompt forbade."""
    answer = str(answer or "").lower().strip()
    for mark in (".", ",", "?", "!"):
        answer = answer.replace(mark, "")
    return answer.strip()


# Decoration a chat judge adds around a one-word verdict even though the prompt
# forbids it: markdown emphasis, bullets, quotes. Stripped, not matched, so the
# verdict itself stays the source's bare token.
_VERDICT_WRAP = " \t*_`\"'‘’“”-•>"
# The colon is optional ("answer is no"); \b keeps "answers vary" intact.
_VERDICT_LABEL = re.compile(r"^(?:the\s+)?answer\b(?:\s+is)?\s*[:\-–]?\s*")


def binary_verdict(free_form_answer: str) -> str:
    """The yes/no verdict of a chat reply ("" if empty): the first word left after dropping
    wrappers and an "Answer:" label. The source's literal ``== "yes"`` would score "**Yes**",
    "Answer: yes" or "yes the shirt is black" 0."""
    normalized = _VERDICT_LABEL.sub("", normalize_binary_answer(free_form_answer).strip(_VERDICT_WRAP))
    words = normalized.strip(_VERDICT_WRAP).split()
    return words[0].strip(_VERDICT_WRAP) if words else ""


def dependency_filtered_scores(
    qid2scores: Dict[int, float], qid2dependency: Dict[int, List[int]],
) -> Tuple[Dict[int, float], Dict[int, bool]]:
    """Zero out the questions whose parent was answered "no" (the source's
    ``calc_vqa_score`` / ``evaluate_image_dsg`` step 3). Parent id 0 is the
    no-parent sentinel. A question without a parsed dependency entry counts as
    a root (the source's fallback when no dependencies are given), and a
    parent id with no parsed question is ignored. Validity is judged against
    the UNFILTERED parent scores, as in the source: a zeroed-out parent that
    itself answered "yes" does not invalidate its children."""
    filtered: Dict[int, float] = {}
    validity: Dict[int, bool] = {}
    for qid, question_score in qid2scores.items():
        parent_ids = qid2dependency.get(qid, [0])
        any_parent_answered_no = any(
            parent_id != 0 and qid2scores.get(parent_id) == 0 for parent_id in parent_ids
        )
        filtered[qid] = 0.0 if any_parent_answered_no else question_score
        validity[qid] = not any_parent_answered_no
    return filtered, validity
