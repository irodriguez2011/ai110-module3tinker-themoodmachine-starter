"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "hopeful",
    "proud",
    "grateful",
    # slang
    "lit",
    "fire",
    "goated",
    "bussin",
    "slay",
    "dope",
    "hyped",
    "stoked",
    "pumped",
    "vibe",
    "vibing",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "bored",
    "annoy",
    "annoyed",
    "annoying",
    "stressful",
    "poorly",
    "anxious",
    "nervous",
    # slang
    "mid",
    "cooked",
    "ugh",
    "lame",
    "trash",
    "sucks",
    "exhausted",
    "drained",
    "cringe",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "I love this food",
    "My day was okay",
    "My interview went poorly, but it was a learning experience",
    "I'm so bored",
    "I'm so angry I could cry",
    "Today was chill",
    "🤣",
    "That assignment was stressful, but I'm happy i got it out of the way",
    "😐",
    "He's lowkey starting to annoy me",
    "oh, great. It's raining on the one day i have somewhere to go",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",  # "Feeling tired but kind of hopeful"
    "neutral",  # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "positive",  # "I love this food"
    "neutral",  # "My day was okay"
    "mixed",  # "My interview went poorly, but it was a learning experience"
    "negative",  # "I'm so bored"
    "negative",  # "I'm so angry I could cry"
    "positive",  # "Today was chill"
    "positive",  # "🤣
    "mixed",  # "That assigment was stressful, but I'm happy i got it out of the way"
    "neutral",  # "😐"
    "negative",  # "He's lowkey starting to annoy me"
    "negative",  # oh, great. It's raining on the one day i have somewhere to go"
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)

# ---------------------------------------------------------------------
# Informal / slang / hard examples (added for the dataset TODO)
# ---------------------------------------------------------------------

_EXTRA_POSTS = [
    ("ngl this pizza is bussin", "positive"),
    ("lowkey cooked for this exam tomorrow", "negative"),
    ("that concert was lit, no cap", "positive"),
    ("ugh mondays, so mid", "negative"),
    ("meh, it is what it is", "neutral"),
    ("I'm so hyped but lowkey nervous", "mixed"),
    ("this day sucks \U0001F62D", "negative"),
    # Hard: sarcasm with no cue phrase the model knows.
    ("wow I just love waiting two hours in traffic", "negative"),
    # Hard: sarcasm cue fires, but there is no positive word to flip.
    ("thanks a lot, my phone died right before the call", "negative"),
]

for _post, _label in _EXTRA_POSTS:
    SAMPLE_POSTS.append(_post)
    TRUE_LABELS.append(_label)

assert len(SAMPLE_POSTS) == len(TRUE_LABELS)
