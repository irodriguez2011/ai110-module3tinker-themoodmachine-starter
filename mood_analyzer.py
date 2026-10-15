# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = (
            positive_words if positive_words is not None else POSITIVE_WORDS
        )
        negative_words = (
            negative_words if negative_words is not None else NEGATIVE_WORDS
        )

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          - Strips whitespace and converts to lowercase
          - Normalizes repeated letters ("soooo" -> "soo")
          - Removes punctuation, keeping apostrophes inside words ("i'm")
          - Keeps text emoticons (":)", ":-(") and emojis as their own tokens
        """
        cleaned = text.strip().lower()

        # Drop emoji variation selectors / joiners so they don't become tokens.
        cleaned = cleaned.replace("️", "").replace("‍", "")

        # "soooo" -> "soo": cap runs of the same word character at two.
        cleaned = re.sub(r"(\w)\1{2,}", r"\1\1", cleaned)

        # Alternatives, in priority order:
        #   1. text emoticons like :) :-( ;) :d
        #   2. a single non-ASCII symbol (emoji)
        #   3. a word, allowing an inner apostrophe
        token_pattern = r"[:;=]-?[()dp]|[^\x00-\x7f\w\s]|\w+(?:'\w+)*"
        return re.findall(token_pattern, cleaned)

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        Improvements over plain word matching:
          - Negation: a negator right before a sentiment word flips it
            ("not happy" -> -1, "not bad" -> +1)
          - Counting: every occurrence counts, not just presence
          - Emojis / emoticons: treated as sentiment signals
          - Sarcasm cues: a known sarcastic phrase ("oh great", "yeah right")
            flips the whole score. This is a partial fix only; see
            SARCASM_CUES below and the Limitations section of model_card.md.
        """
        positive_total, negative_total = self._signal_totals(text)
        return positive_total + negative_total

    def _signal_totals(self, text: str) -> Tuple[int, int]:
        """
        Return (positive_total, negative_total) for the text, where
        negative_total is <= 0. Negation can move a word to the other side,
        and a sarcasm cue swaps the two sides.
        """
        tokens = self.preprocess(text)
        positive_total = 0
        negative_total = 0

        for i, token in enumerate(tokens):
            token = self._canonical(token)
            if token in self.WORD_WEIGHTS:
                value = self.WORD_WEIGHTS[token]
            elif token in self.positive_words or token in self.POSITIVE_SYMBOLS:
                value = 1
            elif token in self.negative_words or token in self.NEGATIVE_SYMBOLS:
                value = -1
            else:
                continue

            if i > 0 and self._is_negator(tokens[i - 1]):
                value = -value

            if value > 0:
                positive_total += value
            else:
                negative_total += value

        if self._has_sarcasm_cue(tokens):
            positive_total, negative_total = -negative_total, -positive_total

        return positive_total, negative_total

    def _has_sarcasm_cue(self, tokens: List[str]) -> bool:
        padded = " " + " ".join(tokens) + " "
        return any(f" {cue} " in padded for cue in self.SARCASM_CUES)

    # Emoji / emoticon signals that are not in the word lists from dataset.py.
    POSITIVE_SYMBOLS = {
        ":)",
        ":-)",
        ":d",
        ":-d",
        ";)",
        ";-)",
        "😂",
        "🤣",
        "😊",
        "😀",
        "😍",
        "❤",
        "🥳",
    }
    NEGATIVE_SYMBOLS = {":(", ":-(", "😢", "😭", "😡", "😞", "😠", "💔"}
    NEGATORS = {"not", "no", "never", "cannot", "without"}

    # Phrases that usually signal sarcasm. Matched against the preprocessed
    # tokens (punctuation removed), so "Oh, great." matches "oh great".
    # Limitation: this is a fixed list. It misses sarcasm with no cue phrase
    # ("he's a jolly little fellow" about a grump) and wrongly flips sincere
    # uses of a cue ("that sure is nice").
    SARCASM_CUES = {
        "oh great",
        "oh wow great",
        "yeah right",
        "thanks a lot",
        "just what i needed",
        "sure is",
        "isn't he",
        "isn't she",
    }

    # Word weights: strength of a word's signal. Words not listed here fall
    # back to +1 / -1 based on the positive / negative word lists.
    WORD_WEIGHTS = {
        "love": 2,
        "amazing": 2,
        "hate": -2,
        "terrible": -2,
        "awful": -2,
        "kill": -3,
        "killing": -3,
        "killed": -3,
        "murder": -3,
        "die": -3,
    }

    def _is_known(self, token: str) -> bool:
        return (
            token in self.WORD_WEIGHTS
            or token in self.positive_words
            or token in self.negative_words
            or token in self.POSITIVE_SYMBOLS
            or token in self.NEGATIVE_SYMBOLS
        )

    def _canonical(self, token: str) -> str:
        """
        preprocess() caps repeated letters at two ("fireee" -> "firee"), which
        can still miss the real word. If the token is unknown, retry with
        every repeated run collapsed to one letter ("firee" -> "fire").
        """
        if self._is_known(token):
            return token
        collapsed = re.sub(r"(\w)\1+", r"\1", token)
        return collapsed if self._is_known(collapsed) else token

    def _is_negator(self, token: str) -> bool:
        return token in self.NEGATORS or token.endswith("n't")

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        Adjustment: if the text has BOTH positive and negative signals and
        they cancel out (score == 0), return "mixed" instead of "neutral".
        "neutral" is reserved for texts with no sentiment signal at all.
        """
        positive_total, negative_total = self._signal_totals(text)
        score = positive_total + negative_total

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        if positive_total > 0 and negative_total < 0:
            return "mixed"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
