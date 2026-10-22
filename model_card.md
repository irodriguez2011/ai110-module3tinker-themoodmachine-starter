# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
Describe whether you used the rule based model, the ML model, or both.  
Example: “I used the rule based model only” or “I compared both models.”

**Intended purpose:**  
What is this model trying to do?  
Example: classify short text messages as moods like positive, negative, neutral, or mixed.

**How it works (brief):**  
For the rule based version, describe the scoring rules you created.  
For the ML version, describe how training works at a high level (no math needed).



## 2. Data

**Dataset description:**  
Summarize how many posts are in `SAMPLE_POSTS` and how you added new ones.

**Labeling process:**  
Explain how you chose labels for your new examples.  
Mention any posts that were hard to label or could have multiple valid labels.

**Important characteristics of your dataset:**  
Examples you might include:  

- Contains slang or emojis  
- Includes sarcasm  
- Some posts express mixed feelings  
- Contains short or ambiguous messages

**Possible issues with the dataset:**  
Think about imbalance, ambiguity, or missing kinds of language.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
Describe the modeling choices you made.  
Examples:  

- How positive and negative words affect score  
- Negation rules you added  
- Weighted words  
- Emoji handling  
- Threshold decisions for labels

**Strengths of this approach:**  
Where does it behave predictably or reasonably well?

**Weaknesses of this approach:**  
Where does it fail?  
Examples: sarcasm, subtlety, mixed moods, unfamiliar slang.

## 4. How the ML Model Works (if used)

**Features used:**  
Describe the representation.  
Example: “Bag of words using CountVectorizer.”

**Training data:**  
State that the model trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

**Training behavior:**  
Did you observe changes in accuracy when you added more examples or changed labels?

**Strengths and weaknesses:**  
Strengths might include learning patterns automatically.  
Weaknesses might include overfitting to the training data or picking up spurious cues.

## 5. Evaluation

## 5. Evaluation

**How you evaluated the model:**  
I compared both models on the 26 labeled posts in `dataset.py`. The rule-based
model scored 88% (23/26) and the ML model (bag-of-words + logistic regression)
scored 92% (24/26). Both scores are training accuracy: each model was scored on
the same posts it was built from, with no held-out test set, so they overstate
how well either model would do on new text. Per class, the rule-based model's
weakest spot is recall on negative (0.82): it missed 2 of 11 negative posts,
both cases of disguised negativity. That matters most if this were used to
flag users who are upset.

**Examples of correct predictions:**  
- "I am not happy about this" -> negative (rule-based). The negation rule flips
  "happy" from positive to negative.
- "ngl this pizza is bussin" -> positive (rule-based). "bussin" is in the slang
  words I added to the positive list.
- "oh, great. It's raining on the one day i have somewhere to go" -> negative
  (rule-based). "oh great" is a sarcasm cue, so the score is flipped.

**Examples of incorrect predictions:**  
Rule-based model:
- "wow I just love waiting two hours in traffic" -> positive (true: negative).
  "love" scores positive, and there's no sarcasm cue phrase to flip it. The
  sarcasm comes from the situation, not the words.
- "thanks a lot, my phone died right before the call" -> neutral (true:
  negative). Neither "thanks" nor "died" is in the word lists, so the score is
  0 and it defaults to neutral.
- "My interview went poorly, but it was a learning experience" -> negative
  (true: mixed). "poorly" counts as negative, but "learning experience" isn't
  in the positive list, so the positive half doesn't register.

ML model:
- "🤣" -> negative (true: positive) and "😐" -> negative (true: neutral).
  CountVectorizer only keeps word tokens, so emoji-only posts become empty and
  the model falls back to the most common label (negative, 11 of 26 posts).
- It got both sarcasm posts right, but only because it was trained on those
  exact sentences. On new text it learned a spurious cue: "day" appears mostly
  in negative posts, so "what a great day" and "I love this day" are both
  predicted negative.

How the failures differ: the rule-based model handles emojis but misses
sarcasm without a cue phrase. The ML model can't see emojis at all and picks up
shortcuts from word frequency instead of meaning.

## 6. Limitations

Describe the most important limitations.  
Examples:  

- The dataset is small  
- The model does not generalize to longer posts  
- It cannot detect sarcasm reliably  
- It depends heavily on the words you chose or labeled

**Informal language and slang:** The model works best on clear, explicit
statements ("I love this class", "today was a terrible day") and worst on
slang and casual phrasing. "Neutral" is the default when no word matches, so a
neutral prediction often means "no recognized words" rather than "no
sentiment". I added a batch of slang words (`lit`, `bussin`, `cooked`, `mid`,
`sucks`, ...) and informal example posts, which helped, but a fixed list can
only cover words someone thought of. Some slang is also ambiguous (`fire`,
`sick`, `dead`), and intensifiers like "lowkey" and "no cap" carry no polarity
of their own, so they are ignored.

**Sarcasm (partial fix only):** `MoodAnalyzer` has a small list of sarcasm
cue phrases (`SARCASM_CUES` in `mood_analyzer.py`, e.g. "oh great",
"yeah right", "sure is"). When one appears, the score is flipped. This fixes
"oh, great. It's raining on the one day i have somewhere to go", but it only
works for cues someone thought to list. It still fails when the sarcasm
comes from the situation, not the words: "well, he's a jolly little fellow,
isn't he" said about a grumpy person reads as positive because nothing in the
sentence signals it. It can also flip sincere text, e.g. "that sure is nice".
Reliable sarcasm detection needs context beyond a single sentence.

## 7. Ethical Considerations

Discuss any potential impacts of using mood detection in real applications.  
Examples: 

- Misclassifying a message expressing distress  
- Misinterpreting mood for certain language communities  
- Privacy considerations if analyzing personal messages

## 8. Ideas for Improvement

List ways to improve either model.  
Possible directions:  

- Add more labeled data  
- Use TF IDF instead of CountVectorizer  
- Add better preprocessing for emojis or slang  
- Use a small neural network or transformer model  
- Improve the rule based scoring method  
- Add a real test set instead of training accuracy only
