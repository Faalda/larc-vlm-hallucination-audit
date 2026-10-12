"""Faithful adaptation of the CHAIR (Rohrbach et al. 2018) object-mention
extraction algorithm, using the real published synonym list
(data/chair_synonyms.txt, fetched from github.com/LisaAnne/Hallucination)
rather than a hand-curated approximation.

Differences from the reference chair.py: (1) we use NLTK's WordNetLemmatizer
(pos='n') in place of the `pattern.en.singularize` the original uses -- same
purpose (map plural nouns to singular before matching), different library, since
`pattern` is effectively unmaintained for modern Python; (2) we track character
offsets into the original caption for every match (the reference implementation
only needs word identity, not position), since downstream code needs to map a
mention back to the generation step that produced it.
"""
import os

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

_SYNONYM_PATH = os.path.join(os.path.dirname(__file__), "data", "chair_synonyms.txt")

_lemmatizer = WordNetLemmatizer()
_tokenizer = TreebankWordTokenizer()


def _load_synonyms(path):
    synonym_dict = {}
    inverse = {}
    with open(path) as f:
        for line in f:
            words = [w.strip() for w in line.strip().split(",") if w.strip()]
            if not words:
                continue
            canonical = words[0]
            synonym_dict[canonical] = words
            for w in words:
                inverse[w] = canonical
    return synonym_dict, inverse


SYNONYM_DICT, INVERSE_SYNONYM_DICT = _load_synonyms(_SYNONYM_PATH)
MSCOCO_OBJECTS = set(INVERSE_SYNONYM_DICT.keys())

# Exact rules from the reference chair.py (github.com/LisaAnne/Hallucination).
_COCO_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light",
    "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
    "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
    "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear",
    "hair drier", "potted plant", "bow tie", "laptop computer", "stove top oven",
    "home plate", "train track",
]
_ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                  "bear", "zebra", "giraffe", "animal", "cub"]
_VEHICLE_WORDS = ["jet", "train"]

DOUBLE_WORD_DICT = {w: w for w in _COCO_DOUBLE_WORDS}
for _a in _ANIMAL_WORDS:
    DOUBLE_WORD_DICT[f"baby {_a}"] = _a
    DOUBLE_WORD_DICT[f"adult {_a}"] = _a
for _v in _VEHICLE_WORDS:
    DOUBLE_WORD_DICT[f"passenger {_v}"] = _v
DOUBLE_WORD_DICT["bow tie"] = "tie"
DOUBLE_WORD_DICT["toilet seat"] = "toilet"
DOUBLE_WORD_DICT["wine glas"] = "wine glass"


def _singularize(word):
    return _lemmatizer.lemmatize(word, pos="n")


def find_object_mentions(caption):
    """Return non-overlapping (category, char_start, char_end, matched_text)
    tuples, following CHAIR's tokenize -> singularize -> merge-double-words ->
    match-against-synonyms pipeline, with character offsets preserved.
    """
    spans = list(_tokenizer.span_tokenize(caption))
    tokens = [caption[s:e].lower() for s, e in spans]
    singular = [_singularize(t) for t in tokens]

    merged = []  # list of (word, span_start, span_end)
    i = 0
    while i < len(singular):
        if i + 1 < len(singular):
            bigram = f"{singular[i]} {singular[i + 1]}"
            if bigram in DOUBLE_WORD_DICT:
                merged.append((DOUBLE_WORD_DICT[bigram], spans[i][0], spans[i + 1][1]))
                i += 2
                continue
        merged.append((singular[i], spans[i][0], spans[i][1]))
        i += 1

    words_present = {w for w, _, _ in merged}
    has_toilet = "toilet" in words_present

    mentions = []
    for word, s, e in merged:
        if word == "seat" and has_toilet:
            continue
        if word not in MSCOCO_OBJECTS:
            continue
        category = INVERSE_SYNONYM_DICT[word]
        mentions.append((category, s, e, caption[s:e]))
    return mentions
