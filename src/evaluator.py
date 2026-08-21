"""
Stage 2 - the evaluator.

Written before any model exists to be evaluated. Getting this wrong in
either direction silently corrupts every number downstream: too lenient and
wrong answers score as correct, too strict and correct answers are thrown
away. Both failures are invisible in the output.

Two decoding regimes, because they measure different things:

  free-form    the model writes whatever it wants; we parse it. Measures
               classification skill AND output-format compliance, mixed.
  constrained  the model scores all 77 labels and we take the argmax; an
               invalid output is impossible. Measures classification only.

Reporting both, and the gap between them, turns a hidden methodology choice
into a stated measurement: how much of a fine-tuning gain was the model
learning to format rather than learning to classify.

Verdicts are three-way, never boolean:

  correct       predicted label equals gold
  wrong_label   predicted a real, different label
  unparseable   no single valid label could be recovered

`unparseable` is not a kind of wrong answer. Collapsing it into `wrong_label`
hides the format story completely, which is the thing we set out to measure.

Match modes are recorded per row rather than baked into a score, so the
sensitivity of any number to the parser's generosity is derivable later
without rerunning anything - same principle as storing near-duplicate
evidence instead of a flag in stage 1.

  exact       output was the canonical label string, nothing else
  normalized  matched after case/separator normalisation ("Card Arrival")
  extracted   a label was found inside surrounding text ("The intent is X.")
  ambiguous   two or more distinct labels found; refused
  none        no valid label present
"""

import re
from collections import Counter

VERDICTS = ("correct", "wrong_label", "unparseable")
STRICT_MODES = ("exact", "normalized")
LENIENT_MODES = ("exact", "normalized", "extracted")


def normalize(s):
    """Lowercase, collapse every non-alphanumeric run to a single underscore.

    'Card Arrival' -> 'card_arrival'
    'The intent is card_arrival.' -> 'the_intent_is_card_arrival'
    """
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _resolve_overlaps(found):
    """Drop any matched label that is a substring of another matched label.

    Three Banking77 labels nest inside others (card_not_working inside
    virtual_card_not_working; exchange_rate inside two). Without this,
    an output of 'virtual_card_not_working' looks like two labels and
    would be refused as ambiguous.
    """
    return [a for a in found
            if not any(a != b and normalize(a) in normalize(b) for b in found)]


def parse_free_form(raw, labels, exclude=None):
    """Recover a single label from unconstrained model output.

    `exclude` is the query text. 5.5% of Banking77 queries contain a label
    as a substring ("I need to know your exchange rates" contains
    exchange_rate), so a model that echoes the question before answering
    would otherwise earn a free extraction from its own echo. Removing the
    query first blocks the verbatim case. A paraphrased echo still slips
    through, which is one reason match_mode is recorded per row and every
    number can be rescored refusing extraction entirely.

    Returns (predicted_label_or_None, match_mode).
    """
    if raw is None or not str(raw).strip():
        return None, "none"

    text = str(raw).strip()
    by_norm = {normalize(lbl): lbl for lbl in labels}

    # Exact: the canonical string and nothing else.
    if text in set(labels):
        return text, "exact"

    # Normalized: right label, cosmetic differences only.
    n = normalize(text)
    if n in by_norm:
        return by_norm[n], "normalized"

    # Extracted: a label sits inside surrounding text. Strip the echoed
    # query first so the model cannot be credited for repeating the input.
    haystack = n
    if exclude:
        echo = normalize(exclude)
        if echo:
            haystack = haystack.replace(echo, "_")
    found = [lbl for key, lbl in by_norm.items() if key and key in haystack]
    found = _resolve_overlaps(found)
    if len(found) == 1:
        return found[0], "extracted"
    if len(found) > 1:
        return None, "ambiguous"
    return None, "none"


def evaluate_free_form(raw, gold, labels, lenient=True, exclude=None):
    """Verdict for one free-form generation.

    lenient=True accepts `extracted` matches; lenient=False treats them as
    unparseable. Both are computable from the same stored record.
    `exclude` is the query text, removed before extraction so an echoed
    question cannot be scored as an answer.
    """
    pred, mode = parse_free_form(raw, labels, exclude=exclude)
    allowed = LENIENT_MODES if lenient else STRICT_MODES
    if pred is None or mode not in allowed:
        verdict = "unparseable"
        pred = None
    elif pred == gold:
        verdict = "correct"
    else:
        verdict = "wrong_label"
    return {"raw": raw, "pred": pred, "gold": gold,
            "match_mode": mode, "verdict": verdict}


def evaluate_constrained(pred, gold, labels):
    """Verdict for a constrained prediction.

    The prediction is an argmax over the label set, so it is always a valid
    label and `unparseable` cannot occur. A prediction outside the label set
    means the caller is broken, and that raises rather than scoring.
    """
    if pred not in set(labels):
        raise ValueError(f"constrained prediction {pred!r} is not a valid label")
    return {"raw": pred, "pred": pred, "gold": gold,
            "match_mode": "constrained",
            "verdict": "correct" if pred == gold else "wrong_label"}


def score(records, subset_ids=None):
    """Aggregate verdicts.

    subset_ids restricts scoring to a subset (the near-duplicate-free test
    subset, for instance). Records must carry an 'id' for that to work.
    """
    rows = records
    if subset_ids is not None:
        ids = set(subset_ids)
        rows = [r for r in records if r.get("id") in ids]
    n = len(rows)
    if n == 0:
        raise ValueError("no records to score")

    verdicts = Counter(r["verdict"] for r in rows)
    modes = Counter(r["match_mode"] for r in rows)
    for v in verdicts:
        if v not in VERDICTS:
            raise ValueError(f"unknown verdict {v!r}")

    return {
        "n": n,
        "accuracy": verdicts["correct"] / n,
        "wrong_label_rate": verdicts["wrong_label"] / n,
        "unparseable_rate": verdicts["unparseable"] / n,
        "verdicts": dict(verdicts),
        "match_modes": dict(modes),
    }


def rescore_strict(records, labels):
    """Recompute verdicts refusing `extracted` matches, from stored records.

    Exists so the parser's generosity can be measured after the fact rather
    than being a decision frozen at generation time.
    """
    return [evaluate_free_form(r["raw"], r["gold"], labels, lenient=False,
                               exclude=r.get("query"))
            | {k: r[k] for k in ("id", "query") if k in r}
            for r in records]
