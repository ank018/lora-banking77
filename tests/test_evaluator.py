"""Evaluator self-tests, in both directions.

An evaluator can fail two ways and neither is visible in its output:

  too lenient  wrong answers score as correct, inflating everything
  too strict   correct answers are discarded, deflating everything

So there are two suites here. The first proves the evaluator accepts answers
that are right but cosmetically odd. The second proves it refuses answers
that look right and are not. A suite that only tests one direction is how a
project ends up confidently reporting a number nobody can reproduce.

    pytest tests/test_evaluator.py -v
"""

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluator import (  # noqa: E402
    evaluate_constrained, evaluate_free_form, normalize, parse_free_form, score,
)

SPLITS = Path("eval/splits")


@pytest.fixture(scope="module")
def labels():
    p = SPLITS / "train_pool.jsonl"
    if not p.exists():
        pytest.skip("splits not built - run src/01b_build_dataset.py")
    with open(p, encoding="utf-8") as f:
        return sorted({json.loads(line)["label"] for line in f})


# ===================================================================
# Direction 1: it must ACCEPT right answers that look wrong
# ===================================================================

def test_accepts_canonical_label(labels):
    r = evaluate_free_form("card_arrival", "card_arrival", labels)
    assert r["verdict"] == "correct" and r["match_mode"] == "exact"


def test_accepts_title_case(labels):
    r = evaluate_free_form("Card Arrival", "card_arrival", labels)
    assert r["verdict"] == "correct" and r["match_mode"] == "normalized"


def test_accepts_spaces_instead_of_underscores(labels):
    r = evaluate_free_form("card arrival", "card_arrival", labels)
    assert r["verdict"] == "correct"


def test_accepts_hyphens(labels):
    r = evaluate_free_form("card-arrival", "card_arrival", labels)
    assert r["verdict"] == "correct"


def test_accepts_surrounding_whitespace_and_newlines(labels):
    r = evaluate_free_form("\n  card_arrival \n", "card_arrival", labels)
    assert r["verdict"] == "correct"


def test_accepts_trailing_punctuation(labels):
    r = evaluate_free_form("card_arrival.", "card_arrival", labels)
    assert r["verdict"] == "correct"


def test_accepts_quoted_output(labels):
    r = evaluate_free_form('"card_arrival"', "card_arrival", labels)
    assert r["verdict"] == "correct"


def test_accepts_label_embedded_in_a_sentence(labels):
    r = evaluate_free_form("The intent is card_arrival.", "card_arrival", labels)
    assert r["verdict"] == "correct" and r["match_mode"] == "extracted"


def test_accepts_the_one_capitalised_label(labels):
    """Refund_not_showing_up is the only capitalised label in Banking77.
    Case-sensitive matching would fail this class silently."""
    assert "Refund_not_showing_up" in labels
    r = evaluate_free_form("refund not showing up", "Refund_not_showing_up", labels)
    assert r["verdict"] == "correct"


def test_accepts_nested_label_without_calling_it_ambiguous(labels):
    """card_not_working is a substring of virtual_card_not_working.
    Longest match must win rather than the pair being refused."""
    r = evaluate_free_form("virtual_card_not_working",
                           "virtual_card_not_working", labels)
    assert r["verdict"] == "correct"


def test_accepts_nested_label_inside_a_sentence(labels):
    r = evaluate_free_form("This is virtual_card_not_working, I think.",
                           "virtual_card_not_working", labels)
    assert r["verdict"] == "correct" and r["match_mode"] == "extracted"


def test_accepts_shorter_nested_label_on_its_own(labels):
    r = evaluate_free_form("card_not_working", "card_not_working", labels)
    assert r["verdict"] == "correct"


# ===================================================================
# Direction 2: it must REFUSE wrong answers that look right
# ===================================================================

def test_refuses_invented_label(labels):
    r = evaluate_free_form("waiting_for_card", "card_arrival", labels)
    assert r["verdict"] == "unparseable" and r["pred"] is None


def test_refuses_two_labels_offered_at_once(labels):
    """A hedge is not an answer. Accepting the first would score a coin flip
    as a prediction."""
    r = evaluate_free_form("card_arrival or card_delivery_estimate",
                           "card_arrival", labels)
    assert r["verdict"] == "unparseable" and r["match_mode"] == "ambiguous"


def test_refuses_empty_output(labels):
    for raw in ("", "   ", None):
        assert evaluate_free_form(raw, "card_arrival", labels)["verdict"] == "unparseable"


def test_refuses_prose_with_no_label(labels):
    r = evaluate_free_form("I'm not sure what the customer means here.",
                           "card_arrival", labels)
    assert r["verdict"] == "unparseable"


def test_wrong_label_is_not_unparseable(labels):
    """A confident wrong answer and an unreadable one are different failures
    and must never be merged."""
    r = evaluate_free_form("card_delivery_estimate", "card_arrival", labels)
    assert r["verdict"] == "wrong_label" and r["pred"] == "card_delivery_estimate"


def test_unparseable_is_not_counted_as_correct(labels):
    r = evaluate_free_form("banana", "card_arrival", labels)
    assert r["verdict"] != "correct"


def test_gold_label_appearing_in_the_question_does_not_count(labels):
    """The model echoing the query must not be scored as an answer unless a
    label genuinely appears in it."""
    r = evaluate_free_form("Query: I am still waiting on my card?",
                           "card_arrival", labels)
    assert r["verdict"] == "unparseable"


def test_strict_mode_refuses_extraction(labels):
    lenient = evaluate_free_form("The intent is card_arrival.",
                                 "card_arrival", labels, lenient=True)
    strict = evaluate_free_form("The intent is card_arrival.",
                                "card_arrival", labels, lenient=False)
    assert lenient["verdict"] == "correct"
    assert strict["verdict"] == "unparseable"


# ===================================================================
# Constrained decoding
# ===================================================================

def test_constrained_cannot_be_unparseable(labels):
    r = evaluate_constrained("card_arrival", "card_arrival", labels)
    assert r["verdict"] == "correct" and r["match_mode"] == "constrained"


def test_constrained_wrong_answer_is_wrong_label(labels):
    r = evaluate_constrained("age_limit", "card_arrival", labels)
    assert r["verdict"] == "wrong_label"


def test_constrained_rejects_an_invalid_prediction(labels):
    """Impossible unless the caller is broken - so it raises rather than
    quietly scoring."""
    with pytest.raises(ValueError):
        evaluate_constrained("not_a_label", "card_arrival", labels)


# ===================================================================
# Aggregation
# ===================================================================

def test_score_counts_all_three_verdicts(labels):
    recs = [
        evaluate_free_form("card_arrival", "card_arrival", labels),
        evaluate_free_form("age_limit", "card_arrival", labels),
        evaluate_free_form("banana", "card_arrival", labels),
        evaluate_free_form("Card Arrival", "card_arrival", labels),
    ]
    s = score(recs)
    assert s["n"] == 4
    assert s["accuracy"] == 0.5
    assert s["wrong_label_rate"] == 0.25
    assert s["unparseable_rate"] == 0.25
    assert abs(s["accuracy"] + s["wrong_label_rate"]
               + s["unparseable_rate"] - 1.0) < 1e-12


def test_score_respects_a_subset(labels):
    recs = []
    for i, (raw, gold) in enumerate([("card_arrival", "card_arrival"),
                                     ("banana", "card_arrival")]):
        r = evaluate_free_form(raw, gold, labels)
        r["id"] = f"test-{i:05d}"
        recs.append(r)
    s = score(recs, subset_ids=["test-00000"])
    assert s["n"] == 1 and s["accuracy"] == 1.0


def test_score_refuses_an_empty_set(labels):
    with pytest.raises(ValueError):
        score([])


def test_normalize_is_idempotent(labels):
    for lbl in labels:
        assert normalize(normalize(lbl)) == normalize(lbl)


def test_every_label_round_trips_through_the_parser(labels):
    """Whole label set, both canonical and spaced forms. A single label that
    fails to parse is a silent one-class zero."""
    for lbl in labels:
        pred, mode = parse_free_form(lbl, labels)
        assert pred == lbl, f"{lbl} did not parse to itself ({mode})"
        spaced = lbl.replace("_", " ")
        pred2, _ = parse_free_form(spaced, labels)
        assert pred2 == lbl, f"{spaced!r} did not parse to {lbl}"


# ===================================================================
# Query echo
# ===================================================================

def test_echoed_query_does_not_earn_an_extraction(labels):
    """5.5% of Banking77 queries contain a label as a substring. A model
    repeating the question must not be credited for it."""
    q = "I need to know your exchange rates."
    assert parse_free_form(q, labels)[0] == "exchange_rate"      # the hazard
    r = evaluate_free_form(f"Query: {q} Answer:", "exchange_rate",
                           labels, exclude=q)
    assert r["verdict"] == "unparseable"


def test_real_answer_after_an_echo_still_parses(labels):
    q = "I need to know your exchange rates."
    r = evaluate_free_form(f"Query: {q} Answer: exchange_rate", "exchange_rate",
                           labels, exclude=q)
    assert r["verdict"] == "correct" and r["match_mode"] == "extracted"


def test_exclude_does_not_affect_a_bare_label(labels):
    r = evaluate_free_form("exchange_rate", "exchange_rate", labels,
                           exclude="I need to know your exchange rates.")
    assert r["verdict"] == "correct" and r["match_mode"] == "exact"
