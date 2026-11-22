"""Task F step R1: sanity checks on the RQ2 scenario set data/rq2_eval/ (facts.csv, queries.csv).
No embeddings are computed here; these checks only look at ids, labels and words (memory_manager.content_words)."""
import csv
import re
from collections import Counter
from pathlib import Path

import pytest

from memory_manager import content_words, jaccard, normalize_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RQ2_DIR = DATA_DIR / "rq2_eval"
D3_DIR = DATA_DIR / "memory_eval"

USERS = tuple(f"u{i:02d}" for i in range(1, 16))
FACTS_PER_USER = 5
CASE_TYPES = ("charged_target", "calm_target_charged_distractor", "calm_target")
INTENSITIES = ("calm", "mild", "strong")
TOPICS = ("study", "work", "family", "pet", "hobbies", "plans", "money", "friends")
MAX_FACT_WORDS = 25            # base.memory.max_fact_words
MAX_SHARED_WORDS = 1           # "mostly paraphrases": a query shares at most this many content words with its fact
DUPLICATE_OVERLAP = 0.8        # memory_manager.DUPLICATE_OVERLAP: store() would merge facts at or above this


def _read(path) -> list:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def facts():
    return _read(RQ2_DIR / "facts.csv")


@pytest.fixture(scope="module")
def queries():
    return _read(RQ2_DIR / "queries.csv")


@pytest.fixture(scope="module")
def fact_by_id(facts):
    return {f["fact_id"]: f for f in facts}


# -- facts --

def test_columns(facts, queries):
    assert list(facts[0]) == ["user_id", "fact_id", "fact", "intensity", "topic"]
    assert list(queries[0]) == ["user_id", "query_id", "query", "relevant_fact_id", "case_type", "distractor_fact_id"]


def test_fifteen_users_with_five_facts_each(facts):
    assert len(facts) == len(USERS) * FACTS_PER_USER == 75
    assert Counter(f["user_id"] for f in facts) == {u: FACTS_PER_USER for u in USERS}
    assert len({f["fact_id"] for f in facts}) == len(facts)
    for f in facts:
        assert f["fact_id"].startswith(f["user_id"] + "_"), f["fact_id"]


def test_facts_look_like_extractor_output(facts):
    for f in facts:
        text = f["fact"]
        assert text.startswith("The user "), text
        assert len(text.split()) <= MAX_FACT_WORDS, text
        assert not re.search(r"\d", text), text


def test_intensity_and_topic_values(facts):
    for f in facts:
        assert f["intensity"] in INTENSITIES, f
        assert f["topic"] in TOPICS, f
    assert {f["topic"] for f in facts} == set(TOPICS)          # every topic is used somewhere


def test_each_user_has_strong_and_calm_fact(facts):
    for u in USERS:
        levels = {f["intensity"] for f in facts if f["user_id"] == u}
        assert {"strong", "calm"} <= levels, u


def test_no_near_duplicate_facts_within_a_user(facts):
    for u in USERS:
        texts = [f["fact"] for f in facts if f["user_id"] == u]
        for i, a in enumerate(texts):
            for b in texts[i + 1:]:
                assert normalize_text(a) != normalize_text(b)
                assert jaccard(a, b) < DUPLICATE_OVERLAP, (a, b)


# -- queries --

def test_forty_five_queries_one_of_each_type_per_user(queries):
    assert len(queries) == 45
    assert len({q["query_id"] for q in queries}) == 45
    for u in USERS:
        types = sorted(q["case_type"] for q in queries if q["user_id"] == u)
        assert types == sorted(CASE_TYPES), u
    for q in queries:
        assert q["query_id"].startswith(q["user_id"] + "_"), q["query_id"]


def test_ids_exist_and_belong_to_same_user(queries, fact_by_id):
    for q in queries:
        rel = q["relevant_fact_id"]
        assert rel in fact_by_id, q
        assert fact_by_id[rel]["user_id"] == q["user_id"], q
        dis = q["distractor_fact_id"]
        if dis:
            assert dis in fact_by_id, q
            assert fact_by_id[dis]["user_id"] == q["user_id"], q
            assert dis != rel, q


def test_case_type_intensity_rules(queries, facts, fact_by_id):
    for q in queries:
        target = fact_by_id[q["relevant_fact_id"]]
        ct, dis = q["case_type"], q["distractor_fact_id"]
        if ct == "charged_target":
            assert target["intensity"] == "strong", q
            assert not dis, q
        elif ct == "calm_target_charged_distractor":
            assert target["intensity"] == "calm", q
            assert dis, q
            assert fact_by_id[dis]["intensity"] == "strong", q
            assert fact_by_id[dis]["topic"] == target["topic"], q          # "nearby topic" = same topic label
        else:                                                               # calm_target
            assert target["intensity"] == "calm", q
            assert not dis, q
            strong_topics = {f["topic"] for f in facts
                             if f["user_id"] == q["user_id"] and f["intensity"] == "strong"}
            assert target["topic"] not in strong_topics, q


def test_queries_are_mostly_paraphrases(queries, fact_by_id):
    for q in queries:
        shared = content_words(q["query"]) & content_words(fact_by_id[q["relevant_fact_id"]]["fact"])
        assert len(shared) <= MAX_SHARED_WORDS, (q["query_id"], shared)


# -- separation from the D3 set --

def test_no_sentence_overlaps_memory_eval(facts, queries):
    d3 = [r["fact"] for r in _read(D3_DIR / "facts.csv")] + [r["query"] for r in _read(D3_DIR / "queries.csv")]
    ours = [f["fact"] for f in facts] + [q["query"] for q in queries]
    d3_norm = {normalize_text(s) for s in d3}
    for s in ours:
        assert normalize_text(s) not in d3_norm, s
        for t in d3:
            assert jaccard(s, t) < DUPLICATE_OVERLAP, (s, t)
