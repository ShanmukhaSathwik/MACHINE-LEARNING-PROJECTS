"""Task D step D3a: sanity checks on the labelled pair set data/memory_eval/ (facts.csv, queries.csv).
No embeddings are computed here; these checks only look at ids, labels and words (memory_manager.content_words)."""
import csv
import random
from pathlib import Path

import pytest

from memory_manager import content_words

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "memory_eval"
CASE_TYPES = ("paraphrase", "direct", "word_trap", "none")
SPLITS = ("tune", "test")
SPLIT_SEED = 42
MAX_FACT_WORDS = 25           # base.memory.max_fact_words


def load_facts(eval_dir=EVAL_DIR) -> list:
    with open(eval_dir / "facts.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_queries(eval_dir=EVAL_DIR) -> list:
    with open(eval_dir / "queries.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def relevant_ids(query_row) -> list:
    return [x for x in query_row["relevant_fact_ids"].split(";") if x]


def assign_splits(query_rows, seed=SPLIT_SEED) -> dict:
    """{query_id: "tune" | "test"}. One random.Random(seed); for each case type in CASE_TYPES order, the query_ids
    of that type are sorted, shuffled with that rng, and the first len // 2 become "tune", the rest "test"."""
    rng = random.Random(seed)
    out = {}
    for ct in CASE_TYPES:
        ids = sorted(r["query_id"] for r in query_rows if r["case_type"] == ct)
        rng.shuffle(ids)
        for i, qid in enumerate(ids):
            out[qid] = "tune" if i < len(ids) // 2 else "test"
    return out


@pytest.fixture(scope="module")
def facts():
    return load_facts()


@pytest.fixture(scope="module")
def queries():
    return load_queries()


def test_columns(facts, queries):
    assert list(facts[0]) == ["fact_id", "fact"]
    assert list(queries[0]) == ["query_id", "query", "relevant_fact_ids", "case_type", "split"]


def test_ids_unique_and_texts_present(facts, queries):
    fids = [r["fact_id"] for r in facts]
    qids = [r["query_id"] for r in queries]
    assert len(fids) == len(set(fids)) and len(qids) == len(set(qids))
    assert all(r["fact"].strip() for r in facts) and all(q["query"].strip() for q in queries)
    assert len({r["fact"].strip().lower() for r in facts}) == len(facts)


def test_facts_look_like_extractor_output(facts):
    for r in facts:
        assert r["fact"].startswith("The user "), r
        assert len(r["fact"].split()) <= MAX_FACT_WORDS, r
        assert not any(ch.isdigit() for ch in r["fact"]), r            # extractor rule: no numbers


def test_every_relevant_fact_id_exists(facts, queries):
    fids = {r["fact_id"] for r in facts}
    for q in queries:
        rel = relevant_ids(q)
        assert len(rel) == len(set(rel)), q
        assert set(rel) <= fids, (q["query_id"], set(rel) - fids)


def test_case_type_and_split_values(queries):
    assert {q["case_type"] for q in queries} <= set(CASE_TYPES)
    assert {q["split"] for q in queries} <= set(SPLITS)


def test_split_is_the_seeded_balanced_assignment(queries):
    assert {q["query_id"]: q["split"] for q in queries} == assign_splits(queries)
    for ct in CASE_TYPES:
        n_tune = sum(q["split"] == "tune" for q in queries if q["case_type"] == ct)
        n_test = sum(q["split"] == "test" for q in queries if q["case_type"] == ct)
        assert abs(n_tune - n_test) <= 1, ct


def test_paraphrase_shares_no_content_word_with_its_relevant_facts(facts, queries):
    text = {r["fact_id"]: r["fact"] for r in facts}
    for q in queries:
        if q["case_type"] != "paraphrase":
            continue
        assert relevant_ids(q), q["query_id"]
        for fid in relevant_ids(q):
            shared = content_words(q["query"]) & content_words(text[fid])
            assert not shared, (q["query_id"], fid, shared)


def test_direct_has_a_relevant_fact(queries):
    assert all(relevant_ids(q) for q in queries if q["case_type"] == "direct")


def test_word_trap_shares_a_content_word_with_some_fact(facts, queries):
    all_fact_words = set().union(*(content_words(r["fact"]) for r in facts))
    for q in queries:
        if q["case_type"] == "word_trap":
            assert content_words(q["query"]) & all_fact_words, q["query_id"]


def test_none_has_no_relevant_fact_and_shares_no_content_word(facts, queries):
    all_fact_words = set().union(*(content_words(r["fact"]) for r in facts))
    for q in queries:
        if q["case_type"] == "none":
            assert not relevant_ids(q), q["query_id"]
            assert not content_words(q["query"]) & all_fact_words, (q["query_id"], content_words(q["query"]) & all_fact_words)
