"""Task D step D2: ChromaMemoryManager in scripts/memory_manager.py.
Every test uses tmp_path. Most use a deterministic fake embedder (hashed bag of content words), so they are fast
and need no model; one test uses the real all-MiniLM-L6-v2 (ONNX) embedder and checks ranking order only.
Every Chroma client is closed in teardown, so Windows can delete tmp_path (Chroma locks its folder while open)."""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("chromadb")
from chromadb.api.types import EmbeddingFunction  # noqa: E402

import memory_manager  # noqa: E402
from memory_manager import (  # noqa: E402
    ChromaMemoryManager,
    MemoryManager,
    content_words,
    make_memory_manager,
)
from safety import consent_given, record_consent  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent.parent
CFG = {"top_k": 3, "w_similarity": 0.7, "w_importance": 0.3, "base_importance": 0.3,
       "spike_threshold": 0.6, "max_fact_words": 25, "min_similarity": 0.0}
MOOD = {"probs": {"neutral": 0.5, "happy": 0.1, "angry": 0.2, "sad": 0.2}, "abstained": False}
RETRIEVE_KEYS = {"memory_id", "fact", "similarity", "importance", "score"}
DIM = 1024


class FakeEmbedder(EmbeddingFunction):
    """Hashed bag of content words (memory_manager.content_words) -> DIM-dim count vector. Index 0 is reserved for
    text with no content words, so no vector is ever all zeros. Cosine of two texts is easy to work out by hand."""

    def __init__(self):
        pass

    def __call__(self, input):
        return [self.vector(t) for t in input]

    @staticmethod
    def vector(text):
        v = np.zeros(DIM, dtype=np.float32)
        for w in content_words(text):
            v[1 + int(hashlib.md5(w.encode()).hexdigest(), 16) % (DIM - 1)] += 1.0
        if not v.any():
            v[0] = 1.0
        return v

    @staticmethod
    def name():
        return "test_fake_bag_of_words"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEmbedder()


def fake_cosine(a, b):
    u, v = FakeEmbedder.vector(a), FakeEmbedder.vector(b)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


@pytest.fixture
def open_store(tmp_path):
    """open_store(path=None, cfg=CFG) -> ChromaMemoryManager with the fake embedder; all are closed in teardown."""
    opened = []

    def _open(path=None, cfg=CFG):
        mm = ChromaMemoryManager(path or tmp_path / "vs", cfg, embedding_function=FakeEmbedder())
        opened.append(mm)
        return mm

    yield _open
    for mm in opened:
        mm.close()


@pytest.fixture
def mm(open_store):
    return open_store()


# ---------------------------------------------------------------- records

def test_store_and_list_round_trip(mm):
    mid = mm.store("alice", "The user has an exam on Monday.", 7, MOOD, 0.9)
    [r] = mm.list_user_facts("alice")
    assert set(r) == {"memory_id", "user_id", "timestamp", "source", "turn_id", "fact", "mood_snapshot", "importance"}
    assert r["memory_id"] == mid and len(mid) == 32 and int(mid, 16) >= 0
    assert r["user_id"] == "alice" and r["turn_id"] == 7 and r["importance"] == 0.9
    assert r["source"] == "user_turn" and r["fact"] == "The user has an exam on Monday."
    assert r["mood_snapshot"] == MOOD and isinstance(r["mood_snapshot"], dict)


def test_same_record_shape_as_json_backend(tmp_path, mm):
    js = MemoryManager(tmp_path / "memories.json", CFG)
    for m in (js, mm):
        m.store("alice", "The user has an exam on Monday.", 7, MOOD, 0.9)
        m.store("alice", "The user likes jazz.", None, {}, 0.3)
    strip = lambda rs: [{k: v for k, v in r.items() if k not in ("memory_id", "timestamp")} for r in rs]  # noqa: E731
    assert strip(mm.list_user_facts("alice")) == strip(js.list_user_facts("alice"))


def test_list_newest_first_even_within_one_clock_tick(mm):
    for i, fact in enumerate(["The user plays the violin.", "The user is moving to a new city.",
                              "The user has a dog named Rex."], start=1):
        mm.store("alice", fact, i, MOOD, 0.3)
    assert [r["turn_id"] for r in mm.list_user_facts("alice")] == [3, 2, 1]


def test_bad_inputs(mm):
    with pytest.raises(ValueError):
        mm.store("alice", "   ", 1, MOOD, 0.5)
    with pytest.raises(ValueError):
        mm.store("alice", "The user likes tea.", 1, MOOD, 1.5)
    with pytest.raises(ValueError):
        mm.store("", "The user likes tea.", 1, MOOD, 0.5)
    with pytest.raises(ValueError):
        mm.list_user_facts("")
    with pytest.raises(ValueError):
        mm.retrieve_relevant("  ", "tea")
    assert mm.list_user_facts("alice") == []


def test_missing_min_similarity_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="min_similarity"):
        ChromaMemoryManager(tmp_path / "vs", {k: v for k, v in CFG.items() if k != "min_similarity"},
                            embedding_function=FakeEmbedder())


# ---------------------------------------------------------------- duplicates

def test_duplicate_updates_instead_of_adding(mm):
    mid = mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.4)
    [before] = mm.list_user_facts("alice")
    assert mm.store("alice", "the user has an exam on monday", 5, MOOD, 0.9) == mid
    [after] = mm.list_user_facts("alice")
    assert after["importance"] == 0.9 and after["timestamp"] >= before["timestamp"]
    assert mm.store("alice", "The user has an exam on Monday!", 6, MOOD, 0.2) == mid
    assert mm.list_user_facts("alice")[0]["importance"] == 0.9            # keeps the higher one
    # word overlap >= 0.8 (4/5 content words) is also a duplicate
    big = mm.store("alice", "The user has a big chemistry exam on Monday morning.", 7, MOOD, 0.5)
    assert mm.store("alice", "The user has a chemistry exam Monday morning.", 8, MOOD, 0.3) == big
    assert len(mm.list_user_facts("alice")) == 2


def test_same_fact_for_two_users_is_not_a_duplicate(mm):
    assert mm.store("alice", "The user likes jazz.", 1, MOOD, 0.3) != mm.store("bob", "The user likes jazz.", 1, MOOD, 0.3)


# ---------------------------------------------------------------- delete / isolation / reset

def test_deleted_fact_never_retrieved_again_even_after_reopening(tmp_path, open_store):
    mm = open_store()
    mid = mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    keep = mm.store("alice", "The user likes jazz.", 2, MOOD, 0.3)
    assert mm.delete("alice", mid) is True
    assert mid not in {h["memory_id"] for h in mm.retrieve_relevant("alice", "exam Monday", top_k=10)}
    assert mm.delete("alice", mid) is False
    mm.close()
    reopened = open_store()
    assert [r["memory_id"] for r in reopened.list_user_facts("alice")] == [keep]
    assert mid not in {h["memory_id"] for h in reopened.retrieve_relevant("alice", "exam Monday", top_k=10)}
    assert reopened._col.get(ids=[mid])["ids"] == []                    # gone from Chroma itself, not just hidden


def test_cannot_delete_another_users_memory(mm):
    mid = mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    assert mm.delete("bob", mid) is False
    assert mm.delete("alice", "0" * 32) is False
    assert mm.delete("alice", None) is False
    assert len(mm.list_user_facts("alice")) == 1


def test_users_never_see_each_others_facts(mm):
    mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    mm.store("bob", "The user has a dentist appointment on Monday.", 1, MOOD, 0.9)
    assert [r["user_id"] for r in mm.list_user_facts("alice")] == ["alice"]
    assert [r["user_id"] for r in mm.list_user_facts("bob")] == ["bob"]
    assert [h["fact"] for h in mm.retrieve_relevant("alice", "worried about Monday", top_k=10)] == [
        "The user has an exam on Monday."]
    assert mm.retrieve_relevant("carol", "worried about Monday") == []


def test_reset_removes_one_user_only(mm):
    mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    mm.store("alice", "The user likes jazz.", 2, MOOD, 0.3)
    mm.store("bob", "The user likes hiking.", 1, MOOD, 0.3)
    assert mm.reset_user_memory("alice") == 2
    assert mm.list_user_facts("alice") == []
    assert len(mm.list_user_facts("bob")) == 1
    assert mm.reset_user_memory("alice") == 0


def _delete_all_scenario(root, store):
    """Same users, consent and log files under root; returns (report, consent_path, logs_dir)."""
    root.mkdir()
    consent = root / "consent.json"
    record_consent("alice", consent)
    record_consent("bob", consent)
    store.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    store.store("bob", "The user likes hiking.", 1, MOOD, 0.3)
    logs = root / "logs"
    logs.mkdir()
    for name in ["session_alice_20261112T100000Z.jsonl", "session_alice_20261113T100000Z.jsonl",
                 "session_bob_20261112T100000Z.jsonl", "session_alice_smith_20261112T100000Z.jsonl", "notes_alice.txt"]:
        (logs / name).write_text("{}\n", encoding="utf-8")
    return store.delete_all_my_data("alice", logs, consent), consent, logs


def test_delete_all_my_data_report_matches_json_backend(tmp_path, open_store):
    js = MemoryManager(tmp_path / "memories.json", CFG)
    ch = open_store()
    js_report, _, _ = _delete_all_scenario(tmp_path / "json_side", js)
    ch_report, consent, logs = _delete_all_scenario(tmp_path / "chroma_side", ch)
    assert ch_report == js_report
    assert ch_report["memories_removed"] == 1 and ch_report["consent_revoked"] is True
    assert sorted(ch_report["logs_removed"]) == ["session_alice_20261112T100000Z.jsonl",
                                                 "session_alice_20261113T100000Z.jsonl"]
    assert not consent_given("alice", consent) and consent_given("bob", consent)
    assert ch.list_user_facts("alice") == [] and len(ch.list_user_facts("bob")) == 1
    assert sorted(p.name for p in logs.iterdir()) == ["notes_alice.txt", "session_alice_smith_20261112T100000Z.jsonl",
                                                      "session_bob_20261112T100000Z.jsonl"]


# ---------------------------------------------------------------- persistence / handles

def test_persistence_across_instances_on_one_folder(tmp_path, open_store):
    a = open_store()
    mid = a.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    b = open_store()                                                   # second instance, same folder, a still open
    assert [r["memory_id"] for r in b.list_user_facts("alice")] == [mid]
    b.store("alice", "The user likes jazz.", 2, MOOD, 0.3)
    assert len(a.list_user_facts("alice")) == 2
    a.close()
    b.close()
    c = open_store()
    assert {r["fact"] for r in c.list_user_facts("alice")} == {"The user has an exam on Monday.", "The user likes jazz."}


def test_close_releases_the_folder(tmp_path):
    store = tmp_path / "vs_release"
    with ChromaMemoryManager(store, CFG, embedding_function=FakeEmbedder()) as mm:
        mm.store("alice", "The user likes jazz.", 1, MOOD, 0.3)
    shutil.rmtree(store)                                               # raises on Windows if a file is still locked
    assert not store.exists()


# ---------------------------------------------------------------- retrieval

def test_retrieve_keys_formula_and_top_k(tmp_path, mm):
    facts = [("The user has an exam on Monday.", 0.9), ("The user dreads Monday meetings at work.", 0.3),
             ("The user plays tennis on Monday evenings.", 0.5), ("The user hears jazz on Monday radio.", 1.0)]
    ids = {mm.store("alice", f, i, MOOD, imp): f for i, (f, imp) in enumerate(facts)}
    query = "stressed about the exam on Monday"
    hits = mm.retrieve_relevant("alice", query, top_k=10)
    js = MemoryManager(tmp_path / "memories.json", CFG)
    js.store("alice", facts[0][0], 1, MOOD, 0.9)
    assert set(hits[0]) == set(js.retrieve_relevant("alice", query)[0]) == RETRIEVE_KEYS
    assert len(hits) == 4                                              # min_similarity 0.0: every fact is scored
    for h in hits:
        assert h["fact"] == ids[h["memory_id"]]
        assert h["similarity"] == pytest.approx(fake_cosine(query, h["fact"]), abs=1e-5)
        assert h["score"] == pytest.approx(0.7 * h["similarity"] + 0.3 * h["importance"])
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    assert hits[0]["fact"] == "The user has an exam on Monday."
    assert len(mm.retrieve_relevant("alice", query)) == 3              # cfg top_k
    assert len(mm.retrieve_relevant("alice", query, top_k=1)) == 1
    assert mm.retrieve_relevant("alice", "   ") == []


def test_min_similarity_is_applied_before_scoring(open_store):
    cfg = {**CFG, "w_similarity": 0.5, "w_importance": 0.5}
    related, unrelated = "The user has a chemistry exam.", "The user eats pizza on Monday."
    query = "exam monday morning chemistry"
    sim_rel, sim_unrel = fake_cosine(query, related), fake_cosine(query, unrelated)
    assert sim_unrel < 0.5 < sim_rel                                   # by hand: 2/(2*sqrt2)=0.707, 1/(2*sqrt3)=0.289
    assert 0.5 * sim_unrel + 0.5 * 1.0 > 0.5 * sim_rel + 0.5 * 0.0     # the unrelated fact's score WOULD win

    no_cut = open_store(cfg={**cfg, "min_similarity": 0.0})
    no_cut.store("alice", related, 1, MOOD, 0.0)
    no_cut.store("alice", unrelated, 2, MOOD, 1.0)
    assert [h["fact"] for h in no_cut.retrieve_relevant("alice", query)] == [unrelated, related]
    no_cut.close()

    cut = open_store(cfg={**cfg, "min_similarity": 0.5})              # same folder, same facts, cutoff 0.5
    assert [h["fact"] for h in cut.retrieve_relevant("alice", query)] == [related]


def test_real_minilm_ranks_exam_fact_first_for_a_stress_query(tmp_path, capsys):
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    ef = DefaultEmbeddingFunction()
    try:
        ef(["warm-up"])
    except Exception as e:                                             # no network and no cached model
        pytest.skip(f"default MiniLM embedder could not load: {e}")
    exam, pizza = "The user gets anxious before exams.", "The user likes pizza."
    query = "I'm stressed about Monday."
    with ChromaMemoryManager(tmp_path / "vs_real", CFG, embedding_function=ef) as mm:
        mm.store("alice", exam, 1, MOOD, 0.5)
        mm.store("alice", pizza, 2, MOOD, 0.5)
        hits = mm.retrieve_relevant("alice", query, top_k=10)
    with capsys.disabled():
        print()
        for h in hits:
            print(f"    [real MiniLM] query {query!r} -> {h['fact']!r}: similarity {h['similarity']:.4f}, "
                  f"score {h['score']:.4f}")
    assert hits and hits[0]["fact"] == exam
    if len(hits) == 2:
        assert hits[0]["similarity"] > hits[1]["similarity"]


# ---------------------------------------------------------------- factory

def test_make_memory_manager_picks_the_backend(tmp_path):
    conf = {"memory": dict(CFG), "memory_file": tmp_path / "memories.json", "log_dir": tmp_path / "logs",
            "vector_store_dir": tmp_path / "vs"}
    assert type(make_memory_manager(conf)) is MemoryManager                       # no backend key -> json
    assert type(make_memory_manager({**conf, "memory": {**CFG, "backend": "json"}})) is MemoryManager
    with make_memory_manager({**conf, "memory": {**CFG, "backend": "chroma"}}, FakeEmbedder()) as ch:
        assert type(ch) is ChromaMemoryManager and ch.path == tmp_path / "vs"
    with pytest.raises(ValueError, match="unknown memory.backend"):
        make_memory_manager({**conf, "memory": {**CFG, "backend": "faiss"}})
    with pytest.raises(ValueError, match="vector_store_dir"):
        make_memory_manager({**conf, "memory": {**CFG, "backend": "chroma"}, "vector_store_dir": None})


def test_cli_backend_flag_uses_chroma(tmp_path, monkeypatch, capsys):
    conf = {"memory": {**CFG, "backend": "json"}, "memory_file": tmp_path / "memories.json",
            "log_dir": tmp_path / "logs", "vector_store_dir": tmp_path / "vs"}
    with ChromaMemoryManager(tmp_path / "vs", CFG, embedding_function=FakeEmbedder()) as mm:
        mm.store("alice", "The user likes jazz.", 1, MOOD, 0.3)
    monkeypatch.setattr(memory_manager, "load_memory_config", lambda: conf)
    real_factory = memory_manager.make_memory_manager
    monkeypatch.setattr(memory_manager, "make_memory_manager", lambda c: real_factory(c, FakeEmbedder()))
    assert memory_manager.main(["--user", "alice", "--list", "--backend", "chroma"]) == 0
    assert "The user likes jazz." in capsys.readouterr().out
    assert conf["memory"]["backend"] == "json"                                    # the CLI override did not leak
    assert not (tmp_path / "memories.json").exists()


# ---------------------------------------------------------------- imports

def test_importing_memory_manager_loads_neither_torch_nor_chromadb():
    code = (f"import sys; sys.path.insert(0, {str(CODE_DIR / 'scripts')!r}); import memory_manager; "
            "print([m for m in ('torch', 'chromadb') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
