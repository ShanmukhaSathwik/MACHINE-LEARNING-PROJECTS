"""Demo step 5: scripts/memory_manager.py. Every test uses tmp_path; the real data/ and results/ are never touched."""
import ast
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

import memory_manager
from memory_manager import (
    MemoryFileError,
    MemoryManager,
    importance_from_turn,
    is_user_session_log,
    jaccard,
    load_memory_config,
    session_log_filename,
)
from safety import consent_given, record_consent

CODE_DIR = Path(__file__).resolve().parent.parent
CFG = {"top_k": 3, "w_similarity": 0.7, "w_importance": 0.3, "base_importance": 0.3,
       "spike_threshold": 0.6, "max_fact_words": 25}
MOOD = {"probs": {"neutral": 0.5, "happy": 0.1, "angry": 0.2, "sad": 0.2}, "abstained": False}


@pytest.fixture
def mm(tmp_path):
    return MemoryManager(tmp_path / "memories.json", CFG)


# ---------------------------------------------------------------- config

def test_config_block_matches_spec():
    conf = load_memory_config()
    assert conf["memory"] == CFG
    assert conf["memory_file"] == CODE_DIR / "data" / "memories.json"
    assert conf["log_dir"] == CODE_DIR / "results" / "logs"


def test_missing_config_key_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="top_k"):
        MemoryManager(tmp_path / "m.json", {k: v for k, v in CFG.items() if k != "top_k"})


# ---------------------------------------------------------------- records

def test_record_fields_and_source(mm):
    mid = mm.store("alice", "The user has an exam on Monday.", 7, MOOD, 0.9)
    [r] = mm.list_user_facts("alice")
    assert set(r) == {"memory_id", "user_id", "timestamp", "source", "turn_id", "fact", "mood_snapshot", "importance"}
    assert r["memory_id"] == mid and len(mid) == 32 and int(mid, 16) >= 0
    assert r["user_id"] == "alice" and r["turn_id"] == 7 and r["importance"] == 0.9
    assert r["source"] == "user_turn"
    assert r["mood_snapshot"] == MOOD
    assert datetime.fromisoformat(r["timestamp"]).utcoffset().total_seconds() == 0


def test_source_cannot_be_set():
    import inspect
    params = inspect.signature(MemoryManager.store).parameters
    assert "source" not in params
    with pytest.raises(TypeError):
        MemoryManager.store(None, "u", "f", 1, {}, 0.5, source="agent_reply")   # noqa


def test_list_newest_first(mm):
    mm.store("alice", "The user plays the violin.", 1, MOOD, 0.3)
    mm.store("alice", "The user is moving to a new city.", 2, MOOD, 0.3)
    assert [r["turn_id"] for r in mm.list_user_facts("alice")] == [2, 1]


def test_bad_inputs(mm):
    with pytest.raises(ValueError):
        mm.store("alice", "   ", 1, MOOD, 0.5)
    with pytest.raises(ValueError):
        mm.store("alice", "The user likes tea.", 1, MOOD, 1.5)
    with pytest.raises(ValueError):
        mm.list_user_facts("")


# ---------------------------------------------------------------- isolation

def test_user_isolation(mm):
    mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    mm.store("bob", "The user has a dentist appointment on Monday.", 1, MOOD, 0.9)
    assert [r["user_id"] for r in mm.list_user_facts("alice")] == ["alice"]
    assert [r["user_id"] for r in mm.list_user_facts("bob")] == ["bob"]
    hits = mm.retrieve_relevant("alice", "worried about Monday")
    assert [h["fact"] for h in hits] == ["The user has an exam on Monday."]
    assert mm.retrieve_relevant("carol", "worried about Monday") == []


# ---------------------------------------------------------------- delete / reset

def test_deleted_memory_never_returns(tmp_path, mm):
    mid = mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    assert mm.delete("alice", mid) is True
    assert mm.retrieve_relevant("alice", "exam Monday") == []
    reloaded = MemoryManager(tmp_path / "memories.json", CFG)
    assert reloaded.list_user_facts("alice") == []
    assert reloaded.retrieve_relevant("alice", "exam Monday") == []
    assert mid not in (tmp_path / "memories.json").read_text(encoding="utf-8")
    assert mm.delete("alice", mid) is False


def test_cannot_delete_another_users_memory(mm):
    mid = mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    assert mm.delete("bob", mid) is False
    assert mm.delete("alice", "0" * 32) is False
    assert len(mm.list_user_facts("alice")) == 1


def test_reset_wipes_one_user_only(mm):
    mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    mm.store("alice", "The user likes jazz.", 2, MOOD, 0.3)
    mm.store("bob", "The user likes hiking.", 1, MOOD, 0.3)
    assert mm.reset_user_memory("alice") == 2
    assert mm.list_user_facts("alice") == []
    assert len(mm.list_user_facts("bob")) == 1
    assert mm.reset_user_memory("alice") == 0


def test_delete_all_my_data(tmp_path, mm):
    consent = tmp_path / "consent.json"
    record_consent("alice", consent)
    record_consent("bob", consent)
    mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    mm.store("bob", "The user likes hiking.", 1, MOOD, 0.3)
    logs = tmp_path / "logs"
    logs.mkdir()
    mine = [session_log_filename("alice"), "session_alice_20261112T100000Z.jsonl"]
    others = ["session_bob_20261112T100000Z.jsonl", "session_alice_smith_20261112T100000Z.jsonl",
              "notes_alice.txt", "session_alice_20261112T100000Z.json"]
    for name in mine + others:
        (logs / name).write_text("{}\n", encoding="utf-8")

    report = mm.delete_all_my_data("alice", logs, consent)

    assert report["memories_removed"] == 1 and report["consent_revoked"] is True
    assert sorted(report["logs_removed"]) == sorted(set(mine))
    assert report["logs_skipped_tracked"] == []
    assert not consent_given("alice", consent) and consent_given("bob", consent)
    assert mm.list_user_facts("alice") == [] and len(mm.list_user_facts("bob")) == 1
    assert sorted(p.name for p in logs.iterdir()) == sorted(others)


def test_delete_all_without_logs_or_consent(tmp_path, mm):
    report = mm.delete_all_my_data("alice", tmp_path / "no_logs_here", tmp_path / "consent.json")
    assert report["memories_removed"] == 0 and report["consent_revoked"] is False and report["logs_removed"] == []
    assert not (tmp_path / "consent.json").exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_delete_all_never_removes_a_git_tracked_log(tmp_path, mm):
    logs = tmp_path / "repo"
    logs.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=logs, check=True)
    tracked = logs / "session_alice_20261101T090000Z.jsonl"
    untracked = logs / "session_alice_20261102T090000Z.jsonl"
    tracked.write_text("{}\n", encoding="utf-8")
    untracked.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", tracked.name], cwd=logs, check=True)

    report = mm.delete_all_my_data("alice", logs, tmp_path / "consent.json")
    assert report["logs_skipped_tracked"] == [tracked.name]
    assert report["logs_removed"] == [untracked.name]
    assert tracked.exists() and not untracked.exists()


def test_session_log_name_rule():
    assert is_user_session_log("session_bob_20261112T120000Z.jsonl", "bob")
    assert not is_user_session_log("session_bob_smith_20261112T120000Z.jsonl", "bob")
    assert is_user_session_log("session_bob_smith_20261112T120000Z.jsonl", "bob_smith")
    assert not is_user_session_log("session_bob_.jsonl", "bob")
    assert is_user_session_log(session_log_filename("a.b"), "a.b")
    assert not is_user_session_log("session_aXb_1.jsonl", "a.b")      # '.' is literal, not a wildcard


# ---------------------------------------------------------------- retrieval

def test_scoring_formula_and_top_k(mm):
    a = mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.9)
    b = mm.store("alice", "The user dreads Monday meetings at work.", 2, MOOD, 0.3)
    c = mm.store("alice", "The user plays tennis on Monday evenings.", 3, MOOD, 0.5)
    mm.store("alice", "The user likes jazz.", 4, MOOD, 1.0)
    query = "stressed about the exam on Monday"
    hits = mm.retrieve_relevant("alice", query)
    assert len(hits) == 3
    assert set(hits[0]) == {"memory_id", "fact", "similarity", "importance", "score"}
    for h in hits:
        assert h["similarity"] == pytest.approx(jaccard(query, h["fact"]))
        assert h["score"] == pytest.approx(0.7 * h["similarity"] + 0.3 * h["importance"])
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    assert hits[0]["memory_id"] == a
    assert {h["memory_id"] for h in hits} == {a, b, c}
    assert len(mm.retrieve_relevant("alice", query, top_k=1)) == 1


def test_similarity_zero_not_returned(mm):
    mm.store("alice", "The user likes jazz.", 1, MOOD, 1.0)
    assert mm.retrieve_relevant("alice", "exam on Monday") == []
    assert mm.retrieve_relevant("alice", "the and of") == []          # stopwords only


def test_jaccard_ignores_case_punctuation_and_stopwords():
    assert jaccard("The user has an EXAM on Monday!", "exam monday") == 1.0
    assert jaccard("I'm stressed about Monday", "The user has an exam on Monday.") == pytest.approx(1 / 3)


# ---------------------------------------------------------------- importance

def test_importance_abstain_is_base():
    imp, spike = importance_from_turn({"neutral": 0.0, "sad": 1.0}, True, CFG)
    assert imp == pytest.approx(0.3) and spike is False


def test_importance_mostly_neutral_near_base():
    imp, spike = importance_from_turn({"neutral": 0.92, "happy": 0.03, "angry": 0.02, "sad": 0.03}, False, CFG)
    assert 0.3 < imp < 0.4 and spike is False
    assert imp == pytest.approx(0.3 + 0.7 * 0.08)


def test_importance_strong_sad_high_and_spike():
    imp, spike = importance_from_turn({"neutral": 0.05, "happy": 0.0, "angry": 0.05, "sad": 0.9}, False, CFG)
    assert imp == pytest.approx(0.3 + 0.7 * 0.95) and imp > 0.9 and spike is True


def test_spike_threshold_is_inclusive():
    _, spike = importance_from_turn({"neutral": 0.4, "sad": 0.6}, False, CFG)
    assert spike is True


# ---------------------------------------------------------------- duplicates

def test_duplicate_updates_instead_of_adding(mm):
    mid = mm.store("alice", "The user has an exam on Monday.", 1, MOOD, 0.4)
    [before] = mm.list_user_facts("alice")
    again = mm.store("alice", "the user has an exam on monday", 5, MOOD, 0.9)
    [after] = mm.list_user_facts("alice")
    assert again == mid
    assert after["importance"] == 0.9 and after["timestamp"] >= before["timestamp"]
    assert mm.store("alice", "The user has an exam on Monday!", 6, MOOD, 0.2) == mid
    assert mm.list_user_facts("alice")[0]["importance"] == 0.9            # keeps the higher one


def test_near_duplicate_by_overlap(mm):
    mid = mm.store("alice", "The user has a big chemistry exam on Monday morning.", 1, MOOD, 0.5)
    # content words {big, chemistry, exam, monday, morning}; new adds nothing new -> overlap 4/5 = 0.8
    assert mm.store("alice", "The user has a chemistry exam Monday morning.", 2, MOOD, 0.3) == mid
    assert len(mm.list_user_facts("alice")) == 1
    other = mm.store("alice", "The user has a chemistry lab report due Friday.", 3, MOOD, 0.3)
    assert other != mid and len(mm.list_user_facts("alice")) == 2


def test_same_fact_for_two_users_is_not_a_duplicate(mm):
    a = mm.store("alice", "The user likes jazz.", 1, MOOD, 0.3)
    b = mm.store("bob", "The user likes jazz.", 1, MOOD, 0.3)
    assert a != b


# ---------------------------------------------------------------- file safety

@pytest.mark.parametrize("content", ["{not json", '["a list"]', '{"memories": [{"fact": "no ids"}]}'])
def test_corrupt_file_raises_and_is_unchanged(tmp_path, content):
    path = tmp_path / "memories.json"
    path.write_text(content, encoding="utf-8")
    mm = MemoryManager(path, CFG)
    for call in (lambda: mm.list_user_facts("alice"), lambda: mm.retrieve_relevant("alice", "x"),
                 lambda: mm.store("alice", "The user likes jazz.", 1, MOOD, 0.3),
                 lambda: mm.delete("alice", "x"), lambda: mm.reset_user_memory("alice")):
        with pytest.raises(MemoryFileError):
            call()
    assert path.read_text(encoding="utf-8") == content
    assert [p.name for p in tmp_path.iterdir()] == ["memories.json"]      # no temp files left


def test_missing_or_empty_file_is_empty(tmp_path):
    path = tmp_path / "memories.json"
    assert MemoryManager(path, CFG).list_user_facts("alice") == []
    assert not path.exists()
    path.write_text("", encoding="utf-8")
    assert MemoryManager(path, CFG).list_user_facts("alice") == []


def test_file_format_and_no_temp_files(tmp_path, mm):
    mm.store("alice", "The user likes jazz.", 1, MOOD, 0.3)
    data = json.loads((tmp_path / "memories.json").read_text(encoding="utf-8"))
    assert list(data) == ["memories"] and len(data["memories"]) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["memories.json"]


# ---------------------------------------------------------------- CLI

def test_cli_list_forget_delete_all(tmp_path, monkeypatch, capsys):
    path = tmp_path / "memories.json"
    mid = MemoryManager(path, CFG).store("alice", "The user likes jazz.", 1, MOOD, 0.3)
    MemoryManager(path, CFG).store("bob", "The user likes tea.", 1, MOOD, 0.3)
    monkeypatch.setattr(memory_manager, "DEFAULT_CONSENT_PATH", tmp_path / "consent.json")
    monkeypatch.setattr(memory_manager, "load_memory_config",
                        lambda: {"memory": CFG, "memory_file": path, "log_dir": tmp_path / "logs"})

    assert memory_manager.main(["--user", "alice", "--list"]) == 0
    assert "The user likes jazz." in capsys.readouterr().out
    assert memory_manager.main(["--user", "bob", "--forget", mid]) == 1           # not bob's
    assert memory_manager.main(["--user", "alice", "--delete-all"], input_fn=lambda p: "no") == 1
    assert len(MemoryManager(path, CFG).list_user_facts("alice")) == 1
    assert memory_manager.main(["--user", "alice", "--delete-all"], input_fn=lambda p: " YES ") == 0
    assert MemoryManager(path, CFG).list_user_facts("alice") == []
    assert len(MemoryManager(path, CFG).list_user_facts("bob")) == 1


# ---------------------------------------------------------------- imports

def test_no_torch_or_gpu_imports():
    tree = ast.parse((CODE_DIR / "scripts" / "memory_manager.py").read_text(encoding="utf-8"))
    gpu_modules = {p.stem for p in (CODE_DIR / "gpu").glob("*.py")} | {"gpu"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "torch" not in imported and not imported & gpu_modules
