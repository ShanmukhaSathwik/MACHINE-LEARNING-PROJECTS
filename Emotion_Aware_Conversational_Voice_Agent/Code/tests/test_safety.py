"""Demo step 4: scripts/safety.py (consent + crisis check). Consent tests use tmp_path only."""
import ast
import json
from datetime import datetime
from pathlib import Path

import pytest

import safety
from safety import (
    CONSENT_TEXT,
    CRISIS_PHRASES,
    SAFE_RESPONSE,
    SAFE_STRATEGY,
    ConsentFileError,
    ask_consent,
    check_crisis,
    consent_given,
    normalize,
    record_consent,
    revoke_consent,
)

CODE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- normalize / crisis check

def test_normalize_rules():
    assert normalize("I’m   SO-tired!!") == "i am so tired"
    assert normalize("Don't, can't; I've") == "do not can not i have"
    assert normalize("im ok") == "i am ok"
    assert normalize(None) == ""
    assert normalize("") == ""


def test_phrases_are_already_normalized():
    for p in CRISIS_PHRASES:
        assert normalize(p) == p, p


def test_no_bare_trigger_words():
    for bare in ("die", "dying", "dead", "kill"):
        assert bare not in CRISIS_PHRASES


@pytest.mark.parametrize("text, phrase", [
    ("I want to kill myself", "kill myself"),
    ("I WANT TO KILL MYSELF.", "kill myself"),
    ("i want to... kill myself!!!", "kill myself"),
    ("I’m going to end my life", "end my life"),
    ("im suicidal", "suicidal"),
    ("I'm suicidal", "suicidal"),
    ("thinking about self-harm", "self harm"),
    ("thinking   about   self  harm", "self harm"),
    ("I just want to   die", "want to die"),
    ("i wish i were dead", "wish i were dead"),
    ("I wish I was dead", "wish i was dead"),
    ("I don’t want to be alive", "do not want to be alive"),
    ("I do not want to be here anymore", "not want to be here anymore"),
    ("there's no reason to live", "no reason to live"),
    ("Sometimes I cut myself", "cut myself"),
    ("i might hurt myself", "hurt myself"),
    ("they'd be better off dead without me", "better off dead"),
])
def test_crisis_phrases_match(text, phrase):
    assert check_crisis(text) == {"matched": True, "phrase": phrase}


@pytest.mark.parametrize("text", [
    "I don't want to live anymore",
    "I don't want to kill myself",
])
def test_negated_sentences_still_match(text):
    assert check_crisis(text)["matched"] is True


@pytest.mark.parametrize("text", [
    "this exam is killing me",
    "I'm dying to see that movie",
    "my phone died",
    "kill the process",
    "I could die of embarrassment",
    "that joke killed me",
    "this traffic is killing me",
    "I'm dead tired",
])
def test_no_false_alarms(text):
    assert check_crisis(text) == {"matched": False, "phrase": None}


@pytest.mark.parametrize("text", ["", None, "   ", "?!"])
def test_empty_not_matched(text):
    assert check_crisis(text) == {"matched": False, "phrase": None}


def test_safe_response_and_strategy():
    assert "988" in SAFE_RESPONSE
    assert "911" in SAFE_RESPONSE
    assert SAFE_STRATEGY == "NEUTRAL"


# ---------------------------------------------------------------- consent (tmp_path only)

def test_no_file_means_no_consent(tmp_path):
    path = tmp_path / "consent.json"
    assert consent_given("alice", path) is False
    assert not path.exists()


def test_empty_file_means_no_consent(tmp_path):
    path = tmp_path / "consent.json"
    path.write_text("", encoding="utf-8")
    assert consent_given("alice", path) is False


def test_record_keep_and_revoke(tmp_path):
    path = tmp_path / "consent.json"
    record_consent("alice", path)
    assert consent_given("alice", path) is True
    record_consent("bob", path)
    assert consent_given("alice", path) and consent_given("bob", path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"alice", "bob"}
    for ts in data.values():
        assert isinstance(ts, str)
        parsed = datetime.fromisoformat(ts)
        assert parsed.utcoffset().total_seconds() == 0   # UTC

    assert revoke_consent("alice", path) is True
    assert consent_given("alice", path) is False
    assert consent_given("bob", path) is True
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"bob"}

    assert revoke_consent("carol", path) is False
    assert revoke_consent("alice", path) is False


def test_revoke_without_file_does_not_create_it(tmp_path):
    path = tmp_path / "consent.json"
    assert revoke_consent("alice", path) is False
    assert not path.exists()


def test_corrupt_file_raises_and_is_untouched(tmp_path):
    path = tmp_path / "consent.json"
    path.write_text("{not json", encoding="utf-8")
    for call in (lambda: consent_given("a", path), lambda: record_consent("a", path),
                 lambda: revoke_consent("a", path)):
        with pytest.raises(ConsentFileError):
            call()
    assert path.read_text(encoding="utf-8") == "{not json"


def test_wrong_shape_file_raises(tmp_path):
    path = tmp_path / "consent.json"
    path.write_text('["alice"]', encoding="utf-8")
    with pytest.raises(ConsentFileError):
        consent_given("alice", path)


@pytest.mark.parametrize("answer", ["yes", "Y", " yes ", "YES", "y"])
def test_ask_consent_yes(tmp_path, answer):
    path = tmp_path / "consent.json"
    printed = []
    assert ask_consent("alice", path, input_fn=lambda prompt="": answer, print_fn=printed.append) is True
    assert consent_given("alice", path) is True
    assert printed[0] == CONSENT_TEXT


@pytest.mark.parametrize("answer", ["no", "", "sure", "yeah", "n", "yes please"])
def test_ask_consent_not_yes(tmp_path, answer):
    path = tmp_path / "consent.json"
    printed = []
    assert ask_consent("alice", path, input_fn=lambda prompt="": answer, print_fn=printed.append) is False
    assert consent_given("alice", path) is False
    assert not path.exists()
    assert printed[0] == CONSENT_TEXT


def test_consent_text_contents():
    t = CONSENT_TEXT.lower()
    assert "recorded" in t
    assert "stored" in t
    assert "delete all my data" in t
    assert "not a crisis service" in t
    assert "counsellor" in t


def test_default_path_is_resolved_from_file():
    assert safety.DEFAULT_CONSENT_PATH == CODE_DIR / "data" / "consent.json"
    assert safety.DEFAULT_CONSENT_PATH.is_absolute()


def test_consent_json_is_gitignored():
    gitignore = CODE_DIR.parent.parent / ".gitignore"
    lines = [ln.strip() for ln in gitignore.read_text(encoding="utf-8").splitlines()]
    assert "Emotion_Aware_Conversational_Voice_Agent/Code/data/consent.json" in lines


# ---------------------------------------------------------------- imports

def test_no_torch_or_gpu_imports():
    tree = ast.parse((CODE_DIR / "scripts" / "safety.py").read_text(encoding="utf-8"))
    gpu_modules = {p.stem for p in (CODE_DIR / "gpu").glob("*.py")} | {"gpu"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "torch" not in imported
    assert not (imported & gpu_modules), imported & gpu_modules
