"""Demo step 5: scripts/fact_extractor.py. Stub LLMs only; no network, no real memory file."""
import ast
import inspect
import re
from pathlib import Path

import pytest

import fact_extractor
from fact_extractor import build_extraction_prompt, extract_fact, parse_extraction

CODE_DIR = Path(__file__).resolve().parent.parent
EMOTION_LABELS = ("neutral", "happy", "angry", "sad", "fearful", "calm", "disgust", "surprised")


class Stub:
    def __init__(self, reply="The user has an exam on Monday."):
        self.reply, self.prompts = reply, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.reply


# ---------------------------------------------------------------- parse

@pytest.mark.parametrize("raw, expected", [
    ("The user has an exam on Monday.", "The user has an exam on Monday."),
    ("  The user likes jazz.  \n", "The user likes jazz."),
    ('"The user likes jazz."', "The user likes jazz."),
    ("“The user likes jazz.”", "The user likes jazz."),
    ("- The user likes jazz.", "The user likes jazz."),
    ("* The user likes jazz.", "The user likes jazz."),
    ("1. The user likes jazz.", "The user likes jazz."),
    ('- "The user likes jazz."', "The user likes jazz."),
    ("Fact: The user likes jazz.", "The user likes jazz."),
    ("The user likes jazz.\nThe user also likes tea.", "The user likes jazz."),
    ("NONE", None),
    ("none.", None),
    ("None", None),
    ('"NONE"', None),
    ("", None),
    ("   \n  ", None),
    (None, None),
])
def test_parse_cases(raw, expected):
    assert parse_extraction(raw) == expected


def test_parse_too_long():
    assert parse_extraction("The user " + "word " * 23) is not None            # 25 words: at the limit
    assert parse_extraction("The user " + "word " * 24) is None                # 26 words: over
    assert parse_extraction(" ".join(["w"] * 25)) == " ".join(["w"] * 25)
    assert parse_extraction(" ".join(["w"] * 26)) is None
    assert parse_extraction("The user likes jazz.", max_fact_words=3) is None


# ---------------------------------------------------------------- prompt

def test_prompt_contains_transcript_and_rules():
    said = "I have my driving test next Thursday and I keep practising parking."
    p = build_extraction_prompt(said)
    assert said in p
    assert "NONE" in p and "The user" in p and "at most 25 words" in p
    assert "third person" in p and "names" in p


def test_prompt_has_no_probabilities_or_emotion_labels():
    for spike in (False, True):
        p = build_extraction_prompt("I have my driving test next Thursday.", emotional_spike=spike).lower()
        assert not re.search(r"\d+\.\d+|\d+\s*%|probabilit|confidence|score", p)
        for label in EMOTION_LABELS:
            assert not re.search(rf"\b{label}\b", p), label


def test_spike_flag_changes_the_instruction():
    calm = build_extraction_prompt("I have my driving test next Thursday.")
    spike = build_extraction_prompt("I have my driving test next Thursday.", emotional_spike=True)
    assert calm != spike
    assert "strongly emotional" in spike and "strongly emotional" not in calm
    assert "include how they feel" in spike


# ---------------------------------------------------------------- extract_fact

def test_extract_fact_takes_only_the_user_transcript():
    params = list(inspect.signature(extract_fact).parameters)
    assert params == ["user_transcript", "llm_fn", "emotional_spike"]
    assert not any(w in p for p in params for w in ("reply", "agent", "response", "assistant"))


def test_extract_fact_happy_path():
    stub = Stub('- "The user has an exam on Monday."')
    assert extract_fact("I have an exam on Monday", stub) == "The user has an exam on Monday."
    assert len(stub.prompts) == 1 and "I have an exam on Monday" in stub.prompts[0]


def test_extract_fact_none_reply():
    assert extract_fact("haha nice", Stub("NONE")) is None


def test_spike_reaches_the_prompt():
    stub = Stub()
    extract_fact("I have an exam on Monday", stub, emotional_spike=True)
    assert "strongly emotional" in stub.prompts[0]


@pytest.mark.parametrize("text", ["I want to kill myself", "I don't want to be alive anymore", "thinking about self-harm"])
def test_crisis_returns_none_without_calling_llm(text):
    stub = Stub()
    assert extract_fact(text, stub) is None
    assert stub.prompts == []


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_transcript_returns_none_without_calling_llm(text):
    stub = Stub()
    assert extract_fact(text, stub) is None
    assert stub.prompts == []


@pytest.mark.parametrize("exc", [RuntimeError("all models failed"), TimeoutError(), ValueError("bad")])
def test_llm_error_returns_none(exc):
    def boom(prompt):
        raise exc
    assert extract_fact("I have an exam on Monday", boom) is None


def test_non_string_llm_reply_returns_none():
    assert extract_fact("I have an exam on Monday", lambda p: None) is None


def test_demo_runs(capsys):
    assert fact_extractor.demo() == 0
    out = capsys.readouterr().out
    assert "exam on Monday" in out and "temporary file removed: True" in out


# ---------------------------------------------------------------- imports

@pytest.mark.parametrize("name", ["fact_extractor.py", "memory_manager.py"])
def test_no_torch_or_gpu_imports(name):
    tree = ast.parse((CODE_DIR / "scripts" / name).read_text(encoding="utf-8"))
    gpu_modules = {p.stem for p in (CODE_DIR / "gpu").glob("*.py")} | {"gpu"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "torch" not in imported, name
    assert not imported & gpu_modules, (name, imported & gpu_modules)
