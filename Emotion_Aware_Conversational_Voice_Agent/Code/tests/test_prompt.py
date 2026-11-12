"""
test_prompt.py — P3 unit test (code decides, LLM speaks).
For every strategy (with and without check_in) the system prompt must not contain an emotion label,
a probability-like number, or the word "distress". Also checks that generate_reply returns the exact
prompt it sent, and that the strategy -> TTS tag mapping comes from the strategy alone.
"""
import json
import re
from pathlib import Path

import pytest

import response_generator as rg
from mood_manager import MoodManager, STRATEGY_FOR_MODE, TTS_TAGS, strategy_for, tts_tags_for

CODE_DIR = Path(__file__).resolve().parent.parent
CLASSES = json.loads((CODE_DIR / "checkpoints" / "xgb_meta.json").read_text())["classes"]
EMOTION_WORDS = sorted(set(CLASSES) | {"happy", "sad", "angry", "happiness", "sadness", "anger"})
EMOTION_RE = re.compile(r"\b(" + "|".join(EMOTION_WORDS) + r")\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d*\.\d+|\d+\s*%")                       # 0.83, .5, 72%
SCORE_WORDS_RE = re.compile(r"probab|confidence|p_distress", re.IGNORECASE)
CASES = [(s, c) for s in rg.STRATEGIES for c in (False, True)]


@pytest.mark.parametrize("strategy,check_in", CASES)
def test_prompt_has_no_emotion_labels_or_scores(strategy, check_in):
    prompt = rg.build_system_prompt(strategy, facts=None, check_in=check_in)
    assert not EMOTION_RE.search(prompt), EMOTION_RE.search(prompt).group(0)
    assert "distress" not in prompt.lower()
    assert not NUMBER_RE.search(prompt), NUMBER_RE.search(prompt).group(0)
    assert not SCORE_WORDS_RE.search(prompt), SCORE_WORDS_RE.search(prompt).group(0)


@pytest.mark.parametrize("strategy,check_in", CASES)
def test_prompt_carries_its_strategy_block(strategy, check_in):
    prompt = rg.build_system_prompt(strategy, check_in=check_in)
    assert rg.STRATEGY_INSTRUCTIONS[strategy] in prompt
    for other in set(rg.STRATEGIES) - {strategy}:
        assert rg.STRATEGY_INSTRUCTIONS[other] not in prompt
    assert (rg.CHECK_IN_INSTRUCTION in prompt) == check_in


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        rg.build_system_prompt("EMPATHETIC")   # a mode, not a strategy


@pytest.mark.parametrize("strategy,check_in", CASES)
def test_generate_reply_returns_exact_prompt_sent(monkeypatch, strategy, check_in):
    sent = {}

    def fake_llm(system_prompt, user_message, history=None, api_key=None, models=None, attempts=None):
        sent["prompt"] = system_prompt
        return "ok", "fake-model"

    monkeypatch.setattr(rg, "_call_llm", fake_llm)
    out = rg.generate_reply(strategy, "hello", check_in=check_in, use_llm=True)
    assert out["source"] == "llm"
    assert out["system_prompt"] == sent["prompt"] == rg.build_system_prompt(strategy, check_in=check_in)


def test_generate_reply_template_path():
    for strategy in rg.STRATEGIES:
        out = rg.generate_reply(strategy, "hello", use_llm=False)
        assert out["source"] == "template" and out["error"] is None and out["system_prompt"] is None
        assert out["text"] == rg.TEMPLATES[strategy]
    assert rg.generate_reply("VALIDATE", "hello", check_in=True, use_llm=False)["text"] == rg.CHECK_IN_TEMPLATE


def test_strategy_and_tts_mapping():
    assert set(STRATEGY_FOR_MODE.values()) == set(rg.STRATEGIES) == set(TTS_TAGS)
    assert strategy_for("DEFAULT") == "NEUTRAL" and strategy_for("EMPATHETIC") == "VALIDATE"
    assert tts_tags_for("NEUTRAL") == [] and tts_tags_for("VALIDATE") == ["<sigh>", "<softly>"]


def test_mood_profile_follows_mode():
    m = MoodManager(gamma=0.7, threshold=0.5, persistence=2)
    keys = {"mode", "strategy", "check_in", "reason", "tts_tags"}
    m.update(0.1, confidence=1.0, tau_conf=0.0)
    p = m.get_current_profile()
    assert set(p) == keys and p["mode"] == "DEFAULT" and p["strategy"] == "NEUTRAL" and p["tts_tags"] == []
    for _ in range(4):
        m.update(1.0, confidence=1.0, tau_conf=0.0)
    p = m.get_current_profile()
    assert p["mode"] == "EMPATHETIC" and p["strategy"] == "VALIDATE" and p["tts_tags"] == TTS_TAGS["VALIDATE"]
    p["tts_tags"].append("<x>")                       # caller can't mutate the shared table
    assert TTS_TAGS["VALIDATE"] == ["<sigh>", "<softly>"]
