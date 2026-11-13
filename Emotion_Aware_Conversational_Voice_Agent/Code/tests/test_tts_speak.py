"""
test_tts_speak.py - step 8b of the GPU demo build (gpu/tts_gpu.py CSMVoice / speak_stream / speak_with_fallback,
scripts/fallback_tts.py). CPU only: no CSM, no sounddevice playback, no pyttsx3 voice, no network.
CSM is a stub TTS; playback, pyttsx3 and the clock-free parts are monkeypatched.
"""
import ast
import inspect
import subprocess
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import fallback_tts
import tts_gpu
from device import list_presets, load_config
from tts_gpu import (
    STRATEGIES,
    CSMVoice,
    VoiceRefsError,
    ref_strategy,
    speak_with_fallback,
    split_sentences,
    strip_tags,
)

CODE_DIR = Path(__file__).resolve().parent.parent
SR = 24000
REF_LEN = {s: 1000 * (i + 1) for i, s in enumerate(STRATEGIES)}   # each reference clip has its own length


# ---------------------------------------------------------------- stubs

class StubTTS:
    """Stands in for the loaded 8a TTS: records every synth call, returns 0.1 s of audio."""

    def __init__(self, fail_on=None):
        self.sr = SR
        self.calls = []
        self.fail_on = fail_on          # text that makes synth raise (a CSM failure mid-reply)

    def synth(self, text, ref_text, ref_audio_24k):
        self.calls.append({"text": text, "ref_text": ref_text, "ref_len": len(ref_audio_24k)})
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("CUDA out of memory (stub)")
        return {"audio": np.full(2400, 0.1, dtype=np.float32), "sr": SR}


def make_refs(folder, strategies=STRATEGIES, speaker="ex02", write=True):
    refs = {}
    for s in strategies:
        if write:
            sf.write(str(Path(folder) / f"{s}.wav"), np.full(REF_LEN[s], 0.05, dtype=np.float32), SR,
                     subtype="PCM_16")
        refs[s] = {"file": f"{s}.wav", "text": f"{s} reference words", "speaker": speaker, "style": "default",
                   "source_id": f"{speaker}_default_0000{STRATEGIES.index(s)}", "seconds": REF_LEN[s] / SR}
    return refs


@pytest.fixture
def voice(tmp_path):
    return CSMVoice(StubTTS(), make_refs(tmp_path), tmp_path)


@pytest.fixture
def no_audio(monkeypatch):
    """Playback that plays nothing; returns the list of played clip lengths."""
    played = []
    monkeypatch.setattr(tts_gpu, "_play_blocking", lambda audio, sr: played.append(audio.size))
    monkeypatch.setattr(tts_gpu, "_stop_playback", lambda: None)
    return played


@pytest.fixture
def fake_pyttsx3(monkeypatch):
    """A pyttsx3 that records what it would say; set .fail = True to make init() raise."""
    mod = types.SimpleNamespace(said=[], fail=False, order=None)

    class Engine:
        def say(self, text):
            mod.said.append(text)

        def runAndWait(self):
            pass

        def stop(self):
            pass

    def init():
        if mod.order is not None:
            mod.order.append("pyttsx3")
        if mod.fail:
            raise RuntimeError("no SAPI5 voice (stub)")
        return Engine()

    mod.init = init
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)
    return mod


# ---------------------------------------------------------------- split_sentences

@pytest.mark.parametrize("text, expected", [
    ("Hi there. How are you? I'm fine!", ["Hi there.", "How are you?", "I'm fine!"]),
    ("Dr. Smith said 3.5 is fine. Okay.", ["Dr. Smith said 3.5 is fine.", "Okay."]),
    ("Use e.g. a timer, i.e. a clock. Then rest.", ["Use e.g. a timer, i.e. a clock.", "Then rest."]),
    ("Mr. and Mrs. Lee vs. Prof. Ng.", ["Mr. and Mrs. Lee vs. Prof. Ng."]),
    ("Well... I think so. Maybe.", ["Well...", "I think so.", "Maybe."]),
    ("Wait… really?! Yes.", ["Wait…", "really?!", "Yes."]),
    ('"Wow!" she said. It costs $4.99 today.', ['"Wow!"', "she said.", "It costs $4.99 today."]),
    ("Version 2.0 is out", ["Version 2.0 is out"]),
    ("One. Two", ["One.", "Two"]),
    ("Line one\nLine two.", ["Line one", "Line two."]),
    ("  lots   of\tspace .  ", ["lots of space ."]),
    ("", []),
    ("   ", []),
    (None, []),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


def test_abbreviations_match_llm_sentence_buffer():
    import llm_gpu

    assert tts_gpu.ABBREVIATIONS == llm_gpu.ABBREVIATIONS
    text = "Dr. Lee said 3.5 hours. Rest now! Then e.g. read? Ok then."
    buf = llm_gpu.SentenceBuffer(min_words=1)
    assert split_sentences(text) == buf.feed(text) + buf.flush()


# ---------------------------------------------------------------- strip_tags

@pytest.mark.parametrize("text, expected", [
    ("<sigh> I'm so sorry. <softly>That sounds hard.", "I'm so sorry. That sounds hard."),
    ("You did it <laughs>!", "You did it!"),
    ("<brightly>Great job</brightly>.", "Great job."),
    ('Pause <break time="1s"/> here.', "Pause here."),
    ("Keep x < 5 and y > 3.", "Keep x < 5 and y > 3."),
    ("I <3 you.", "I <3 you."),
    ("No tags at all.", "No tags at all."),
    ("<sigh>", ""),
    (None, ""),
])
def test_strip_tags(text, expected):
    assert strip_tags(text) == expected


def test_tags_flag_is_off_in_every_preset():
    for preset in list_presets():
        assert load_config(preset)["tts"]["tags_enabled"] is False


def test_tags_stripped_before_csm_unless_enabled(tmp_path):
    off = CSMVoice(StubTTS(), make_refs(tmp_path), tmp_path)
    off.speak_sentence("<sigh> I'm sorry.", "VALIDATE")
    assert off.tts.calls[-1]["text"] == "I'm sorry."
    on = CSMVoice(StubTTS(), make_refs(tmp_path, write=False), tmp_path, tags_enabled=True)
    on.speak_sentence("<sigh> I'm sorry.", "VALIDATE")
    assert on.tts.calls[-1]["text"] == "<sigh> I'm sorry."


def test_tag_only_sentence_skips_csm(voice):
    audio, sr = voice.speak_sentence("<sigh>", "VALIDATE")
    assert audio.size == 0 and audio.dtype == np.float32 and sr == SR
    assert voice.tts.calls == []


# ---------------------------------------------------------------- strategy -> reference

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_each_strategy_uses_its_own_reference(voice, strategy):
    audio, sr = voice.speak_sentence("Hello there friend.", strategy)
    call = voice.tts.calls[-1]
    assert call["ref_text"] == f"{strategy} reference words"
    assert call["ref_len"] == REF_LEN[strategy]
    assert audio.dtype == np.float32 and audio.ndim == 1 and sr == SR
    assert voice.last_sentence["ref_strategy"] == strategy


def test_reference_audio_is_resampled_to_the_tts_rate(tmp_path):
    refs = make_refs(tmp_path, write=False)
    for s in STRATEGIES:                                    # 48 kHz stereo, like raw Expresso
        sf.write(str(tmp_path / f"{s}.wav"), np.full((48000, 2), 0.05, dtype=np.float32), 48000)
    v = CSMVoice(StubTTS(), refs, tmp_path)
    assert all(abs(r["audio"].size - SR) <= 2 and r["audio"].ndim == 1 for r in v.refs.values())


def test_unknown_strategy(voice):
    with pytest.raises(ValueError, match="unknown strategy"):
        voice.speak_sentence("Hello.", "HAPPY")


# ---------------------------------------------------------------- missing references

def test_missing_strategy_raises_clear_error(tmp_path):
    v = CSMVoice(StubTTS(), make_refs(tmp_path, strategies=("VALIDATE", "ENCOURAGE", "NEUTRAL")), tmp_path)
    with pytest.raises(VoiceRefsError, match=r"no reference clip for CALM.*--set-ref CALM"):
        v.speak_sentence("Breathe.", "CALM")
    assert v.tts.calls == []
    v.speak_sentence("Nice work.", "ENCOURAGE")              # the others still speak
    assert v.tts.calls[-1]["ref_text"] == "ENCOURAGE reference words"


def test_missing_clip_file_raises_clear_error(tmp_path):
    refs = make_refs(tmp_path)
    (tmp_path / "CALM.wav").unlink()
    v = CSMVoice(StubTTS(), refs, tmp_path)
    with pytest.raises(VoiceRefsError, match=r"CALM.*not found"):
        v.speak_sentence("Breathe.", "CALM")


def test_refused_reference_sets(tmp_path):
    with pytest.raises(VoiceRefsError, match="no usable reference"):
        CSMVoice(StubTTS(), {}, tmp_path)
    refs = make_refs(tmp_path)
    refs["CALM"]["speaker"] = "ex03"
    with pytest.raises(VoiceRefsError, match="one speaker"):
        CSMVoice(StubTTS(), refs, tmp_path)
    refs = make_refs(tmp_path)
    del refs["NEUTRAL"]["text"]
    with pytest.raises(VoiceRefsError, match="missing"):
        CSMVoice(StubTTS(), refs, tmp_path)


def test_real_voice_refs_json_passes_the_check(tmp_path):
    import json

    data = json.loads((CODE_DIR / "gpu" / "voice_refs.json").read_text(encoding="utf-8"))
    for s in STRATEGIES:
        (tmp_path / data[s]["file"]).write_bytes(b"RIFF")
    present, missing = tts_gpu.check_voice_refs(data, tmp_path)
    assert set(present) == set(STRATEGIES) and missing == {}


# ---------------------------------------------------------------- crisis -> NEUTRAL

def test_ref_strategy():
    assert [ref_strategy(s) for s in STRATEGIES] == list(STRATEGIES)
    assert [ref_strategy(s, crisis=True) for s in STRATEGIES] == ["NEUTRAL"] * 4
    assert ref_strategy("not-a-strategy", crisis=True) == "NEUTRAL"   # a crisis turn never fails on the name
    with pytest.raises(ValueError):
        ref_strategy("not-a-strategy")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_crisis_forces_neutral_reference(voice, strategy):
    voice.speak_sentence("Please reach out to someone.", strategy, crisis=True)
    assert voice.tts.calls[-1]["ref_text"] == "NEUTRAL reference words"
    assert voice.tts.calls[-1]["ref_len"] == REF_LEN["NEUTRAL"]


def test_crisis_stream_and_fallback_use_neutral(voice, no_audio):
    list(voice.speak_stream("You matter. Please call someone now.", "ENCOURAGE", crisis=True))
    assert {c["ref_text"] for c in voice.tts.calls} == {"NEUTRAL reference words"}
    assert voice.last_stream["ref_strategy"] == "NEUTRAL" and voice.last_stream["crisis"] is True
    rec = speak_with_fallback("You matter.", "VALIDATE", voice=voice, crisis=True)
    assert rec["path"] == "csm" and voice.tts.calls[-1]["ref_text"] == "NEUTRAL reference words"


def test_crisis_without_neutral_reference_raises(tmp_path):
    v = CSMVoice(StubTTS(), make_refs(tmp_path, strategies=("VALIDATE", "CALM", "ENCOURAGE")), tmp_path)
    with pytest.raises(VoiceRefsError, match=r"NEUTRAL \(crisis turns always use NEUTRAL\)"):
        v.speak_sentence("Please reach out.", "VALIDATE", crisis=True)


# ---------------------------------------------------------------- speak_stream

def test_stream_yields_one_clip_per_sentence_and_times_it(voice, no_audio):
    out = list(voice.speak_stream("First one here. Second one here. Third one here.", "CALM"))
    assert len(out) == 3 and all(sr == SR and a.dtype == np.float32 for a, sr in out)
    assert [c["text"] for c in voice.tts.calls] == ["First one here.", "Second one here.", "Third one here."]
    st = voice.last_stream
    assert st["generated"] == 3 and st["played"] == 3 and no_audio == [2400] * 3
    assert st["time_to_first_audio"] is not None and st["time_to_first_audio"] >= 0
    assert st["audio_seconds"] == pytest.approx(0.3)
    assert st["rtf"] == pytest.approx(st["gen_seconds"] / st["audio_seconds"], abs=1e-3)


def test_stream_accepts_an_iterable_and_skips_tag_only_sentences(voice, no_audio):
    out = list(voice.speak_stream(iter(["<sigh>", "Okay then."]), "VALIDATE"))
    assert len(out) == 1 and voice.last_stream["generated"] == 1


def test_stream_no_play_never_touches_playback(voice, monkeypatch):
    def boom(*a):
        raise AssertionError("played with play=False")

    monkeypatch.setattr(tts_gpu, "_play_blocking", boom)
    assert len(list(voice.speak_stream("One. Two.", "NEUTRAL", play=False))) == 2
    assert voice.last_stream["played"] == 0


def test_stream_generates_next_sentence_while_previous_plays(tmp_path, monkeypatch):
    gen2_started = threading.Event()

    class Tts(StubTTS):
        def synth(self, text, ref_text, ref_audio_24k):
            if text.startswith("Second"):
                gen2_started.set()
            return super().synth(text, ref_text, ref_audio_24k)

    seen = []

    def play(audio, sr):
        if not seen:                       # still playing clip 1: has clip 2 started generating?
            seen.append(gen2_started.wait(timeout=5))
        else:
            seen.append(True)

    monkeypatch.setattr(tts_gpu, "_play_blocking", play)
    v = CSMVoice(Tts(), make_refs(tmp_path), tmp_path)
    list(v.speak_stream("First one here. Second one here.", "NEUTRAL"))
    assert seen == [True, True]


def test_stream_closed_early_stops_playback(voice, monkeypatch):
    gate, stopped = threading.Event(), []
    monkeypatch.setattr(tts_gpu, "_play_blocking", lambda a, sr: gate.wait(timeout=5))
    monkeypatch.setattr(tts_gpu, "_stop_playback", lambda: (stopped.append(1), gate.set()))
    gen = voice.speak_stream("One here. Two here. Three here.", "NEUTRAL")
    next(gen)
    next(gen)
    gen.close()
    assert stopped == [1]
    assert len(voice.tts.calls) == 2                 # the third sentence was never generated
    assert voice.last_stream["played"] <= 1          # the queued second clip was dropped


def test_stream_playback_error_is_raised(voice, monkeypatch):
    def broken(audio, sr):
        raise OSError("no output device (stub)")

    monkeypatch.setattr(tts_gpu, "_play_blocking", broken)
    with pytest.raises(RuntimeError, match="playback failed"):
        list(voice.speak_stream("One here. Two here.", "NEUTRAL"))


# ---------------------------------------------------------------- fallback chain

def test_chain_csm_ok_never_calls_fallback(voice, no_audio, monkeypatch):
    monkeypatch.setattr(fallback_tts, "speak", lambda t: pytest.fail("fallback called"))
    rec = speak_with_fallback("All good. Talk soon.", "NEUTRAL", voice=voice)
    assert rec["path"] == "csm" and rec["csm_sentences"] == 2 and rec["error"] is None
    assert rec["stats"]["generated"] == 2


def test_chain_csm_fails_then_pyttsx3(fake_pyttsx3, caplog):
    class BrokenVoice:
        def speak_stream(self, *a, **k):
            raise RuntimeError("CSM crashed (stub)")

    rec = speak_with_fallback("<sigh> I'm sorry. That is hard.", "VALIDATE", voice=BrokenVoice())
    assert rec["path"] == "pyttsx3" and "CSM crashed" in rec["error"]
    assert fake_pyttsx3.said == ["I'm sorry. That is hard."]            # tags stripped for pyttsx3 too
    assert "path=pyttsx3" in caplog.text and "fallback" in caplog.text


def test_chain_order_csm_then_pyttsx3_then_text(fake_pyttsx3, capsys):
    order = []
    fake_pyttsx3.order, fake_pyttsx3.fail = order, True

    class BrokenVoice:
        def speak_stream(self, *a, **k):
            order.append("csm")
            raise RuntimeError("CSM crashed (stub)")

    rec = speak_with_fallback("Take a breath.", "CALM", voice=BrokenVoice())
    out = capsys.readouterr().out
    if "Take a breath." in out:
        order.append("text")
    assert order == ["csm", "pyttsx3", "text"]
    assert rec["path"] == "text"


def test_chain_no_voice_goes_straight_to_fallback(fake_pyttsx3):
    rec = speak_with_fallback("Hello.", "NEUTRAL", voice=None)
    assert rec["path"] == "pyttsx3" and rec["error"] == "no CSMVoice loaded"
    assert fake_pyttsx3.said == ["Hello."]


def test_chain_missing_reference_falls_back(tmp_path, fake_pyttsx3, no_audio):
    v = CSMVoice(StubTTS(), make_refs(tmp_path, strategies=("NEUTRAL",)), tmp_path)
    rec = speak_with_fallback("Breathe in.", "CALM", voice=v)
    assert rec["path"] == "pyttsx3" and "VoiceRefsError" in rec["error"] and "CALM" in rec["error"]
    assert fake_pyttsx3.said == ["Breathe in."]


@pytest.mark.parametrize("play", [False, True])
def test_chain_mid_reply_failure_speaks_only_the_rest(tmp_path, fake_pyttsx3, no_audio, play):
    v = CSMVoice(StubTTS(fail_on="Second"), make_refs(tmp_path), tmp_path)
    rec = speak_with_fallback("First one here. Second one here. Third one here.", "NEUTRAL", voice=v, play=play)
    assert rec["path"] == "pyttsx3" and rec["csm_sentences"] == 1
    assert fake_pyttsx3.said == ["Second one here. Third one here."]
    assert no_audio == ([2400] if play else [])


def test_chain_playback_failure_repeats_nothing_unplayed(voice, fake_pyttsx3, monkeypatch):
    def broken(audio, sr):
        raise OSError("no output device (stub)")

    monkeypatch.setattr(tts_gpu, "_play_blocking", broken)
    rec = speak_with_fallback("One here. Two here.", "NEUTRAL", voice=voice)
    assert rec["path"] == "pyttsx3" and rec["csm_sentences"] == 0
    assert fake_pyttsx3.said == ["One here. Two here."]


def test_chain_empty_text(voice):
    assert speak_with_fallback("   ", "NEUTRAL", voice=voice)["path"] == "none"
    assert voice.tts.calls == []


# ---------------------------------------------------------------- scripts/fallback_tts.py

def test_fallback_tts_paths(fake_pyttsx3, capsys, monkeypatch):
    assert fallback_tts.speak("Hello   there.") == "pyttsx3" and fake_pyttsx3.said == ["Hello there."]
    fake_pyttsx3.fail = True
    assert fallback_tts.speak("Printed instead.") == "text"
    assert "Printed instead." in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "pyttsx3", None)               # pyttsx3 not installed
    assert fallback_tts.speak("Still printed.") == "text"
    assert "Still printed." in capsys.readouterr().out
    assert fallback_tts.speak("") == "none" and fallback_tts.speak(None) == "none"


def test_fallback_tts_never_raises(monkeypatch):
    class BadStr:
        def __str__(self):
            raise ValueError("cannot be a string")

    monkeypatch.setitem(sys.modules, "pyttsx3", None)
    assert fallback_tts.speak(BadStr()) == "none"
    monkeypatch.setattr("builtins.print", lambda *a, **k: (_ for _ in ()).throw(UnicodeEncodeError(
        "cp1252", "x", 0, 1, "stub")))
    assert fallback_tts.speak("emoji \U0001F600") == "text"


def test_fallback_tts_imports_no_torch_and_nothing_from_gpu():
    path = CODE_DIR / "scripts" / "fallback_tts.py"
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    gpu_modules = {p.stem for p in (CODE_DIR / "gpu").glob("*.py")}
    assert "torch" not in imported and not imported & gpu_modules, imported
    # and at run time: importing it in a fresh interpreter loads no torch
    code = (f"import sys; sys.path.insert(0, {str(CODE_DIR / 'scripts')!r}); import fallback_tts; "
            "print('torch' in sys.modules, 'device' in sys.modules)")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"]


# ---------------------------------------------------------------- CLI / CSV / GPU guard (no model)

def test_fallback_test_cli_uses_only_the_fallback(monkeypatch):
    said = []
    monkeypatch.setattr(fallback_tts, "speak", lambda t: said.append(t) or "pyttsx3")
    monkeypatch.setattr(tts_gpu.TTS, "from_preset", classmethod(lambda *a, **k: pytest.fail("CSM loaded")))
    monkeypatch.setattr(sys, "argv", ["tts_gpu.py", "--fallback-test"])
    assert tts_gpu._main() == 0
    assert said == [tts_gpu.FALLBACK_TEST_LINE]


def test_speak_cli_needs_strategy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tts_gpu.py", "--speak", "Hello."])
    with pytest.raises(SystemExit):
        tts_gpu._main()


def test_csm_voice_from_preset_refuses_cpu(tmp_path, monkeypatch):
    import explore_expresso

    monkeypatch.setattr(explore_expresso, "load_env", lambda: None)
    monkeypatch.setenv("GPU_OUTPUTS_DIR", str(tmp_path))
    (tmp_path / "voice_refs").mkdir()
    refs = make_refs(tmp_path / "voice_refs")
    json_path = tmp_path / "voice_refs.json"
    json_path.write_text(__import__("json").dumps(refs), encoding="utf-8")
    with pytest.raises(RuntimeError, match="GPU only"):
        CSMVoice.from_preset("smoke_cpu", json_path=json_path)
    with pytest.raises(VoiceRefsError, match="not found"):
        CSMVoice.from_preset("gpu_24gb", json_path=tmp_path / "nope.json")


def test_speak_csv_rows_and_never_overwrite(voice, no_audio, tmp_path):
    list(voice.speak_stream("One here. Two here.", "CALM", play=False))
    meta = {"run_at": "2026-11-13 12:00:00", "preset": "gpu_24gb", "model": "sesame/csm-1b", "dtype": "bfloat16",
            "gpu": "stub", "peak_vram_gb": None}
    rows = tts_gpu.speak_csv_rows(voice.last_stream, meta)
    assert [r["sentence"] for r in rows] == [1, 2] and rows[0]["text"] == "One here."
    assert set(rows[0]) == set(tts_gpu.SPEAK_CSV_FIELDS)
    assert rows[0]["time_to_first_audio"] == voice.last_stream["time_to_first_audio"]
    first = tts_gpu.write_speak_csv(rows, tmp_path / "tts_speak_csm.csv")
    second = tts_gpu.write_speak_csv(rows, tmp_path / "tts_speak_csm.csv")
    assert first.name == "tts_speak_csm.csv" and second.name.startswith("tts_speak_csm_") and first != second
    assert tts_gpu.SPEAK_CSV == "results/metrics/tts_speak_csm.csv"


# ---------------------------------------------------------------- P6 and signatures

def test_speaking_api_takes_no_emotion_inputs():
    for fn in (CSMVoice.speak_sentence, CSMVoice.speak_stream, speak_with_fallback):
        for p in inspect.signature(fn).parameters:
            assert not any(w in p.lower() for w in ("prob", "emotion", "label", "mood", "confidence")), (fn, p)
    assert list(inspect.signature(CSMVoice.speak_sentence).parameters)[:3] == ["self", "text", "strategy"]
    assert list(inspect.signature(speak_with_fallback).parameters)[:3] == ["text", "strategy", "voice"]
