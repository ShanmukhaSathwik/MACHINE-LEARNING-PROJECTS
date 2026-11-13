"""
test_bench_gpu.py - step 10 (gpu/bench_gpu.py, gpu/bench_turns.yaml). CPU only: made-up timings, stub parts,
fake torch. No model, no network.
"""
import csv
import datetime
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import bench_gpu as bg
import safety
from device import load_config

CODE_DIR = Path(__file__).resolve().parent.parent
EXPECT_GPU = {"llm_backend": "qwen", "voice": "csm"}


# ---------------------------------------------------------------- summary / median logic

def run_row(turn, p, **vals):
    row = {"row_type": "run", "pass": p, "turn": turn, "label": f"t{turn}", "status": "ok",
           "voice_path": "csm", "llm_backend": "qwen", "fallback_flags": ""}
    row.update({f: None for f in bg.NUMERIC})
    row.update(vals)
    return row


def test_summary_median_and_spread_with_made_up_timings():
    rows = [{**run_row(1, 0, asr_s=9.0, turn_total_s=99.0), "row_type": "warmup"},   # discarded
            run_row(1, 1, asr_s=0.10, turn_total_s=2.0, turn_first_audio_s=1.0),
            run_row(1, 2, asr_s=0.30, turn_total_s=4.0, turn_first_audio_s=None),      # missing value ignored
            run_row(1, 3, asr_s=0.20, turn_total_s=3.0, turn_first_audio_s=1.4),
            run_row(2, 1, asr_s=0.5), run_row(2, 2, asr_s=0.7)]
    s = {x["turn"]: x for x in bg.summarize(rows)}
    assert s[1]["n"] == 3
    assert s[1]["asr_s"] == {"median": 0.2, "min": 0.1, "max": 0.3}
    assert s[1]["turn_total_s"] == {"median": 3.0, "min": 2.0, "max": 4.0}
    assert s[1]["turn_first_audio_s"] == {"median": 1.2, "min": 1.0, "max": 1.4}             # median of 2 values
    assert s[1]["emotion_s"] is None
    assert s[2]["asr_s"]["median"] == pytest.approx(0.6)


def test_turn_first_audio_is_stages_before_tts_plus_voice_ttfa():
    t = {"consent": 0.01, "asr": 0.2, "safety": 0.001, "emotion": 0.1, "abstain": 0.0, "mood": 0.002,
         "memory": 0.01, "prompt": 0.001, "llm": 0.5, "tts": 3.0, "extract": 0.4, "total": 4.3}
    assert bg.turn_first_audio(t, 0.6) == pytest.approx(0.01 + 0.2 + 0.001 + 0.1 + 0.002 + 0.01 + 0.001 + 0.5 + 0.6)
    crisis = {"consent": 0.0, "asr": 0.2, "safety": 0.001, "tts": 2.0, "total": 2.3}
    assert bg.turn_first_audio(crisis, 0.4) == pytest.approx(0.601)
    assert bg.turn_first_audio(t, None) is None and bg.turn_first_audio({}, 0.5) is None


# ---------------------------------------------------------------- fallback flags

@pytest.mark.parametrize("row, flags", [
    ({"ser_fallback": False, "llm_backend": "qwen", "voice_path": "csm"}, []),
    ({"ser_fallback": True, "llm_backend": "qwen", "voice_path": "csm"}, ["SER: T3 -> pretrained"]),
    ({"ser_fallback": False, "llm_backend": "gemini", "voice_path": "csm"}, ["LLM: qwen -> gemini"]),
    ({"ser_fallback": False, "llm_backend": "template", "voice_path": "pyttsx3"},
     ["LLM: qwen -> template", "voice: csm -> pyttsx3"]),
    ({"ser_fallback": False, "llm_backend": "qwen", "llm_fell_back": True, "voice_path": "text"},
     ["LLM: model switch inside qwen", "voice: csm -> text"]),
    ({"ser_fallback": False, "llm_backend": None, "voice_path": "csm"}, []),      # crisis / delete turns: no LLM
])
def test_fallback_flags(row, flags):
    assert bg.fallback_flags(row, EXPECT_GPU) == flags


def test_pyttsx3_is_not_a_fallback_when_it_is_the_configured_voice():
    assert bg.fallback_flags({"voice_path": "pyttsx3", "llm_backend": "gemini"},
                             {"llm_backend": "gemini", "voice": "pyttsx3"}) == []


# ---------------------------------------------------------------- file naming / GPU memory / imports

def test_bench_file_name_never_overwrites(tmp_path):
    when = datetime.datetime(2026, 11, 13, 9, 30, 5)
    first = bg.new_bench_path("gpu_24gb", when, tmp_path)
    assert first.name == "bench_gpu_24gb_20261113_093005.csv"
    first.write_text("x")
    second = bg.new_bench_path("gpu_24gb", when, tmp_path)
    second.write_text("y")
    third = bg.new_bench_path("gpu_24gb", when, tmp_path)
    assert (second.name, third.name) == ("bench_gpu_24gb_20261113_093005_2.csv", "bench_gpu_24gb_20261113_093005_3.csv")
    assert first.read_text() == "x"


def fake_torch(cuda, peak_bytes=0):
    cuda_ns = types.SimpleNamespace(is_available=lambda: cuda, max_memory_allocated=lambda: peak_bytes)
    return types.SimpleNamespace(cuda=cuda_ns)


def test_gpu_memory_is_na_on_cpu_and_measured_on_cuda():
    assert bg.peak_gpu_gb(fake_torch(False)) == "n/a"
    assert bg.peak_gpu_gb(fake_torch(True, 3 * 1024 ** 3)) == 3.0


def test_cpu_run_writes_na_for_gpu_memory(tmp_path):
    rows = [run_row(1, 1, asr_s=0.1)]
    path = bg.write_csv(tmp_path / "b.csv", rows, bg.summarize(rows), "n/a", "n/a", "smoke_cpu")
    df = pd.read_csv(path, keep_default_na=False)
    assert set(df["gpu_peak_after_load_gb"]) == {"n/a"} and set(df["gpu_peak_after_run_gb"]) == {"n/a"}
    assert list(df["row_type"]) == ["run", "median", "min", "max"]


def test_module_import_loads_no_torch():
    code = (f"import sys; sys.path.insert(0, {str(CODE_DIR / 'gpu')!r}); import bench_gpu; "
            "print([m for m in ('torch', 'transformers', 'device', 'asr_gpu', 'ser_gpu', 'llm_gpu', 'tts_gpu') "
            "if m in sys.modules])")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------- bench file and config

def test_bench_turns_file_uses_test_split_clips_only_and_has_crisis_and_delete_cases():
    turns, texts = bg.load_bench_turns(CODE_DIR / "gpu" / "bench_turns.yaml")
    man = pd.read_csv(CODE_DIR / "data" / "manifests" / "ser_4class.csv")
    split = {(CODE_DIR / p).resolve(): s for p, s in zip(man["wav_path"], man["split"])}
    assert all(split[Path(t["wav"]).resolve()] == "test" for t in turns)
    assert not any("my_voice" in str(t["wav"]) for t in turns)
    assert len(texts) == len(turns)
    assert any(t and safety.check_crisis(t)["matched"] for t in texts)
    i = next(k for k, t in enumerate(texts) if t and safety.wants_delete_all(t))
    assert texts[i + 1] and not safety.is_spoken_yes(texts[i + 1])          # the follow-up cancels, never deletes


def test_bench_settings_come_from_config():
    for preset in ("smoke_cpu", "gpu_24gb"):
        b = load_config(preset)["bench"]
        assert isinstance(b["repeats"], int) and b["repeats"] >= 1
        assert b["turns_file"] == "gpu/bench_turns.yaml"


def test_my_voice_clips_are_refused(tmp_path):
    folder = tmp_path / "my_voice_v2"
    folder.mkdir()
    sf.write(str(folder / "a.wav"), np.zeros(160, np.float32), 16000)
    (tmp_path / "b.yaml").write_text(f"turns:\n  - wav: {(folder / 'a.wav').as_posix()}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="my_voice"):
        bg.load_bench_turns(tmp_path / "b.yaml")


# ---------------------------------------------------------------- end to end on stub parts

def test_run_bench_end_to_end_on_stubs(tmp_path):
    for n in ("a.wav", "b.wav"):
        sf.write(str(tmp_path / n), np.zeros(1600, np.float32), 16000)
    (tmp_path / "bench.yaml").write_text(
        "turns:\n  - {wav: a.wav, label: neutral}\n  - {wav: b.wav, label: crisis, text: I want to kill myself.}\n"
        "  - {wav: a.wav, label: del, text: Please delete all my data.}\n  - {wav: a.wav, label: no, text: No.}\n",
        encoding="utf-8")
    turns, texts = bg.load_bench_turns(tmp_path / "bench.yaml")
    cfg = load_config("smoke_cpu")
    parts, ser_info, expect = bg.stub_parts()
    bench, consent = bg.build_bench_agent(cfg, {**parts, "ser_model": ser_info}, tmp_path / "bench")
    rows = bg.run_bench(bench, turns, texts, 2, consent, ser_info, expect, print_fn=lambda *a: None)
    assert [r["row_type"] for r in rows] == ["warmup"] + ["run"] * 8
    assert [r["status"] for r in rows if r["pass"] == 1] == ["ok", "crisis", "delete_confirm", "delete_cancelled"]
    ok = next(r for r in rows if r["pass"] == 1 and r["turn"] == 1)
    assert ok["asr_s"] is not None and ok["llm_first_sentence_s"] is not None and ok["turn_total_s"] > 0
    assert ok["ser_model"] == "stub" and ok["ser_fallback"] is False and ok["llm_backend"] == "stub"
    assert all(r["fallback_flags"] == "" for r in rows)
    # bench data stays in the bench folder: real memory / consent / logs untouched
    assert (tmp_path / "bench" / "consent.json").exists() and (tmp_path / "bench" / "logs").is_dir()
    assert {p.name.split("_")[1] for p in (tmp_path / "bench" / "logs").iterdir()} == {"bench"}
    summary = bg.summarize(rows)
    assert [s["n"] for s in summary] == [2, 2, 2, 2]
    out = bg.write_csv(bg.new_bench_path("smoke_cpu-stub", datetime.datetime.now(), tmp_path), rows, summary,
                       "n/a", "n/a", "smoke_cpu-stub")
    with open(out, encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == len(rows) + 3 * len(summary) and set(written[0]) == set(bg.FIELDS)


def test_text_override_still_runs_the_real_asr():
    calls = []

    class ASR:
        def transcribe(self, y, sr=None):
            calls.append(1)
            return {"text": "real words", "skipped": True, "reason": "silence"}

    w = bg.TextOverrideASR(ASR())
    assert w.transcribe(None, 16000)["text"] == "real words"
    w.override = "scripted words"
    out = w.transcribe(None, 16000)
    assert out == {"text": "scripted words", "skipped": False, "reason": None} and len(calls) == 2


def test_fallback_is_flagged_in_the_printed_summary():
    rows = [run_row(1, 1, voice_path="pyttsx3", fallback_flags="voice: csm -> pyttsx3")]
    out = []
    bg.print_summary(bg.summarize(rows), rows, "n/a", "n/a", print_fn=out.append)
    text = "\n".join(out)
    assert "FALLBACK USED" in text and "voice: csm -> pyttsx3" in text and "n/a (no CUDA)" in text
