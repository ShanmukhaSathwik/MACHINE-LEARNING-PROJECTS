"""
test_asr_gpu.py - step 6 of the GPU demo build (gpu/asr_gpu.py).
Every test uses a stub model, except one integration test that runs only when openai/whisper-tiny.en
is already in the Hugging Face cache and RAVDESS is in data/raw (it never downloads).
"""
import ast
import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import asr_gpu
from asr_gpu import ASR, build_generate_kwargs, normalize_for_wer, ravdess_statement, wer
from device import load_config

CODE_DIR = Path(__file__).resolve().parent.parent
SR = asr_gpu.SAMPLE_RATE


class StubPipe:
    """Stands in for the transformers pipeline: records every call, returns fixed text."""

    def __init__(self, text="  hello there  "):
        self.text = text
        self.calls = []

    def __call__(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return {"text": self.text}


def make_asr(preset="smoke_cpu", model_name=None):
    asr = ASR(load_config(preset), "cpu", torch.float32, model_name=model_name)
    asr.pipe = StubPipe()
    return asr


def tone(seconds=1.0, sr=SR, amp=0.3, freq=220.0):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------- generate kwargs

def test_generate_kwargs_english_only_model():
    cfg = load_config("smoke_cpu")
    kw = build_generate_kwargs("openai/whisper-tiny.en", cfg)
    assert kw == {"num_beams": 1}
    assert "language" not in kw and "task" not in kw


def test_generate_kwargs_multilingual_model():
    cfg = load_config("gpu_16gb")
    kw = build_generate_kwargs("openai/whisper-large-v3-turbo", cfg)
    assert kw == {"num_beams": 5, "language": "en", "task": "transcribe"}


def test_generate_kwargs_beam_from_config():
    cfg = load_config("smoke_cpu")
    cfg["asr"]["beam_size"] = 3
    assert build_generate_kwargs("openai/whisper-base", cfg)["num_beams"] == 3


# ---------------------------------------------------------------- silence gate

@pytest.mark.parametrize("y", [
    np.zeros(SR, dtype=np.float32),
    np.random.default_rng(0).normal(0, 1e-4, SR).astype(np.float32),   # about -80 dBFS
])
def test_silence_is_skipped_and_model_never_called(y):
    asr = make_asr()
    out = asr.transcribe(y, sr=SR)
    assert out["skipped"] is True and out["reason"] == "silence" and out["text"] == ""
    assert asr.pipe.calls == []


def test_short_burst_below_min_speech_is_silence():
    asr = make_asr()
    y = np.zeros(SR, dtype=np.float32)
    y[:int(0.1 * SR)] = tone(0.1)          # 0.1 s of sound < min_speech_seconds 0.3
    assert asr.transcribe(y, sr=SR)["reason"] == "silence"
    assert asr.pipe.calls == []


def test_loud_signal_is_transcribed():
    asr = make_asr()
    out = asr.transcribe(tone(1.0), sr=SR)
    assert out["skipped"] is False and out["reason"] is None
    assert out["text"] == "hello there"                    # stripped
    assert len(asr.pipe.calls) == 1
    inputs, kwargs = asr.pipe.calls[0]
    assert inputs["sampling_rate"] == SR
    assert kwargs["generate_kwargs"] == {"num_beams": 1}
    assert "chunk_length_s" not in kwargs
    assert set(out) == {"text", "skipped", "reason", "audio_seconds", "asr_seconds", "model"}
    assert out["model"] == "openai/whisper-tiny.en"
    assert out["audio_seconds"] == pytest.approx(1.0)


def test_long_audio_uses_30s_chunking():
    asr = make_asr()
    asr.transcribe(tone(31.0), sr=SR)
    assert asr.pipe.calls[0][1]["chunk_length_s"] == 30


def test_warm_up_bypasses_silence_gate():
    asr = make_asr()
    asr.warm_up()
    assert len(asr.pipe.calls) == 1
    assert asr.pipe.calls[0][0]["raw"].size == SR


def test_transcribe_before_load_is_a_clear_error():
    asr = ASR(load_config("smoke_cpu"), "cpu", torch.float32)
    with pytest.raises(RuntimeError, match="load"):
        asr.transcribe(tone(1.0), sr=SR)


def test_model_name_override():
    asr = make_asr(model_name="openai/whisper-base")
    assert asr.model_name == "openai/whisper-base"
    assert asr.generate_kwargs["language"] == "en"


# ---------------------------------------------------------------- inputs

def test_path_and_array_give_the_same_audio(tmp_path):
    y = tone(1.0)
    wav = tmp_path / "clip.wav"
    sf.write(wav, y, SR, subtype="FLOAT")
    asr = make_asr()
    out_path = asr.transcribe(wav)
    out_str = asr.transcribe(str(wav))
    out_arr = asr.transcribe(y, sr=SR)
    assert not out_path["skipped"] and not out_str["skipped"] and not out_arr["skipped"]
    a, b, c = (call[0]["raw"] for call in asr.pipe.calls)
    np.testing.assert_allclose(a, c, atol=1e-6)
    np.testing.assert_allclose(b, c, atol=1e-6)


@pytest.mark.parametrize("layout", ["frames_channels", "channels_frames"])
def test_stereo_becomes_mono(layout):
    left, right = tone(1.0, amp=0.4), tone(1.0, amp=0.2)
    stereo = np.stack([left, right], axis=1 if layout == "frames_channels" else 0)
    asr = make_asr()
    asr.transcribe(stereo, sr=SR)
    raw = asr.pipe.calls[0][0]["raw"]
    assert raw.ndim == 1 and raw.dtype == np.float32
    np.testing.assert_allclose(raw, (left + right) / 2, atol=1e-6)


def test_48k_is_resampled_to_16k(tmp_path):
    asr = make_asr()
    out = asr.transcribe(tone(1.0, sr=48000), sr=48000)
    raw = asr.pipe.calls[0][0]["raw"]
    assert abs(raw.size - SR) <= 2
    assert out["audio_seconds"] == pytest.approx(1.0, abs=1e-3)
    wav = tmp_path / "clip48k.wav"                      # same through a file
    sf.write(wav, tone(1.0, sr=48000), 48000)
    asr.transcribe(wav)
    assert abs(asr.pipe.calls[1][0]["raw"].size - SR) <= 2


def test_int16_array_is_scaled():
    asr = make_asr()
    pcm = (tone(1.0, amp=0.5) * 32767).astype(np.int16)
    asr.transcribe(pcm, sr=SR)
    assert np.abs(asr.pipe.calls[0][0]["raw"]).max() == pytest.approx(0.5, abs=1e-3)


def test_empty_array_is_skipped_empty():
    asr = make_asr()
    out = asr.transcribe(np.zeros(0, dtype=np.float32), sr=SR)
    assert out["skipped"] is True and out["reason"] == "empty" and out["text"] == ""
    assert asr.pipe.calls == []


def test_array_without_sr_is_a_clear_error():
    asr = make_asr()
    with pytest.raises(ValueError, match="sr is required"):
        asr.transcribe(tone(1.0))


# ---------------------------------------------------------------- WER helpers

def test_wer_values():
    assert wer("kids are talking by the door", "kids are talking by the door") == 0.0
    assert wer("dogs are sitting by the door", "dogs are standing by the door") == pytest.approx(1 / 6)
    assert wer("one two three four five", "one two six four five") == pytest.approx(0.2)
    assert wer("one two three four five", "") == 1.0
    assert wer("Kids are talking by the door.", "kids, are talking by the DOOR") == 0.0
    assert wer("", "") == 0.0


def test_normalize_for_wer():
    assert normalize_for_wer("Kids ARE talking, by the door!") == "kids are talking by the door"
    assert normalize_for_wer("  Don't   stop.  ") == "dont stop"


def test_ravdess_statement():
    assert ravdess_statement("03-01-01-01-01-01-16.wav") == "kids are talking by the door"
    assert ravdess_statement("data/raw/Actor_12/03-01-05-02-02-01-12.wav") == "dogs are sitting by the door"
    with pytest.raises(ValueError):
        ravdess_statement("03_angry.wav")


def test_ravdess_clips_balanced_and_seeded():
    manifest = CODE_DIR / asr_gpu.MANIFEST
    if not manifest.exists():
        pytest.skip("manifest not present")
    with open(manifest, newline="", encoding="utf-8") as f:
        first = next(r for r in csv.DictReader(f) if r["corpus"] == "ravdess" and r["split"] == "test")
    if not (CODE_DIR / first["wav_path"]).exists():
        pytest.skip("RAVDESS audio not in data/raw")
    a, b = asr_gpu.ravdess_test_clips(24), asr_gpu.ravdess_test_clips(24)
    assert a == b
    assert [sum(e == c for _, e in a) for c in asr_gpu.CLASSES] == [6, 6, 6, 6]
    assert len({p for p, _ in a}) == 24


def test_new_csv_path_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(asr_gpu, "code_path", lambda rel: tmp_path / rel)
    (tmp_path / "results" / "metrics").mkdir(parents=True)
    first = asr_gpu.new_csv_path("ravdess", "openai/whisper-tiny.en")
    assert first.name == "asr_ravdess_whisper-tiny.en.csv"
    first.write_text("x")
    second = asr_gpu.new_csv_path("ravdess", "openai/whisper-tiny.en")
    assert second != first and second.name.startswith("asr_ravdess_whisper-tiny.en_")


def test_find_sentence_list(tmp_path):
    assert asr_gpu.find_sentence_list(tmp_path) == ({}, None)
    (tmp_path / "sentences.csv").write_text("file,text\n01_happy,I got the job\n", encoding="utf-8")
    mapping, path = asr_gpu.find_sentence_list(tmp_path)
    assert mapping == {"01_happy.wav": "I got the job"} and path.name == "sentences.csv"


# ---------------------------------------------------------------- config

def test_config_presets():
    cpu, gpu = load_config("smoke_cpu"), load_config("gpu_16gb")
    assert cpu["asr"]["model"] == "openai/whisper-tiny.en" and cpu["asr"]["beam_size"] == 1
    assert gpu["asr"]["model"] == "openai/whisper-large-v3-turbo" and gpu["asr"]["beam_size"] == 5
    for cfg in (cpu, gpu):
        assert cfg["asr"]["language"] == "en"
        assert cfg["asr"]["silence_dbfs"] == -45
        assert cfg["asr"]["min_speech_seconds"] == 0.3


# ---------------------------------------------------------------- imports

def test_does_not_import_cpu_baseline():
    tree = ast.parse((CODE_DIR / "gpu" / "asr_gpu.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"asr", "emotion_predictor", "feature_extraction", "scripts"}


# ---------------------------------------------------------------- integration (no downloads)

def _tiny_en_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001
        return False
    return all(isinstance(try_to_load_from_cache("openai/whisper-tiny.en", f), str)
               for f in ("config.json", "model.safetensors"))


def _neutral_ravdess_clip():
    manifest = CODE_DIR / asr_gpu.MANIFEST
    if not manifest.exists():
        return None
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["corpus"] == "ravdess" and r["split"] == "test" and r["label"] == "neutral":
                p = CODE_DIR / r["wav_path"]
                return p if p.exists() else None
    return None


def test_integration_tiny_en_on_neutral_ravdess(monkeypatch):
    clip = _neutral_ravdess_clip()
    if not _tiny_en_cached() or clip is None:
        pytest.skip("needs openai/whisper-tiny.en in the HF cache and RAVDESS in data/raw")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")   # never download from a test
    asr = ASR.from_preset("smoke_cpu")
    out = asr.transcribe(clip)
    assert out["skipped"] is False
    assert wer(ravdess_statement(clip.name), out["text"]) <= 0.2, out["text"]
