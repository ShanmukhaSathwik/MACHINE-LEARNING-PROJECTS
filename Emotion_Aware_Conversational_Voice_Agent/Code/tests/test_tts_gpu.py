"""
test_tts_gpu.py - step 8a of the GPU demo build (gpu/tts_gpu.py), logic only.
No model, no Expresso, no GPU: reference-clip selection, resampling, voice_refs.json validation, P6 checks.
"""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

import tts_gpu
from device import load_config
from tts_gpu import (
    STRATEGIES,
    TTS,
    VoiceRefsError,
    audio_clean,
    build_conversation,
    check_same_speaker,
    max_new_frames,
    prepare_audio,
    select_ref_clips,
    validate_voice_refs,
)

CODE_DIR = Path(__file__).resolve().parent.parent


def row(id_, speaker="ex02", style="default", duration=5.0, text="Some words were spoken here."):
    return {"id": id_, "speaker_id": speaker, "style": style, "duration": duration, "text": text}


def refs(speaker="ex02", **overrides):
    data = {s: {"file": f"{s}.wav", "text": f"{s} text", "speaker": speaker, "style": "default",
                "source_id": f"{speaker}_default_{i:05d}", "seconds": 5.0} for i, s in enumerate(STRATEGIES)}
    data.update(overrides)
    return data


# ---------------------------------------------------------------- reference-clip selection

def test_select_first_by_id_within_4_to_8_seconds():
    rows = [
        row("ex02_default_00005", duration=6.0),
        row("ex02_default_00001", duration=3.9),          # too short
        row("ex02_default_00002", duration=8.1),          # too long
        row("ex02_default_00003", duration=4.0),          # edge: allowed
        row("ex02_default_00004", duration=8.0),          # edge: allowed
    ]
    assert [r["id"] for r in select_ref_clips(rows, "ex02", "default", n=2)] == \
        ["ex02_default_00003", "ex02_default_00004"]
    assert [r["id"] for r in select_ref_clips(rows, "ex02", "default", n=1)] == ["ex02_default_00003"]


def test_select_filters_speaker_style_and_transcript():
    rows = [
        row("ex01_default_00001", speaker="ex01"),
        row("ex02_sad_00001", style="sad"),
        row("ex02_default_00001", text="   "),
        row("ex02_default_00002", duration=None),
        row("ex02_default_00003"),
    ]
    assert [r["id"] for r in select_ref_clips(rows, "ex02", "default", n=5)] == ["ex02_default_00003"]
    assert [r["id"] for r in select_ref_clips(rows, "ex02", "sad", n=5)] == ["ex02_sad_00001"]
    assert select_ref_clips(rows, "ex03", "default") == []


def test_select_calls_is_clean_only_on_eligible_rows():
    rows = [row("ex02_default_00001", duration=2.0), row("ex02_default_00002"), row("ex02_default_00003"),
            row("ex02_default_00004")]
    seen = []

    def is_clean(r):
        seen.append(r["id"])
        return r["id"] != "ex02_default_00002"            # this one is "clipped"

    picked = select_ref_clips(rows, "ex02", "default", n=1, is_clean=is_clean)
    assert [r["id"] for r in picked] == ["ex02_default_00003"]
    assert seen == ["ex02_default_00002", "ex02_default_00003"]   # stops once n are found; never the 2 s clip


def test_shared_expresso_ids_become_unique_and_bare_id_is_refused():
    # as in Expresso: each speaker's 2 longform rows reuse the id of their first default clip
    rows = [row("ex02_default_00001"), row("ex02_default_00001", style="longform", duration=7.2),
            row("ex02_default_00001", style="longform", duration=7.2), row("ex02_default_00002")]
    shared = tts_gpu.make_clip_ids_unique(rows)
    ids = [r["id"] for r in rows]
    assert ids == ["ex02_default_00001_default1", "ex02_default_00001_longform1",
                   "ex02_default_00001_longform2", "ex02_default_00002"]
    assert [r["expresso_id"] for r in rows] == ["ex02_default_00001"] * 3 + ["ex02_default_00002"]
    assert shared == {"ex02_default_00001": ids[:3]}
    by_id = {r["id"]: r for r in rows}
    assert len(by_id) == len(rows)
    assert [r["id"] for r in select_ref_clips(rows, "ex02", "longform", n=2)] == ids[1:3]   # two distinct clips
    with pytest.raises(VoiceRefsError, match="matches 3 Expresso clips"):
        tts_gpu.resolve_clip(by_id, shared, "ex02_default_00001")
    with pytest.raises(VoiceRefsError, match="unknown clip id"):
        tts_gpu.resolve_clip(by_id, shared, "ex09_sad_00001")
    assert tts_gpu.resolve_clip(by_id, shared, "ex02_default_00001_longform2")["style"] == "longform"
    assert tts_gpu.resolve_clip(by_id, shared, "ex02_default_00002")["style"] == "default"


def test_audio_clean():
    sr = 24000
    t = np.arange(sr) / sr
    assert audio_clean(0.1 * np.sin(2 * np.pi * 220 * t), sr) is True
    assert audio_clean(np.clip(1.5 * np.sin(2 * np.pi * 220 * t), -1, 1), sr) is False   # clipped
    assert audio_clean(1e-4 * np.sin(2 * np.pi * 220 * t), sr) is False                  # far too quiet
    assert audio_clean(np.zeros(0), sr) is False


# ---------------------------------------------------------------- resampling

@pytest.mark.parametrize("layout", ["frames_channels", "channels_frames"])
def test_48k_stereo_to_24k_mono(layout):
    sr = 48000
    t = np.arange(sr) / sr
    left, right = 0.4 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 220 * t)
    stereo = np.stack([left, right], axis=1 if layout == "frames_channels" else 0).astype(np.float32)
    y = prepare_audio(stereo, sr, 24000)
    assert y.ndim == 1 and y.dtype == np.float32
    assert abs(y.size - 24000) <= 2
    assert np.abs(y).max() == pytest.approx(0.3, abs=0.02)             # mean of the two channels


def test_24k_mono_unchanged():
    y = np.linspace(-0.5, 0.5, 24000, dtype=np.float32)
    np.testing.assert_array_equal(prepare_audio(y, 24000, 24000), y)


# ---------------------------------------------------------------- CSM inputs

def test_conversation_is_ref_then_text_as_speaker_0():
    audio = np.zeros(24000, dtype=np.float32)
    conv = build_conversation("ref words", audio, "new line", "0")
    assert [m["role"] for m in conv] == ["0", "0"]
    assert conv[0]["content"][0] == {"type": "text", "text": "ref words"}
    assert conv[0]["content"][1]["type"] == "audio" and conv[0]["content"][1]["path"] is audio
    assert conv[1]["content"] == [{"type": "text", "text": "new line"}]


def test_generation_limit():
    assert max_new_frames(12) == 150           # 12 s x 12.5 frames/s
    assert max_new_frames(0.5) == 7            # rounds up
    assert max_new_frames(10, 25) == 250


# ---------------------------------------------------------------- voice_refs.json

def test_voice_refs_valid(tmp_path):
    for s in STRATEGIES:
        (tmp_path / f"{s}.wav").write_bytes(b"RIFF")
    assert validate_voice_refs(refs(), tmp_path) == refs()
    assert validate_voice_refs(refs()) == refs()                       # without a folder: fields + speaker only


def test_voice_refs_missing_strategy():
    data = refs()
    del data["CALM"]
    with pytest.raises(VoiceRefsError, match="CALM"):
        validate_voice_refs(data)


def test_voice_refs_two_speakers():
    data = refs(CALM={**refs()["CALM"], "speaker": "ex03"})
    with pytest.raises(VoiceRefsError, match="one speaker"):
        validate_voice_refs(data)


def test_voice_refs_missing_field_and_file(tmp_path):
    data = refs()
    del data["NEUTRAL"]["text"]
    with pytest.raises(VoiceRefsError, match="text"):
        validate_voice_refs(data)
    with pytest.raises(VoiceRefsError, match="not found"):
        validate_voice_refs(refs(), tmp_path)                           # no wavs in the folder


def test_set_ref_speaker_check():
    existing = {"VALIDATE": {"speaker": "ex02"}, "CALM": {"speaker": "ex02"}}
    check_same_speaker(existing, "ENCOURAGE", "ex02")
    check_same_speaker(existing, "VALIDATE", "ex02")
    check_same_speaker({}, "VALIDATE", "ex04")
    with pytest.raises(VoiceRefsError, match="one speaker"):
        check_same_speaker(existing, "ENCOURAGE", "ex01")
    check_same_speaker({"VALIDATE": {"speaker": "ex01"}}, "VALIDATE", "ex02")   # replacing the only one is fine


# ---------------------------------------------------------------- P6 / imports / config

def test_synth_takes_text_and_reference_only():
    params = list(inspect.signature(TTS.synth).parameters)
    assert params == ["self", "text", "ref_text", "ref_audio_24k"]
    for p in params:
        assert not any(w in p.lower() for w in ("prob", "emotion", "label", "mood", "confidence"))


def test_no_forbidden_imports():
    tree = ast.parse((CODE_DIR / "gpu" / "tts_gpu.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"ser_gpu", "emotion_predictor", "mood_manager"}


def test_config():
    base = load_config("gpu_24gb")["tts"]
    assert base["out_sample_rate"] == 24000 and base["max_audio_seconds"] == 12
    assert base["speaker_id"] == "0" and base["seed"] == 42
    assert base["backend"] == "csm" and base["model"] == "sesame/csm-1b"
    assert load_config("smoke_cpu")["tts"]["out_sample_rate"] == 24000


def test_from_preset_refuses_cpu():
    with pytest.raises(RuntimeError, match="GPU only"):
        TTS.from_preset("smoke_cpu")


def test_audition_lines_and_style_lists():
    assert set(tts_gpu.AUDITION_LINES) == set(STRATEGIES) == set(tts_gpu.STYLE_LISTS)
    assert tts_gpu.STYLE_LISTS["CALM"] == ["calm", "enunciated", "longform", "whisper", "default"]
