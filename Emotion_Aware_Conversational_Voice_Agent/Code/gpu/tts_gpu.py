"""
tts_gpu.py - the agent's voice for the GPU demo build.
Step 8a: voice audition with CSM-1B on CUDA. Step 8b: speaking real replies, sentence by sentence, with a
pyttsx3 fallback (scripts/fallback_tts.py).

GPU only (CSM). Base sesame/csm-1b, conditioned on ONE Expresso reference clip (transcript + audio). 8a auditions
speakers and styles by ear and records the chosen clip per strategy in gpu/voice_refs.json.

P6: the voice comes only from MoodManager's strategy. synth takes text + a reference clip; nothing here receives
emotion probabilities or labels, and this file does not import ser_gpu, emotion_predictor or mood_manager.

Step 8b (the live agent's voice):
    CSMVoice.from_preset(preset)    loads CSM once (the 8a TTS class), reads gpu/voice_refs.json and the clips in
                                    voice_refs/; speak_sentence(text, strategy) -> (audio float32, sr) uses that
                                    strategy's reference as context; speak_stream(sentences, strategy) yields audio
                                    per sentence while a background thread plays the previous one (sounddevice).
    crisis=True                     always uses the NEUTRAL reference, whatever the strategy.
    speak_with_fallback(text, strategy, voice)   CSM; on any exception the rest of the reply goes to pyttsx3,
                                    and if that fails too, the text is printed. The path used is logged and returned.
    Voice tags (<sigh>, <softly>): base CSM-1B is not trained on them, so with tts.tags_enabled false (the default)
    strip_tags removes any <...> tag before CSM or pyttsx3 sees the text.
    time_to_first_audio = seconds from the start of speak_stream until the first sentence's audio is ready to play
    (after the one-off load and warm-up). rtf = generation seconds / audio seconds (< 1 = faster than real time).

CSM conversation = [speaker "0": reference transcript + reference audio (24 kHz mono)], then [speaker "0": text].
Expresso audio is 48 kHz, so references are resampled to 24 kHz mono first. The seed (tts.seed) is set before
every generation so voices are compared fairly. Generation limit: CSM generates one Mimi codec frame per step
(12.5 frames/s = 80 ms), so max_new_tokens = ceil(max_audio_seconds x frame_rate) (12 s -> 150).

Outputs (device.outputs_dir(): gpu/outputs on the GPU machine, $GPU_OUTPUTS_DIR at home; audio is never committed):
    tts_audition/speakers/<speaker>_ref.wav, <speaker>_<STRATEGY>.wav
    tts_audition/styles/<speaker>/<STRATEGY>_<style>_<clip_id>.wav (+ ..._ref.wav)
    voice_refs/<STRATEGY>.wav   (--set-ref), recorded in gpu/voice_refs.json (text + metadata only, tracked)
    tts_speak/speak_<STRATEGY>_<timestamp>.wav   (--speak)
An existing audio file is never overwritten: a timestamp is added to the new name instead.
--speak also writes per-sentence timings to results/metrics/tts_speak_csm.csv (a timestamp is added to the name
if that file exists). Only --speak writes it, and --speak only runs after CSM has loaded on CUDA.

Usage (from Code/):
    python gpu/tts_gpu.py --audition-speakers --preset gpu_24gb
    python gpu/tts_gpu.py --audition-styles --speaker ex02 --preset gpu_24gb
    python gpu/tts_gpu.py --set-ref VALIDATE ex02_sad_00042
    python gpu/tts_gpu.py --speak "I'm sorry. That sounds hard." --strategy VALIDATE --preset gpu_24gb [--no-play]
    python gpu/tts_gpu.py --fallback-test ["some line"]       (pyttsx3 only; no GPU, no CSM)
"""

import argparse
import csv
import datetime
import io
import json
import logging
import math
import os
import queue
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from device import code_path, load_config, outputs_dir, set_seed, setup

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import fallback_tts  # noqa: E402  (scripts/; no torch, no gpu imports)

log = logging.getLogger("tts_gpu")
VOICE_REFS_JSON = HERE / "voice_refs.json"
STRATEGIES = ("VALIDATE", "CALM", "ENCOURAGE", "NEUTRAL")
MIMI_FRAME_RATE = 12.5            # CSM codec frames per second; read from the model config when it is loaded
REF_MIN_S, REF_MAX_S = 4.0, 8.0   # reference clip length
REF_MAX_PEAK = 0.99               # louder = clipped
REF_RMS_DBFS = (-35.0, -10.0)     # too quiet / too hot
REF_FIELDS = ("file", "text", "speaker", "style", "source_id", "seconds")
SPEAKER_STYLE = "default"         # --audition-speakers compares speakers on their everyday voice
CLIPS_PER_STYLE = 2               # --audition-styles
CRISIS_STRATEGY = "NEUTRAL"       # a crisis turn always speaks with this reference
# Same set as llm_gpu.ABBREVIATIONS (kept equal by a test), so LLM streaming and TTS split sentences alike.
ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "st", "vs", "e.g", "i.e", "prof", "jr", "sr"}
WARMUP_TEXT = "Hello there."
FALLBACK_TEST_LINE = "This is the fallback voice. The GPU voice is not being used."
SPEAK_CSV = "results/metrics/tts_speak_csm.csv"

AUDITION_LINES = {
    "VALIDATE": "I'm really sorry you're going through this. That sounds exhausting.",
    "CALM": "Let's take this one step at a time. Just start with the first chapter tonight.",
    "ENCOURAGE": "You did it! All that studying really paid off.",
    "NEUTRAL": "Hey, good to hear from you. How was your day?",
}
# Styles to audition per strategy, in this order; only those the speaker has are used.
STYLE_LISTS = {
    "VALIDATE": ["sympathetic", "sad", "default"],
    "CALM": ["calm", "enunciated", "longform", "whisper", "default"],
    "ENCOURAGE": ["happy", "laughing", "default"],
    "NEUTRAL": ["default"],
}


class VoiceRefsError(RuntimeError):
    """gpu/voice_refs.json or a reference clip is missing or inconsistent."""


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------

def prepare_audio(y, sr, out_sr=24000) -> np.ndarray:
    """Mono float32 at out_sr (48 kHz Expresso -> 24 kHz). Accepts (frames, channels) or (channels, frames)."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=int(np.argmin(y.shape)))
    if y.size and sr != out_sr:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=out_sr)
    return np.ascontiguousarray(y, dtype=np.float32)


def build_conversation(ref_text, ref_audio, text, speaker_id="0") -> list:
    """[speaker: reference transcript + audio], then [same speaker: the new text]."""
    role = str(speaker_id)
    return [
        {"role": role, "content": [{"type": "text", "text": ref_text}, {"type": "audio", "path": ref_audio}]},
        {"role": role, "content": [{"type": "text", "text": text}]},
    ]


def max_new_frames(max_audio_seconds, frame_rate=MIMI_FRAME_RATE) -> int:
    """CSM generates one codec frame per step: max_new_tokens = ceil(seconds x frames per second)."""
    return int(math.ceil(float(max_audio_seconds) * float(frame_rate)))


def audio_clean(y, sr) -> bool:
    """Not clipped, and RMS loudness in a normal speaking range."""
    y = prepare_audio(y, sr, sr)
    if y.size == 0 or float(np.abs(y).max()) >= REF_MAX_PEAK:
        return False
    rms_db = 20 * np.log10(float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 1e-12)
    return bool(REF_RMS_DBFS[0] <= rms_db <= REF_RMS_DBFS[1])


def select_ref_clips(rows, speaker, style, n=1, min_s=REF_MIN_S, max_s=REF_MAX_S, is_clean=None) -> list:
    """First n rows by id with this speaker and style, min_s <= duration <= max_s, a transcript, and
    is_clean(row) True (only called on rows that pass the other checks, so audio is decoded only for those).
    Each row: {"id", "speaker_id", "style", "text", "duration"} (Expresso read speech: one speaker per clip)."""
    picked = []
    for row in sorted(rows, key=lambda r: r["id"]):
        if row["speaker_id"] != speaker or row["style"] != style:
            continue
        dur = row.get("duration")
        if dur is None or not min_s <= dur <= max_s or not str(row.get("text") or "").strip():
            continue
        if is_clean is not None and not is_clean(row):
            continue
        picked.append(row)
        if len(picked) == n:
            break
    return picked


def validate_voice_refs(data, refs_dir=None) -> dict:
    """All four strategies with every field, one speaker for all, and (if refs_dir is given) every file present."""
    if not isinstance(data, dict):
        raise VoiceRefsError("voice_refs.json must be an object {strategy: {...}}")
    missing = [s for s in STRATEGIES if s not in data]
    if missing:
        raise VoiceRefsError(f"voice_refs.json has no reference for {missing}; set each with --set-ref STRATEGY CLIP_ID")
    for s in STRATEGIES:
        lacking = [f for f in REF_FIELDS if f not in data[s]]
        if lacking:
            raise VoiceRefsError(f"voice_refs.json entry {s} is missing {lacking}")
    speakers = {data[s]["speaker"] for s in STRATEGIES}
    if len(speakers) != 1:
        raise VoiceRefsError(f"all four references must come from one speaker, found {sorted(speakers)}")
    if refs_dir is not None:
        for s in STRATEGIES:
            path = Path(refs_dir) / data[s]["file"]
            if not path.is_file():
                raise VoiceRefsError(f"reference clip for {s} not found: {path} (audio is not in git)")
    return data


def check_same_speaker(existing, strategy, speaker) -> None:
    """Refuse a reference whose speaker differs from the ones already set for the other strategies."""
    others = {s: r["speaker"] for s, r in (existing or {}).items() if s != strategy}
    wrong = {s: spk for s, spk in others.items() if spk != speaker}
    if wrong:
        raise VoiceRefsError(f"refused: {strategy} would be speaker {speaker}, but {sorted(wrong)} already use "
                             f"{sorted(set(wrong.values()))}. All four references must be one speaker.")


def make_clip_ids_unique(rows) -> dict:
    """Expresso reuses some ids (each speaker's 2 longform rows carry the id exNN_default_00001, the same as
    that speaker's first default clip). Every row that shares an id gets id = <expresso id>_<style><k>
    (e.g. ex02_default_00001_longform2); unique ids stay as they are. The original is kept in row["expresso_id"].
    Returns {shared expresso id: [the new ids]}, so a bare shared id can be refused instead of guessed."""
    counts = {}
    for r in rows:
        counts[r["id"]] = counts.get(r["id"], 0) + 1
    shared, seen = {}, {}
    for r in rows:
        raw = r["id"]
        r["expresso_id"] = raw
        if counts[raw] > 1:
            key = (raw, r["style"])
            seen[key] = seen.get(key, 0) + 1
            r["id"] = f"{raw}_{r['style']}{seen[key]}"
            shared.setdefault(raw, []).append(r["id"])
    return shared


def resolve_clip(by_id, shared, clip_id):
    """The row for clip_id. A shared Expresso id is refused (it matches several clips), as is an unknown one."""
    if clip_id in shared:
        raise VoiceRefsError(f"{clip_id!r} matches {len(shared[clip_id])} Expresso clips; "
                             f"use one of: {', '.join(shared[clip_id])}")
    if clip_id not in by_id:
        raise VoiceRefsError(f"unknown clip id {clip_id!r} (use an id printed by --audition-speakers / --audition-styles)")
    return by_id[clip_id]


def _check_strategy(strategy):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


# Step 8b helpers ---------------------------------------------------------------------------------

# A sentence ends at . ! ? ... or the ellipsis character (plus closing quotes/brackets) before whitespace or the
# end of the text, or at a newline. "3.5" never matches (no whitespace after the dot).
_SENT_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)|\n+")
_TAG = re.compile(r"</?[A-Za-z][^<>]{0,60}>")      # <sigh>, </b>, <break time="1s"/>; not "x < 5" or "<3"


def _is_abbreviation(text_before, terminator) -> bool:
    """A single "." after a word in ABBREVIATIONS ("Dr.", "e.g.") does not end the sentence."""
    if terminator.strip("\"'”’)]") != ".":
        return False
    tokens = text_before.split()
    return bool(tokens) and tokens[-1].lstrip("(\"'[").lower() in ABBREVIATIONS


def split_sentences(text) -> list:
    """Sentences to speak one at a time. Splits after . ! ? ... (and at newlines); keeps decimals ("3.5") and the
    abbreviations in ABBREVIATIONS together; a last piece without an end mark is kept. Whitespace is collapsed."""
    text = str(text or "")
    out, start = [], 0
    for m in _SENT_END.finditer(text):
        if m.group().startswith("\n"):
            piece = text[start:m.start()]
        elif _is_abbreviation(text[start:m.start()], m.group()):
            continue
        else:
            piece = text[start:m.end()]
        piece = " ".join(piece.split())
        if piece:
            out.append(piece)
        start = m.end()
    rest = " ".join(text[start:].split())
    if rest:
        out.append(rest)
    return out


def strip_tags(text) -> str:
    """Remove every <...> style tag (base CSM-1B is not trained on them); tidy the spaces left behind."""
    t = " ".join(_TAG.sub(" ", str(text or "")).split())
    return re.sub(r"\s+([.,!?;:…])", r"\1", t).strip()


def ref_strategy(strategy, crisis=False) -> str:
    """The strategy whose reference clip is used: CRISIS_STRATEGY on a crisis turn (whatever strategy says),
    else the strategy itself (which must be one of STRATEGIES)."""
    if crisis:
        return CRISIS_STRATEGY
    _check_strategy(strategy)
    return strategy


def check_voice_refs(data, refs_dir) -> tuple:
    """Usable references for CSMVoice: ({strategy: entry}, {strategy: why it is missing}).
    Unlike validate_voice_refs (which --set-ref's end state must pass), a strategy may be missing here: speaking
    with it raises a clear error, and speak_with_fallback then uses pyttsx3. Refused outright: a malformed entry,
    two speakers among the usable entries, or no usable entry at all."""
    if not isinstance(data, dict):
        raise VoiceRefsError("voice_refs.json must be an object {strategy: {...}}")
    present, missing = {}, {}
    for s in STRATEGIES:
        entry = data.get(s)
        if entry is None:
            missing[s] = f"not in gpu/voice_refs.json (set it with --set-ref {s} CLIP_ID)"
            continue
        lacking = [f for f in REF_FIELDS if f not in entry]
        if lacking:
            raise VoiceRefsError(f"voice_refs.json entry {s} is missing {lacking}")
        path = Path(refs_dir) / entry["file"]
        if not path.is_file():
            missing[s] = f"reference clip not found: {path} (audio is not in git)"
            continue
        present[s] = entry
    speakers = sorted({e["speaker"] for e in present.values()})
    if len(speakers) > 1:
        raise VoiceRefsError(f"all references must come from one speaker, found {speakers}")
    if not present:
        raise VoiceRefsError("no usable reference clip: " + "; ".join(f"{s}: {why}" for s, why in missing.items()))
    return present, missing


def _read_ref_audio(path, out_sr) -> np.ndarray:
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32")
    return prepare_audio(y, sr, out_sr)


# ---------------------------------------------------------------------------------------------
# TTS (CSM only in step 8a)
# ---------------------------------------------------------------------------------------------

class TTS:
    """sesame/csm-1b on CUDA. Load once, warm up once, then synth(text, ref_text, ref_audio_24k)."""

    def __init__(self, cfg, device, dtype, model_name=None):
        tts = cfg["tts"]
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.model_name = model_name or tts.get("model")
        self.sr = int(tts["out_sample_rate"])
        self.max_audio_seconds = float(tts["max_audio_seconds"])
        self.speaker_id = str(tts["speaker_id"])
        self.seed = int(tts["seed"])
        self.max_new_tokens = max_new_frames(self.max_audio_seconds)
        self.processor = self.model = None

    @classmethod
    def from_preset(cls, preset, model_name=None):
        """device.setup(preset) -> CSM on cuda with the preset dtype, warmed up once. GPU only."""
        cfg, device, dtype = setup(preset)
        if device != "cuda":
            raise RuntimeError(f"step 8a is GPU only; preset {cfg['preset']} resolves to {device}. "
                               "Use a gpu_* preset on the GPU machine.")
        if cfg["tts"].get("backend") != "csm":
            raise RuntimeError(f"preset {cfg['preset']} has tts.backend {cfg['tts'].get('backend')!r}, not csm")
        return cls(cfg, device, dtype, model_name=model_name).load().warm_up()

    def load(self):
        from transformers import AutoProcessor, CsmForConditionalGeneration

        if not self.model_name:
            raise ValueError("tts.model is not set (expected sesame/csm-1b)")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = CsmForConditionalGeneration.from_pretrained(
            self.model_name, dtype=self.dtype, device_map=self.device)
        self.model.eval()
        codec = getattr(self.model.config, "codec_config", None)
        self.max_new_tokens = max_new_frames(self.max_audio_seconds,
                                             getattr(codec, "frame_rate", None) or MIMI_FRAME_RATE)
        return self

    def warm_up(self):
        """One short text-only generation (first call pays CUDA kernel and cache setup)."""
        self._generate([{"role": self.speaker_id, "content": [{"type": "text", "text": "Hello there."}]}])
        return self

    def _generate(self, conversation):
        import torch

        set_seed(self.seed)
        inputs = self.processor.apply_chat_template(conversation, tokenize=True, return_dict=True)
        inputs = {k: (v.to(self.model.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.model.device))
                  for k, v in inputs.items()}
        with torch.inference_mode():
            audio = self.model.generate(**inputs, output_audio=True, max_new_tokens=self.max_new_tokens)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return audio[0].float().cpu().numpy().astype(np.float32).reshape(-1)

    def synth(self, text, ref_text, ref_audio_24k) -> dict:
        """{audio, sr, audio_seconds, gen_seconds, rtf}. rtf = gen_seconds / audio_seconds (< 1 = faster than real time)."""
        if self.model is None:
            raise RuntimeError("CSM not loaded (use TTS.from_preset)")
        t0 = time.perf_counter()
        audio = self._generate(build_conversation(ref_text, ref_audio_24k, text, self.speaker_id))
        gen = time.perf_counter() - t0
        secs = audio.size / self.sr
        return {"audio": audio, "sr": self.sr, "audio_seconds": round(secs, 3), "gen_seconds": round(gen, 3),
                "rtf": round(gen / secs, 3) if secs else None}


# ---------------------------------------------------------------------------------------------
# Step 8b: speaking replies (CSMVoice), background playback, pyttsx3 fallback
# ---------------------------------------------------------------------------------------------

def _play_blocking(audio, sr):
    """Play one clip on the default output device and wait until it ends (sounddevice)."""
    import sounddevice as sd

    sd.play(audio, sr)
    sd.wait()


def _stop_playback():
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:  # noqa: BLE001
        pass


class Player:
    """Plays clips in order on a background thread, so the caller can generate the next sentence meanwhile.
    put(audio, sr) returns at once. close(wait=True) blocks until every queued clip has played; close(wait=False)
    stops the current clip and drops the rest. A playback error stops playback; the next put() raises it."""

    def __init__(self):
        self._q = queue.Queue()
        self._abort = threading.Event()
        self.played = 0
        self.error = None
        self._thread = threading.Thread(target=self._run, name="tts-playback", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            if self._abort.is_set() or self.error is not None:
                continue
            try:
                _play_blocking(*item)
                self.played += 1
            except Exception as e:  # noqa: BLE001
                self.error = e

    def put(self, audio, sr):
        if self.error is not None:
            raise RuntimeError(f"playback failed: {self.error}") from self.error
        self._q.put((audio, sr))

    def close(self, wait=True):
        if not wait:
            self._abort.set()
            _stop_playback()
        self._q.put(None)
        self._thread.join()


class CSMVoice:
    """CSM-1B speaking real replies: the 8a TTS class, with each strategy's reference clip as context.

    CSMVoice(tts, refs, refs_dir, tags_enabled=False): tts is a loaded TTS (anything with .sr and
    .synth(text, ref_text, ref_audio_24k) -> {"audio", "sr", ...}); refs is the gpu/voice_refs.json dict.
    Use CSMVoice.from_preset(preset) for the real thing (GPU only)."""

    def __init__(self, tts, refs, refs_dir, tags_enabled=False):
        self.tts = tts
        self.sr = int(tts.sr)
        self.tags_enabled = bool(tags_enabled)
        self.refs_dir = Path(refs_dir)
        present, self.missing = check_voice_refs(refs, self.refs_dir)
        self.refs = {s: {"text": e["text"], "file": e["file"], "speaker": e["speaker"],
                         "audio": _read_ref_audio(self.refs_dir / e["file"], self.sr)} for s, e in present.items()}
        self.last_sentence = None   # timings of the last speak_sentence
        self.last_stream = None     # timings of the last (or current) speak_stream

    @classmethod
    def from_preset(cls, preset=None, model_name=None, json_path=VOICE_REFS_JSON, warm_up=True):
        """Check gpu/voice_refs.json and the clips first (no point loading CSM without them), then load CSM once
        on CUDA (TTS.from_preset: refuses a CPU preset) and warm up once with a reference clip as context."""
        cfg = load_config(preset)
        tcfg = cfg["tts"]
        refs_dir = code_path(tcfg.get("voice_refs_dir") or "gpu/outputs/voice_refs")
        json_path = Path(json_path)
        if not json_path.is_file():
            raise VoiceRefsError(f"{json_path} not found; set the references with --set-ref STRATEGY CLIP_ID")
        refs = json.loads(json_path.read_text(encoding="utf-8"))
        _, missing = check_voice_refs(refs, refs_dir)
        if missing:
            print("WARNING: no reference for " + "; ".join(f"{s} ({why})" for s, why in missing.items())
                  + ". Those strategies will use the fallback voice.", file=sys.stderr)
        import explore_expresso

        explore_expresso.load_env()   # HF_TOKEN from Code/.env for the gated model; never printed or stored here
        tts = TTS.from_preset(preset, model_name=model_name)
        voice = cls(tts, refs, refs_dir, tags_enabled=tcfg.get("tags_enabled", False))
        if warm_up:
            voice.warm_up()
        return voice

    def warm_up(self):
        """One generation with a reference as context (TTS.warm_up is text-only), so the first reply's
        time_to_first_audio does not include one-off setup for the longer input."""
        key = CRISIS_STRATEGY if CRISIS_STRATEGY in self.refs else next(iter(self.refs))
        self.speak_sentence(WARMUP_TEXT, key)
        return self

    def reference(self, strategy, crisis=False):
        """(strategy whose clip is used, {"text", "audio", ...}). Raises VoiceRefsError if it has no clip."""
        key = ref_strategy(strategy, crisis)
        if key not in self.refs:
            why = self.missing.get(key, "not loaded")
            note = f" (crisis turns always use {CRISIS_STRATEGY})" if crisis else ""
            raise VoiceRefsError(f"no reference clip for {key}{note}: {why}")
        return key, self.refs[key]

    def prepare_text(self, text) -> str:
        t = str(text or "") if self.tags_enabled else strip_tags(text)
        return " ".join(t.split())

    def speak_sentence(self, text, strategy, crisis=False):
        """(audio float32 mono, sample rate) for one sentence, with the strategy's reference clip as context
        (NEUTRAL's on a crisis turn). Text that is empty after strip_tags gives empty audio without calling CSM."""
        key, ref = self.reference(strategy, crisis)
        text = self.prepare_text(text)
        sr, gen = self.sr, 0.0
        audio = np.zeros(0, dtype=np.float32)
        if text:
            t0 = time.perf_counter()
            r = self.tts.synth(text, ref["text"], ref["audio"])
            gen = time.perf_counter() - t0
            audio = np.asarray(r["audio"], dtype=np.float32).reshape(-1)
            sr = int(r.get("sr", self.sr))
        secs = audio.size / sr
        self.last_sentence = {"text": text, "ref_strategy": key, "gen_seconds": round(gen, 3),
                              "audio_seconds": round(secs, 3), "rtf": round(gen / secs, 3) if secs else None}
        return audio, sr

    def speak_stream(self, sentences, strategy, crisis=False, play=True):
        """Generator: (audio, sr) per sentence, in order. With play=True each clip is queued on a background
        Player as soon as it is made, so sentence N+1 is generated while sentence N plays; when the generator is
        exhausted it waits for playback to end. sentences: a string (split with split_sentences) or any iterable
        of sentences (e.g. the LLM's stream). Timings go to self.last_stream as they happen."""
        self.last_stream = None   # never leave the previous reply's counts behind if this one fails early
        if isinstance(sentences, str):
            sentences = split_sentences(sentences)
        stats = {"strategy": strategy, "ref_strategy": ref_strategy(strategy, crisis), "crisis": bool(crisis),
                 "played_live": bool(play), "sentences": [], "generated": 0, "played": 0,
                 "time_to_first_audio": None, "gen_seconds": 0.0, "audio_seconds": 0.0, "rtf": None}
        self.last_stream = stats
        player = Player() if play else None
        aborted = False
        t0 = time.perf_counter()
        try:
            for text in sentences:
                audio, sr = self.speak_sentence(text, strategy, crisis=crisis)
                if audio.size == 0:
                    continue
                if stats["time_to_first_audio"] is None:
                    stats["time_to_first_audio"] = round(time.perf_counter() - t0, 3)
                info = self.last_sentence
                stats["sentences"].append(info)
                stats["generated"] += 1
                stats["gen_seconds"] = round(stats["gen_seconds"] + info["gen_seconds"], 3)
                stats["audio_seconds"] = round(stats["audio_seconds"] + info["audio_seconds"], 3)
                stats["rtf"] = round(stats["gen_seconds"] / stats["audio_seconds"], 3)
                if player is not None:
                    player.put(audio, sr)
                yield audio, sr
        except GeneratorExit:
            aborted = True    # the caller stopped early: cut playback
            raise
        finally:
            if player is not None:
                player.close(wait=not aborted)
                stats["played"] = player.played
        if player is not None and player.error is not None:
            raise RuntimeError(f"playback failed: {player.error}") from player.error


def speak_with_fallback(text, strategy, voice=None, crisis=False, play=True) -> dict:
    """What the live agent calls once per reply. Tries voice (a loaded CSMVoice) sentence by sentence; on any
    exception (or with no voice) the part of the reply not yet spoken goes to scripts/fallback_tts.speak:
    pyttsx3, else the text is printed. Never raises for a CSM failure. Returns and logs the path used:
    {"path": "csm" | "pyttsx3" | "text" | "none", "csm_sentences", "error", "strategy", "crisis", "stats"}."""
    sentences = split_sentences(text)
    record = {"path": None, "strategy": strategy, "crisis": bool(crisis), "csm_sentences": 0,
              "error": None, "stats": None}
    if not sentences:
        record["path"] = "none"
        return record
    if voice is None:
        record["error"] = "no CSMVoice loaded"
    else:
        try:
            for _ in voice.speak_stream(sentences, strategy, crisis=crisis, play=play):
                pass
            record.update(path="csm", csm_sentences=len(sentences), stats=getattr(voice, "last_stream", None))
            log.info("tts path=csm strategy=%s crisis=%s sentences=%d", strategy, bool(crisis), len(sentences))
            return record
        except Exception as e:  # noqa: BLE001  (any CSM failure -> fallback)
            record["error"] = f"{type(e).__name__}: {e}"
            stats = getattr(voice, "last_stream", None) or {}
            record["stats"] = stats or None
            # what was already spoken by CSM is not repeated: played clips (live), else generated ones
            record["csm_sentences"] = int(stats.get("played" if play else "generated", 0) or 0)
    rest = strip_tags(" ".join(sentences[record["csm_sentences"]:]))
    record["path"] = fallback_tts.speak(rest)
    log.warning("tts path=%s (fallback) strategy=%s crisis=%s csm_sentences=%d reason=%s",
                record["path"], strategy, bool(crisis), record["csm_sentences"], record["error"])
    return record


def speak_iter_with_fallback(sentences, strategy, voice=None, crisis=False, play=True) -> dict:
    """speak_with_fallback for a STREAM of sentences (step 9a: the live agent feeds LLM.stream_reply in, so CSM
    starts on the first sentence while the LLM is still writing the rest). Same record, same log line.
    On any CSM exception (or with no voice) the sentences not yet spoken, plus whatever the stream still holds,
    go to scripts/fallback_tts.speak (tags stripped). Never raises for a CSM failure."""
    it = iter(sentences)
    consumed = []

    def feed():
        for s in it:
            consumed.append(s)
            yield s

    record = {"path": None, "strategy": strategy, "crisis": bool(crisis), "csm_sentences": 0,
              "error": None, "stats": None}
    if voice is None:
        record["error"] = "no CSMVoice loaded"
    else:
        try:
            for _ in voice.speak_stream(feed(), strategy, crisis=crisis, play=play):
                pass
            if not consumed:
                record["path"] = "none"
                return record
            record.update(path="csm", csm_sentences=len(consumed), stats=getattr(voice, "last_stream", None))
            log.info("tts path=csm strategy=%s crisis=%s sentences=%d", strategy, bool(crisis), len(consumed))
            return record
        except Exception as e:  # noqa: BLE001  (any CSM failure -> fallback)
            record["error"] = f"{type(e).__name__}: {e}"
            stats = getattr(voice, "last_stream", None) or {}
            record["stats"] = stats or None
            record["csm_sentences"] = int(stats.get("played" if play else "generated", 0) or 0)
    try:
        remaining = list(it)
    except Exception:  # noqa: BLE001  (a broken stream: speak what we have)
        remaining = []
    rest = strip_tags(" ".join(consumed[record["csm_sentences"]:] + remaining))
    record["path"] = fallback_tts.speak(rest)
    log.warning("tts path=%s (fallback) strategy=%s crisis=%s csm_sentences=%d reason=%s",
                record["path"], strategy, bool(crisis), record["csm_sentences"], record["error"])
    return record


# ---------------------------------------------------------------------------------------------
# Expresso access (through gpu/explore_expresso.py)
# ---------------------------------------------------------------------------------------------

class Expresso:
    """Read-speech Expresso, loaded with explore_expresso.load_expresso (audio not decoded by datasets)."""

    def __init__(self):
        import explore_expresso as ee

        self.ee = ee
        ee.load_env()
        self.ds = ee.load_expresso()
        meta = self.ds.select_columns(["speaker_id", "style", "id", "text"])
        self.rows = [{"index": i, "id": r["id"], "speaker_id": r["speaker_id"], "style": r["style"],
                      "text": (r["text"] or "").strip(), "duration": None} for i, r in enumerate(meta)]
        self.shared_ids = make_clip_ids_unique(self.rows)
        self.by_id = {r["id"]: r for r in self.rows}
        if len(self.by_id) != len(self.rows):
            raise RuntimeError("clip ids are still not unique after make_clip_ids_unique")
        self.speakers = sorted({r["speaker_id"] for r in self.rows})

    def styles_of(self, speaker) -> list:
        return sorted({r["style"] for r in self.rows if r["speaker_id"] == speaker})

    def _raw(self, row):
        _, raw, _ = self.ee.audio_info(self.ds[row["index"]]["audio"])
        return raw

    def cell(self, speaker, style) -> list:
        """Rows of one speaker x style, with durations filled in from the file headers."""
        import soundfile as sf

        cell = [r for r in self.rows if r["speaker_id"] == speaker and r["style"] == style]
        for r in cell:
            if r["duration"] is None:
                raw = self._raw(r)
                if raw is not None:
                    info = sf.info(io.BytesIO(raw))
                    r["duration"] = info.frames / info.samplerate
        return cell

    def audio(self, row, out_sr=24000):
        import soundfile as sf

        y, sr = sf.read(io.BytesIO(self._raw(row)), dtype="float32")
        return prepare_audio(y, sr, out_sr)

    def is_clean(self, row) -> bool:
        import soundfile as sf

        y, sr = sf.read(io.BytesIO(self._raw(row)), dtype="float32")
        return audio_clean(y, sr)

    def pick(self, speaker, style, n):
        return select_ref_clips(self.cell(speaker, style), speaker, style, n, is_clean=self.is_clean)


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def _new_path(path) -> Path:
    path = Path(path)
    if path.exists():
        path = path.with_name(f"{path.stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    return path


def _write_wav(path, audio, sr, keep_existing=False) -> Path:
    """Write a 16-bit wav. Never overwrites: a new timestamped name, or (keep_existing) the file already there."""
    import soundfile as sf

    path = Path(path)
    if keep_existing and path.exists():
        return path
    path = _new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def _peak_vram_gb():
    import torch

    return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if torch.cuda.is_available() else None


def _short(text, n=60):
    return text if len(text) <= n else text[: n - 3] + "..."


def audition_speakers(tts, ex):
    import torch

    out_dir = outputs_dir() / "tts_audition" / "speakers"
    print("Styles per speaker:")
    for spk in ex.speakers:
        print(f"  {spk}: {', '.join(ex.styles_of(spk))}")
    print()
    rows = []
    for spk in ex.speakers:
        picked = ex.pick(spk, SPEAKER_STYLE, 1)
        if not picked:
            print(f"{spk}: no clean {SPEAKER_STYLE} clip of {REF_MIN_S:g}-{REF_MAX_S:g} s; skipped")
            continue
        ref = picked[0]
        ref_audio = ex.audio(ref, tts.sr)
        _write_wav(out_dir / f"{spk}_ref.wav", ref_audio, tts.sr, keep_existing=True)
        torch.cuda.reset_peak_memory_stats()
        lines = {}
        for st in STRATEGIES:
            r = tts.synth(AUDITION_LINES[st], ref["text"], ref_audio)
            _write_wav(out_dir / f"{spk}_{st}.wav", r["audio"], r["sr"])
            lines[st] = r
        rows.append((spk, ref, lines, _peak_vram_gb()))
        print(f"{spk} done ({ref['id']})", flush=True)

    print(f"\n{'speaker':<8} {'ref id':<20} " + " ".join(f"{st + ' gen/RTF':>18}" for st in STRATEGIES)
          + f" {'peak VRAM':>10}  ref transcript")
    for spk, ref, lines, vram in rows:
        cells = " ".join(f"{lines[st]['gen_seconds']:>9.2f}s/{lines[st]['rtf']:<7}" for st in STRATEGIES)
        print(f"{spk:<8} {ref['id']:<20} {cells} {str(vram) + ' GB':>10}  {_short(ref['text'])}")
    print(f"\nAudio: {out_dir}")
    print("Next: python gpu/tts_gpu.py --audition-styles --speaker SPEAKER --preset PRESET")


def audition_styles(tts, ex, speaker):
    import torch

    if speaker not in ex.speakers:
        raise SystemExit(f"Unknown speaker {speaker!r}; Expresso speakers: {ex.speakers}")
    out_dir = outputs_dir() / "tts_audition" / "styles" / speaker
    have = set(ex.styles_of(speaker))
    print(f"{speaker} styles: {', '.join(sorted(have))}\n")
    torch.cuda.reset_peak_memory_stats()
    table = []
    for st in STRATEGIES:
        styles = [s for s in STYLE_LISTS[st] if s in have]
        skipped = [s for s in STYLE_LISTS[st] if s not in have]
        if skipped:
            print(f"{st}: {speaker} has no {skipped}; auditioning {styles}")
        for style in styles:
            for ref in ex.pick(speaker, style, CLIPS_PER_STYLE):
                ref_audio = ex.audio(ref, tts.sr)
                stem = f"{st}_{style}_{ref['id']}"
                _write_wav(out_dir / f"{stem}_ref.wav", ref_audio, tts.sr, keep_existing=True)
                r = tts.synth(AUDITION_LINES[st], ref["text"], ref_audio)
                _write_wav(out_dir / f"{stem}.wav", r["audio"], r["sr"])
                table.append((st, style, ref, r))
                print(f"  {stem} done", flush=True)

    print(f"\n{'strategy':<10} {'style':<11} {'clip id':<22} {'gen':>7} {'RTF':>6} {'audio':>6}  ref transcript")
    for st, style, ref, r in table:
        print(f"{st:<10} {style:<11} {ref['id']:<22} {r['gen_seconds']:>6.2f}s {r['rtf']:>6} "
              f"{r['audio_seconds']:>5.2f}s  {_short(ref['text'])}")
    print(f"\npeak VRAM {_peak_vram_gb()} GB   audio: {out_dir}")
    print("Next, one per strategy (same speaker): python gpu/tts_gpu.py --set-ref STRATEGY CLIP_ID")


def _atomic_write_json(path, data):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def set_ref(cfg, strategy, clip_id, ex=None, json_path=VOICE_REFS_JSON):
    """Copy Expresso clip clip_id (24 kHz mono) to voice_refs/<STRATEGY>.wav and record it in gpu/voice_refs.json.
    All four must be one speaker. An existing <STRATEGY>.wav is kept as <STRATEGY>_<timestamp>.wav."""
    _check_strategy(strategy)
    json_path = Path(json_path)
    existing = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    ex = ex or Expresso()
    row = resolve_clip(ex.by_id, ex.shared_ids, clip_id)
    check_same_speaker(existing, strategy, row["speaker_id"])
    sr = int(cfg["tts"]["out_sample_rate"])
    audio = ex.audio(row, sr)
    refs_dir = code_path(cfg["tts"].get("voice_refs_dir") or "gpu/outputs/voice_refs")
    refs_dir.mkdir(parents=True, exist_ok=True)
    dest = refs_dir / f"{strategy}.wav"
    if dest.exists():
        dest.rename(refs_dir / f"{strategy}_{datetime.datetime.now():%Y%m%d_%H%M%S}.wav")
    import soundfile as sf

    sf.write(str(dest), audio, sr, subtype="PCM_16")
    existing[strategy] = {"file": dest.name, "text": row["text"], "speaker": row["speaker_id"], "style": row["style"],
                          "source_id": clip_id, "seconds": round(audio.size / sr, 2)}
    _atomic_write_json(json_path, {s: existing[s] for s in STRATEGIES if s in existing})
    print(f"{strategy} -> {clip_id} ({row['speaker_id']}, {row['style']}, {existing[strategy]['seconds']}s): {row['text']}")
    print(f"audio: {dest}\nrecorded in: {json_path}")
    still = [s for s in STRATEGIES if s not in existing]
    print("All four set (one speaker)." if not still else f"Still to set: {still}")


SPEAK_CSV_FIELDS = ("run_at", "preset", "model", "dtype", "gpu", "strategy", "ref_strategy", "crisis", "played_live",
                    "sentence", "text", "chars", "gen_seconds", "audio_seconds", "rtf",
                    "time_to_first_audio", "reply_gen_seconds", "reply_audio_seconds", "reply_rtf", "peak_vram_gb")


def speak_csv_rows(stats, meta) -> list:
    """One row per spoken sentence; the reply-level numbers (time_to_first_audio, totals, rtf) repeat on each row."""
    rows = []
    for i, s in enumerate(stats["sentences"], 1):
        rows.append({**meta, "strategy": stats["strategy"], "ref_strategy": stats["ref_strategy"],
                     "crisis": stats["crisis"], "played_live": stats["played_live"], "sentence": i,
                     "text": s["text"], "chars": len(s["text"]), "gen_seconds": s["gen_seconds"],
                     "audio_seconds": s["audio_seconds"], "rtf": s["rtf"],
                     "time_to_first_audio": stats["time_to_first_audio"], "reply_gen_seconds": stats["gen_seconds"],
                     "reply_audio_seconds": stats["audio_seconds"], "reply_rtf": stats["rtf"]})
    return rows


def write_speak_csv(rows, path=None) -> Path:
    """results/metrics/tts_speak_csm.csv, or a timestamped name next to it if it exists (never overwrites)."""
    path = _new_path(path or code_path(SPEAK_CSV))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SPEAK_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def speak_cli(voice, text, strategy, crisis=False, play=True):
    """--speak: stream one reply through CSM (playing it unless --no-play), save the wav and the timings."""
    import torch

    sentences = split_sentences(text)
    if not sentences:
        raise SystemExit("--speak: nothing to say")
    run_at = datetime.datetime.now()
    torch.cuda.reset_peak_memory_stats()
    chunks = [audio for audio, _ in voice.speak_stream(sentences, strategy, crisis=crisis, play=play)]
    stats = voice.last_stream
    if not chunks:
        raise SystemExit("--speak: CSM produced no audio")
    wav = _write_wav(outputs_dir() / "tts_speak" / f"speak_{strategy}_{run_at:%Y%m%d_%H%M%S}.wav",
                     np.concatenate(chunks), voice.sr)
    vram = _peak_vram_gb()
    print(f"{'#':>2} {'gen':>7} {'audio':>7} {'RTF':>6}  text")
    for i, s in enumerate(stats["sentences"], 1):
        print(f"{i:>2} {s['gen_seconds']:>6.2f}s {s['audio_seconds']:>6.2f}s {s['rtf']:>6}  {_short(s['text'], 70)}")
    print(f"\nreference: {stats['ref_strategy']}" + ("  (crisis -> NEUTRAL)" if crisis else ""))
    print(f"time_to_first_audio: {stats['time_to_first_audio']:.3f} s")
    print(f"real-time factor:    {stats['rtf']}  ({stats['gen_seconds']:.2f} s to generate "
          f"{stats['audio_seconds']:.2f} s of audio; < 1 = faster than real time)")
    print(f"peak VRAM {vram} GB   played: {'yes' if play else 'no (--no-play)'}")
    print(f"audio: {wav}")
    meta = {"run_at": f"{run_at:%Y-%m-%d %H:%M:%S}", "preset": voice.tts.cfg["preset"], "model": voice.tts.model_name,
            "dtype": str(voice.tts.dtype).replace("torch.", ""), "gpu": torch.cuda.get_device_name(0),
            "peak_vram_gb": vram}
    print(f"timings: {write_speak_csv(speak_csv_rows(stats, meta))}")


def _main():
    parser = argparse.ArgumentParser(description="CSM-1B voice (GPU demo build, steps 8a/8b; CSM is GPU only).")
    parser.add_argument("--preset", help="a gpu_* preset (default: $AGENT_PRESET)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audition-speakers", action="store_true",
                      help="each Expresso speaker's default-style clip as reference, 4 lines each")
    mode.add_argument("--audition-styles", action="store_true", help="every style per strategy for --speaker")
    mode.add_argument("--set-ref", nargs=2, metavar=("STRATEGY", "CLIP_ID"),
                      help="use an Expresso clip as that strategy's reference (no GPU needed)")
    mode.add_argument("--speak", metavar="TEXT", help="speak TEXT with CSM, sentence by sentence (needs --strategy)")
    mode.add_argument("--fallback-test", nargs="?", const=FALLBACK_TEST_LINE, metavar="TEXT",
                      help="speak one line with the fallback only (pyttsx3, else printed); no GPU")
    parser.add_argument("--speaker", help="Expresso speaker for --audition-styles (e.g. ex02)")
    parser.add_argument("--strategy", help=f"--speak: one of {', '.join(STRATEGIES)}")
    parser.add_argument("--crisis", action="store_true", help="--speak: crisis turn (always the NEUTRAL reference)")
    parser.add_argument("--no-play", action="store_true", help="--speak: generate and save only, no playback")
    parser.add_argument("--model", help="override tts.model")
    args = parser.parse_args()
    if args.audition_styles and not args.speaker:
        parser.error("--audition-styles needs --speaker")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)   # this file's path log at INFO; other libraries (comtypes) stay quiet

    if args.fallback_test is not None:
        rec = speak_with_fallback(args.fallback_test, (args.strategy or CRISIS_STRATEGY).upper(), voice=None)
        print(f"path used: {rec['path']}")
        return 0

    if args.speak is not None:
        if not args.strategy:
            parser.error("--speak needs --strategy")
        strategy = args.strategy.upper()
        if strategy not in STRATEGIES:
            parser.error(f"unknown strategy {args.strategy!r}; expected one of {STRATEGIES}")
        try:
            t0 = time.perf_counter()
            voice = CSMVoice.from_preset(args.preset, model_name=args.model)
            load_s = time.perf_counter() - t0
        except (ValueError, RuntimeError, OSError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        tts = voice.tts
        print(f"preset {tts.cfg['preset']}  device {tts.device}  dtype {str(tts.dtype).replace('torch.', '')}  "
              f"model {tts.model_name}  seed {tts.seed}  max_new_tokens {tts.max_new_tokens}  "
              f"tags_enabled {voice.tags_enabled}")
        print(f"references: {', '.join(sorted(voice.refs))}" + (f"  (missing: {', '.join(voice.missing)})"
                                                                  if voice.missing else ""))
        print(f"loaded + warm-up in {load_s:.1f}s\n")
        speak_cli(voice, args.speak, strategy, crisis=args.crisis, play=not args.no_play)
        return 0

    if args.set_ref:
        try:
            set_ref(load_config(args.preset), args.set_ref[0].upper(), args.set_ref[1])
        except (VoiceRefsError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        return 0

    try:
        t0 = time.perf_counter()
        tts = TTS.from_preset(args.preset, model_name=args.model)
        load_s = time.perf_counter() - t0
    except (ValueError, RuntimeError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"preset {tts.cfg['preset']}  device {tts.device}  dtype {str(tts.dtype).replace('torch.', '')}  "
          f"model {tts.model_name}  seed {tts.seed}  max_new_tokens {tts.max_new_tokens} ({tts.max_audio_seconds:g}s)")
    print(f"loaded + warm-up in {load_s:.1f}s   peak VRAM so far {_peak_vram_gb()} GB\n")
    ex = Expresso()
    if args.audition_speakers:
        audition_speakers(tts, ex)
    else:
        audition_styles(tts, ex, args.speaker)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
