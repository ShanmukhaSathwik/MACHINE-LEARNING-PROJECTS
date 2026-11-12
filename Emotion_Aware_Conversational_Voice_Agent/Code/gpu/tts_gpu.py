"""
tts_gpu.py - the agent's voice for the GPU demo build (step 8a: voice audition with CSM-1B on CUDA).

GPU only. Base sesame/csm-1b, conditioned on ONE Expresso reference clip (transcript + audio). This step
auditions speakers and styles by ear and records the chosen clip per strategy; the pyttsx3 fallback and
sentence streaming come in a later step.

P6: the voice comes only from MoodManager's strategy. synth takes text + a reference clip; nothing here receives
emotion probabilities or labels, and this file does not import ser_gpu, emotion_predictor or mood_manager.

CSM conversation = [speaker "0": reference transcript + reference audio (24 kHz mono)], then [speaker "0": text].
Expresso audio is 48 kHz, so references are resampled to 24 kHz mono first. The seed (tts.seed) is set before
every generation so voices are compared fairly. Generation limit: CSM generates one Mimi codec frame per step
(12.5 frames/s = 80 ms), so max_new_tokens = ceil(max_audio_seconds x frame_rate) (12 s -> 150).

Outputs (device.outputs_dir(): gpu/outputs on the GPU machine, $GPU_OUTPUTS_DIR at home; audio is never committed):
    tts_audition/speakers/<speaker>_ref.wav, <speaker>_<STRATEGY>.wav
    tts_audition/styles/<speaker>/<STRATEGY>_<style>_<clip_id>.wav (+ ..._ref.wav)
    voice_refs/<STRATEGY>.wav   (--set-ref), recorded in gpu/voice_refs.json (text + metadata only, tracked)
An existing audio file is never overwritten: a timestamp is added to the new name instead.

Usage (from Code/):
    python gpu/tts_gpu.py --audition-speakers --preset gpu_24gb
    python gpu/tts_gpu.py --audition-styles --speaker ex02 --preset gpu_24gb
    python gpu/tts_gpu.py --set-ref VALIDATE ex02_sad_00042
"""

import argparse
import datetime
import io
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from device import code_path, load_config, outputs_dir, set_seed, setup

HERE = Path(__file__).resolve().parent
VOICE_REFS_JSON = HERE / "voice_refs.json"
STRATEGIES = ("VALIDATE", "CALM", "ENCOURAGE", "NEUTRAL")
MIMI_FRAME_RATE = 12.5            # CSM codec frames per second; read from the model config when it is loaded
REF_MIN_S, REF_MAX_S = 4.0, 8.0   # reference clip length
REF_MAX_PEAK = 0.99               # louder = clipped
REF_RMS_DBFS = (-35.0, -10.0)     # too quiet / too hot
REF_FIELDS = ("file", "text", "speaker", "style", "source_id", "seconds")
SPEAKER_STYLE = "default"         # --audition-speakers compares speakers on their everyday voice
CLIPS_PER_STYLE = 2               # --audition-styles

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


def _main():
    parser = argparse.ArgumentParser(description="CSM-1B voice audition (GPU demo build, step 8a; GPU only).")
    parser.add_argument("--preset", help="a gpu_* preset (default: $AGENT_PRESET)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audition-speakers", action="store_true",
                      help="each Expresso speaker's default-style clip as reference, 4 lines each")
    mode.add_argument("--audition-styles", action="store_true", help="every style per strategy for --speaker")
    mode.add_argument("--set-ref", nargs=2, metavar=("STRATEGY", "CLIP_ID"),
                      help="use an Expresso clip as that strategy's reference (no GPU needed)")
    parser.add_argument("--speaker", help="Expresso speaker for --audition-styles (e.g. ex02)")
    parser.add_argument("--model", help="override tts.model")
    args = parser.parse_args()
    if args.audition_styles and not args.speaker:
        parser.error("--audition-styles needs --speaker")

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
