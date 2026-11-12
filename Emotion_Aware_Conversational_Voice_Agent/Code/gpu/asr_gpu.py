"""
asr_gpu.py - Whisper speech-to-text for the GPU demo build.

Step 6 of the GPU demo build. Wraps the transformers automatic-speech-recognition pipeline
behind one call that takes a file or an array and returns the text plus timing.
GPU-first (gpu_* presets: openai/whisper-large-v3-turbo, beam 5); smoke_cpu runs whisper-tiny.en at home.
Independent of the CPU baseline: it does not import scripts/asr.py, emotion_predictor or feature_extraction.

Pure helpers (no model, unit-tested in tests/test_asr_gpu.py):
  - build_generate_kwargs(model_name, cfg)  num_beams; language/task only for multilingual models
  - speech_seconds(y, sr)                   seconds of 25 ms frames (10 ms hop) louder than silence_dbfs
  - normalize_for_wer(text), wer(ref, hyp)  word error rate by word-level edit distance (no jiwer)
  - ravdess_statement(filename)             the sentence spoken in a RAVDESS clip (field 5)

ASR(cfg, device, dtype, model_name=None)
    load() once, warm_up() once, then transcribe(audio, sr=None) -> dict
    {text, skipped, reason, audio_seconds, asr_seconds, model}.
    Audio becomes mono float32 at 16 kHz. It is not trimmed or loudness-normalized: Whisper does its
    own feature extraction. Silence (too little speech) is skipped without calling the model.

Usage (works from any folder):
    python gpu/asr_gpu.py --preset smoke_cpu --audio data/my_voice_v2
    python gpu/asr_gpu.py --preset smoke_cpu --audio data/my_voice_v2/03_angry.wav
    python gpu/asr_gpu.py --preset smoke_cpu --ravdess 24
    python gpu/asr_gpu.py --preset gpu_16gb --ravdess 24 --model openai/whisper-small.en

Output: results/metrics/asr_<set>_<model-short>.csv, e.g. asr_ravdess_whisper-tiny.en.csv
(a timestamp is added to the name if that file already exists; nothing is overwritten).
"""

import argparse
import csv
import datetime
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

from device import StageTimer, code_path, setup

SAMPLE_RATE = 16000
FRAME_SECONDS = 0.025      # silence gate: 25 ms frames
HOP_SECONDS = 0.010        # ... every 10 ms
CHUNK_SECONDS = 30         # Whisper's window; longer audio goes through the pipeline's chunking
WARMUP_NOISE_STD = 1e-3    # warm-up input: 1 s of low noise (about -60 dBFS)

CLASSES = ["neutral", "happy", "angry", "sad"]
MANIFEST = "data/manifests/ser_4class.csv"
RAVDESS_SEED = 42
RAVDESS_STATEMENTS = {"01": "kids are talking by the door", "02": "dogs are sitting by the door"}
REFERENCE_TEXT_COLUMNS = ("text", "sentence", "transcript", "reference")
CSV_FIELDS = ["set", "file", "emotion", "reference", "text", "wer", "skipped", "reason",
              "audio_seconds", "asr_seconds", "model"]


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------

def is_english_only(model_name) -> bool:
    """openai/whisper-tiny.en -> True. English-only Whisper checkpoints end in '.en'."""
    return str(model_name).rstrip("/\\").lower().endswith(".en")


def build_generate_kwargs(model_name, cfg) -> dict:
    """num_beams = asr.beam_size. English-only (.en) models get NO language/task (they reject them);
    multilingual models get language (asr.language, "en") and task "transcribe" (no language detection,
    no translation)."""
    asr_cfg = cfg["asr"]
    kwargs = {"num_beams": int(asr_cfg.get("beam_size", 1))}
    if not is_english_only(model_name):
        kwargs["language"] = asr_cfg.get("language") or "en"
        kwargs["task"] = "transcribe"
    return kwargs


def speech_seconds(y, sr, silence_dbfs) -> float:
    """Seconds of audio in 25 ms frames (10 ms hop) whose RMS is above silence_dbfs.
    Each frame louder than the threshold counts for one hop (10 ms). Audio shorter than one frame is one frame."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return 0.0
    frame = max(1, int(round(FRAME_SECONDS * sr)))
    hop = max(1, int(round(HOP_SECONDS * sr)))
    if y.size < frame:
        frames = y[None, :]
    else:
        frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    db = 20.0 * np.log10(rms + 1e-12)
    return float(np.count_nonzero(db > silence_dbfs)) * hop / sr


def normalize_for_wer(text) -> str:
    """Lowercase, apostrophes removed ("don't" -> "dont"), other punctuation -> space, spaces collapsed."""
    t = str(text or "").lower().replace("’", "'").replace("'", "")
    t = re.sub(r"[^\w\s]|_", " ", t)
    return " ".join(t.split())


def _edit_counts(ref, hyp):
    """(word edits, reference words) after normalize_for_wer, by Levenshtein distance over words."""
    r, h = normalize_for_wer(ref).split(), normalize_for_wer(hyp).split()
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1,                  # deletion
                         cur[j - 1] + 1,               # insertion
                         prev[j - 1] + (rw != hw))     # substitution / match
        prev = cur
    return prev[-1], len(r)


def wer(ref, hyp) -> float:
    """Word error rate = (substitutions + deletions + insertions) / reference words, on normalized text.
    An empty reference gives 0.0 if the hypothesis is empty too, else 1.0."""
    edits, n = _edit_counts(ref, hyp)
    if n == 0:
        return 0.0 if edits == 0 else 1.0
    return edits / n


def ravdess_statement(filename) -> str:
    """'03-01-05-01-02-01-12.wav' -> 'dogs are sitting by the door' (field 5: 01 kids, 02 dogs)."""
    parts = Path(str(filename)).stem.split("-")
    if len(parts) != 7 or parts[4] not in RAVDESS_STATEMENTS:
        raise ValueError(f"Not a RAVDESS file name with statement 01/02 in field 5: {filename!r}")
    return RAVDESS_STATEMENTS[parts[4]]


def _is_ravdess_name(filename) -> bool:
    try:
        ravdess_statement(filename)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------

class ASR:
    """Whisper via the transformers ASR pipeline. Load once, warm up once, transcribe many."""

    def __init__(self, cfg, device, dtype, model_name=None):
        import torch

        asr_cfg = cfg["asr"]
        self.cfg = cfg
        self.model_name = model_name or asr_cfg["model"]
        self.device = device
        self.dtype = dtype if device == "cuda" else torch.float32
        self.silence_dbfs = float(asr_cfg["silence_dbfs"])
        self.min_speech_seconds = float(asr_cfg["min_speech_seconds"])
        self.generate_kwargs = build_generate_kwargs(self.model_name, cfg)
        self.pipe = None   # set by load(); tests put a stub here

    def load(self):
        from transformers import pipeline

        self.pipe = pipeline("automatic-speech-recognition", model=self.model_name,
                             dtype=self.dtype, device=self.device)
        return self

    def warm_up(self):
        """One model call on 1 s of low noise, bypassing the silence gate (first call pays CUDA/kernel setup)."""
        noise = np.random.default_rng(0).normal(0.0, WARMUP_NOISE_STD, SAMPLE_RATE).astype(np.float32)
        self._run_model(noise)
        return self

    @classmethod
    def from_preset(cls, preset=None, model_name=None):
        """setup(preset) -> loaded, warmed-up ASR. model_name overrides the preset's asr.model."""
        cfg, device, dtype = setup(preset)
        return cls(cfg, device, dtype, model_name=model_name).load().warm_up()

    # -- audio in --

    @staticmethod
    def to_mono_16k(audio, sr=None):
        """(float32 mono 16 kHz array, original sr). audio is a file path or an array (then sr is required).
        No trimming, no loudness normalization."""
        if isinstance(audio, (str, Path)):
            import soundfile as sf

            y, sr = sf.read(str(audio), dtype="float32")
        else:
            if sr is None:
                raise ValueError("sr is required when audio is an array (e.g. transcribe(y, sr=16000))")
            y = np.asarray(audio)
            if np.issubdtype(y.dtype, np.integer):      # int PCM -> [-1, 1]
                y = y.astype(np.float32) / float(np.iinfo(y.dtype).max)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:   # soundfile gives (frames, channels); librosa gives (channels, frames)
            y = y.mean(axis=int(np.argmin(y.shape)))
        elif y.ndim > 2:
            raise ValueError(f"audio must be 1-D (mono) or 2-D (stereo), got shape {y.shape}")
        if y.size and sr != SAMPLE_RATE:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        return np.ascontiguousarray(y, dtype=np.float32), sr

    # -- model call --

    def _run_model(self, y) -> str:
        if self.pipe is None:
            raise RuntimeError("ASR model not loaded: call load() (or use ASR.from_preset).")
        kwargs = {"generate_kwargs": dict(self.generate_kwargs)}
        if y.size > CHUNK_SECONDS * SAMPLE_RATE:
            kwargs["chunk_length_s"] = CHUNK_SECONDS
        out = self.pipe({"raw": y, "sampling_rate": SAMPLE_RATE}, **kwargs)
        return str(out["text"]).strip()

    def transcribe(self, audio, sr=None) -> dict:
        """{text, skipped, reason, audio_seconds, asr_seconds, model}. reason is "empty", "silence" or None.
        asr_seconds covers the whole call (read, resample, silence gate, model)."""
        t0 = time.perf_counter()
        y, _ = self.to_mono_16k(audio, sr)
        result = {"text": "", "skipped": False, "reason": None,
                  "audio_seconds": round(y.size / SAMPLE_RATE, 3), "asr_seconds": 0.0, "model": self.model_name}
        if y.size == 0:
            result.update(skipped=True, reason="empty")
        elif speech_seconds(y, SAMPLE_RATE, self.silence_dbfs) < self.min_speech_seconds:
            result.update(skipped=True, reason="silence")
        else:
            result["text"] = self._run_model(y)
        result["asr_seconds"] = round(time.perf_counter() - t0, 4)
        return result


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def _load_env():
    """Load Code/.env (HF_TOKEN) if python-dotenv is installed; shell variables win."""
    try:
        from dotenv import load_dotenv
    except Exception:  # noqa: BLE001
        return
    load_dotenv(code_path(".env"), override=False)


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else code_path(p)


def _rel(path) -> str:
    try:
        return Path(path).resolve().relative_to(code_path("").resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def model_short(model_name) -> str:
    """openai/whisper-tiny.en -> whisper-tiny.en"""
    return str(model_name).rstrip("/\\").replace("\\", "/").split("/")[-1]


def new_csv_path(set_name, model_name) -> Path:
    out_dir = code_path("results/metrics")
    stem = f"asr_{set_name}_{model_short(model_name)}"
    path = out_dir / f"{stem}.csv"
    if path.exists():
        path = out_dir / f"{stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    return path


def find_sentence_list(folder):
    """{wav file name: reference text} from a .csv next to the clips that has a file column and a text column
    (text / sentence / transcript / reference). Returns (mapping, csv path) or ({}, None)."""
    for csv_path in sorted(Path(folder).glob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
            file_col = cols.get("file") or cols.get("filename") or cols.get("wav")
            text_col = next((cols[c] for c in REFERENCE_TEXT_COLUMNS if c in cols), None)
            if not (file_col and text_col):
                continue
            mapping = {}
            for row in reader:
                name = Path(row[file_col].strip()).name
                mapping[name if name.lower().endswith(".wav") else name + ".wav"] = row[text_col]
            return mapping, csv_path
    return {}, None


def ravdess_test_clips(n, seed=RAVDESS_SEED):
    """[(path, emotion)]: n RAVDESS test-split clips from the 4-class manifest, balanced across CLASSES
    (n // 4 each; any remainder goes to the first classes in CLASSES order), sampled with random.Random(seed)."""
    with open(code_path(MANIFEST), newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["corpus"] == "ravdess" and r["split"] == "test" and r["in_4class"] == "True"
                and r["label"] in CLASSES]
    rng = random.Random(seed)
    clips = []
    for i, label in enumerate(CLASSES):
        k = n // len(CLASSES) + (1 if i < n % len(CLASSES) else 0)
        pool = sorted((r for r in rows if r["label"] == label), key=lambda r: r["utt_id"])
        if k > len(pool):
            raise ValueError(f"Asked for {k} {label} clips but the RAVDESS test split has only {len(pool)}")
        clips += [(_resolve(r["wav_path"]), label) for r in rng.sample(pool, k)]
    missing = [p for p, _ in clips if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} RAVDESS clip(s) not found, e.g. {missing[0]}. "
                                "Is RAVDESS in data/raw/Actor_XX/?")
    return clips


def _run_clips(asr, set_name, clips):
    """clips = [(path, reference or None, emotion or "")]. Prints one line per clip; returns CSV rows."""
    rows = []
    for path, ref, emotion in clips:
        out = asr.transcribe(path)
        w = wer(ref, out["text"]) if ref is not None else None
        rows.append({
            "set": set_name, "file": _rel(path), "emotion": emotion, "reference": ref or "",
            "text": out["text"], "wer": "" if w is None else round(w, 4),
            "skipped": out["skipped"], "reason": out["reason"] or "",
            "audio_seconds": out["audio_seconds"], "asr_seconds": out["asr_seconds"], "model": asr.model_name,
        })
        status = f"SKIPPED ({out['reason']})" if out["skipped"] else "ok"
        wer_txt = f"  WER {w:.2f}" if w is not None else ""
        emo_txt = f"  [{emotion}]" if emotion else ""
        print(f"  {Path(path).name:28s}{emo_txt}  {out['audio_seconds']:5.2f}s audio  "
              f"{out['asr_seconds']:.2f}s asr  {status}{wer_txt}")
        print(f"      text: {out['text']!r}" + (f"\n      ref : {ref!r}" if ref is not None and w else ""))
    return rows


def _corpus_wer(rows, refs) -> float:
    edits = words = 0
    for r, ref in zip(rows, refs):
        e, n = _edit_counts(ref, r["text"])
        edits, words = edits + e, words + n
    return edits / words if words else 0.0


def _summary(rows, refs, by_emotion=False):
    scored = [(r, ref) for r, ref in zip(rows, refs) if ref is not None]
    skipped = sum(r["skipped"] for r in rows)
    mean_s = np.mean([r["asr_seconds"] for r in rows]) if rows else 0.0
    print(f"\nclips {len(rows)}   skipped {skipped}   mean seconds per clip {mean_s:.3f}")
    if not scored:
        return
    rs, fs = zip(*scored)
    clip_wers = [wer(ref, r["text"]) for r, ref in scored]
    print(f"overall WER {_corpus_wer(rs, fs):.3f}  (all word errors / all reference words, {len(scored)} clips; "
          f"mean per-clip WER {np.mean(clip_wers):.3f}; skipped clips count as all words deleted)")
    if by_emotion:
        for emo in CLASSES:
            sub = [(r, ref) for r, ref in scored if r["emotion"] == emo]
            if sub:
                er, ef = zip(*sub)
                print(f"  {emo:8s} WER {_corpus_wer(er, ef):.3f}  ({len(sub)} clips, "
                      f"mean seconds {np.mean([r['asr_seconds'] for r in er]):.3f})")


def _write_csv(rows, set_name, model_name):
    path = new_csv_path(set_name, model_name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-clip rows: {_rel(path)}")


def run_audio(asr, target):
    path = _resolve(target)
    if path.is_dir():
        wavs = sorted(path.glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No .wav files in {path}")
        refs_by_name, list_path = find_sentence_list(path)
        if list_path:
            print(f"Sentence list: {_rel(list_path)} ({len(refs_by_name)} rows) -> WER per clip")
        else:
            print(f"Sentence list: none found (looked for a .csv with file + text columns in {_rel(path)}); "
                  "WER is computed only for RAVDESS-named clips")
        set_name = path.name
    elif path.is_file():
        wavs, refs_by_name, set_name = [path], {}, path.stem
    else:
        raise FileNotFoundError(f"Not found: {path}")
    refs = [refs_by_name.get(w.name) or (ravdess_statement(w.name) if _is_ravdess_name(w.name) else None)
            for w in wavs]
    rows = _run_clips(asr, set_name, [(w, ref, "") for w, ref in zip(wavs, refs)])
    _summary(rows, refs)
    _write_csv(rows, set_name, asr.model_name)


def run_ravdess(asr, n):
    clips = ravdess_test_clips(n)
    print(f"RAVDESS test split: {len(clips)} clips, seed {RAVDESS_SEED}, "
          + ", ".join(f"{c} {sum(e == c for _, e in clips)}" for c in CLASSES))
    refs = [ravdess_statement(p.name) for p, _ in clips]
    rows = _run_clips(asr, "ravdess", [(p, ref, emo) for (p, emo), ref in zip(clips, refs)])
    _summary(rows, refs, by_emotion=True)
    _write_csv(rows, "ravdess", asr.model_name)


def _main():
    parser = argparse.ArgumentParser(description="Whisper speech-to-text (GPU demo build, step 6).")
    parser.add_argument("--preset", help="preset name (default: $AGENT_PRESET, else smoke_cpu)")
    parser.add_argument("--model", help="override the preset's asr.model (e.g. openai/whisper-base.en)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audio", metavar="PATH", help="a .wav file or a folder of .wav files (relative to Code/)")
    mode.add_argument("--ravdess", metavar="N", type=int, nargs="?", const=24,
                      help="N RAVDESS test-split clips, balanced across the 4 emotions, seed 42 (default 24)")
    args = parser.parse_args()
    if args.ravdess is not None and args.ravdess < 1:
        parser.error("--ravdess N must be at least 1")

    _load_env()
    try:
        cfg, device, dtype = setup(args.preset)
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    asr = ASR(cfg, device, dtype, model_name=args.model)
    timer = StageTimer()
    with timer("load"):
        asr.load()
    with timer("warm_up"):
        asr.warm_up()
    print(f"preset {cfg['preset']}  device {device}  dtype {str(asr.dtype).replace('torch.', '')}  "
          f"model {asr.model_name}  generate {asr.generate_kwargs}")
    print(f"loaded in {timer.times['load']:.1f}s, warm-up {timer.times['warm_up']:.2f}s   "
          f"silence gate: < {asr.min_speech_seconds}s above {asr.silence_dbfs} dBFS -> skipped\n")

    try:
        if args.audio:
            run_audio(asr, args.audio)
        else:
            run_ravdess(asr, args.ravdess)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
