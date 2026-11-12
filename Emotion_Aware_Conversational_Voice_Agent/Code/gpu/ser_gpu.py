"""
ser_gpu.py - pretrained speech emotion recognition (SER) for the GPU demo build.

Step 2 of the GPU demo build. Wraps a Hugging Face audio-classification model
(the superb/*-superb-er family) behind one call that returns the same 4 classes
the rest of the agent uses, plus p_distress for MoodManager.

Pure helpers (no downloads, unit-tested in tests/test_ser_gpu.py):
  - map_labels(id2label)        model labels ("neu"/"hap"/"ang"/"sad" or full words) -> CLASSES
  - build_output(logits, map)   softmax -> probs, label, confidence, margin, p_distress
  - true_label_from_filename()  "03_happy.wav" -> "happy"

SERModel(cfg, device, dtype)    loads the model once, then predict_array / predict_file.
No loudness normalization is added here: the clip goes to the model's own feature extractor as recorded.

Usage (works from any folder):
    python gpu/ser_gpu.py --preset smoke_cpu --file data/my_voice_v2/03_angry.wav
    python gpu/ser_gpu.py --preset smoke_cpu --compare \
        --models superb/hubert-large-superb-er superb/wav2vec2-base-superb-er \
        --folders data/my_voice data/my_voice_v2

Output of --compare: results/metrics/ser_candidates_my_voice.csv
(a timestamp is added to the name if that file already exists; nothing is overwritten).
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np

from device import StageTimer, code_path, setup

CLASSES = ["neutral", "happy", "angry", "sad"]
SAMPLE_RATE = 16000
TRIM_TOP_DB = 30          # same trim level as scripts/feature_extraction.py
MIN_TRIMMED_SECONDS = 0.5  # if trimming leaves less than this, keep the untrimmed clip

LABEL_ALIASES = {
    "neu": "neutral", "neutral": "neutral",
    "hap": "happy", "happy": "happy", "happiness": "happy",
    "ang": "angry", "angry": "angry", "anger": "angry",
    "sad": "sad", "sadness": "sad",
}

# Fixed in advance (step 2 plan), before any result was seen.
V2_FOLDER = "data/my_voice_v2"
V1_FOLDER = "data/my_voice"
PASS_RATIO = 10 / 15
SMALLER_MODEL = "superb/wav2vec2-base-superb-er"
XGB_REFERENCE = "XGBoost baseline on my_voice_v2: 5/15"
CSV_NAME = "ser_candidates_my_voice"
CSV_FIELDS = ["model", "folder", "file", "true_label", "pred_label", "confidence", "margin",
              "p_neutral", "p_happy", "p_angry", "p_sad", "seconds"]


# Pure helpers

def map_labels(id2label) -> dict:
    """{model output index: one of CLASSES}. Raises ValueError naming any label it does not know."""
    index_map, unknown = {}, []
    for idx, label in id2label.items():
        cls = LABEL_ALIASES.get(str(label).strip().lower())
        if cls is None:
            unknown.append(label)
        else:
            index_map[int(idx)] = cls
    if unknown:
        raise ValueError(f"Unknown emotion label(s) {unknown}; expected one of {sorted(LABEL_ALIASES)}")
    return index_map


def build_output(logits, index_map) -> dict:
    """Softmax over the model outputs, summed into CLASSES (a class the model lacks gets 0)."""
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    p = np.exp(z - z.max())
    p /= p.sum()
    probs = {c: 0.0 for c in CLASSES}
    for idx, cls in index_map.items():
        probs[cls] += float(p[idx])
    ranked = sorted(probs.values(), reverse=True)
    return {
        "probs": probs,
        "label": max(CLASSES, key=lambda c: probs[c]),
        "confidence": ranked[0],
        "margin": ranked[0] - ranked[1],
        "p_distress": probs["angry"] + probs["sad"],
    }


def true_label_from_filename(name):
    """'03_happy.wav' -> 'happy'. Returns None if the last '_' part is not one of CLASSES."""
    tag = Path(name).stem.split("_")[-1].lower()
    return tag if tag in CLASSES else None


# Model

class SERModel:
    def __init__(self, cfg, device, dtype, model_name=None):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.torch = torch
        self.model_name = model_name or cfg["ser"]["model"]
        self.max_seconds = cfg["ser"]["max_seconds"]
        self.device = device
        self.dtype = dtype if device == "cuda" else torch.float32
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.index_map = map_labels(self.model.config.id2label)
        self._infer(np.zeros(SAMPLE_RATE, dtype=np.float32))   # warm-up: 1 s of silence

    def _infer(self, y) -> dict:
        inputs = self.extractor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: (v.to(self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device))
                  for k, v in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0]
        return build_output(logits.float().cpu().numpy(), self.index_map)

    def predict_array(self, y, sr) -> dict:
        import librosa

        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:   # soundfile gives (frames, channels), librosa gives (channels, frames)
            y = y.mean(axis=int(np.argmin(y.shape)))
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        y_trim, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
        if y_trim.size >= MIN_TRIMMED_SECONDS * SAMPLE_RATE:
            y = y_trim
        y = y[: int(self.max_seconds * SAMPLE_RATE)]
        return self._infer(np.ascontiguousarray(y, dtype=np.float32))

    def predict_file(self, path) -> dict:
        import soundfile as sf

        y, sr = sf.read(str(path), dtype="float32")
        return self.predict_array(y, sr)


# Command line

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
        return path.resolve().relative_to(code_path("").resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _new_csv_path() -> Path:
    out_dir = code_path("results/metrics")
    path = out_dir / f"{CSV_NAME}.csv"
    if path.exists():
        path = out_dir / f"{CSV_NAME}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    return path


def _print_folder_summary(model, folder, rows):
    scored = [r for r in rows if r["true_label"]]
    correct = sum(r["true_label"] == r["pred_label"] for r in scored)
    print(f"\n[{model}]  {folder}: {correct}/{len(scored)} correct "
          f"({len(rows)} clips, {len(rows) - len(scored)} without a true label)")
    print(f"  mean confidence {np.mean([r['confidence'] for r in rows]):.3f}   "
          f"mean seconds/clip {np.mean([r['seconds'] for r in rows]):.3f}")
    print("  confusion (rows = true, cols = predicted):")
    print("  " + " " * 9 + "".join(f"{c:>9s}" for c in CLASSES))
    for t in CLASSES:
        if not any(r["true_label"] == t for r in scored):
            continue
        counts = [sum(r["true_label"] == t and r["pred_label"] == p for r in scored) for p in CLASSES]
        print(f"  {t:>9s}" + "".join(f"{n:>9d}" for n in counts))
    return correct, len(scored)


def _pick_winner(models, acc):
    """acc[(model, folder)] = (correct, n). Returns (winner, passed, explanation)."""
    def ratio(m, folder):
        c, n = acc.get((m, folder), (0, 0))
        return c / n if n else 0.0

    best_v2 = max(ratio(m, V2_FOLDER) for m in models)
    tied = [m for m in models if ratio(m, V2_FOLDER) == best_v2]
    why = f"higher accuracy on {V2_FOLDER}"
    if len(tied) > 1:
        best_v1 = max(ratio(m, V1_FOLDER) for m in tied)
        tied = [m for m in tied if ratio(m, V1_FOLDER) == best_v1]
        why = f"tie on {V2_FOLDER}, higher accuracy on {V1_FOLDER}"
        if len(tied) > 1:
            why = f"tie on both folders, smaller model ({SMALLER_MODEL})"
            tied = [SMALLER_MODEL] if SMALLER_MODEL in tied else tied[:1]
    winner = tied[0]
    return winner, ratio(winner, V2_FOLDER) >= PASS_RATIO, why


def compare(cfg, device, dtype, models, folders):
    folder_paths = [_resolve(f) for f in folders]
    for fp in folder_paths:
        if not fp.is_dir():
            raise FileNotFoundError(f"Folder not found: {fp}")
    out_path = _new_csv_path()
    rows, acc = [], {}

    for model_name in models:
        load = StageTimer()
        with load("load"):
            ser = SERModel(cfg, device, dtype, model_name=model_name)
        print(f"\n=== {model_name}  (loaded + warm-up in {load.times['load']:.1f}s on {device}) ===")
        for fp in folder_paths:
            folder = _rel(fp)
            folder_rows = []
            for wav in sorted(fp.glob("*.wav")):
                timer = StageTimer()
                with timer("ser"):
                    out = ser.predict_file(wav)
                folder_rows.append({
                    "model": model_name, "folder": folder, "file": wav.name,
                    "true_label": true_label_from_filename(wav.name) or "",
                    "pred_label": out["label"],
                    "confidence": round(out["confidence"], 4), "margin": round(out["margin"], 4),
                    **{f"p_{c}": round(out["probs"][c], 4) for c in CLASSES},
                    "seconds": round(timer.times["ser"], 4),
                })
            acc[(model_name, folder)] = _print_folder_summary(model_name, folder, folder_rows)
            rows.extend(folder_rows)
        del ser

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-clip rows: {out_path}")

    print("\n=== Decision (rule fixed in advance) ===")
    print(f"  winner = higher accuracy on {V2_FOLDER}, must be >= 10/15 to pass; "
          f"tie -> {V1_FOLDER}; still tied -> {SMALLER_MODEL}")
    for m in models:
        parts = [f"{f}: {c}/{n}" for (mm, f), (c, n) in acc.items() if mm == m]
        print(f"  {m:40s} " + "   ".join(parts))
    if not any(f == V2_FOLDER for (_, f) in acc):
        print(f"  {V2_FOLDER} was not in --folders, so the rule cannot be applied.")
        return
    winner, passed, why = _pick_winner(models, acc)
    c, n = acc[(winner, V2_FOLDER)]
    print(f"  WINNER: {winner}  ({why})  {V2_FOLDER}: {c}/{n}  -> {'PASS' if passed else 'FAIL'}")
    print(f"  Reference: {XGB_REFERENCE}")
    print("  gpu_config.yaml is NOT changed by this script.")


def _main():
    parser = argparse.ArgumentParser(description="Pretrained speech emotion model: one clip, or compare candidates.")
    parser.add_argument("--preset", help="preset name (default: $AGENT_PRESET, else smoke_cpu)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file", help="one .wav file to classify")
    mode.add_argument("--compare", action="store_true", help="run every model on every .wav in --folders")
    parser.add_argument("--models", nargs="+", help="model names for --compare (default: ser.candidates, else ser.model)")
    parser.add_argument("--folders", nargs="+", default=[V1_FOLDER, V2_FOLDER],
                        help="folders of .wav clips, relative to Code/ (default: data/my_voice data/my_voice_v2)")
    args = parser.parse_args()

    _load_env()
    try:
        cfg, device, dtype = setup(args.preset)
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"preset {cfg['preset']}  device {device}  dtype {str(dtype).replace('torch.', '')}")

    if args.file:
        ser = SERModel(cfg, device, dtype)
        timer = StageTimer()
        with timer("ser"):
            out = ser.predict_file(_resolve(args.file))
        print(f"model   : {ser.model_name}")
        print(f"file    : {args.file}  (true label from name: {true_label_from_filename(args.file)})")
        print(json.dumps({**out, "seconds": round(timer.times["ser"], 4)}, indent=2))
        return 0

    models = args.models or cfg["ser"].get("candidates") or [cfg["ser"]["model"]]
    compare(cfg, device, dtype, models, args.folders)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
