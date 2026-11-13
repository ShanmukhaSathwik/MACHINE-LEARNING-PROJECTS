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
    python gpu/ser_gpu.py --preset smoke_cpu --compare \
        --models superb/hubert-large-superb-er superb/wav2vec2-base-superb-er --manifest-split test

Output of --compare: results/metrics/ser_candidates_my_voice.csv (folders) or
results/metrics/ser_candidates_ravdess_<split>.csv (--manifest-split; Check A wrapper sanity rule)
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
# Check A (wrapper sanity check on RAVDESS), fixed in advance: OK if accuracy >= 50% and no class gets > 70% of predictions.
MANIFEST = "data/manifests/manifest.csv"
MANIFEST_CSV_PREFIX = "ser_candidates_ravdess_"
WRAPPER_MIN_ACC = 0.50
WRAPPER_MAX_SHARE = 0.70
# Check A2 (--verify), fixed in advance: WRAPPER CORRECT if our probs match the HF pipeline within 1e-3 on every clip.
VERIFY_MODELS = ["superb/hubert-large-superb-er", "superb/wav2vec2-base-superb-er"]
VERIFY_N_CLIPS = 20
VERIFY_SEED = 42
VERIFY_TOL = 1e-3
DEMO_REPO, DEMO_CONFIG, DEMO_SPLIT = "anton-l/superb_demo", "er", "session1"
VERIFY_FIELDS = ["model", "check", "file", "true_label", "pred_label", "compare_label", "max_abs_diff",
                 "confidence", "p_neutral", "p_happy", "p_angry", "p_sad"]
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


class LabelOrderError(RuntimeError):
    """A fine-tuned checkpoint's saved label order (best/id2label.json) disagrees with the model it holds.
    Never caught by the pretrained fallback: a model whose outputs mean something else must not run."""


def resolve_checkpoint(cfg):
    """ser.checkpoint (relative to Code/; gpu/outputs/... goes to $GPU_OUTPUTS_DIR when set) -> Path, or None."""
    value = (cfg.get("ser") or {}).get("checkpoint")
    return code_path(value) if value else None


def check_label_order(saved, model_id2label) -> dict:
    """saved: best/id2label.json ({"0": {"class": "neutral", "model_label": "neu"}, ...}, written by finetune_ser.py);
    model_id2label: the loaded model's config.id2label. Returns {index: class}; raises LabelOrderError on any
    difference (indices, the raw label at an index, or the class it maps to)."""
    want = {int(i): (str(v["class"]), str(v["model_label"])) for i, v in dict(saved).items()}
    have = {int(i): str(lab) for i, lab in dict(model_id2label).items()}
    if set(want) != set(have):
        raise LabelOrderError(f"id2label.json has indices {sorted(want)}, the model has {sorted(have)}")
    wrong = {i: (want[i][1], have[i]) for i in want if want[i][1] != have[i]}
    if wrong:
        raise LabelOrderError(f"label at index differs (id2label.json, model): {wrong}")
    mapped = map_labels(have)
    classes = {i: want[i][0] for i in want}
    if mapped != classes:
        raise LabelOrderError(f"id2label.json classes {classes} != model labels mapped to classes {mapped}")
    return mapped


def load_ser_with_fallback(cfg, device, dtype, model_cls=None, warn=print):
    """(model, info). The ser.checkpoint folder (T3) if it exists, loads and passes check_label_order; otherwise
    the pretrained ser.model with ONE warning. info = {"name", "fallback", "checkpoint", "reason"}.
    A LabelOrderError is raised, never hidden behind the fallback."""
    model_cls = model_cls or SERModel
    ckpt = resolve_checkpoint(cfg)
    reason = None
    if ckpt is None:
        reason = "ser.checkpoint is not set"
    elif not ckpt.is_dir():
        reason = f"checkpoint folder not found: {ckpt}"
    elif not (ckpt / "id2label.json").is_file():
        reason = f"id2label.json not found in {ckpt}"
    else:
        try:
            model = model_cls(cfg, device, dtype, model_name=str(ckpt))
        except Exception as e:  # noqa: BLE001  (any load failure -> pretrained)
            reason = f"checkpoint failed to load ({type(e).__name__}: {str(e)[:200]})"
        else:
            saved = json.loads((ckpt / "id2label.json").read_text(encoding="utf-8"))
            check_label_order(saved, model.model.config.id2label)          # raises LabelOrderError
            return model, {"name": str(ckpt), "fallback": False, "checkpoint": str(ckpt), "reason": None}
    name = cfg["ser"]["model"]
    warn(f"WARNING: emotion model fallback: {reason}. Using the pretrained {name} instead.")
    return model_cls(cfg, device, dtype, model_name=name), {"name": name, "fallback": True,
                                                            "checkpoint": str(ckpt) if ckpt else None,
                                                            "reason": reason}


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

    def preprocess(self, y, sr, trim=True) -> np.ndarray:
        """Mono, 16 kHz, trim silence (unless trim=False), cut to max_seconds. No loudness normalization."""
        import librosa

        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:   # soundfile gives (frames, channels), librosa gives (channels, frames)
            y = y.mean(axis=int(np.argmin(y.shape)))
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        if trim:
            y_trim, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
            if y_trim.size >= MIN_TRIMMED_SECONDS * SAMPLE_RATE:
                y = y_trim
        y = y[: int(self.max_seconds * SAMPLE_RATE)]
        return np.ascontiguousarray(y, dtype=np.float32)

    def predict_array(self, y, sr, trim=True) -> dict:
        return self._infer(self.preprocess(y, sr, trim))

    def predict_file(self, path, trim=True) -> dict:
        import soundfile as sf

        y, sr = sf.read(str(path), dtype="float32")
        return self.predict_array(y, sr, trim)


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


def _new_csv_path(name=CSV_NAME) -> Path:
    out_dir = code_path("results/metrics")
    path = out_dir / f"{name}.csv"
    if path.exists():
        path = out_dir / f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    return path


def _folder_clips(folder):
    """[(path, true label or None)] for every .wav in folder, sorted; label from the file name."""
    return [(wav, true_label_from_filename(wav.name)) for wav in sorted(folder.glob("*.wav"))]


def _manifest_clips(split):
    """[(path, label)] for the manifest rows where split == split, in manifest order."""
    with open(code_path(MANIFEST), newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if not rows:
        raise ValueError(f"No rows with split == {split!r} in {MANIFEST}")
    for r in rows:
        if r["label"] not in CLASSES:
            raise ValueError(f"Manifest label {r['label']!r} ({r['wav_path']}) is not one of {CLASSES}")
    return [(_resolve(r["wav_path"]), r["label"]) for r in rows]


def _prediction_shares(rows) -> dict:
    return {c: sum(r["pred_label"] == c for r in rows) / len(rows) for c in CLASSES}


def _print_folder_summary(model, folder, rows):
    scored = [r for r in rows if r["true_label"]]
    correct = sum(r["true_label"] == r["pred_label"] for r in scored)
    pct = f" = {correct / len(scored):.1%}" if scored else ""
    print(f"\n[{model}]  {folder}: {correct}/{len(scored)} correct{pct} "
          f"({len(rows)} clips, {len(rows) - len(scored)} without a true label)")
    print("  share of predictions: " + "  ".join(f"{c} {s:.1%}" for c, s in _prediction_shares(rows).items()))
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


def _run_sources(cfg, device, dtype, models, sources, out_path):
    """sources = [(name, [(path, true label or None), ...])]. Runs every model on every clip,
    prints a summary per model x source, writes the per-clip CSV. Returns (acc, rows_by_key)."""
    rows, acc, by_key = [], {}, {}
    for model_name in models:
        load = StageTimer()
        with load("load"):
            ser = SERModel(cfg, device, dtype, model_name=model_name)
        print(f"\n=== {model_name}  (loaded + warm-up in {load.times['load']:.1f}s on {device}) ===")
        for source, clips in sources:
            folder_rows = []
            for wav, true_label in clips:
                timer = StageTimer()
                with timer("ser"):
                    out = ser.predict_file(wav)
                folder_rows.append({
                    "model": model_name, "folder": source,
                    "file": wav.name if not source.startswith("manifest:") else _rel(wav),
                    "true_label": true_label or "",
                    "pred_label": out["label"],
                    "confidence": round(out["confidence"], 4), "margin": round(out["margin"], 4),
                    **{f"p_{c}": round(out["probs"][c], 4) for c in CLASSES},
                    "seconds": round(timer.times["ser"], 4),
                })
            acc[(model_name, source)] = _print_folder_summary(model_name, source, folder_rows)
            by_key[(model_name, source)] = folder_rows
            rows.extend(folder_rows)
        del ser

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-clip rows: {out_path}")
    return acc, by_key


def compare_manifest(cfg, device, dtype, models, split):
    """Check A: wrapper sanity check on a manifest split (RAVDESS), with its own fixed rule."""
    source = f"manifest:{split}"
    sources = [(source, _manifest_clips(split))]
    out_path = _new_csv_path(f"{MANIFEST_CSV_PREFIX}{split}")
    acc, by_key = _run_sources(cfg, device, dtype, models, sources, out_path)

    print(f"\n=== Wrapper check (rule fixed in advance): OK if accuracy >= {WRAPPER_MIN_ACC:.0%} "
          f"AND no class gets > {WRAPPER_MAX_SHARE:.0%} of predictions ===")
    for m in models:
        c, n = acc[(m, source)]
        shares = _prediction_shares(by_key[(m, source)])
        top = max(shares, key=shares.get)
        ok = c / n >= WRAPPER_MIN_ACC and shares[top] <= WRAPPER_MAX_SHARE
        print(f"  {m:40s} accuracy {c}/{n} = {c / n:.1%}   largest share {top} {shares[top]:.1%}   "
              f"-> {'wrapper OK' if ok else 'possible wrapper bug'}")


def _pipeline_probs(clf, y) -> dict:
    """Run a transformers audio-classification pipeline on a 16 kHz array; probs summed into CLASSES."""
    out = clf({"raw": y, "sampling_rate": SAMPLE_RATE}, top_k=None)
    probs = {c: 0.0 for c in CLASSES}
    for d in out:
        probs[map_labels({0: d["label"]})[0]] += float(d["score"])
    return probs


def _max_diff(p, q) -> float:
    return max(abs(p[c] - q[c]) for c in CLASSES)


def _load_demo_clips():
    """[(file name, waveform, sr, true class)] from the model-card demo set, read with soundfile (no datasets decoding)."""
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(DEMO_REPO, DEMO_CONFIG, split=DEMO_SPLIT)
    names = ds.features["label"].names
    ds = ds.cast_column("audio", Audio(decode=False))
    clips = []
    for row in ds:
        a = row["audio"]
        raw = a.get("bytes") or Path(a["path"]).read_bytes()
        y, sr = sf.read(io.BytesIO(raw), dtype="float32")
        clips.append((Path(a.get("path") or row["file"]).name, y, sr, map_labels({0: names[row["label"]]})[0]))
    return clips


def verify(cfg, device, dtype, models):
    """Check A2: (b) ours vs HF pipeline on the same arrays, (c) trim on vs off, (d) model-card demo set."""
    import random

    import soundfile as sf
    import torch
    from transformers import pipeline

    def row(model, check, file, true, out, compare_label="", diff=""):
        return {"model": model, "check": check, "file": file, "true_label": true or "",
                "pred_label": out["label"], "compare_label": compare_label,
                "max_abs_diff": f"{diff:.2e}" if diff != "" else "", "confidence": round(out["confidence"], 4),
                **{f"p_{c}": round(out["probs"][c], 4) for c in CLASSES}}

    if device != "cpu" or dtype != torch.float32:
        print(f"NOTE: our model runs on {device}/{dtype}; the pipeline runs on CPU float32, "
              f"so the {VERIFY_TOL:g} tolerance assumes --preset smoke_cpu.")
    test_clips = _manifest_clips("test")
    sample = random.Random(VERIFY_SEED).sample(test_clips, VERIFY_N_CLIPS)
    demo = _load_demo_clips()
    out_path = _new_csv_path("ser_verify")
    rows, verdicts = [], {}

    for model_name in models:
        ser = SERModel(cfg, device, dtype, model_name=model_name)
        clf = pipeline("audio-classification", model=model_name, top_k=None, device="cpu")
        print(f"\n=== {model_name} ===")

        # (b) same preprocessed array -> ours vs pipeline
        print(f"(b) ours vs transformers pipeline on {VERIFY_N_CLIPS} RAVDESS test clips (seed {VERIFY_SEED}), same array:")
        diffs = []
        for wav, true in sample:
            y, sr = sf.read(str(wav), dtype="float32")
            y16 = ser.preprocess(y, sr)
            ours = ser._infer(y16)
            theirs = _pipeline_probs(clf, y16)
            d = _max_diff(ours["probs"], theirs)
            diffs.append((d, _rel(wav), ours["label"], max(theirs, key=theirs.get)))
            rows.append(row(model_name, "pipeline_vs_ours", _rel(wav), true, ours, max(theirs, key=theirs.get), d))
        for d, f, a, b in diffs:
            print(f"    {f:48s} ours={a:7s} pipeline={b:7s} max|diff|={d:.2e}")
        worst = max(diffs)
        ok = worst[0] < VERIFY_TOL
        verdicts[model_name] = (ok, worst)
        print(f"    max over clips: {worst[0]:.2e}  ->  {'WRAPPER CORRECT' if ok else 'WRAPPER BUG'}"
              + ("" if ok else f"  (largest difference: {worst[1]})"))

        # (c) trim on vs off on all test clips
        changed = []
        for wav, true in test_clips:
            on, off = ser.predict_file(wav, trim=True), ser.predict_file(wav, trim=False)
            if on["label"] != off["label"]:
                changed.append((_rel(wav), true, on["label"], off["label"]))
            rows.append(row(model_name, "trim_on_vs_off", _rel(wav), true, on, off["label"],
                            _max_diff(on["probs"], off["probs"])))
        acc_on = sum(r["pred_label"] == r["true_label"] for r in rows
                     if r["model"] == model_name and r["check"] == "trim_on_vs_off")
        acc_off = sum(r["compare_label"] == r["true_label"] for r in rows
                      if r["model"] == model_name and r["check"] == "trim_on_vs_off")
        print(f"(c) trim OFF vs ON on {len(test_clips)} RAVDESS test clips: {len(changed)} predictions change   "
              f"(accuracy trim on {acc_on}/{len(test_clips)}, trim off {acc_off}/{len(test_clips)})")
        for f, t, a, b in changed:
            print(f"    {f:48s} true={t:6s} on={a:7s} off={b}")

        # (d) model-card demo set
        correct = 0
        print(f"(d) {DEMO_REPO} / {DEMO_CONFIG} / {DEMO_SPLIT} (IEMOCAP, the models' home data):")
        for name, y, sr, true in demo:
            out = ser.predict_array(y, sr)
            correct += out["label"] == true
            rows.append(row(model_name, "superb_demo", name, true, out))
            print(f"    {name:30s} true={true:7s} pred={out['label']:7s} conf={out['confidence']:.3f}"
                  f"  {'ok' if out['label'] == true else 'WRONG'}")
        print(f"    {correct}/{len(demo)} correct")
        del ser, clf

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VERIFY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRows: {out_path}")
    print(f"\n=== Verdict (rule fixed in advance: max|diff| < {VERIFY_TOL:g} on every clip in (b)) ===")
    for m, (ok, worst) in verdicts.items():
        print(f"  {m:40s} max|diff| {worst[0]:.2e}  -> {'WRAPPER CORRECT' if ok else 'WRAPPER BUG (' + worst[1] + ')'}")


def compare(cfg, device, dtype, models, folders):
    folder_paths = [_resolve(f) for f in folders]
    for fp in folder_paths:
        if not fp.is_dir():
            raise FileNotFoundError(f"Folder not found: {fp}")
    sources = [(_rel(fp), _folder_clips(fp)) for fp in folder_paths]
    acc, _ = _run_sources(cfg, device, dtype, models, sources, _new_csv_path())

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
    mode.add_argument("--verify", action="store_true",
                      help="Check A2: compare against the transformers pipeline, trim on/off, model-card demo set")
    parser.add_argument("--models", nargs="+", help="model names for --compare (default: ser.candidates, else ser.model)")
    parser.add_argument("--folders", nargs="+",
                        help="folders of .wav clips, relative to Code/ (default: data/my_voice data/my_voice_v2)")
    parser.add_argument("--manifest-split", metavar="SPLIT",
                        help=f"--compare on the {MANIFEST} rows with this split (e.g. test) instead of folders")
    args = parser.parse_args()
    if args.manifest_split and args.folders:
        parser.error("use either --folders or --manifest-split, not both")
    if args.manifest_split and not args.compare:
        parser.error("--manifest-split works with --compare only")

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

    if args.verify:
        verify(cfg, device, dtype, args.models or VERIFY_MODELS)
        return 0

    models = args.models or cfg["ser"].get("candidates") or [cfg["ser"]["model"]]
    if args.manifest_split:
        compare_manifest(cfg, device, dtype, models, args.manifest_split)
    else:
        compare(cfg, device, dtype, models, args.folders or [V1_FOLDER, V2_FOLDER])
    return 0


if __name__ == "__main__":
    sys.exit(_main())
