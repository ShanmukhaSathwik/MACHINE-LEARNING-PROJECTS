"""
eval_my_voice_v3.py - test the fine-tuned emotion model (training T3) on my own recordings. CPU only.

Clips : data/my_voice_v3/NN_<emotion>.wav (32 clips, same sentence; the rejected/ subfolder is ignored).
        True label = the emotion in the file name (neutral / happy / angry / sad).
Model : the T3 best/ folder (hubert-large-superb-er fine-tuned, 4 classes), loaded through gpu/ser_gpu.SERModel
        with the smoke_cpu preset, so preprocessing is identical to the agent (mono, 16 kHz, trim, no normalization).
        Label order is read from best/id2label.json and must match the model's own id2label.

Rule (fixed in advance, before running): PASS if at least 22 of 32 clips are predicted correctly (2/3).
Labels are the emotion I performed, with no independent check. Nothing is tuned or trained on this data.

Output:
  - per-clip table, accuracy x/32, per-class accuracy, 4x4 confusion matrix (rows true, columns predicted), PASS/FAIL
  - mean loudness (RMS dBFS of the trimmed 16 kHz audio the model sees) per true and per predicted emotion
  - results/metrics/my_voice_v3_ser_eval.csv and results/figures/my_voice_v3_confusion.png
    (a timestamp is added to the name if the file already exists; nothing is overwritten)

Usage (from Code/):
    python scripts/eval_my_voice_v3.py
    python scripts/eval_my_voice_v3.py --model C:/emotion-agent/gpu_outputs/ser_finetune/gpu_24gb_20261112_1500/best
"""
import argparse
import csv
import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR / "gpu"))

from ser_gpu import CLASSES  # noqa: E402

CLIP_DIR = CODE_DIR / "data" / "my_voice_v3"
DEFAULT_MODEL = Path(r"C:\emotion-agent\gpu_outputs\ser_finetune\gpu_24gb_20261112_1500\best")
PRESET = "smoke_cpu"
# Fixed in advance (before any result was seen)
N_EXPECTED = 32
MIN_CORRECT = 22
LABEL_NOTE = "Labels are the emotion I performed, with no independent check."
CSV_PATH = CODE_DIR / "results" / "metrics" / "my_voice_v3_ser_eval.csv"
FIG_PATH = CODE_DIR / "results" / "figures" / "my_voice_v3_confusion.png"
CLIP_RE = re.compile(r"(\d+)_(" + "|".join(CLASSES) + r")")
CSV_FIELDS = (["clip", "true_label", "pred_label", "correct", "confidence", "margin"]
              + [f"p_{c}" for c in CLASSES] + ["rms_dbfs", "model"])


# Scoring (pure; unit-tested in tests/test_eval_my_voice_v3.py)

def score(true_labels, pred_labels, classes=CLASSES) -> dict:
    """Accuracy, per-class accuracy and confusion matrix (rows = true, columns = predicted, in `classes` order)."""
    if len(true_labels) != len(pred_labels):
        raise ValueError(f"{len(true_labels)} true labels but {len(pred_labels)} predictions")
    idx = {c: i for i, c in enumerate(classes)}
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if t not in idx or p not in idx:
            raise ValueError(f"label not in {classes}: true {t!r}, predicted {p!r}")
        conf[idx[t], idx[p]] += 1
    n_correct, n = int(np.trace(conf)), int(conf.sum())
    per_class = {c: {"correct": int(conf[i, i]), "total": int(conf[i].sum()),
                     "accuracy": float(conf[i, i] / conf[i].sum()) if conf[i].sum() else float("nan")}
                 for c, i in idx.items()}
    return {"n_correct": n_correct, "n": n, "accuracy": n_correct / n if n else float("nan"),
            "per_class": per_class, "confusion": conf}


def verdict(n_correct, min_correct=MIN_CORRECT) -> str:
    return "PASS" if n_correct >= min_correct else "FAIL"


def rms_dbfs(y) -> float:
    y = np.asarray(y, dtype=np.float64)
    return float(20 * np.log10(max(np.sqrt(np.mean(y ** 2)) if y.size else 0.0, 1e-10)))


def mean_by(rows, key, value="rms_dbfs", classes=CLASSES) -> dict:
    """{class: (mean of `value`, count)} over rows grouped by rows[key]; classes with no rows are left out."""
    out = {}
    for c in classes:
        vals = [r[value] for r in rows if r[key] == c]
        if vals:
            out[c] = (float(np.mean(vals)), len(vals))
    return out


# Clips, model, files

def list_clips(folder) -> list:
    """[(path, true label)] for NN_<emotion>.wav directly in folder (subfolders such as rejected/ are ignored)."""
    clips, bad = [], []
    for p in sorted(folder.glob("*.wav")):
        m = CLIP_RE.fullmatch(p.stem)
        if m:
            clips.append((p, m.group(2)))
        else:
            bad.append(p.name)
    if bad:
        raise ValueError(f"file names not of the form NN_<{'|'.join(CLASSES)}>.wav: {bad}")
    return clips


def label_order(model_dir) -> dict:
    """{output index: class} from best/id2label.json (written by gpu/finetune_ser.py)."""
    raw = json.loads((Path(model_dir) / "id2label.json").read_text(encoding="utf-8"))
    return {int(i): (v["class"] if isinstance(v, dict) else v) for i, v in raw.items()}


def free_path(path) -> Path:
    path = Path(path)
    if path.exists():
        path = path.with_name(f"{path.stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    return path


def save_confusion_png(conf, classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(conf, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (performed)")
    ax.set_title(title, fontsize=10)
    top = conf.max() if conf.size else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                    color="white" if top and conf[i, j] > top / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _rel(path) -> str:
    try:
        return Path(path).resolve().relative_to(CODE_DIR.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def main():
    ap = argparse.ArgumentParser(description="Test the fine-tuned SER model on data/my_voice_v3 (CPU).")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="fine-tuned best/ folder")
    args = ap.parse_args()
    model_dir = Path(args.model)

    clips = list_clips(CLIP_DIR)
    if len(clips) != N_EXPECTED:
        raise SystemExit(f"Expected {N_EXPECTED} clips in {_rel(CLIP_DIR)}, found {len(clips)}; "
                         f"the {MIN_CORRECT}/{N_EXPECTED} rule is defined for {N_EXPECTED}.")
    order = label_order(model_dir)

    import soundfile as sf
    from device import setup
    from ser_gpu import SERModel

    cfg, device, dtype = setup(PRESET)
    print(f"Loading {model_dir} on {device} (preset {PRESET})...", flush=True)
    model = SERModel(cfg, device, dtype, model_name=str(model_dir))
    if model.index_map != order:
        raise SystemExit(f"id2label.json order {order} differs from the model's id2label {model.index_map}")
    print("Label order (id2label.json): " + ", ".join(f"{i}={c}" for i, c in sorted(order.items())))

    rows = []
    for path, true in clips:
        y, sr = sf.read(str(path), dtype="float32")
        x = model.preprocess(y, sr)                     # exactly what the model sees
        out = model._infer(x)                           # = model.predict_array(y, sr)
        rows.append({"clip": path.name, "true_label": true, "pred_label": out["label"],
                     "correct": out["label"] == true, "confidence": round(out["confidence"], 4),
                     "margin": round(out["margin"], 4),
                     **{f"p_{c}": round(out["probs"][c], 4) for c in CLASSES},
                     "rms_dbfs": round(rms_dbfs(x), 2), "model": _rel(model_dir)})

    # 1. Per-clip table
    print(f"\n{'clip':16s} {'true':8s} {'predicted':9s} {'conf':>6s} {'dBFS':>7s}")
    for r in rows:
        mark = "" if r["correct"] else "  x"
        print(f"{r['clip']:16s} {r['true_label']:8s} {r['pred_label']:9s} {r['confidence']:6.3f} "
              f"{r['rms_dbfs']:7.1f}{mark}")

    # 2. Accuracy, per class, confusion
    s = score([r["true_label"] for r in rows], [r["pred_label"] for r in rows])
    print(f"\nOverall accuracy: {s['n_correct']}/{s['n']} = {s['accuracy']:.3f}")
    print("Per class:")
    for c in CLASSES:
        pc = s["per_class"][c]
        print(f"  {c:8s} {pc['correct']}/{pc['total']} = {pc['accuracy']:.3f}")
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(f"  {'':8s}" + "".join(f"{c:>9s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:8s}" + "".join(f"{v:>9d}" for v in s["confusion"][i]))

    # 3. Verdict
    v = verdict(s["n_correct"])
    print(f"\nRule (fixed in advance): PASS if >= {MIN_CORRECT}/{N_EXPECTED} correct.  "
          f"Result: {s['n_correct']}/{s['n']} -> {v}")
    print(f"Note: {LABEL_NOTE}")

    # 5. Loudness
    print("\nMean loudness (RMS dBFS of the trimmed 16 kHz audio the model sees):")
    by_true, by_pred = mean_by(rows, "true_label"), mean_by(rows, "pred_label")
    print(f"  {'emotion':8s} {'by true label':>16s} {'by predicted label':>20s}")
    for c in CLASSES:
        t = f"{by_true[c][0]:6.1f} (n={by_true[c][1]})" if c in by_true else "-"
        p = f"{by_pred[c][0]:6.1f} (n={by_pred[c][1]})" if c in by_pred else "-"
        print(f"  {c:8s} {t:>16s} {p:>20s}")

    # 4. Files
    csv_path, fig_path = free_path(CSV_PATH), free_path(FIG_PATH)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    save_confusion_png(s["confusion"], CLASSES, fig_path,
                       f"my_voice_v3, fine-tuned hubert-large: {s['n_correct']}/{s['n']} correct ({v})")
    print(f"\nSaved {_rel(csv_path)}\nSaved {_rel(fig_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
