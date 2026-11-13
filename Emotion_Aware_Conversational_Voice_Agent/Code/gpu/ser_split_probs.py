"""
ser_split_probs.py - per-clip emotion probabilities of the T3 checkpoint on ONE manifest split (step 9b).

Feeds scripts/abstain_curve_t3.py (tau_conf from validation only). Every clip goes through ser_gpu.SERModel, the
exact path the live agent uses (one clip at a time: mono, 16 kHz, trim, cut to ser.max_seconds). The checkpoint
must be ser.checkpoint and pass check_label_order: there is NO pretrained fallback here.

For --split val it also prints accuracy next to the run's own metrics.csv val_acc at the best epoch (highest
val_macro_f1, as finetune_ser.py saves best/). They can differ slightly: training evaluated in padded batches with
bf16 autocast on the GPU; here clips are one at a time (float32 on the CPU).

Usage (from Code/; CPU is fine):
    python gpu/ser_split_probs.py --split val                  # -> results/metrics/ser_t3_val_probs.csv
    python gpu/ser_split_probs.py --split test                 # information only, never for choosing anything
    python gpu/ser_split_probs.py --split val --limit 20       # timing check, nothing is written
    python gpu/ser_split_probs.py --split val --preset gpu_24gb
(a timestamp is added to the output name if the file exists; nothing is overwritten)
"""

import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path

from device import code_path, setup
from ser_gpu import CLASSES, SERModel, check_label_order, resolve_checkpoint

FIELDS = ["utt_id", "corpus", "speaker_id", "split", "true_label", "pred_label", "confidence", "margin",
          "p_neutral", "p_happy", "p_angry", "p_sad"]


def manifest_rows(cfg, split) -> list:
    from finetune_ser import load_rows, split_rows   # same manifest reading as training (in_4class rows)

    ft = cfg["finetune"]
    return split_rows(load_rows(code_path(ft["manifest"]), ft.get("in_4class_only", True)))[split]


def best_epoch_val_acc(run_dir):
    """(epoch, val_acc) at the highest val_macro_f1 in <run_dir>/metrics.csv, or None if it is not there."""
    path = Path(run_dir) / "metrics.csv"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["val_macro_f1"]))
    return int(best["epoch"]), float(best["val_acc"])


def _new_path(path) -> Path:
    path = Path(path)
    if path.exists():
        path = path.with_name(f"{path.stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    return path


def main():
    ap = argparse.ArgumentParser(description="T3 per-clip probabilities on one split (agent's SERModel path).")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--preset", default="smoke_cpu", help="device preset (default smoke_cpu: CPU)")
    ap.add_argument("--limit", type=int, help="only the first N clips, print timing, write nothing")
    ap.add_argument("--out", help="output CSV (default results/metrics/ser_t3_<split>_probs.csv)")
    args = ap.parse_args()

    cfg, device, dtype = setup(args.preset)
    ckpt = resolve_checkpoint(cfg)
    if ckpt is None or not (ckpt / "id2label.json").is_file():
        print(f"ERROR: T3 checkpoint not found at {ckpt} (ser.checkpoint). Nothing computed.", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    model = SERModel(cfg, device, dtype, model_name=str(ckpt))
    check_label_order(json.loads((ckpt / "id2label.json").read_text(encoding="utf-8")), model.model.config.id2label)
    print(f"checkpoint {ckpt}  device {device}  loaded in {time.perf_counter() - t0:.1f}s; label order checked")

    rows = manifest_rows(cfg, args.split)
    if args.limit:
        rows = rows[: args.limit]
    out, t0 = [], time.perf_counter()
    for i, r in enumerate(rows, 1):
        o = model.predict_file(code_path(r["wav_path"]))
        out.append({"utt_id": r["utt_id"], "corpus": r["corpus"], "speaker_id": r["speaker_id"], "split": args.split,
                    "true_label": r["label"], "pred_label": o["label"], "confidence": round(o["confidence"], 6),
                    "margin": round(o["margin"], 6), **{f"p_{c}": round(o["probs"][c], 6) for c in CLASSES}})
        if i % 100 == 0 or i == len(rows):
            el = time.perf_counter() - t0
            print(f"  {i}/{len(rows)} clips  {el:.0f}s  ({el / i:.2f}s per clip)", flush=True)
    acc = sum(r["true_label"] == r["pred_label"] for r in out) / len(out)
    print(f"\n{args.split}: {len(out)} clips, accuracy {acc:.5f} (no abstain)")
    if args.split == "val":
        ref = best_epoch_val_acc(ckpt.parent)
        if ref:
            print(f"run metrics.csv best epoch {ref[0]}: val_acc {ref[1]:.5f}  (difference {acc - ref[1]:+.5f})")
    if args.limit:
        print("--limit: timing check only, nothing written")
        return 0
    path = _new_path(args.out or code_path(f"results/metrics/ser_t3_{args.split}_probs.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
