"""
abstain_curve_t3.py - tau_conf for the T3 fine-tuned emotion model (step 9b). Brain only: pandas/numpy, no torch.

Rule (Rule B, fixed before anything was computed, as in A2.2): tau_conf = the threshold with the highest
VALIDATION accuracy among thresholds whose validation coverage is at least 0.85; ties -> the smallest threshold
(most coverage). Validation = the RAVDESS + CREMA-D val split only. Test is reported as information only and is
never used to choose. my_voice_v3 is never read here.

confidence = top-1 of the 4 class probabilities; a clip is kept (not abstained) if confidence >= tau, the same
comparison MoodManagerV2 uses. Thresholds: 0 (no abstain), then 1/4 (the lowest possible top-1 with 4 classes) to
TAU_MAX in steps of TAU_STEP (same step and upper end as scripts/abstain_curve.py).

Inputs : results/metrics/ser_t3_val_probs.csv (gpu/ser_split_probs.py --split val)
         optional test CSV for information (default ser_t3_test_probs.csv if present)
Output : results/metrics/abstain_curve_t3.csv (a timestamp is added if it exists; nothing is overwritten)
Usage  : python scripts/abstain_curve_t3.py
"""
import argparse
import datetime
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
METRICS = CODE_DIR / "results" / "metrics"
VAL_CSV = METRICS / "ser_t3_val_probs.csv"
TEST_CSV = METRICS / "ser_t3_test_probs.csv"
OUT_CSV = METRICS / "abstain_curve_t3.csv"
RUN_ID = "gpu_24gb_20261112_1500"
CLASSES = ["neutral", "happy", "angry", "sad"]
MIN_COVERAGE = 0.85        # Rule B
TAU_STEP, TAU_MAX = 0.01, 0.95


def tau_grid(n_classes=len(CLASSES), step=TAU_STEP, tau_max=TAU_MAX) -> list:
    lo = math.ceil(round(100 / n_classes, 9)) / 100           # 0.25 for 4 classes
    return [0.0] + [round(t, 2) for t in np.arange(lo, tau_max + 1e-9, step)]


def with_predictions(df) -> pd.DataFrame:
    """confidence and pred recomputed from the p_<class> columns (top-1, argmax)."""
    p = df[[f"p_{c}" for c in CLASSES]].to_numpy(dtype=float)
    out = df.copy()
    out["confidence"] = p.max(axis=1)
    out["pred"] = np.array(CLASSES)[p.argmax(axis=1)]
    return out


def split_stats(df, tau) -> dict:
    kept = df[df["confidence"] >= tau]
    row = {"n": len(df), "n_kept": len(kept), "coverage": len(kept) / len(df) if len(df) else np.nan,
           "accuracy": float((kept["pred"] == kept["true_label"]).mean()) if len(kept) else np.nan}
    for c in CLASSES:
        n_c = int((df["true_label"] == c).sum())
        row[f"coverage_{c}"] = int((kept["true_label"] == c).sum()) / n_c if n_c else np.nan
    return row


def curve(val, test=None, taus=None) -> pd.DataFrame:
    val = with_predictions(val)
    test = with_predictions(test) if test is not None else None
    rows = []
    for tau in taus or tau_grid():
        row = {"tau": tau, **{f"val_{k}": v for k, v in split_stats(val, tau).items()}}
        if test is not None:
            row.update({f"test_{k}": v for k, v in split_stats(test, tau).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def choose_tau(cur, min_coverage=MIN_COVERAGE):
    """(tau, its curve row) by the rule above, using val_* columns ONLY. Raises if no threshold qualifies."""
    grid = cur[cur["tau"] > 0].sort_values("tau")
    eligible = grid[grid["val_n_kept"] >= min_coverage * grid["val_n"] - 1e-9]
    if eligible.empty:
        raise ValueError(f"no threshold keeps at least {min_coverage:.0%} of validation clips")
    best = eligible.loc[eligible["val_accuracy"].idxmax()]      # idxmax keeps the first (smallest tau) on ties
    return float(best["tau"]), best


def _new_path(path) -> Path:
    path = Path(path)
    if path.exists():
        path = path.with_name(f"{path.stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rule B tau_conf for the T3 model (validation only).")
    ap.add_argument("--val", default=str(VAL_CSV))
    ap.add_argument("--test", help=f"information only (default {TEST_CSV.name} if it exists)")
    args = ap.parse_args(argv)
    val_path = Path(args.val)
    if not val_path.is_file():
        print(f"{val_path} not found. Compute it first:  python gpu/ser_split_probs.py --split val")
        return 1
    val = pd.read_csv(val_path)
    if set(val["split"]) != {"val"}:
        raise SystemExit(f"{val_path} must hold validation clips only, found splits {sorted(set(val['split']))}")
    test_path = Path(args.test) if args.test else (TEST_CSV if TEST_CSV.is_file() else None)
    test = pd.read_csv(test_path) if test_path else None

    cur = curve(val, test)
    tau, best = choose_tau(cur)
    base = cur[cur["tau"] == 0].iloc[0]
    out = _new_path(OUT_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    cur.round(5).to_csv(out, index=False)

    print(f"Rule B (T3 run {RUN_ID}): max val accuracy with val coverage >= {MIN_COVERAGE:.2f} "
          f"-> tau_conf = {tau:.2f}\n")
    splits = ["val"] + (["test"] if test is not None else [])
    for name, r in (("no abstain", base), (f"tau {tau:.2f}", best)):
        for s in splits:
            per = "  ".join(f"{c} {r[f'{s}_coverage_{c}']:.3f}" for c in CLASSES)
            note = "   (information only)" if s == "test" else ""
            print(f"{name:11s} {s:4s} coverage {r[f'{s}_coverage']:.4f} ({int(r[f'{s}_n_kept'])}/{int(r[f'{s}_n'])})  "
                  f"accuracy {r[f'{s}_accuracy']:.4f}   per-class coverage: {per}{note}")
    print(f"\nval: {val_path.name}" + (f"   test: {test_path.name}" if test_path else "   test: none"))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
