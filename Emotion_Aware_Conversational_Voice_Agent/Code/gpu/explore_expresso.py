"""
explore_expresso.py - Step 3 of the GPU kit (runs on CPU, at home).

Looks at the Expresso dataset (read-speech part, with transcripts) before we
spend GPU time fine-tuning CSM-1B on it. Answers:
  1. Which of the 4 speakers should be the agent's voice?  -> listen to samples
  2. How much audio does each speaker have per style?      -> table
  3. Do the styles cover our 4 strategies?                 -> coverage check

Usage (from Code/, venv active):
    python gpu/explore_expresso.py
    python gpu/explore_expresso.py --per-cell 5        # more samples per speaker x style

Outputs (<outputs>/expresso/; <outputs> = $GPU_OUTPUTS_DIR if set, else gpu/outputs, ignored by git):
    style_table.csv        speaker x style: clips, minutes
    samples/*.wav|flac     a few clips per speaker x style
    samples/samples.csv    file, speaker, style, duration, transcript
    listen.html            open in a browser: every sample with a player + transcript

Dataset: ylacombe/expresso (Nguyen et al., Interspeech 2023), CC BY-NC 4.0.
First run downloads it to the Hugging Face cache (C: by default), a few GB.
Audio is NOT decoded by `datasets` (avoids the torchcodec dependency on Windows);
soundfile reads only file headers for durations.
"""

import argparse
import csv
import html
import io
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
REPO = "ylacombe/expresso"

# Strategy -> preferred Expresso styles (first match that exists is used).
STRATEGY_STYLES = {
    "VALIDATE": ["sad", "default"],
    "CALM": ["whisper", "narration", "default"],
    "ENCOURAGE": ["happy", "laughing"],
    "NEUTRAL": ["default"],
}
MIN_MINUTES_PER_STYLE = 10  # rough floor for a usable LoRA style (per speaker)


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(CODE_DIR / ".env", override=False)
    except Exception:  # noqa: BLE001
        pass


def load_expresso():
    from datasets import Audio, get_dataset_config_names, load_dataset

    try:
        configs = get_dataset_config_names(REPO)
    except Exception:  # noqa: BLE001
        configs = []
    config = "read" if "read" in configs else (configs[0] if configs else None)
    print(f"Loading {REPO} (config={config}) ... first run downloads it, please wait.")
    ds = load_dataset(REPO, config, split="train") if config else load_dataset(REPO, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def label_fn(ds, col):
    """ClassLabel columns come back as ints; turn them into names."""
    feat = ds.features.get(col)
    if hasattr(feat, "int2str"):
        return lambda v: feat.int2str(v) if isinstance(v, int) else str(v)
    return lambda v: str(v)


def audio_info(a):
    """Return (duration_s, raw_bytes, ext) without decoding the whole file."""
    import soundfile as sf

    raw = a.get("bytes")
    path = a.get("path") or ""
    if raw is None and path and os.path.exists(path):
        raw = Path(path).read_bytes()
    if raw is None:
        return None, None, None
    try:
        info = sf.info(io.BytesIO(raw))
        dur = info.frames / info.samplerate
    except Exception:  # noqa: BLE001
        dur = None
    ext = Path(path).suffix.lower() if Path(path).suffix else ".wav"
    return dur, raw, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=3, help="samples per speaker x style")
    ap.add_argument("--min-s", type=float, default=3.0, help="min sample length (s)")
    ap.add_argument("--max-s", type=float, default=12.0, help="max sample length (s)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    load_env()
    from device import outputs_dir   # imports torch; this script already needs the GPU kit libraries
    print(f"outputs : {outputs_dir()}")
    out_dir = outputs_dir() / "expresso"
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    ds = load_expresso()
    print(f"Rows: {len(ds)} | columns: {ds.column_names}")
    spk_name = label_fn(ds, "speaker_id")
    sty_name = label_fn(ds, "style")

    # ---- pass 1: durations per speaker x style --------------------------
    stats = defaultdict(lambda: {"clips": 0, "seconds": 0.0})
    candidates = defaultdict(list)  # (spk, style) -> row indices in sample range
    bad = 0
    for i, row in enumerate(ds):
        spk, sty = spk_name(row["speaker_id"]), sty_name(row["style"])
        dur, _, _ = audio_info(row["audio"])
        if dur is None:
            bad += 1
            continue
        s = stats[(spk, sty)]
        s["clips"] += 1
        s["seconds"] += dur
        if args.min_s <= dur <= args.max_s:
            candidates[(spk, sty)].append(i)
        if (i + 1) % 2000 == 0:
            print(f"  scanned {i + 1}/{len(ds)}")
    if bad:
        print(f"WARNING: {bad} clips could not be read")

    speakers = sorted({k[0] for k in stats})
    styles = sorted({k[1] for k in stats})

    # ---- table ------------------------------------------------------------
    with open(out_dir / "style_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "style", "clips", "minutes"])
        for spk in speakers:
            for sty in styles:
                s = stats.get((spk, sty), {"clips": 0, "seconds": 0})
                w.writerow([spk, sty, s["clips"], round(s["seconds"] / 60, 1)])

    print("\nMinutes of audio (speaker x style):")
    header = f"{'style':<14}" + "".join(f"{spk:>9}" for spk in speakers)
    print(header)
    print("-" * len(header))
    for sty in styles:
        cells = "".join(f"{stats.get((spk, sty), {'seconds': 0})['seconds'] / 60:>9.1f}" for spk in speakers)
        print(f"{sty:<14}{cells}")
    totals = "".join(
        f"{sum(stats.get((spk, s), {'seconds': 0})['seconds'] for s in styles) / 60:>9.1f}" for spk in speakers
    )
    print("-" * len(header))
    print(f"{'TOTAL':<14}{totals}")

    # ---- strategy coverage ------------------------------------------------
    print(f"\nStrategy coverage (needs >= {MIN_MINUTES_PER_STYLE} min per speaker):")
    for strat, wanted in STRATEGY_STYLES.items():
        found = next((s for s in wanted if s in styles), None)
        if not found:
            print(f"  {strat:<10} -> none of {wanted} exist  ** MISSING **")
            continue
        mins = {spk: stats.get((spk, found), {"seconds": 0})["seconds"] / 60 for spk in speakers}
        low = [spk for spk, m in mins.items() if m < MIN_MINUTES_PER_STYLE]
        note = f"  low for {', '.join(low)}" if low else "  ok for all speakers"
        print(f"  {strat:<10} -> {found:<10}" + " ".join(f"{spk}:{m:.0f}m" for spk, m in mins.items()) + note)

    # ---- samples + listening page -----------------------------------------
    rng = random.Random(args.seed)
    rows_out = []
    for spk in speakers:
        for sty in styles:
            idxs = candidates.get((spk, sty), [])
            for n, i in enumerate(rng.sample(idxs, min(args.per_cell, len(idxs))), start=1):
                row = ds[i]
                dur, raw, ext = audio_info(row["audio"])
                name = f"{spk}_{sty}_{n}{ext}"
                (sample_dir / name).write_bytes(raw)
                rows_out.append(
                    {"file": name, "speaker": spk, "style": sty,
                     "duration_s": round(dur, 2), "text": row.get("text", "")}
                )

    with open(sample_dir / "samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "style", "duration_s", "text"])
        w.writeheader()
        w.writerows(rows_out)

    by_cell = defaultdict(list)
    for r in rows_out:
        by_cell[(r["speaker"], r["style"])].append(r)
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Expresso voices</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px;vertical-align:top;min-width:220px}"
        "th{background:#eee;position:sticky;top:0}.t{font-size:12px;color:#555;margin:2px 0 8px}"
        "audio{width:210px;height:32px}</style>",
        "<h2>Expresso: pick the agent's voice</h2>",
        "<p>Columns = speakers, rows = styles. Listen to <b>default</b> first (everyday voice), "
        "then <b>sad</b>, <b>whisper</b>, <b>happy</b>.</p><table><tr><th>style</th>",
    ]
    parts += [f"<th>{html.escape(spk)}</th>" for spk in speakers]
    parts.append("</tr>")
    order = [s for s in ["default", "sad", "whisper", "happy", "laughing", "narration"] if s in styles]
    order += [s for s in styles if s not in order]
    for sty in order:
        parts.append(f"<tr><th>{html.escape(sty)}</th>")
        for spk in speakers:
            cell = []
            for r in by_cell.get((spk, sty), []):
                cell.append(
                    f"<audio controls preload='none' src='samples/{html.escape(r['file'])}'></audio>"
                    f"<div class='t'>{html.escape(r['text'][:160])} ({r['duration_s']}s)</div>"
                )
            parts.append("<td>" + ("".join(cell) or "-") + "</td>")
        parts.append("</tr>")
    parts.append("</table>")
    (out_dir / "listen.html").write_text("\n".join(parts), encoding="utf-8")

    print(f"\nSaved {len(rows_out)} samples to {sample_dir}")
    print(f"Table : {out_dir / 'style_table.csv'}")
    print(f"LISTEN: open {out_dir / 'listen.html'} in your browser")
    return 0


if __name__ == "__main__":
    sys.exit(main())
