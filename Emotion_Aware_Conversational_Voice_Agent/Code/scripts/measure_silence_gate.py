"""
measure_silence_gate.py - measure (not guess) the ASR silence gate (step 9b). Brain only: no torch, nothing from gpu/.

The gate (gpu/asr_gpu.speech_seconds): 25 ms frames every 10 ms on 16 kHz mono audio; a frame counts as speech
if its RMS level in dBFS is above asr.silence_dbfs; a turn passes if at least asr.min_speech_seconds of frames do.
So for one recording the useful number is its PASS LEVEL: the level of its k-th loudest frame, with
k = min_speech_seconds / hop. Any gate below that level lets the recording through; any gate at or above it
blocks it.

  (a) speech: pass level of every TRAIN-split clip in data/manifests/ser_4class.csv (RAVDESS + CREMA-D), with the
      quietest clips and the quietest per corpus x label (sad especially) printed.
  (b) room noise: pass level of a 10-second recording of the room with nobody talking
      (data/room_noise/room_noise.wav, made with scripts/record.py).
  gate = halfway between (b) and the quietest (a), margin = half the gap, shown on both sides.

Caveat printed with the result: the corpora were recorded with their own microphones and gain; the room noise
is recorded through YOUR microphone. my_voice_v3 is never read here.

Output: results/metrics/silence_gate.csv (a timestamp is added if it exists; nothing is overwritten).
The gate default in gpu/gpu_config.yaml is NOT changed by this script.

Usage:  python scripts/measure_silence_gate.py
        python scripts/measure_silence_gate.py --speech-only        # (a) only, printed, nothing written
        python scripts/measure_silence_gate.py --noise path/to/noise.wav
"""
import argparse
import csv
import datetime
import math
import sys
from pathlib import Path

import numpy as np
import yaml

CODE_DIR = Path(__file__).resolve().parent.parent
CONFIG = CODE_DIR / "gpu" / "gpu_config.yaml"
MANIFEST = CODE_DIR / "data" / "manifests" / "ser_4class.csv"
NOISE_WAV = CODE_DIR / "data" / "room_noise" / "room_noise.wav"
OUT_CSV = CODE_DIR / "results" / "metrics" / "silence_gate.csv"
SAMPLE_RATE = 16000
FRAME_SECONDS, HOP_SECONDS = 0.025, 0.010    # must equal gpu/asr_gpu.py (a test checks)
RECORD_CMD = "python scripts/record.py room_noise --seconds 10 --out data/room_noise"


def asr_config(path=CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return dict(yaml.safe_load(f)["base"]["asr"])


def frame_dbfs(y, sr=SAMPLE_RATE) -> np.ndarray:
    """dBFS of every 25 ms frame (10 ms hop), computed exactly like asr_gpu.speech_seconds."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros(0)
    frame = max(1, int(round(FRAME_SECONDS * sr)))
    hop = max(1, int(round(HOP_SECONDS * sr)))
    frames = y[None, :] if y.size < frame else np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return 20.0 * np.log10(rms + 1e-12)


def frames_needed(min_speech_seconds, sr=SAMPLE_RATE) -> int:
    hop = max(1, int(round(HOP_SECONDS * sr)))
    return max(1, math.ceil(round(min_speech_seconds * sr / hop, 9)))


def pass_level(db, k) -> float:
    """Level of the k-th loudest frame: the recording passes the gate iff gate < this. -inf if fewer than k frames."""
    if len(db) < k:
        return float("-inf")
    return float(np.sort(db)[::-1][k - 1])


def choose_gate(quietest_speech, noise) -> dict:
    """Halfway between the noise and the quietest speech (dB); margin = half the gap. gate None if they overlap."""
    gap = quietest_speech - noise
    if gap <= 0:
        return {"gate": None, "margin": None, "gap": gap}
    return {"gate": noise + gap / 2, "margin": gap / 2, "gap": gap}


def load_16k_mono(path) -> np.ndarray:
    """Same conversion as the ASR: mono (channel mean), resampled to 16 kHz."""
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=int(np.argmin(y.shape)))
    if y.size and sr != SAMPLE_RATE:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(y, dtype=np.float32)


def measure_speech(k, manifest=MANIFEST, split="train"):
    import pandas as pd

    df = pd.read_csv(manifest)
    df = df[(df["split"] == split) & (df["in_4class"] == True)]   # noqa: E712
    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        rows.append({"utt_id": r.utt_id, "corpus": r.corpus, "label": r.label,
                     "dbfs": pass_level(frame_dbfs(load_16k_mono(CODE_DIR / r.wav_path)), k)})
        if i % 1000 == 0:
            print(f"  {i}/{len(df)} clips", flush=True)
    return pd.DataFrame(rows)


def _new_path(path) -> Path:
    path = Path(path)
    if path.exists():
        path = path.with_name(f"{path.stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    return path


def print_record_help(noise_path):
    print(f"Room-noise recording not found: {noise_path}")
    print("Record 10 seconds of your room with nobody talking (same mic and position as for the agent):")
    print(f"    {RECORD_CMD}")
    print("Press Enter, then stay silent for the full 10 seconds. record.py will warn that the clip is almost")
    print("silent; for this recording that is expected. Then run this script again.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the ASR silence gate from real speech and room noise.")
    ap.add_argument("--noise", default=str(NOISE_WAV), help=f"room-noise wav (default {NOISE_WAV.relative_to(CODE_DIR)})")
    ap.add_argument("--speech-only", action="store_true", help="measure and print the speech side only; write nothing")
    args = ap.parse_args(argv)
    noise_path = Path(args.noise)
    if "my_voice_v3" in noise_path.parts:
        raise SystemExit("my_voice_v3 is test-only and is never used to choose a setting.")
    if not args.speech_only and not noise_path.is_file():
        print_record_help(noise_path)
        return 0

    asr = asr_config()
    k = frames_needed(asr["min_speech_seconds"])
    print(f"gate rule: >= {asr['min_speech_seconds']} s of 25 ms frames above the gate -> a recording passes if "
          f"gate < its {k}th-loudest frame level.  current asr.silence_dbfs = {asr['silence_dbfs']}")
    print("measuring train-split speech (RAVDESS + CREMA-D) ...")
    sp = measure_speech(k)
    quiet = sp.nsmallest(5, "dbfs")
    q_min = float(sp["dbfs"].min())
    print(f"\n(a) speech, {len(sp)} train clips: quietest pass level {q_min:.1f} dBFS   "
          f"1st pct {sp['dbfs'].quantile(0.01):.1f}   5th pct {sp['dbfs'].quantile(0.05):.1f}   "
          f"median {sp['dbfs'].median():.1f}")
    print("    quietest clips: " + "; ".join(f"{r.utt_id} ({r.corpus} {r.label}) {r.dbfs:.1f}"
                                         for r in quiet.itertuples()))
    per = sp.groupby(["corpus", "label"])["dbfs"].min()
    print("    quietest per corpus x label: " + "  ".join(f"{c}/{l} {v:.1f}" for (c, l), v in per.items()))
    blocked = int((sp["dbfs"] <= asr["silence_dbfs"]).sum())
    print(f"    train clips the CURRENT gate ({asr['silence_dbfs']} dBFS) would block: {blocked}/{len(sp)}")

    if args.speech_only:
        if not noise_path.is_file():
            print()
            print_record_help(noise_path)
        print("\n--speech-only: nothing written")
        return 0

    nd = frame_dbfs(load_16k_mono(noise_path))
    n_level = pass_level(nd, k)
    g = choose_gate(q_min, n_level)
    print(f"\n(b) room noise {noise_path.name} ({len(nd) * HOP_SECONDS:.1f} s): pass level {n_level:.1f} dBFS   "
          f"median frame {np.median(nd):.1f}   loudest frame {nd.max():.1f}")
    if g["gate"] is None:
        print(f"\nNO GATE separates them: room noise ({n_level:.1f}) is not quieter than the quietest speech "
              f"({q_min:.1f}). Reduce the noise or move the mic; the default is left unchanged.")
    else:
        print(f"\ngate = {g['gate']:.1f} dBFS  (noise {n_level:.1f} + {g['margin']:.1f} dB margin | "
              f"{g['margin']:.1f} dB margin + {q_min:.1f} quietest speech)")
    print("Caveat: the speech side comes from the corpora's own recording chain, the noise side from your mic.")
    print("gpu_config.yaml asr.silence_dbfs was NOT changed.")

    out = _new_path(OUT_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["measure", "corpus", "label", "utt_id", "dbfs", "note"])
        w.writerow(["speech_quietest", "", "", quiet.iloc[0]["utt_id"], round(q_min, 2), f"{len(sp)} train clips"])
        for pct in (0.01, 0.05, 0.5):
            w.writerow([f"speech_p{int(pct * 100):02d}", "", "", "", round(float(sp["dbfs"].quantile(pct)), 2), ""])
        for r in quiet.itertuples():
            w.writerow(["speech_quiet_clip", r.corpus, r.label, r.utt_id, round(r.dbfs, 2), ""])
        for (c, l), v in per.items():
            w.writerow(["speech_quietest_by_class", c, l, "", round(float(v), 2), ""])
        w.writerow(["room_noise", "", "", str(noise_path.name), round(n_level, 2), f"{len(nd) * HOP_SECONDS:.1f} s"])
        w.writerow(["gate_midpoint", "", "", "", "" if g["gate"] is None else round(g["gate"], 2),
                    "no gate separates them" if g["gate"] is None else f"margin {g['margin']:.2f} dB each side"])
        w.writerow(["current_silence_dbfs", "", "", "", asr["silence_dbfs"], "gpu_config.yaml, unchanged"])
        w.writerow(["min_speech_seconds", "", "", "", asr["min_speech_seconds"], f"k = {k} frames"])
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
