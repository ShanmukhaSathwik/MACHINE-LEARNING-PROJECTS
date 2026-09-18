# Emotion-Aware Voice Agent — Step 1: Feature Extraction + Manifest

**Goal of this step:** adapt/write `feature_extraction.py` so that in a single pass over RAVDESS it produces one CSV that is *both* your manifest (`utt_id, speaker_id, wav_path, label, split, duration`) *and* your classical-baseline feature set (39 feature columns). This is the fastest path to preliminary results and feeds directly into Step 2 (Random Forest / XGBoost).

---

## VS Code or Google Colab?

**VS Code, running locally.** Everything in Steps 1–2 is CPU-only: reading `.wav` files, extracting features, training RF/XGBoost. No GPU needed. Colab is reserved for the single GPU-hungry piece later — the CSM-1B voice model (Phase 8).

Concrete reason it matters: Colab wipes its filesystem on disconnect, so your manifest, cached features, and checkpoints would vanish and you'd re-upload 215 MB of RAVDESS every session. Locally, everything persists and the rest of your pipeline can import it.

---

## Phase 0 — Set up the project (one time)

Open a terminal in VS Code (`` Ctrl+` ``). Use **Python 3.10 or 3.11** (librosa's audio backend is happiest there).

```bash
# 1. make and enter the project folder
mkdir emotion-voice-agent
cd emotion-voice-agent

# 2. create an isolated virtual environment
python -m venv .venv

# 3. activate it
#    Windows (PowerShell):
.venv\Scripts\Activate.ps1
#    macOS / Linux:
source .venv/bin/activate

# 4. install what Steps 1 and 2 need
pip install numpy pandas librosa soundfile scikit-learn xgboost tqdm
```

The venv is active when your prompt shows `(.venv)`. Re-run step 3 each time you return to work.

Create the folders:

```bash
# Windows (PowerShell):
mkdir data\raw, data\manifests, data\splits

# macOS / Linux:
mkdir -p data/raw data/manifests data/splits
```

- `data/raw/` — RAVDESS audio (never commit to git)
- `data/manifests/` — the CSV we build
- `data/splits/` — record of which actor went to train/val/test

---

## Phase 1 — Get the RAVDESS audio

Download the **speech** zip (215 MB, 1,440 clips): `Audio_Speech_Actors_01-24.zip` from Zenodo:
<https://zenodo.org/records/1188976>

Ignore the video and song files. Unzip **into `data/raw/`** so you get:

```
data/raw/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-03-02-01-01-01.wav
│   └── ...
├── Actor_02/
└── ... up to Actor_24/
```

The script searches recursively, so nested `Actor_XX` folders are fine — don't flatten them.

---

## Phase 2 — The key concept: the filename *is* the label

RAVDESS has no separate labels file. Each clip's emotion and speaker are encoded in its filename. `03-01-06-01-02-01-12.wav` is seven hyphen-separated numbers:

| Position | Field | `06` / `12` example |
|---|---|---|
| 1 | Modality | 03 = audio-only |
| 2 | Vocal channel | 01 = speech |
| **3** | **Emotion** | **06 = fearful** ← we read this |
| 4 | Intensity | 01 = normal |
| 5 | Statement | 02 = "dogs are sitting by the door" |
| 6 | Repetition | 01 = 1st take |
| **7** | **Actor** | **12 = actor #12 (female)** ← we read this |

Full emotion code map:
`01`=neutral, `02`=calm, `03`=happy, `04`=sad, `05`=angry, `06`=fearful, `07`=disgust, `08`=surprised.

We commit to **5 classes**: neutral, happy, sad, angry, fearful — keep codes `01, 03, 04, 05, 06` and **drop** calm/disgust/surprised. (Changed via one dictionary in the config.)

**Why the actor number matters most:** if clips from the same actor land in both train and test, the model can memorize that actor's voice instead of learning emotion ("speaker leakage") — the #1 risk in your risk table. So we split **by actor**, never by clip.

---

## Phase 3 — The script, section by section

Create `feature_extraction.py` at the root of `emotion-voice-agent/`. (In the final modular layout this becomes `src/perception/features.py`; the root keeps it easy to run now.)

### Block 1 — Imports and config

```python
"""
feature_extraction.py
Step 1: build the manifest + classical-baseline feature set from RAVDESS in one pass.
Output -> data/manifests/manifest.csv
Columns: utt_id, speaker_id, wav_path, label, split, duration, <39 feature columns>
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG — the only part you'll ever edit
# ---------------------------------------------------------------------------
RAW_DIR    = Path("data/raw")                       # where Actor_XX folders live
OUT_CSV    = Path("data/manifests/manifest.csv")    # the file we produce
SPLIT_JSON = Path("data/splits/actor_splits.json")  # record of the speaker split

SAMPLE_RATE = 16000   # resample every clip to 16 kHz mono (project standard)
TRIM_TOP_DB = 30      # silence-trim aggressiveness (higher = trims less)
N_MFCC      = 13      # number of MFCC coefficients
SEED        = 42      # makes the speaker split reproducible
USE_PITCH   = True    # F0 features are informative but slow; set False for a fast first run

# RAVDESS emotion code (filename field 3) -> our label.
# Any code NOT listed here is dropped (calm=02, disgust=07, surprised=08).
EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",   # rename to "anxious" here to match your mood axis if you prefer
}

# Speaker-independent split: of the 24 actors, how many go to val and test.
N_VAL_SPEAKERS  = 4
N_TEST_SPEAKERS = 4   # the remaining 16 actors become the train set
```

Everything tunable lives at the top. `USE_PITCH = False` is your escape hatch: pitch extraction (`pyin`) is the slowest part, so flip it off for a fast first run, then back on later.

### Block 2 — Read the label and actor out of the filename

```python
def parse_filename(wav_path: Path):
    """
    RAVDESS name '03-01-06-01-02-01-12.wav' -> 7 fields.
      field 3 (index 2) = emotion code
      field 7 (index 6) = actor number
    Returns (label, actor_id). label is None if the emotion isn't one we keep.
    """
    parts = wav_path.stem.split("-")     # ['03','01','06','01','02','01','12']
    if len(parts) != 7:                  # skip anything not shaped like a RAVDESS file
        return None, None
    emotion_code = parts[2]
    actor_id     = int(parts[6])
    label        = EMOTION_MAP.get(emotion_code)   # None if calm/disgust/surprised
    return label, actor_id
```

`wav_path.stem` is the filename without `.wav`. We split on `-`, grab the emotion code and actor, and look the emotion up in our map. Dropped emotions return `label = None` and get skipped later.

### Block 3 — Turn one waveform into a fixed-length feature vector

RF/XGBoost need **one row of numbers per clip**, but audio is a long, variable-length wave. The trick: compute a feature over every short frame (giving a sequence), then summarize that sequence with its **mean and standard deviation**. Mean = the typical value; std = how much it moves — and emotional speech is largely about movement (wobbling pitch, bursts of energy).

```python
def extract_features(y: np.ndarray, sr: int) -> dict:
    """Summarize one waveform into ~39 numbers (a dict of named features)."""
    feats = {}

    # --- MFCCs: 13 coefficients describing vocal-tract shape / timbre ---
    # Summarize each coefficient's sequence by mean and std -> 13*2 = 26 features.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)   # shape [13, n_frames]
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"]  = float(np.std(mfcc[i]))

    # --- RMS energy: loudness. Angry/happy louder; sad/neutral quieter. ---
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"]  = float(np.std(rms))

    # --- Zero-crossing rate: how "buzzy/noisy" vs "tonal" the signal is. ---
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"]  = float(np.std(zcr))

    # --- Spectral shape: where energy sits in frequency (brightness). ---
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["centroid_mean"] = float(np.mean(cent))
    feats["centroid_std"]  = float(np.std(cent))

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats["bandwidth_mean"] = float(np.mean(bw))
    feats["bandwidth_std"]  = float(np.std(bw))

    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats["rolloff_mean"] = float(np.mean(roll))
    feats["rolloff_std"]  = float(np.std(roll))

    # --- Pitch (F0): the single strongest emotion cue in prosody. ---
    if USE_PITCH:
        f0, voiced_flag, _ = librosa.pyin(
            y, sr=sr,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz  (low male voice floor)
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz (high ceiling)
        )
        has_voice = np.any(~np.isnan(f0))
        feats["f0_mean"]      = float(np.nanmean(f0)) if has_voice else 0.0
        feats["f0_std"]       = float(np.nanstd(f0))  if has_voice else 0.0
        feats["voiced_ratio"] = float(np.mean(voiced_flag))   # fraction of voiced frames
    else:
        feats["f0_mean"] = feats["f0_std"] = feats["voiced_ratio"] = 0.0

    return feats
```

`pyin` returns pitch in Hz for voiced frames and `NaN` for silent/unvoiced ones, plus a `voiced_flag`. We use `np.nanmean`/`np.nanstd` to average only real pitch values, and `voiced_ratio` tells us how much of the clip carried a pitched voice.

### Block 4 — Assign each actor to train / val / test

```python
def assign_speaker_splits(actor_ids) -> dict:
    """Deterministically map each actor number to 'train'/'val'/'test'.
       Splitting by SPEAKER (not by clip) prevents speaker leakage."""
    actors = sorted(set(int(a) for a in actor_ids))
    shuffled = np.array(actors)
    np.random.default_rng(SEED).shuffle(shuffled)   # same shuffle every run (SEED)
    shuffled = [int(a) for a in shuffled]

    test_actors = set(shuffled[:N_TEST_SPEAKERS])
    val_actors  = set(shuffled[N_TEST_SPEAKERS:N_TEST_SPEAKERS + N_VAL_SPEAKERS])

    split_of = {}
    for a in actors:
        if   a in test_actors: split_of[a] = "test"
        elif a in val_actors:  split_of[a] = "val"
        else:                  split_of[a] = "train"
    return split_of
```

A **fixed seed** gives the identical split every run (the "reproducible" part of your plan). 16 actors → train, 4 → val, 4 → test. Because assignment is keyed on the actor, all of an actor's clips move together.

### Block 5 — The main loop

```python
def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(RAW_DIR.rglob("*.wav"))   # recursive: finds Actor_XX/*.wav
    if not wav_files:
        raise SystemExit(f"No .wav files under {RAW_DIR.resolve()} — is RAVDESS unzipped there?")

    # Pass 1: collect actors present, decide the split.
    actor_ids = [a for w in wav_files for _, a in [parse_filename(w)] if a is not None]
    split_of = assign_speaker_splits(actor_ids)

    # Pass 2: extract features clip by clip.
    rows, kept, dropped = [], 0, 0
    for w in tqdm(wav_files, desc="Extracting"):
        label, actor = parse_filename(w)
        if label is None:          # not a kept emotion (or malformed name)
            dropped += 1
            continue
        try:
            y, sr = librosa.load(w, sr=SAMPLE_RATE, mono=True)   # 16 kHz mono
            duration = float(librosa.get_duration(y=y, sr=sr))   # full clip length
            y_trim, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)  # cut edge silence
            if y_trim.size < 512:      # trimming ate everything -> fall back to original
                y_trim = y
            feats = extract_features(y_trim, sr)
        except Exception as e:
            print(f"  skipping {w.name}: {e}")
            dropped += 1
            continue

        rows.append({
            "utt_id":     w.stem,
            "speaker_id": actor,
            "wav_path":   w.as_posix(),
            "label":      label,
            "split":      split_of[actor],
            "duration":   round(duration, 3),
            **feats,
        })
        kept += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    SPLIT_JSON.write_text(json.dumps(split_of, indent=2))

    # --- report so you can eyeball that it worked ---
    print(f"\nDone. kept {kept} clips, dropped {dropped}.")
    print(f"Manifest -> {OUT_CSV.resolve()}")
    print("\nLabel distribution:\n", df["label"].value_counts())
    print("\nClips per split:\n", df["split"].value_counts())
    print("\nActors per split:")
    for s in ["train", "val", "test"]:
        print(f"  {s:5s}: {sorted(df[df.split == s].speaker_id.unique())}")


if __name__ == "__main__":
    main()
```

Two design choices worth understanding:

1. Features are extracted on the trimmed **but not volume-normalized** waveform — for this classical model, absolute loudness (`rms_mean`) is a genuine emotion cue we want to keep, and tree models don't care about feature scale. (Waveform volume-normalization matters later, for the Bi-LSTM's raw input.)
2. `duration` is measured on the full clip *before* trimming, so it reflects real file length for dataset stats.

The per-file body is wrapped in `try/except` so one corrupt clip prints a warning and is skipped instead of crashing a long run.

---

## The complete file (`feature_extraction.py`)

```python
"""
feature_extraction.py
Step 1: build the manifest + classical-baseline feature set from RAVDESS in one pass.
Output -> data/manifests/manifest.csv
Columns: utt_id, speaker_id, wav_path, label, split, duration, <39 feature columns>
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG — the only part you'll ever edit
# ---------------------------------------------------------------------------
RAW_DIR    = Path("data/raw")
OUT_CSV    = Path("data/manifests/manifest.csv")
SPLIT_JSON = Path("data/splits/actor_splits.json")

SAMPLE_RATE = 16000
TRIM_TOP_DB = 30
N_MFCC      = 13
SEED        = 42
USE_PITCH   = True   # set False for a much faster first run

EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",   # rename to "anxious" to match your mood axis if you prefer
}

N_VAL_SPEAKERS  = 4
N_TEST_SPEAKERS = 4


def parse_filename(wav_path: Path):
    parts = wav_path.stem.split("-")
    if len(parts) != 7:
        return None, None
    emotion_code = parts[2]
    actor_id     = int(parts[6])
    return EMOTION_MAP.get(emotion_code), actor_id


def extract_features(y: np.ndarray, sr: int) -> dict:
    feats = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"]  = float(np.std(mfcc[i]))

    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"]  = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"]  = float(np.std(zcr))

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["centroid_mean"] = float(np.mean(cent))
    feats["centroid_std"]  = float(np.std(cent))

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats["bandwidth_mean"] = float(np.mean(bw))
    feats["bandwidth_std"]  = float(np.std(bw))

    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats["rolloff_mean"] = float(np.mean(roll))
    feats["rolloff_std"]  = float(np.std(roll))

    if USE_PITCH:
        f0, voiced_flag, _ = librosa.pyin(
            y, sr=sr,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        has_voice = np.any(~np.isnan(f0))
        feats["f0_mean"]      = float(np.nanmean(f0)) if has_voice else 0.0
        feats["f0_std"]       = float(np.nanstd(f0))  if has_voice else 0.0
        feats["voiced_ratio"] = float(np.mean(voiced_flag))
    else:
        feats["f0_mean"] = feats["f0_std"] = feats["voiced_ratio"] = 0.0

    return feats


def assign_speaker_splits(actor_ids) -> dict:
    actors = sorted(set(int(a) for a in actor_ids))
    shuffled = np.array(actors)
    np.random.default_rng(SEED).shuffle(shuffled)
    shuffled = [int(a) for a in shuffled]

    test_actors = set(shuffled[:N_TEST_SPEAKERS])
    val_actors  = set(shuffled[N_TEST_SPEAKERS:N_TEST_SPEAKERS + N_VAL_SPEAKERS])

    split_of = {}
    for a in actors:
        if   a in test_actors: split_of[a] = "test"
        elif a in val_actors:  split_of[a] = "val"
        else:                  split_of[a] = "train"
    return split_of


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(RAW_DIR.rglob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No .wav files under {RAW_DIR.resolve()} — is RAVDESS unzipped there?")

    actor_ids = [a for w in wav_files for _, a in [parse_filename(w)] if a is not None]
    split_of = assign_speaker_splits(actor_ids)

    rows, kept, dropped = [], 0, 0
    for w in tqdm(wav_files, desc="Extracting"):
        label, actor = parse_filename(w)
        if label is None:
            dropped += 1
            continue
        try:
            y, sr = librosa.load(w, sr=SAMPLE_RATE, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))
            y_trim, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
            if y_trim.size < 512:
                y_trim = y
            feats = extract_features(y_trim, sr)
        except Exception as e:
            print(f"  skipping {w.name}: {e}")
            dropped += 1
            continue

        rows.append({
            "utt_id":     w.stem,
            "speaker_id": actor,
            "wav_path":   w.as_posix(),
            "label":      label,
            "split":      split_of[actor],
            "duration":   round(duration, 3),
            **feats,
        })
        kept += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    SPLIT_JSON.write_text(json.dumps(split_of, indent=2))

    print(f"\nDone. kept {kept} clips, dropped {dropped}.")
    print(f"Manifest -> {OUT_CSV.resolve()}")
    print("\nLabel distribution:\n", df["label"].value_counts())
    print("\nClips per split:\n", df["split"].value_counts())
    print("\nActors per split:")
    for s in ["train", "val", "test"]:
        print(f"  {s:5s}: {sorted(df[df.split == s].speaker_id.unique())}")


if __name__ == "__main__":
    main()
```

---

## Run it

With the venv active and RAVDESS unzipped in `data/raw/`:

```bash
python feature_extraction.py
```

A `tqdm` bar appears. With `USE_PITCH = True`, expect a few minutes; with `False`, well under a minute.

Success looks roughly like (we keep 5 of 8 emotions):

```
Done. kept 1056 clips, dropped 384.
Label distribution:
 neutral    96
 happy     192
 sad       192
 angry     192
 fearful   192
Clips per split:
 train    ~700
 val      ~180
 test     ~180
Actors per split:
  train: [3, 4, 5, 7, 8, 9, 10, 11, 13, 16, 17, 18, 19, 21, 22, 24]
  val:   [1, 6, 20, 23]
  test:  [2, 12, 14, 15]
```

**Two sanity checks:**

1. **Neutral has ~96 clips** while the others have ~192 — RAVDESS neutral has no "strong" intensity version, so it's genuinely half-sized. This real class imbalance is why the plan uses class weights and **macro-F1** rather than plain accuracy.
2. **No actor number appears in more than one split** — that's the proof your split is speaker-independent.

Open `data/manifests/manifest.csv` (VS Code previews CSVs) and confirm the metadata columns followed by 39 feature columns, one row per clip.

---

## Next: Step 2 — Train RF + XGBoost on this CSV

Key discipline for Step 2: train only on `split == "train"`, tune on `val`, and touch `test` exactly once. Report per-class precision/recall/F1 and macro-F1, plus a confusion matrix (watch angry↔fearful, sad↔neutral).
