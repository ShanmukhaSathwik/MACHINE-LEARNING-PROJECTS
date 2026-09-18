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

# Config

RAW_DIR    = Path("data/raw")
OUT_CSV    = Path("data/manifests/manifest.csv")
SPLIT_JSON = Path("data/splits/actor_splits.json")

# The Auido/Proccessing settings

SAMPLE_RATE = 16000   # resample every clip to 16 kHz mono
TRIM_TOP_DB = 30      # silence-trim aggressiveness (higher = trims less)
N_MFCC      = 13      # number of MFCC coefficients
SEED        = 42      # makes the speaker split reproducible
USE_PITCH   = True    # F0 features are informative but slow; set False for a fast run

# Emotion Map
# Keeping only 3 classes for preliminary results: happy, sad, angry.

EMOTION_MAP = {
    "03": "happy",
    "04": "sad",
    "05": "angry",
}

# The split size settings, 70/15/15
# Speaker-independent split: of the 24 actors
N_VAL_SPEAKERS  = 4
N_TEST_SPEAKERS = 4   # the remaining 16 actors become the train set

#Parsing/Metadata Extraction

def parse_filename(wav_path: Path):

    """
    RAVDESS name '03-01-06-01-02-01-12.wav' -> 7 fields.
    field 3 (index 2) = emotion code
    field 7 (index 6) = actor number
    Returns (label, actor_id). label is None if the emotion isn't one we keep.
    """
    parts = wav_path.stem.split("-")
    if len(parts) != 7:             #skip anything not shaped like a RAVDESS file
        return None, None

    emotion_code =parts[2]
    actor_id = int(parts[6])
    label =EMOTION_MAP.get(emotion_code) #None if calm/disgust/surprised, etc.
    return label, actor_id

#Extract Features

""" y = waveform, sr =sample rate"""
def extract_features(y: np.ndarray, sr: int) -> dict:
    """Summarize one waveform into ~39 numbers (a dict of named features)."""
    feats ={}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))

        #RMS energy: loudness. Angry/happy = louder; sad/neutral= quieter.
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))

    # Zero-crossing rate: hpw "buzzy/noisy" vs "tonal" the signal is.               
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))

    # Spectral shape: where energy sits in frequency (brightness)
    """
    Spectral centroid, if the frequenct spectram has:
    Higher = the sound is dominated by higher frequencies (brighter, sharper).
    Lower = dominated by lower frequencies (duller, warmer).
    """
    cent =librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["centroid_mean"] = float(np.mean(cent))
    feats["centroid_std"] = float(np.std(cent))

    """
    Spectral bandwidth — how spread out the frequencies are around that centroid.
    Wide bandwidth = energy spread across many frequencies (noisy/complex).
    Narrow = energy concentrated (pure tone-like)
    """
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats["bandwidth_mean"] = float(np.mean(bw))
    feats["bandwidth_std"]  = float(np.std(bw))

    """Spectral rolloff —the frequency below which most of the total energy"""
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats["rolloff_mean"] = float(np.mean(roll))
    feats["rolloff_std"]  = float(np.std(roll))

    # Pitch

    if USE_PITCH:
        f0, voiced_flag, _ = librosa.pyin(
        y, sr=sr,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz  (low male voice floor)
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz (high ceiling)
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
        if a in test_actors:
            split_of[a] = "test"
        elif a in val_actors:
            split_of[a] = "val"
        else:
            split_of[a] = "train"
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

   

               
                 


        

        

        
                
        