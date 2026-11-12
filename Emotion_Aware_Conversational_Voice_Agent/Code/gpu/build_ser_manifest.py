"""
build_ser_manifest.py - combined RAVDESS + CREMA-D emotion manifest for SER fine-tuning.

Training prep T1. Builds a manifest only; trains nothing, changes no audio.

Output:
  data/manifests/ser_4class.csv
      utt_id, corpus, speaker_id, wav_path (relative to Code/), label, split, duration, in_4class
  data/splits/crema_d_splits.json
      {"1001": "train", ...}, same format as data/splits/actor_splits.json
  (if a file already exists, the new one gets a timestamp; nothing is overwritten)

Labels: neutral, happy, angry, sad, fearful. in_4class = False only for fearful
(kept for a possible 5-class run; not used in the first training run).
  RAVDESS  (data/raw/Actor_XX/*.wav, speech only): 01 neutral + 02 calm -> neutral, 03 happy, 04 sad,
           05 angry, 06 fearful; 07 disgust and 08 surprised dropped.
           Splits: data/splits/actor_splits.json, unchanged, so train/val/test actors match the baseline.
  CREMA-D  (data/raw/crema_d/AudioWAV/AAAA_SSS_EEE_LL.wav): NEU, HAP, ANG, SAD, FEA -> fearful; DIS dropped.
           Splits: speaker-independent by actor, 70/15/15, seed 42 (sorted actor IDs shuffled with
           random.Random(42); train = round(0.70 n), val = round(0.15 n), test = the rest).
speaker_id is unique across corpora: rav_16, crema_1001.

Duplicate recordings (checked before any row is built):
  Every audio file of both corpora is hashed (SHA-1 of the file bytes), including the files whose
  emotion is dropped (DIS / disgust / surprised count as their own label here).
  - Byte-identical files whose labels differ: the true label is unknown, so EVERY file in that group is left out.
  - Byte-identical files with the same label: the first by path is kept, the rest are left out.
  Every excluded file is printed with its reason. The expected result is exactly 7 files:
  3 CREMA-D pairs with different labels (1006_TIE_HAP/NEU, 1013_WSI_DIS/SAD, 1017_IWW_ANG/FEA, all 6 out)
  and 1 RAVDESS same-label copy (Actor_07/03-01-03-01-02-02-07 = ...-02-01-07, happy; the second copy out).
  If the check finds anything else, it prints it and stops before writing any output.
  The RAVDESS baseline check counts the excluded RAVDESS copy as expected (287/288 present + 1 duplicate = PASS).

Usage (works from any folder):
    python gpu/build_ser_manifest.py
"""

import csv
import datetime
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent   # Code/gpu/
CODE_DIR = HERE.parent                    # Code/
RAVDESS_DIR = CODE_DIR / "data" / "raw"
CREMA_DIR = CODE_DIR / "data" / "raw" / "crema_d" / "AudioWAV"
RAVDESS_SPLITS = CODE_DIR / "data" / "splits" / "actor_splits.json"
CREMA_SPLITS = CODE_DIR / "data" / "splits" / "crema_d_splits.json"
OUT_CSV = CODE_DIR / "data" / "manifests" / "ser_4class.csv"
BASELINE_MANIFEST = CODE_DIR / "data" / "manifests" / "manifest.csv"

LABELS = ["neutral", "happy", "angry", "sad", "fearful"]
FOUR_CLASS = {"neutral", "happy", "angry", "sad"}
SPLITS = ["train", "val", "test"]
FIELDS = ["utt_id", "corpus", "speaker_id", "wav_path", "label", "split", "duration", "in_4class"]

RAVDESS_EMOTION = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fearful"}
RAVDESS_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$")  # mod-vocal-emo-int-stmt-rep-actor

CREMA_EMOTION = {"NEU": "neutral", "HAP": "happy", "ANG": "angry", "SAD": "sad", "FEA": "fearful"}
CREMA_ALL_EMOTIONS = set(CREMA_EMOTION) | {"DIS"}
CREMA_RE = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})$")   # actor_sentence_emotion_level
CREMA_EXPECTED_FILES = 7442
CREMA_SEED, CREMA_TRAIN, CREMA_VAL = 42, 0.70, 0.15

RAVDESS_DROPPED = {"07": "disgust", "08": "surprised"}
# The duplicate check must find exactly these 7 files: 3 byte-identical CREMA-D pairs with different labels
# (whole pair out) + 1 RAVDESS file identical to another take with the same label (second copy out).
EXPECTED_EXCLUDED = {
    "data/raw/crema_d/AudioWAV/1006_TIE_HAP_XX.wav", "data/raw/crema_d/AudioWAV/1006_TIE_NEU_XX.wav",
    "data/raw/crema_d/AudioWAV/1013_WSI_DIS_XX.wav", "data/raw/crema_d/AudioWAV/1013_WSI_SAD_XX.wav",
    "data/raw/crema_d/AudioWAV/1017_IWW_ANG_XX.wav", "data/raw/crema_d/AudioWAV/1017_IWW_FEA_XX.wav",
    "data/raw/Actor_07/03-01-03-01-02-02-07.wav",   # same bytes as 03-01-03-01-02-01-07.wav (happy, both train)
}


def free_name(path: Path) -> Path:
    """path if it does not exist yet, else the same name with a timestamp before the extension."""
    if not path.exists():
        return path
    return path.with_name(f"{path.stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}{path.suffix}")


def rel(path: Path) -> str:
    return path.relative_to(CODE_DIR).as_posix()


def duration_of(path: Path) -> float:
    """Seconds, from the header only. Raises if the file is not real audio (e.g. a Git LFS pointer)."""
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF":
            raise ValueError(f"{rel(path)} is not a WAV file (Git LFS pointer not downloaded?)")
    info = sf.info(str(path))
    return info.frames / info.samplerate


def catalog(crema_wavs):
    """[(wav_path relative to Code/, label)] for EVERY audio file of both corpora; dropped emotions keep their own name."""
    items = []
    for wav in sorted(RAVDESS_DIR.glob("Actor_*/*.wav")):
        m = RAVDESS_RE.match(wav.stem)
        if not m:
            raise ValueError(f"Unexpected RAVDESS file name: {rel(wav)}")
        emo = m.group(3)
        items.append((rel(wav), RAVDESS_EMOTION.get(emo) or RAVDESS_DROPPED.get(emo, f"code {emo}")))
    for wav in crema_wavs:
        emo = CREMA_RE.match(wav.stem).group(3)
        items.append((rel(wav), CREMA_EMOTION.get(emo, "disgust")))
    return items


def find_duplicates(items):
    """SHA-1 of every file. Returns ({excluded path: reason}, number of duplicate groups)."""
    groups = defaultdict(list)
    for path, label in items:
        groups[hashlib.sha1((CODE_DIR / path).read_bytes()).hexdigest()].append((path, label))
    excluded, n_groups = {}, 0
    for digest, members in groups.items():
        if len(members) < 2:
            continue
        n_groups += 1
        members.sort()
        desc = ", ".join(f"{Path(p).name} ({lab})" for p, lab in members)
        if len({lab for _, lab in members}) > 1:
            for path, _ in members:
                excluded[path] = f"identical bytes, different labels -> whole group left out [{desc}] sha1 {digest[:10]}"
        else:
            for path, _ in members[1:]:
                excluded[path] = f"identical bytes, same label -> duplicate of {Path(members[0][0]).name} sha1 {digest[:10]}"
    return excluded, n_groups


def ravdess_rows(exclude=frozenset()):
    splits = json.loads(RAVDESS_SPLITS.read_text(encoding="utf-8"))
    rows, dropped = [], defaultdict(int)
    for wav in sorted(RAVDESS_DIR.glob("Actor_*/*.wav")):
        if rel(wav) in exclude:
            dropped["duplicate"] += 1
            continue
        m = RAVDESS_RE.match(wav.stem)
        if not m:
            raise ValueError(f"Unexpected RAVDESS file name: {rel(wav)}")
        _, vocal, emo, _, _, _, actor = m.groups()
        if vocal != "01":            # 01 = speech, 02 = song
            dropped["song"] += 1
            continue
        label = RAVDESS_EMOTION.get(emo)
        if label is None:
            dropped[RAVDESS_DROPPED.get(emo, f"code {emo}")] += 1
            continue
        actor = str(int(actor))
        rows.append({
            "utt_id": wav.stem, "corpus": "ravdess", "speaker_id": f"rav_{actor}", "wav_path": rel(wav),
            "label": label, "split": splits[actor], "duration": round(duration_of(wav), 3),
            "in_4class": label in FOUR_CLASS,
        })
    return rows, dict(dropped)


def crema_files():
    """All CREMA-D wavs, after checking the count and the naming pattern."""
    wavs = sorted(CREMA_DIR.glob("*.wav"))
    bad = [w.name for w in wavs if not (CREMA_RE.match(w.stem) and CREMA_RE.match(w.stem).group(3) in CREMA_ALL_EMOTIONS)]
    if bad:
        raise ValueError(f"{len(bad)} CREMA-D files do not match AAAA_SSS_EEE_LL.wav, e.g. {bad[:5]}")
    if len(wavs) != CREMA_EXPECTED_FILES:
        raise ValueError(f"Expected {CREMA_EXPECTED_FILES} CREMA-D wav files, found {len(wavs)} in {rel(CREMA_DIR)}")
    return wavs


def crema_split(actors):
    """Speaker-independent 70/15/15 split by actor ID, seed 42."""
    order = sorted(actors)
    random.Random(CREMA_SEED).shuffle(order)
    n_train, n_val = round(CREMA_TRAIN * len(order)), round(CREMA_VAL * len(order))
    split = {a: "train" for a in order[:n_train]}
    split.update({a: "val" for a in order[n_train:n_train + n_val]})
    split.update({a: "test" for a in order[n_train + n_val:]})
    return {a: split[a] for a in sorted(split, key=int)}


def crema_rows(wavs, splits, exclude=frozenset()):
    rows, dropped, unreadable = [], defaultdict(int), []
    for wav in wavs:
        actor, _, emo, _ = CREMA_RE.match(wav.stem).groups()
        if rel(wav) in exclude:
            dropped["duplicate"] += 1
            continue
        label = CREMA_EMOTION.get(emo)
        if label is None:
            dropped["disgust"] += 1
            continue
        try:
            dur = duration_of(wav)
        except Exception as e:  # noqa: BLE001
            unreadable.append(f"{wav.name}: {type(e).__name__}: {e}")
            continue
        if dur <= 0:
            unreadable.append(f"{wav.name}: zero length")
            continue
        rows.append({
            "utt_id": wav.stem, "corpus": "crema_d", "speaker_id": f"crema_{actor}", "wav_path": rel(wav),
            "label": label, "split": splits[actor], "duration": round(dur, 3),
            "in_4class": label in FOUR_CLASS,
        })
    return rows, dict(dropped), unreadable


def report(rows):
    corpora = sorted({r["corpus"] for r in rows}, reverse=True)   # ravdess, crema_d
    print("\nClips per corpus x label x split   (* = in_4class)")
    print(f"  {'corpus':9s} {'label':10s}" + "".join(f"{s:>8s}" for s in SPLITS) + f"{'total':>8s}")
    for corpus in corpora + ["ALL"]:
        for label in LABELS:
            counts = [sum(1 for r in rows if (corpus == "ALL" or r["corpus"] == corpus)
                          and r["label"] == label and r["split"] == s) for s in SPLITS]
            mark = "*" if label in FOUR_CLASS else " "
            print(f"  {corpus:9s} {mark}{label:9s}" + "".join(f"{c:>8d}" for c in counts) + f"{sum(counts):>8d}")
        for name, keep in (("4-class", True), ("all", None)):
            counts = [sum(1 for r in rows if (corpus == "ALL" or r["corpus"] == corpus) and r["split"] == s
                          and (keep is None or r["in_4class"] == keep)) for s in SPLITS]
            print(f"  {corpus:9s} {'= ' + name:10s}" + "".join(f"{c:>8d}" for c in counts) + f"{sum(counts):>8d}")
        print()

    print("Actors per split")
    for corpus in corpora:
        counts = [len({r["speaker_id"] for r in rows if r["corpus"] == corpus and r["split"] == s}) for s in SPLITS]
        print(f"  {corpus:9s}" + "".join(f"  {s} {c:3d}" for s, c in zip(SPLITS, counts)) + f"   total {sum(counts)}")
    test_ids = {c: sorted({r["speaker_id"] for r in rows if r["corpus"] == c and r["split"] == "test"},
                          key=lambda x: int(x.split("_")[1])) for c in corpora}
    for c, ids in test_ids.items():
        print(f"  {c} test speakers: {', '.join(ids)}")

    print("\nHours per split (all labels / 4-class only)")
    for corpus in corpora + ["ALL"]:
        cells = []
        for s in SPLITS:
            sel = [r for r in rows if (corpus == "ALL" or r["corpus"] == corpus) and r["split"] == s]
            cells.append(f"{s} {sum(r['duration'] for r in sel) / 3600:5.2f} / "
                         f"{sum(r['duration'] for r in sel if r['in_4class']) / 3600:5.2f}")
        print(f"  {corpus:9s}  " + "   ".join(cells))

    print("\nSpeaker leakage check")
    speaker_splits = defaultdict(set)
    for r in rows:
        speaker_splits[r["speaker_id"]].add(r["split"])
    leaks = {spk: sorted(s) for spk, s in speaker_splits.items() if len(s) > 1}
    if leaks:
        print(f"  FAIL: {len(leaks)} speakers appear in more than one split: {leaks}")
    else:
        print(f"  PASS: all {len(speaker_splits)} speakers appear in exactly one split (no speaker in two splits)")
    return not leaks


def baseline_check(rows, excluded=()):
    """RAVDESS angry/happy/sad rows must carry the same split as the baseline manifest.
    Baseline clips left out by the duplicate check count as expected, not as failures."""
    with open(BASELINE_MANIFEST, newline="", encoding="utf-8") as f:
        base = {r["utt_id"]: r["split"] for r in csv.DictReader(f)}
    ours = {r["utt_id"]: r["split"] for r in rows if r["corpus"] == "ravdess" and r["utt_id"] in base}
    diff = sorted(u for u in ours if ours[u] != base[u])
    dup_ids = {Path(p).stem for p in excluded if p.startswith("data/raw/Actor_")}
    missing = sorted(set(base) - set(ours))
    as_duplicate = [u for u in missing if u in dup_ids]
    unexplained = [u for u in missing if u not in dup_ids]
    for u in as_duplicate:
        print(f"  excluded as a duplicate (expected): {u} (baseline split {base[u]})")
    ok = not diff and not unexplained
    print(f"  {'PASS' if ok else 'FAIL'}: {len(ours)}/{len(base)} baseline manifest clips present with the same split"
          f" + {len(as_duplicate)} excluded as a duplicate"
          + ("" if ok else f"; {len(diff)} with a different split {diff[:5]}, {len(unexplained)} missing {unexplained[:5]}"))
    return ok


def main():
    if not CREMA_DIR.is_dir():
        print(f"ERROR: {rel(CREMA_DIR)} not found. Download CREMA-D AudioWAV first.", file=sys.stderr)
        return 1

    wavs = crema_files()
    print(f"CREMA-D : {len(wavs)} wav files found, all named AAAA_SSS_EEE_LL.wav")

    items = catalog(wavs)
    print(f"\nDuplicate check: SHA-1 of all {len(items)} audio files "
          f"({sum(p.startswith('data/raw/Actor_') for p, _ in items)} RAVDESS + {len(wavs)} CREMA-D, dropped emotions included)")
    excluded, n_groups = find_duplicates(items)
    for path in sorted(excluded):
        print(f"  EXCLUDED {path}\n           {excluded[path]}")
    by_corpus = defaultdict(int)
    for path in excluded:
        by_corpus["crema_d" if "/crema_d/" in path else "ravdess"] += 1
    print(f"  totals: {n_groups} duplicate groups, {len(excluded)} files excluded {dict(by_corpus)}")
    if set(excluded) != EXPECTED_EXCLUDED:
        print(f"  STOP: the duplicate check does not match the expected {len(EXPECTED_EXCLUDED)} files "
              "(3 CREMA-D pairs + 1 RAVDESS copy).")
        print(f"    unexpected: {sorted(set(excluded) - EXPECTED_EXCLUDED)}")
        print(f"    expected but not found: {sorted(EXPECTED_EXCLUDED - set(excluded))}")
        print("  Nothing was written.")
        return 1
    print(f"  matches the expected {len(EXPECTED_EXCLUDED)} files (3 CREMA-D pairs + 1 RAVDESS copy)\n")

    rav, rav_dropped = ravdess_rows(set(excluded))
    print(f"RAVDESS : {len(rav)} clips kept, dropped {rav_dropped}")

    actors = sorted({CREMA_RE.match(w.stem).group(1) for w in wavs}, key=int)
    splits = crema_split(actors)
    crema, crema_dropped, unreadable = crema_rows(wavs, splits, set(excluded))
    print(f"CREMA-D : {len(crema)} clips kept from {len(actors)} actors, dropped {crema_dropped}")
    if unreadable:
        print(f"CREMA-D : {len(unreadable)} unreadable / empty files left out:")
        for u in unreadable:
            print(f"    {u}")

    rows = sorted(rav + crema, key=lambda r: (r["corpus"] != "ravdess", int(r["speaker_id"].split("_")[1]), r["utt_id"]))
    ok = report(rows)
    print("\nRAVDESS splits vs baseline data/manifests/manifest.csv")
    ok = baseline_check(rows, excluded) and ok

    splits_path = free_name(CREMA_SPLITS)
    splits_path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    csv_path = free_name(OUT_CSV)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {rel(csv_path)} ({len(rows)} rows) and {rel(splits_path)} ({len(splits)} actors)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
