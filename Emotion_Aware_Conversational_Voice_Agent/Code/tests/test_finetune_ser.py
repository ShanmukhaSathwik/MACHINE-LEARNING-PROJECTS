"""
test_finetune_ser.py - training T2 (gpu/finetune_ser.py). No downloads, no model loading.
1. Row selection: only in_4class rows; no speaker in two splits (on the real ser_4class.csv).
2. Label ids follow the model's own id2label and cover exactly CLASSES.
3. Class weights: rarer class -> larger weight; always finite.
4. Augmentation: float32 mono at 16 kHz, length close to the input; val/test are never augmented.
"""
import numpy as np
import pytest
import soundfile as sf

import finetune_ser as ft
from device import code_path, load_config
from ser_gpu import CLASSES, SAMPLE_RATE

CFG = load_config("smoke_cpu")
AUG = CFG["finetune"]["augment"]
MAX_SECONDS = CFG["ser"]["max_seconds"]
# speed s resamples the clip to length / s, so the configured [0.9, 1.1] gives lengths x0.909 .. x1.111
LEN_LO, LEN_HI = 1 / max(AUG["speed"]) - 0.005, 1 / min(AUG["speed"]) + 0.005


def test_row_selection_and_no_speaker_leak():
    rows = ft.load_rows(code_path(CFG["finetune"]["manifest"]), in_4class_only=True)
    assert rows and all(r["in_4class"] is True for r in rows)
    assert {r["label"] for r in rows} == set(CLASSES)          # fearful never selected
    assert ft.speaker_leaks(rows) == {}
    all_rows = ft.load_rows(code_path(CFG["finetune"]["manifest"]), in_4class_only=False)
    assert len(all_rows) > len(rows) and "fearful" in {r["label"] for r in all_rows}
    leaky = [{"speaker_id": "x", "split": "train"}, {"speaker_id": "x", "split": "test"}]
    assert ft.speaker_leaks(leaky) == {"x": ["test", "train"]}


def test_label_ids_follow_model_id2label():
    assert ft.label_ids({0: "neu", 1: "hap", 2: "ang", 3: "sad"}) == {"neutral": 0, "happy": 1, "angry": 2, "sad": 3}
    shuffled = ft.label_ids({0: "sad", 1: "neu", 2: "ang", 3: "hap"})   # a different model order is kept as-is
    assert shuffled == {"sad": 0, "neutral": 1, "angry": 2, "happy": 3}
    assert sorted(shuffled) == sorted(CLASSES)
    with pytest.raises(ValueError):
        ft.label_ids({0: "neu", 1: "hap", 2: "ang"})                 # a class missing
    with pytest.raises(ValueError):
        ft.label_ids({0: "neu", 1: "hap", 2: "ang", 3: "calm"})      # unknown label


def test_cli_overrides_and_sanity_run_id():
    import datetime
    from types import SimpleNamespace

    cfg = load_config("gpu_16gb")
    args = SimpleNamespace(max_steps=20, max_train_clips=200, max_val_clips=50, num_workers=0)
    applied = ft.apply_overrides(cfg["finetune"], args)
    assert {k: new for k, _, new in applied} == {"max_steps": 20, "max_train_clips": 200,
                                                 "max_val_clips": 50, "num_workers": 0}
    assert dict((k, old) for k, old, _ in applied)["num_workers"] == 4          # old value reported
    f = cfg["finetune"]
    assert (f["max_steps"], f["max_train_clips"], f["max_val_clips"], f["num_workers"]) == (20, 200, 50, 0)
    assert f["batch_size"] == 8                                                   # untouched keys stay
    when = datetime.datetime(2026, 11, 12, 14, 5)
    assert ft.make_run_id(cfg["preset"], when, sanity=bool(applied)) == "gpu_16gb_20261112_1405_sanity"
    # no flags given -> nothing changes and the run id has no suffix
    plain = load_config("gpu_16gb")
    none = SimpleNamespace(max_steps=None, max_train_clips=None, max_val_clips=None, num_workers=None)
    assert ft.apply_overrides(plain["finetune"], none) == []
    assert ft.make_run_id("gpu_16gb", when, sanity=False) == "gpu_16gb_20261112_1405"


def test_class_weights():
    ids = [0] * 10 + [1] * 5 + [2] * 20 + [3] * 1
    w = ft.class_weights(ids, 4)
    assert np.all(np.isfinite(w)) and w.dtype == np.float32
    assert w[3] > w[1] > w[0] > w[2]                                 # rarer -> larger
    w0 = ft.class_weights([0, 0, 1, 1, 2, 2], 4)                     # class 3 has no clips
    assert np.all(np.isfinite(w0)) and w0[3] > w0[0]


def _tone(seconds, sr):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


@pytest.mark.parametrize("seed", range(6))
def test_augment_output_format_and_length(seed):
    y = _tone(2.0, SAMPLE_RATE)
    out = ft.augment(y, {**AUG, "prob": 1.0}, np.random.default_rng(seed))   # all three steps
    assert out.dtype == np.float32 and out.ndim == 1 and np.all(np.isfinite(out))
    assert LEN_LO <= len(out) / len(y) <= LEN_HI
    assert not np.allclose(out[: min(len(out), len(y))], y[: min(len(out), len(y))])
    same = ft.augment(y, {**AUG, "prob": 0.0}, np.random.default_rng(seed))  # nothing applied
    assert np.array_equal(same, y)


def test_speed_extremes_and_dataset_16k(tmp_path):
    y = _tone(2.0, SAMPLE_RATE)
    for s in AUG["speed"]:
        out = ft.augment(y, {"speed": [s, s], "pitch_semitones": [0, 0], "noise_snr_db": [30, 30], "prob": 1.0},
                         np.random.default_rng(0))
        assert abs(len(out) / len(y) - 1 / s) < 0.005
    # A 48 kHz stereo file comes out of the dataset as 16 kHz mono float32
    wav = tmp_path / "x.wav"
    sf.write(wav, np.stack([_tone(2.0, 48000)] * 2, axis=1), 48000)
    rows = [{"wav_path": str(wav), "label": "happy"}]
    item, label = ft.SERDataset(rows, {"happy": 1}, MAX_SECONDS)[0]
    assert item.dtype == np.float32 and item.ndim == 1 and label == 1
    assert abs(len(item) - 2.0 * SAMPLE_RATE) <= 0.1 * 2.0 * SAMPLE_RATE


def test_val_and_test_are_never_augmented(tmp_path):
    wav = tmp_path / "y.wav"
    sf.write(wav, _tone(2.0, SAMPLE_RATE), SAMPLE_RATE)
    rows = [{"wav_path": str(wav), "label": "sad"}]
    splits = {"train": rows, "val": rows, "test": rows}
    ds = ft.make_datasets(splits, {"sad": 3}, MAX_SECONDS, {**AUG, "prob": 1.0}, seed=0)
    assert ds["train"].augment_cfg is not None
    assert ds["val"].augment_cfg is None and ds["test"].augment_cfg is None
    clean = ft.preprocess(*sf.read(str(wav), dtype="float32"), MAX_SECONDS)
    for split in ("val", "test"):
        a, _ = ds[split][0]
        b, _ = ds[split][0]
        assert np.array_equal(a, clean) and np.array_equal(a, b)
    t, _ = ds["train"][0]
    assert not (len(t) == len(clean) and np.array_equal(t, clean))
