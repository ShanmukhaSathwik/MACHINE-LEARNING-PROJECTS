"""
test_eval_my_voice_v3.py - scoring logic of scripts/eval_my_voice_v3.py only (made-up predictions, no model loading).
1. Accuracy, per-class accuracy, confusion matrix (rows = true, columns = predicted, CLASSES order).
2. PASS / FAIL at the fixed 22/32 rule.
3. Clip names -> labels; rejected/ ignored; mean loudness grouping.
"""
import numpy as np
import pytest

import eval_my_voice_v3 as ev
from ser_gpu import CLASSES


def test_score_accuracy_per_class_and_confusion():
    true = ["neutral", "neutral", "happy", "happy", "angry", "angry", "sad", "sad"]
    pred = ["neutral", "sad", "happy", "angry", "angry", "angry", "neutral", "neutral"]
    s = ev.score(true, pred)
    assert (s["n_correct"], s["n"]) == (4, 8) and s["accuracy"] == 0.5
    assert {c: (v["correct"], v["total"]) for c, v in s["per_class"].items()} == {
        "neutral": (1, 2), "happy": (1, 2), "angry": (2, 2), "sad": (0, 2)}
    assert s["per_class"]["angry"]["accuracy"] == 1.0 and s["per_class"]["sad"]["accuracy"] == 0.0
    assert CLASSES == ["neutral", "happy", "angry", "sad"]
    expected = np.array([[1, 0, 0, 1],    # true neutral
                         [0, 1, 1, 0],    # true happy
                         [0, 0, 2, 0],    # true angry
                         [2, 0, 0, 0]])   # true sad
    assert np.array_equal(s["confusion"], expected)
    assert s["confusion"].sum() == len(true) and np.trace(s["confusion"]) == s["n_correct"]


def test_score_class_never_seen_and_bad_input():
    s = ev.score(["happy", "happy"], ["happy", "angry"])
    assert s["per_class"]["happy"]["accuracy"] == 0.5
    assert s["per_class"]["sad"]["total"] == 0 and np.isnan(s["per_class"]["sad"]["accuracy"])
    with pytest.raises(ValueError):
        ev.score(["happy"], ["happy", "sad"])           # length mismatch
    with pytest.raises(ValueError):
        ev.score(["calm"], ["happy"])                    # label outside the 4 classes


def test_verdict_at_the_22_of_32_rule():
    assert (ev.N_EXPECTED, ev.MIN_CORRECT) == (32, 22)
    assert ev.verdict(22) == "PASS" and ev.verdict(32) == "PASS"
    assert ev.verdict(21) == "FAIL" and ev.verdict(0) == "FAIL"
    true = [c for c in CLASSES for _ in range(8)]                                  # 32 clips, 8 per class
    pred = true[:22] + ["neutral" if t != "neutral" else "sad" for t in true[22:]]  # first 22 right, rest wrong
    assert ev.verdict(ev.score(true, pred)["n_correct"]) == "PASS"
    pred[0] = "sad"                                                                 # 21 right
    assert ev.verdict(ev.score(true, pred)["n_correct"]) == "FAIL"


def test_list_clips_ignores_rejected_and_rejects_bad_names(tmp_path):
    for name in ["01_neutral.wav", "02_happy.wav", "03_angry.wav", "04_sad.wav"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "rejected").mkdir()
    (tmp_path / "rejected" / "06_angry_223650.wav").write_bytes(b"")
    assert [(p.name, lab) for p, lab in ev.list_clips(tmp_path)] == [
        ("01_neutral.wav", "neutral"), ("02_happy.wav", "happy"), ("03_angry.wav", "angry"), ("04_sad.wav", "sad")]
    (tmp_path / "05_calm.wav").write_bytes(b"")
    with pytest.raises(ValueError):
        ev.list_clips(tmp_path)


def test_rms_dbfs_and_mean_by():
    assert ev.rms_dbfs(np.ones(100)) == pytest.approx(0.0)
    assert ev.rms_dbfs(np.full(100, 0.1)) == pytest.approx(-20.0)
    rows = [{"true_label": "sad", "pred_label": "neutral", "rms_dbfs": -30.0},
            {"true_label": "sad", "pred_label": "sad", "rms_dbfs": -20.0},
            {"true_label": "angry", "pred_label": "angry", "rms_dbfs": -10.0}]
    assert ev.mean_by(rows, "true_label") == {"angry": (-10.0, 1), "sad": (-25.0, 2)}
    assert ev.mean_by(rows, "pred_label") == {"neutral": (-30.0, 1), "angry": (-10.0, 1), "sad": (-20.0, 1)}
