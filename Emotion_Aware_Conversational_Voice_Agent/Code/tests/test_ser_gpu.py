"""
test_ser_gpu.py - step 2 of the GPU demo build (gpu/ser_gpu.py).
Tests only the pure helpers: label mapping, softmax output, and true labels from file names.
No test downloads or loads a model.
"""
import numpy as np
import pytest

import ser_gpu


def test_map_labels_short_and_full_words():
    superb = {0: "neu", 1: "hap", 2: "ang", 3: "sad"}
    assert ser_gpu.map_labels(superb) == {0: "neutral", 1: "happy", 2: "angry", 3: "sad"}
    full = {"0": "Angry", "1": "neutral", "2": "SAD", "3": "happy"}   # string keys, any case
    assert ser_gpu.map_labels(full) == {0: "angry", 1: "neutral", 2: "sad", 3: "happy"}
    with pytest.raises(ValueError, match="calm"):
        ser_gpu.map_labels({0: "neu", 1: "calm"})


def test_build_output():
    index_map = {0: "neutral", 1: "happy", 2: "angry", 3: "sad"}
    out = ser_gpu.build_output(np.array([0.2, 2.5, -1.0, 0.7]), index_map)
    probs = out["probs"]
    assert set(probs) == set(ser_gpu.CLASSES)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert out["confidence"] == pytest.approx(max(probs.values()))
    assert out["margin"] >= 0
    assert out["p_distress"] == pytest.approx(probs["angry"] + probs["sad"])
    assert out["label"] == max(probs, key=probs.get) == "happy"


def test_true_label_from_filename():
    assert ser_gpu.true_label_from_filename("03_happy.wav") == "happy"
    assert ser_gpu.true_label_from_filename("07_calm.wav") is None
    assert ser_gpu.true_label_from_filename("00_test2.wav") is None
