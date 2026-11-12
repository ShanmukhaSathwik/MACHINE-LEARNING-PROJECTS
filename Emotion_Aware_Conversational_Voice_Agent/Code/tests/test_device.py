"""
test_device.py - step 1 of the GPU demo build (gpu/gpu_config.yaml + gpu/device.py).
Checks that every preset loads with its required keys, that smoke_cpu resolves to cpu/float32,
that bad presets and missing GPUs fail with clear errors (never a silent CPU fallback),
and that StageTimer records time. Loads no models.
"""
import importlib.util
import time
from pathlib import Path

import pytest
import torch

CODE_DIR = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("gpu_device", CODE_DIR / "gpu" / "device.py")
device = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(device)

PRESETS = device.list_presets()
GPU_PRESETS = [p for p in PRESETS if p.startswith("gpu_")]


@pytest.fixture(autouse=True)
def no_env_preset(monkeypatch):
    """A preset set in the shell must not change what these tests load."""
    monkeypatch.delenv(device.ENV_VAR, raising=False)


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_loads_with_required_keys(preset):
    cfg = device.load_config(preset)
    assert cfg["preset"] == preset
    assert cfg["device"] in ("cpu", "cuda")
    assert cfg["dtype"]
    assert cfg["asr"]["model"] and cfg["ser"]["model"]
    assert cfg["llm"]["backend"] and cfg["tts"]["backend"]
    # base values survive the merge
    assert cfg["seed"] == 42 and cfg["sample_rate"] == 16000 and cfg["asr"]["language"] == "en"
    assert cfg["brain"]["gamma"] == 0.7 and cfg["llm"]["max_new_tokens"] == 150


def test_smoke_cpu_is_cpu_float32():
    cfg, dev, dtype = device.setup("smoke_cpu")
    assert dev == "cpu"
    assert dtype == torch.float32


def test_unknown_preset_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown preset 'nope'") as e:
        device.load_config("nope")
    for name in PRESETS:
        assert name in str(e.value)


@pytest.mark.parametrize("preset", GPU_PRESETS)
def test_gpu_preset_needs_cuda(preset):
    cfg = device.load_config(preset)
    if torch.cuda.is_available():
        assert device.pick_device(cfg) == "cuda"
    else:
        with pytest.raises(RuntimeError, match=f"Preset {preset} needs a CUDA GPU but none was found"):
            device.pick_device(cfg)


def test_stage_timer_records_positive_time():
    timer = device.StageTimer()
    with timer("work"):
        time.sleep(0.01)
    assert timer.times["work"] > 0
