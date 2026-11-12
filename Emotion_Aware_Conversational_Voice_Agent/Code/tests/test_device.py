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


# Outputs folder: $GPU_OUTPUTS_DIR if set, else gpu/outputs (conftest unsets it for every test)

GPU_OUTPUTS = CODE_DIR / "gpu" / "outputs"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_outputs_dir_unset_or_empty_is_gpu_outputs(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(device.OUTPUTS_ENV_VAR, value)
    assert device.outputs_dir() == GPU_OUTPUTS
    # every gpu/outputs path is exactly what it was before (CODE_DIR / relative)
    ft = device.load_config("gpu_16gb")["finetune"]
    for rel in (ft["output_dir"], "gpu/outputs/voice_refs", "gpu/outputs"):
        assert device.code_path(rel) == CODE_DIR / rel


def test_outputs_dir_env_set_routes_outputs_only(monkeypatch, tmp_path):
    target = tmp_path / "emotion-agent" / "gpu_outputs"      # does not exist yet
    monkeypatch.setenv(device.OUTPUTS_ENV_VAR, str(target))
    assert device.outputs_dir() == target and target.is_dir()  # created
    cfg = device.load_config("gpu_16gb")
    assert device.code_path(cfg["finetune"]["output_dir"]) == target / "ser_finetune"
    assert device.code_path(cfg["tts"]["voice_refs_dir"]) == target / "voice_refs"
    assert device.code_path("gpu/outputs") == target
    # everything else stays in the repo
    assert device.code_path("results/metrics") == CODE_DIR / "results" / "metrics"
    assert device.code_path(cfg["finetune"]["manifest"]) == CODE_DIR / "data" / "manifests" / "ser_4class.csv"
    assert device.code_path("gpu/outputs_old/x") == CODE_DIR / "gpu" / "outputs_old" / "x"
    assert device.code_path("") == CODE_DIR


def test_outputs_dir_relative_value_is_from_code_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(device, "CODE_DIR", tmp_path)          # keep the created folder out of the repo
    monkeypatch.setenv(device.OUTPUTS_ENV_VAR, "outs")
    assert device.outputs_dir() == tmp_path / "outs" and (tmp_path / "outs").is_dir()


def test_code_path_keeps_absolute_paths(monkeypatch, tmp_path):
    run = tmp_path / "gpu_16gb_20261112_1400"
    assert device.code_path(run) == run
    monkeypatch.setenv(device.OUTPUTS_ENV_VAR, str(tmp_path / "elsewhere"))
    assert device.code_path(run) == run


def test_check_env_shows_outputs_but_saves_report_in_repo(monkeypatch, tmp_path):
    import check_env

    assert check_env.REPORT_DIR == GPU_OUTPUTS                     # env reports: repo's gpu/outputs (tracked)
    assert check_env.resolved_outputs_dir() == str(GPU_OUTPUTS)    # unset: shown as gpu/outputs
    elsewhere = tmp_path / "gpu_outputs"
    monkeypatch.setenv(device.OUTPUTS_ENV_VAR, str(elsewhere))
    assert check_env.resolved_outputs_dir() == str(elsewhere)      # set: the report shows the variable works
    # ...but the report file ignores it (REPORT_DIR pointed at tmp so the real gpu/outputs is not touched)
    monkeypatch.setattr(check_env, "REPORT_DIR", tmp_path / "repo_gpu_outputs")
    out = check_env.save_report({"system": {"computer": "test pc"}, "outputs_dir": str(elsewhere)})
    assert out.parent == tmp_path / "repo_gpu_outputs" and out.name.startswith("env_report_test_pc_")
    assert not list(elsewhere.glob("env_report_*"))
