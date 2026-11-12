"""
device.py - preset config, device / dtype choice, seeding and timing for the GPU demo build.

Step 1 of the GPU demo build. Loads no models and downloads nothing.

What it gives the other GPU scripts:
  - load_config(preset)  -> dict: base + preset from gpu/gpu_config.yaml, deep-merged
  - pick_device(cfg)     -> "cuda" or "cpu"; never falls back to CPU silently
  - pick_dtype(cfg, dev) -> torch dtype ("auto" = bfloat16 if supported, else float16 on cuda)
  - set_seed(seed)       -> python, numpy, torch (and CUDA) seeded together
  - vram_report()        -> GPU name and memory in GB, or {"device": "cpu"}
  - StageTimer           -> seconds per named pipeline stage
  - setup(preset)        -> (cfg, device, dtype), the one call most scripts need

Preset choice (first match wins): the preset argument, the AGENT_PRESET
environment variable, then "smoke_cpu".

Usage (works from any folder):
    python gpu/device.py --list
    python gpu/device.py --preset smoke_cpu
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent   # Code/gpu/
CODE_DIR = HERE.parent                    # Code/ (relative paths in the config resolve from here)
CONFIG_PATH = HERE / "gpu_config.yaml"
DEFAULT_PRESET = "smoke_cpu"
ENV_VAR = "AGENT_PRESET"

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _read_yaml(path=CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """New dict: base with override on top. Nested dicts merge key by key; anything else is replaced."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def list_presets() -> list:
    return list(_read_yaml()["presets"])


def load_config(preset=None) -> dict:
    """base + preset, deep-merged. The chosen name is stored in cfg["preset"]."""
    raw = _read_yaml()
    name = preset or os.environ.get(ENV_VAR) or DEFAULT_PRESET
    presets = raw["presets"]
    if name not in presets:
        raise ValueError(f"Unknown preset {name!r}. Available presets: {', '.join(presets)}")
    cfg = _deep_merge(raw["base"], presets[name])
    cfg["preset"] = name
    return cfg


def code_path(relative) -> Path:
    """Resolve a config path (relative to Code/) to an absolute path."""
    return CODE_DIR / relative


def pick_device(cfg) -> str:
    wanted = cfg["device"]
    if wanted == "cpu":
        return "cpu"
    if wanted == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Preset {cfg['preset']} needs a CUDA GPU but none was found. "
                "Use --preset smoke_cpu at home, or run gpu/check_env.py on the GPU machine."
            )
        return "cuda"
    raise ValueError(f"Preset {cfg['preset']}: device must be 'cpu' or 'cuda', got {wanted!r}")


def pick_dtype(cfg, device):
    """An explicit dtype in the config wins. "auto": bfloat16 / float16 on cuda, float32 on cpu."""
    wanted = cfg["dtype"]
    if wanted != "auto":
        if wanted not in DTYPES:
            raise ValueError(f"Preset {cfg['preset']}: dtype must be 'auto' or one of {list(DTYPES)}, got {wanted!r}")
        return DTYPES[wanted]
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vram_report() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    gb = 1024 ** 3
    idx = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(idx)
    return {
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(idx),
        "total_gb": round(total / gb, 2),
        "free_gb": round(free / gb, 2),
        "allocated_gb": round(torch.cuda.memory_allocated(idx) / gb, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(idx) / gb, 2),
    }


class StageTimer:
    """Seconds per named stage. A stage timed more than once adds up.

        timer = StageTimer()
        with timer("asr"):
            ...
        timer.times   # {"asr": 0.42}
    """

    def __init__(self):
        self.times = {}

    @contextmanager
    def __call__(self, stage):
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            self.times[stage] = self.times.get(stage, 0.0) + time.perf_counter() - t0

    def total(self) -> float:
        return sum(self.times.values())

    def __repr__(self):
        return "StageTimer(" + ", ".join(f"{k}={v:.3f}s" for k, v in self.times.items()) + ")"


def setup(preset=None):
    """load_config, pick_device, pick_dtype, set_seed, in that order. Returns (cfg, device, dtype)."""
    cfg = load_config(preset)
    device = pick_device(cfg)
    dtype = pick_dtype(cfg, device)
    set_seed(cfg["seed"])
    return cfg, device, dtype


def _main():
    parser = argparse.ArgumentParser(description="Show a GPU demo preset and the device it resolves to.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preset", help=f"preset name (default: ${ENV_VAR}, else {DEFAULT_PRESET})")
    group.add_argument("--list", action="store_true", help="print the preset names and exit")
    args = parser.parse_args()

    if args.list:
        for name in list_presets():
            print(name)
        return 0

    try:
        cfg, device, dtype = setup(args.preset)
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"preset      : {cfg['preset']}")
    print(f"device      : {device}")
    print(f"dtype       : {str(dtype).replace('torch.', '')}")
    print(f"seed        : {cfg['seed']}")
    print(f"sample_rate : {cfg['sample_rate']}")
    print(f"brain       : {json.dumps(cfg['brain'])}")
    for section in ("asr", "ser", "llm", "tts"):
        print(f"{section:12s}: {json.dumps(cfg.get(section))}")
    print(f"log_dir     : {code_path(cfg['logging']['log_dir'])}")
    print(f"memory_file : {code_path(cfg['logging']['memory_file'])}")
    print(f"vram        : {json.dumps(vram_report())}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
