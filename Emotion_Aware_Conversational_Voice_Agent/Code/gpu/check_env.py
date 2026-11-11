"""
check_env.py - GPU environment report for the emotion voice agent.

Step 1 of the GPU kit. Run it first on any machine (your CPU PC or the
professor's GPU machine). It never fails: it only reports what it finds.

What it reports:
  - OS, Python, torch, CUDA, GPU name, total/free VRAM, bf16 support
  - versions of the libraries the GPU scripts will need
  - free disk space and Hugging Face login status
  - a recommendation for CSM-1B LoRA batch size and Qwen size, based on VRAM

Usage (from Code/):
    python gpu/check_env.py

Output:
    printed report + gpu/outputs/env_report_<computer>_<YYYYMMDD_HHMM>.json
"""

import datetime
import importlib
import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
MIN_FREE_DISK_GB = 40

LIBRARIES = [
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "huggingface_hub",
    "soundfile",
    "librosa",
    "bitsandbytes",
]


def gb(n_bytes):
    return round(n_bytes / 1024**3, 2)


def system_info():
    return {
        "computer": socket.gethostname(),
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
    }


def torch_info():
    info = {"torch": "not installed"}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
        return info

    info["torch"] = torch.__version__
    info["torch_cuda_build"] = torch.version.cuda  # None = CPU-only build
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["gpus"] = []

    if info["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            try:
                free_b, total_b = torch.cuda.mem_get_info(i)
            except Exception:  # noqa: BLE001
                free_b, total_b = None, props.total_memory
            info["gpus"].append(
                {
                    "index": i,
                    "name": props.name,
                    "vram_total_gb": gb(total_b),
                    "vram_free_gb": gb(free_b) if free_b is not None else None,
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        try:
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception:  # noqa: BLE001
            info["bf16_supported"] = None
    elif info["torch_cuda_build"] is None:
        info["note"] = "This torch build is CPU-only. On a GPU machine, install a CUDA build of torch."
    return info


def library_versions():
    versions = {}
    for name in LIBRARIES:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "installed (no version)")
        except Exception:  # noqa: BLE001
            versions[name] = "not installed"
    return versions


def disk_info():
    usage = shutil.disk_usage(HERE)
    free = gb(usage.free)
    return {
        "path": str(HERE.anchor or HERE),
        "total_gb": gb(usage.total),
        "free_gb": free,
        "warning": f"Less than {MIN_FREE_DISK_GB} GB free" if free < MIN_FREE_DISK_GB else None,
    }


def hf_info():
    try:
        from huggingface_hub import whoami
    except Exception:  # noqa: BLE001
        return {"logged_in": False, "user": None, "note": "huggingface_hub not installed"}
    try:
        me = whoami()
        return {"logged_in": True, "user": me.get("name")}
    except Exception:  # noqa: BLE001
        return {"logged_in": False, "user": None}


def recommendation(t):
    if not t.get("cuda_available"):
        return {"tier": "cpu", "text": "No CUDA GPU - CPU only (fine for smoke tests)"}
    if not t.get("gpus"):
        return {"tier": "unknown", "text": "CUDA reports available but no GPU was listed - check drivers."}
    vram = max(g["vram_total_gb"] for g in t["gpus"])
    if vram < 12:
        return {
            "tier": "small",
            "text": f"{vram} GB VRAM: CSM LoRA batch 1 + gradient accumulation; Qwen3-4B or a 4-bit model.",
        }
    if vram < 24:
        qwen = "Qwen3-8B in bf16" if vram >= 16 else "Qwen3-8B in 4-bit"
        return {"tier": "medium", "text": f"{vram} GB VRAM: CSM LoRA batch 2-4; {qwen}."}
    return {"tier": "large", "text": f"{vram} GB VRAM: CSM LoRA batch 4-8; Qwen3-14B possible."}


def print_report(r):
    line = "-" * 60
    print(line)
    print("GPU ENVIRONMENT REPORT")
    print(line)
    s = r["system"]
    print(f"Computer : {s['computer']}")
    print(f"OS       : {s['os']} {s['os_release']} ({s['os_version']})")
    print(f"Python   : {s['python']}  [{s['python_executable']}]")

    t = r["torch"]
    print(line)
    print(f"torch          : {t.get('torch')}")
    print(f"torch CUDA build: {t.get('torch_cuda_build')}")
    print(f"CUDA available : {t.get('cuda_available')}")
    for g in t.get("gpus", []):
        print(
            f"  GPU {g['index']}: {g['name']} | VRAM {g['vram_total_gb']} GB total, "
            f"{g['vram_free_gb']} GB free | compute {g['compute_capability']}"
        )
    if "bf16_supported" in t:
        print(f"bf16 supported : {t['bf16_supported']}")
    if t.get("note"):
        print(f"NOTE: {t['note']}")

    print(line)
    print("Libraries:")
    for k, v in r["libraries"].items():
        print(f"  {k:<16} {v}")

    d = r["disk"]
    print(line)
    print(f"Disk ({d['path']}): {d['free_gb']} GB free of {d['total_gb']} GB")
    if d["warning"]:
        print(f"WARNING: {d['warning']}")

    h = r["huggingface"]
    print(f"Hugging Face   : {'logged in as ' + h['user'] if h['logged_in'] else 'not logged in'}")

    print(line)
    print(f"RECOMMENDATION : {r['recommendation']['text']}")
    print(line)


def main():
    report = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "system": system_info(),
        "torch": torch_info(),
        "libraries": library_versions(),
        "disk": disk_info(),
        "huggingface": hf_info(),
    }
    report["recommendation"] = recommendation(report["torch"])

    print_report(report)

    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        name = report["system"]["computer"].replace(" ", "_")
        out = OUT_DIR / f"env_report_{name}_{stamp}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved: {out}")
    except Exception as e:  # noqa: BLE001
        print(f"Could not save report: {e}")

    sys.exit(0)


if __name__ == "__main__":
    main()
