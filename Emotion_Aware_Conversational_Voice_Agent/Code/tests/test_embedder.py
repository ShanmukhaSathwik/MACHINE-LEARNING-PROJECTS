"""Task D step D1: scripts/check_embedder.py (Chroma's ONNX MiniLM embedder on CPU, no torch)."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

CODE_DIR = Path(__file__).resolve().parent.parent
SCRIPT = CODE_DIR / "scripts" / "check_embedder.py"


def test_check_embedder_script_passes():
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=600, cwd=CODE_DIR)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS" in out.stdout, out.stdout


def test_check_embedder_imports_no_torch_or_gpu():
    gpu_modules = {p.stem for p in (CODE_DIR / "gpu").glob("*.py")}
    text = SCRIPT.read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", text, flags=re.M))
    assert "torch" not in imported and not imported & gpu_modules, imported
