"""Make scripts/ importable from tests (the scripts import each other as top-level modules)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gpu"))


@pytest.fixture(autouse=True)
def no_gpu_outputs_dir(monkeypatch):
    """GPU_OUTPUTS_DIR set on this PC must never send a test's paths to the real outputs folder.
    Tests that need it set it themselves (to a tmp_path)."""
    monkeypatch.delenv("GPU_OUTPUTS_DIR", raising=False)
