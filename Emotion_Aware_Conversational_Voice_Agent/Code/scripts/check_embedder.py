"""
check_embedder.py
Task D step D1: check that Chroma's built-in sentence embedder runs on CPU without torch.
Brain only: no torch, no models of our own, nothing imported from gpu/.

Embedder: chromadb.utils.embedding_functions.DefaultEmbeddingFunction
    = all-MiniLM-L6-v2 run through onnxruntime (no torch). The ONNX model is downloaded once on the
    first call and cached in Chroma's ONNX cache folder (printed at the end).

Embeds three sentences and prints what is measured:
    embedding size, cosine similarity A-B and A-C,
    time of the first call (includes the one-time model download/load) and of a second call.
    A and B are about the same worry (stress before an event); C is unrelated - the numbers are only
    printed, nothing is asserted about them here.

PASS = the embedder ran and "torch" is not in sys.modules afterwards. FAIL exits with code 1.

CLI:  python scripts/check_embedder.py
"""

import sys
import time

import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

SENTENCES = {
    "A": "I'm stressed about Monday.",
    "B": "The user gets anxious before exams.",
    "C": "The user likes pizza.",
}


def cosine(u, v):
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def main():
    cache_dir = ONNXMiniLM_L6_V2.DOWNLOAD_PATH
    cached_before = (cache_dir / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME / "model.onnx").exists()

    embed = DefaultEmbeddingFunction()
    texts = list(SENTENCES.values())

    t0 = time.perf_counter()
    vecs = embed(texts)
    first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    embed(texts)
    second_s = time.perf_counter() - t0

    a, b, c = vecs
    print(f"Embedder           : {type(embed).__name__} (all-MiniLM-L6-v2, ONNX, CPU)")
    for key, text in SENTENCES.items():
        print(f"  {key} = {text!r}")
    print(f"Embedding size     : {len(a)}")
    print(f"Cosine A-B         : {cosine(a, b):.4f}")
    print(f"Cosine A-C         : {cosine(a, c):.4f}")
    print(f"First call         : {first_s:.3f} s  (model already cached before run: {cached_before})")
    print(f"Second call        : {second_s:.3f} s")
    print(f"Model cache folder : {cache_dir}")

    if "torch" in sys.modules:
        print("FAIL: torch was imported")
        return 1
    print("PASS: embedder ran on CPU and torch is not in sys.modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
