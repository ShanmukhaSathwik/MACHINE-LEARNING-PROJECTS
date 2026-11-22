"""Task F step R2: scripts/run_rq2_eval.py with a fake embedder (hashed bag of content words, no model).
Cosines of the fake embedding can be worked out by hand: cos = shared content words / sqrt(|A| * |B|)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("chromadb")
from chromadb.api.types import EmbeddingFunction  # noqa: E402

import run_rq2_eval as rq2  # noqa: E402
from memory_manager import ChromaMemoryManager, content_words, importance_from_turn  # noqa: E402

CFG = {"top_k": 3, "w_similarity": 0.7, "w_importance": 0.3, "base_importance": 0.3,
       "spike_threshold": 0.6, "max_fact_words": 25, "min_similarity": 0.15}
DIM = 1024


class FakeEmbedder(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            v = np.zeros(DIM, dtype=np.float32)
            for w in content_words(text):
                v[1 + int(hashlib.md5(w.encode()).hexdigest(), 16) % (DIM - 1)] += 1.0
            if not v.any():
                v[0] = 1.0
            out.append(v)
        return out

    @staticmethod
    def name():
        return "test_fake_bag_of_words_rq2"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEmbedder()


# One user, three facts. Query "piano lessons" (content words {piano, lessons}):
#   a_f1 strong {loves, piano}                  cos 1/sqrt(2*2) = 0.5
#   a_f2 calm   {takes, piano, lessons, weekly} cos 2/sqrt(2*4) = 0.707
#   a_f3 calm   {grows, tomatoes}               cos 0 -> cut by min_similarity 0.15
# Weighted: a_f1 0.7*0.5 + 0.3*0.93 = 0.629 > a_f2 0.7*0.707 + 0.3*0.30 = 0.585, so the strong fact jumps ahead;
# similarity-only keeps a_f2 first.
FACTS = pd.DataFrame([
    ("a", "a_f1", "The user loves piano.", "strong", "hobbies"),
    ("a", "a_f2", "The user takes piano lessons weekly.", "calm", "hobbies"),
    ("a", "a_f3", "The user grows tomatoes.", "calm", "plans"),
], columns=["user_id", "fact_id", "fact", "intensity", "topic"])
QUERIES = pd.DataFrame([
    ("a", "a_q1", "piano lessons", "a_f2", "calm_target_charged_distractor", "a_f1"),
    ("a", "a_q2", "rainy weather outside", "a_f3", "calm_target", ""),
    ("a", "a_q3", "loves piano", "a_f1", "charged_target", ""),
], columns=["user_id", "query_id", "query", "relevant_fact_id", "case_type", "distractor_fact_id"])


def run(tmp_path):
    return rq2.evaluate(FACTS, QUERIES, CFG, tmp_path, embedding_function=FakeEmbedder())


def row(df, qid, cond):
    return df[(df["query_id"] == qid) & (df["condition"] == cond)].iloc[0]


def test_intensity_mapping_and_importance():
    for intensity, p_n in (("calm", 1.0), ("mild", 0.5), ("strong", 0.1)):
        p = rq2.intensity_probs(intensity)
        assert p["neutral"] == p_n and sum(p.values()) == pytest.approx(1.0)
        assert p["happy"] == p["angry"] == p["sad"]
    imp = {k: importance_from_turn(rq2.intensity_probs(k), False, CFG)[0] for k in ("calm", "mild", "strong")}
    assert imp == pytest.approx({"calm": 0.30, "mild": 0.65, "strong": 0.93})


def test_weighting_changes_the_ranking_and_similarity_only_does_not(tmp_path):
    df = run(tmp_path)
    w, s = row(df, "a_q1", "weighted"), row(df, "a_q1", "similarity_only")
    assert (w["rank"], w["top1_fact_id"], w["distractor_rank"], w["distractor_in_top3"]) == (2, "a_f1", 1, 1)
    assert (s["rank"], s["top1_fact_id"], s["distractor_rank"], s["distractor_in_top3"]) == (1, "a_f2", 2, 1)
    assert w["reciprocal_rank"] == 0.5 and s["reciprocal_rank"] == 1.0
    assert w["relevant_similarity"] == pytest.approx(2 / np.sqrt(8), abs=1e-6)
    assert w["relevant_importance"] == pytest.approx(0.30)
    assert w["n_returned"] == s["n_returned"] == 2                     # a_f3 cut in both


def test_reciprocal_rank_zero_when_cut_by_min_similarity(tmp_path):
    df = run(tmp_path)
    for cond in rq2.CONDITIONS:
        r = row(df, "a_q2", cond)
        assert pd.isna(r["rank"]) and r["reciprocal_rank"] == 0.0
        assert r["hit_at_1"] == 0 and r["hit_at_3"] == 0
        assert r["relevant_similarity"] == pytest.approx(0.0)          # read by the diagnostic call only
        assert pd.isna(r["distractor_in_top3"])                        # no distractor on this row


def test_ranking_is_read_from_retrieve_relevant(tmp_path, monkeypatch):
    """The spy reverses what retrieve_relevant returns on the condition calls; evaluate must follow it,
    and must call it with only the weights changed, min_similarity from the cfg and top_k 5."""
    real = ChromaMemoryManager.retrieve_relevant
    calls = []

    def spy(self, user_id, query, top_k=None):
        hits = real(self, user_id, query, top_k=top_k)
        calls.append((self.cfg["w_similarity"], self.cfg["w_importance"], self.cfg["min_similarity"], top_k))
        return hits if self.cfg["min_similarity"] < 0 else list(reversed(hits))

    monkeypatch.setattr(ChromaMemoryManager, "retrieve_relevant", spy)
    df = run(tmp_path)
    assert row(df, "a_q3", "similarity_only")["rank"] == 2             # a_f1 is truly first; reversed -> second
    assert row(df, "a_q1", "similarity_only")["rank"] == 2
    assert row(df, "a_q1", "weighted")["rank"] == 1
    condition_calls = [c for c in calls if c[2] >= 0]
    assert {c[:2] for c in condition_calls} == {(0.7, 0.3), (1.0, 0.0)}
    assert all(c[2] == 0.15 and c[3] == 5 for c in condition_calls)
    assert len(condition_calls) == 2 * len(QUERIES)
    diagnostic = [c for c in calls if c[2] < 0]
    assert len(diagnostic) == len(QUERIES) and all(c[:2] == (0.7, 0.3) for c in diagnostic)


def test_retrieve_with_restores_cfg(tmp_path):
    mm = ChromaMemoryManager(tmp_path / "vs", CFG, embedding_function=FakeEmbedder())
    try:
        before = dict(mm.cfg)
        rq2.retrieve_with(mm, "a", "piano", 5, w_similarity=1.0, w_importance=0.0)
        assert mm.cfg == before
    finally:
        mm.close()


def test_summary_counts(tmp_path):
    summ = rq2.summarize(run(tmp_path)).set_index(["condition", "case_type"])
    w_all, s_all = summ.loc[("weighted", "all")], summ.loc[("similarity_only", "all")]
    assert w_all["n"] == 3 and w_all["n_relevant_cut"] == 1 and w_all["n_with_distractor"] == 1
    assert w_all["mrr"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)
    assert s_all["mrr"] == pytest.approx((1.0 + 0.0 + 1.0) / 3)
    assert w_all["distractor_in_top3_rate"] == 1.0
    assert np.isnan(summ.loc[("weighted", "calm_target"), "distractor_in_top3_rate"])


def test_bootstrap_is_deterministic_with_seed_42():
    a = np.array([1.0, 0.5, 0.0, 1.0, 0.333, 0.25, 1.0, 0.5])
    b = np.array([1.0, 1.0, 0.0, 0.5, 0.333, 1.0, 1.0, 0.2])
    r1, r2 = rq2.paired_bootstrap(a, b), rq2.paired_bootstrap(a, b)
    assert r1 == r2
    d = a - b
    idx = np.random.default_rng(42).integers(0, len(d), size=(10_000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    assert (r1["ci_low"], r1["ci_high"]) == (float(lo), float(hi))
    assert r1["mean_diff"] == pytest.approx(d.mean()) and r1["n_resamples"] == 10_000 and r1["seed"] == 42
    assert rq2.paired_bootstrap(a, b, seed=7) != r1


@pytest.mark.parametrize("lo, hi, want", [
    (0.01, 0.20, "helps"),
    (-0.20, -0.01, "hurts"),
    (-0.10, 0.10, "no detectable difference"),
    (0.0, 0.10, "no detectable difference"),      # lower bound must be strictly above 0
    (-0.10, 0.0, "no detectable difference"),
])
def test_verdict(lo, hi, want):
    assert rq2.verdict(lo, hi) == want


def test_guardrail_flag():
    n = 30
    sim_only = np.zeros(n, dtype=int)
    sim_only[:10] = 1                                   # 10/30 hit@1 with similarity only
    weighted_3 = np.zeros(n, dtype=int)
    weighted_3[:7] = 1                                  # drop 3/30 = 0.10 exactly -> not flagged
    weighted_4 = np.zeros(n, dtype=int)
    weighted_4[:6] = 1                                  # drop 4/30 > 0.10 -> flagged
    assert rq2.guardrail(sim_only, weighted_3)["flagged"] is False
    g = rq2.guardrail(sim_only, weighted_4)
    assert g["flagged"] is True and g["drop"] == pytest.approx(4 / 30)
    assert rq2.guardrail(sim_only, sim_only)["flagged"] is False
