"""
choose_min_similarity.py
Task D step D3b: choose memory.min_similarity for the Chroma memory backend from the approved labelled pairs in
data/memory_eval/ (facts.csv, queries.csv), by the rule fixed beforehand in data/memory_eval/README.md.
Brain only: no torch, nothing imported from gpu/. chromadb is imported only to embed; matplotlib only to plot.

Pairs: every query x every fact; positive if the fact is in the query's relevant_fact_ids, otherwise negative.
Similarity: exact cosine (memory_manager.cosine_similarities) between MiniLM embeddings from chromadb's
    DefaultEmbeddingFunction (all-MiniLM-L6-v2 on ONNX, as ChromaMemoryManager); query as-is, fact.strip() as stored.
Cutoffs 0.00..1.00 step 0.01; recalled = similarity >= cutoff.
    FRR = recalled negatives / negatives; recall = recalled positives / positives.
Rule: min_similarity = the LOWEST cutoff with FRR <= 0.10 on the tune split (1e-12 tolerance, so exactly 0.10
    qualifies). Test is reported, never used to choose. No qualifying cutoff -> reported, nothing chosen.
Reported per split at the chosen cutoff: recall, FRR, recall@3 (positives >= cutoff AND among the query's 3
    highest-similarity facts, ties by fact_id), paraphrase recall, direct recall, word_trap queries with >= 1 false
    recall, none queries with >= 1 recall.

This script does not change gpu/gpu_config.yaml; the chosen value is copied there by hand.

Outputs: results/metrics/memory_pairs.csv          (query_id, fact_id, case_type, split, label, similarity)
         results/metrics/memory_threshold.csv      (per cutoff: tune/test recall, FRR and counts)
         results/metrics/memory_threshold_summary.json
         results/figures/memory_threshold.png      (300 dpi)

CLI:  python scripts/choose_min_similarity.py [--eval-dir DIR] [--metrics-dir DIR] [--figures-dir DIR]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from memory_manager import cosine_similarities

CODE_DIR = Path(__file__).resolve().parent.parent          # Code/
EVAL_DIR = CODE_DIR / "data" / "memory_eval"
METRICS_DIR = CODE_DIR / "results" / "metrics"
FIGURES_DIR = CODE_DIR / "results" / "figures"

MAX_FRR = 0.10
STEP = 0.01
TOP_K = 3                     # base.memory.top_k: facts the agent recalls per turn
FRR_TOLERANCE = 1e-12
SPLITS = ("tune", "test")
CASE_TYPES = ("paraphrase", "direct", "word_trap", "none")
EMBEDDER = "chromadb DefaultEmbeddingFunction (all-MiniLM-L6-v2, ONNX)"


# ---------------------------------------------------------------------------------------------
# Rule (pure functions on arrays)
# ---------------------------------------------------------------------------------------------

def candidate_cutoffs(step=STEP) -> np.ndarray:
    """0.00, step, ..., 1.00 (rounded, so 0.1 is the float 0.1)."""
    n = int(round(1.0 / step))
    return np.round(np.arange(n + 1) * step, 6)


def recall_frr(similarities, labels, cutoff) -> tuple:
    """(recall, frr, recalled_pos, n_pos, recalled_neg, n_neg) at one cutoff; recalled = similarity >= cutoff.
    recall / frr are nan when there are no positives / negatives."""
    s = np.asarray(similarities, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    hit = s >= cutoff
    rp, n_pos = int((hit & y).sum()), int(y.sum())
    rn, n_neg = int((hit & ~y).sum()), int((~y).sum())
    recall = rp / n_pos if n_pos else float("nan")
    frr = rn / n_neg if n_neg else float("nan")
    return recall, frr, rp, n_pos, rn, n_neg


def threshold_table(similarities, labels, step=STEP) -> pd.DataFrame:
    """One row per candidate cutoff: cutoff, recall, frr, recalled_pos, n_pos, recalled_neg, n_neg."""
    rows = [(c, *recall_frr(similarities, labels, c)) for c in candidate_cutoffs(step)]
    return pd.DataFrame(rows, columns=["cutoff", "recall", "frr", "recalled_pos", "n_pos", "recalled_neg", "n_neg"])


def choose_cutoff(similarities, labels, max_frr=MAX_FRR, step=STEP):
    """The lowest candidate cutoff whose FRR <= max_frr (1e-12 tolerance), or None if there is none."""
    for c in candidate_cutoffs(step):
        frr = recall_frr(similarities, labels, c)[1]
        if not np.isnan(frr) and frr <= max_frr + FRR_TOLERANCE:
            return float(c)
    return None


def top_k_mask(query_ids, fact_ids, similarities, k=TOP_K) -> np.ndarray:
    """True for pairs among their query's k highest similarities (ties broken by fact_id, ascending)."""
    df = pd.DataFrame({"q": list(query_ids), "f": list(fact_ids), "s": np.asarray(similarities, dtype=np.float64)})
    df["i"] = np.arange(len(df))
    df = df.sort_values(["q", "s", "f"], ascending=[True, False, True], kind="mergesort")
    df["rank"] = df.groupby("q").cumcount()
    mask = np.zeros(len(df), dtype=bool)
    mask[df.loc[df["rank"] < k, "i"].to_numpy()] = True
    return mask


def recall_at_k(query_ids, fact_ids, similarities, labels, cutoff, k=TOP_K) -> float:
    """Share of positives that are >= cutoff AND among their query's k highest-similarity facts (nan if no positives)."""
    s = np.asarray(similarities, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    if not y.any():
        return float("nan")
    ok = (s >= cutoff) & top_k_mask(query_ids, fact_ids, s, k) & y
    return int(ok.sum()) / int(y.sum())


def split_report(pairs: pd.DataFrame, cutoff) -> dict:
    """Every reported number for one split's pairs (columns query_id, fact_id, case_type, label, similarity)."""
    s, y = pairs["similarity"].to_numpy(), pairs["label"].to_numpy()
    recall, frr, rp, n_pos, rn, n_neg = recall_frr(s, y, cutoff)
    hit = s >= cutoff

    def case_recall(ct):
        m = (pairs["case_type"] == ct).to_numpy() & y.astype(bool)
        return (int((hit & m).sum()) / int(m.sum())) if m.any() else float("nan"), int((hit & m).sum()), int(m.sum())

    def queries_with(ct, mask):
        sub = pairs.loc[(pairs["case_type"] == ct).to_numpy()]
        flagged = pairs.loc[(pairs["case_type"] == ct).to_numpy() & mask, "query_id"].unique()
        return int(len(flagged)), int(sub["query_id"].nunique())

    para, para_hit, para_n = case_recall("paraphrase")
    direct, direct_hit, direct_n = case_recall("direct")
    trap_k, trap_n = queries_with("word_trap", hit & ~y.astype(bool))
    none_k, none_n = queries_with("none", hit)
    return {
        "n_queries": int(pairs["query_id"].nunique()), "n_pairs": int(len(pairs)),
        "n_positives": n_pos, "n_negatives": n_neg,
        "recall": recall, "recalled_positives": rp, "frr": frr, "recalled_negatives": rn,
        "recall_at_3": recall_at_k(pairs["query_id"], pairs["fact_id"], s, y, cutoff, TOP_K),
        "paraphrase_recall": para, "paraphrase_recalled": para_hit, "paraphrase_positives": para_n,
        "direct_recall": direct, "direct_recalled": direct_hit, "direct_positives": direct_n,
        "word_trap_queries_with_false_recall": trap_k, "word_trap_queries": trap_n,
        "none_queries_with_recall": none_k, "none_queries": none_n,
    }


# ---------------------------------------------------------------------------------------------
# Data, embedding, outputs
# ---------------------------------------------------------------------------------------------

def load_eval(eval_dir=EVAL_DIR) -> tuple:
    facts = pd.read_csv(Path(eval_dir) / "facts.csv", dtype=str, keep_default_na=False)
    queries = pd.read_csv(Path(eval_dir) / "queries.csv", dtype=str, keep_default_na=False)
    return facts, queries


def build_pairs(facts: pd.DataFrame, queries: pd.DataFrame, embed_fn) -> pd.DataFrame:
    """All query x fact pairs with label and exact cosine similarity. embed_fn(list[str]) -> list of vectors."""
    fact_vecs = np.asarray(embed_fn([f.strip() for f in facts["fact"]]), dtype=np.float64)
    query_vecs = np.asarray(embed_fn(list(queries["query"])), dtype=np.float64)
    rows = []
    for q, qv in zip(queries.itertuples(index=False), query_vecs):
        relevant = {x for x in q.relevant_fact_ids.split(";") if x}
        sims = cosine_similarities(qv, fact_vecs)
        for fid, sim in zip(facts["fact_id"], sims):
            rows.append((q.query_id, fid, q.case_type, q.split, int(fid in relevant), float(sim)))
    return pd.DataFrame(rows, columns=["query_id", "fact_id", "case_type", "split", "label", "similarity"])


def default_embed_fn():
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


def plot(pairs: pd.DataFrame, tune_table: pd.DataFrame, chosen, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(tune_table["cutoff"], tune_table["recall"], label="Recall (tune)", color="#1f77b4", lw=2)
    ax1.plot(tune_table["cutoff"], tune_table["frr"], label="False-recall rate (tune)", color="#d62728", lw=2)
    ax1.axhline(MAX_FRR, color="#d62728", ls=":", lw=1.2, label=f"FRR limit {MAX_FRR:.2f}")
    if chosen is not None:
        ax1.axvline(chosen, color="black", ls="--", lw=1.2, label=f"chosen min_similarity = {chosen:.2f}")
    ax1.set_xlabel("Cutoff (cosine similarity)", fontsize=12)
    ax1.set_ylabel("Rate", fontsize=12)
    ax1.set_title("Tune split: recall and false recalls vs cutoff", fontsize=13)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10)

    bins = np.linspace(min(-0.2, pairs["similarity"].min()), 1.0, 41)
    styles = {("tune", 1): ("#1f77b4", "-"), ("test", 1): ("#1f77b4", "--"),
              ("tune", 0): ("#d62728", "-"), ("test", 0): ("#d62728", "--")}
    for (split, label), (color, ls) in styles.items():
        s = pairs.loc[(pairs["split"] == split) & (pairs["label"] == label), "similarity"]
        name = "positive" if label else "negative"
        ax2.hist(s, bins=bins, histtype="step", color=color, ls=ls, lw=1.8, density=True,
                 label=f"{name} pairs, {split} (n={len(s)})")
    if chosen is not None:
        ax2.axvline(chosen, color="black", ls="--", lw=1.2, label=f"chosen {chosen:.2f}")
    ax2.set_xlabel("Cosine similarity (query, fact)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Similarity of positive vs negative pairs", fontsize=13)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Choose memory.min_similarity from data/memory_eval/ (Task D step D3b).")
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    ap.add_argument("--metrics-dir", default=str(METRICS_DIR))
    ap.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = ap.parse_args(argv)
    metrics_dir, figures_dir = Path(args.metrics_dir), Path(args.figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    facts, queries = load_eval(args.eval_dir)
    pairs = build_pairs(facts, queries, default_embed_fn())
    pairs.to_csv(metrics_dir / "memory_pairs.csv", index=False)

    by_split = {sp: pairs[pairs["split"] == sp] for sp in SPLITS}
    tables = {sp: threshold_table(d["similarity"], d["label"]) for sp, d in by_split.items()}
    table = tables["tune"][["cutoff"]].copy()
    for sp in SPLITS:
        for col in ("recall", "frr", "recalled_pos", "n_pos", "recalled_neg", "n_neg"):
            table[f"{sp}_{col}"] = tables[sp][col].to_numpy()
    table.to_csv(metrics_dir / "memory_threshold.csv", index=False)

    chosen = choose_cutoff(by_split["tune"]["similarity"], by_split["tune"]["label"])
    summary = {
        "rule": f"lowest cutoff in 0.00..1.00 step {STEP} with tune FRR <= {MAX_FRR} (see data/memory_eval/README.md)",
        "embedder": EMBEDDER, "top_k": TOP_K,
        "n_queries": int(queries.shape[0]), "n_facts": int(facts.shape[0]), "n_pairs": int(len(pairs)),
        "chosen_min_similarity": chosen,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits": {sp: (split_report(d, chosen) if chosen is not None else None) for sp, d in by_split.items()},
    }
    with open(metrics_dir / "memory_threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot(pairs, tables["tune"], chosen, figures_dir / "memory_threshold.png")

    print(f"Pairs: {len(pairs)} ({queries.shape[0]} queries x {facts.shape[0]} facts)")
    if chosen is None:
        print(f"No cutoff has tune FRR <= {MAX_FRR}; min_similarity is NOT chosen (leave the config unchanged).")
    else:
        print(f"Chosen min_similarity: {chosen:.2f}")
        for sp in SPLITS:
            r = summary["splits"][sp]
            print(f"  {sp:4s}: positives {r['n_positives']}, negatives {r['n_negatives']} | "
                  f"recall {r['recall']:.4f} ({r['recalled_positives']}/{r['n_positives']}), "
                  f"FRR {r['frr']:.4f} ({r['recalled_negatives']}/{r['n_negatives']}), recall@3 {r['recall_at_3']:.4f}")
            print(f"        paraphrase recall {r['paraphrase_recall']:.4f} ({r['paraphrase_recalled']}/{r['paraphrase_positives']}), "
                  f"direct recall {r['direct_recall']:.4f} ({r['direct_recalled']}/{r['direct_positives']}), "
                  f"word_trap queries with a false recall {r['word_trap_queries_with_false_recall']}/{r['word_trap_queries']}, "
                  f"none queries with a recall {r['none_queries_with_recall']}/{r['none_queries']}")
    text = pairs.merge(queries[["query_id", "query"]], on="query_id").merge(facts[["fact_id", "fact"]], on="fact_id")
    for title, part in (("5 positive pairs with the LOWEST similarity",
                         text[text["label"] == 1].nsmallest(5, "similarity")),
                        ("5 negative pairs with the HIGHEST similarity",
                         text[text["label"] == 0].nlargest(5, "similarity"))):
        print(title + ":")
        for r in part.itertuples(index=False):
            print(f"  {r.similarity:.4f}  [{r.split}]  {r.query_id} {r.query!r}  <->  {r.fact_id} {r.fact!r}")
    print(f"Wrote {metrics_dir / 'memory_pairs.csv'}, memory_threshold.csv, memory_threshold_summary.json, "
          f"{figures_dir / 'memory_threshold.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
