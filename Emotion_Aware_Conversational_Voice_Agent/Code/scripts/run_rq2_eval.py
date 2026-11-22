"""
run_rq2_eval.py
Task F step R2 (RQ2): weighted vs similarity-only memory retrieval on the approved scenario set data/rq2_eval/,
by the decision rule fixed beforehand in data/rq2_eval/README.md ("Decision rule (fixed before running)").
Brain only: no torch, nothing imported from gpu/. chromadb is used only inside ChromaMemoryManager (its version is
read from package metadata); matplotlib only to plot.

Per user: a fresh temporary Chroma store (tempfile, never vector_store/) with the real MiniLM embedder
    (ChromaMemoryManager's default, chromadb DefaultEmbeddingFunction); that user's 5 facts stored in fact_id order,
    turn_id 1..5, empty mood_snapshot, importance = memory_manager.importance_from_turn(probs, abstained=False, cfg)
    with probs from intensity (README): calm p_neutral 1.0, mild 0.5, strong 0.1, the rest split evenly over
    happy / angry / sad. Both conditions query the same store.
Per query: the real ChromaMemoryManager.retrieve_relevant, called once per condition with only the weights changed
    in the manager's cfg (weighted 0.7 / 0.3, similarity-only 1.0 / 0.0); min_similarity from gpu_config.yaml;
    top_k = 5, so the full ranking after the cutoff is read. Scoring is never reimplemented here.
    A third, diagnostic call (min_similarity -1.0, weights unchanged) only reads the relevant fact's similarity
    when the cutoff removed it; it is not used for any metric.
Metrics (overall and per case_type): MRR (reciprocal rank 0 when the relevant fact is cut), hit@1, hit@3,
    distractor-in-top-3 rate (rows with a distractor only).
Verdict: paired bootstrap of d_i = RR_weighted - RR_similarity_only over the 45 queries,
    rng = numpy.random.default_rng(42), idx = rng.integers(0, n, size=(10000, n)), 95% CI = numpy.percentile of the
    resample means at 2.5 / 97.5. Lower bound > 0 -> helps; upper bound < 0 -> hurts; else no detectable difference.
    Per-case-type CIs: same procedure within each type with a fresh default_rng(42), information only.
Guardrail (reported, verdict unchanged): on calm targets (calm_target + calm_target_charged_distractor),
    hit@1(similarity-only) - hit@1(weighted) > 0.10 -> flagged.

Outputs: results/metrics/rq2_per_query.csv   (one row per query x condition)
         results/metrics/rq2_summary.csv     (condition x case_type, incl. "all")
         results/metrics/rq2_summary.json    (all numbers, CI, verdict, guardrail, config, chromadb version, date)
         results/figures/rq2_mrr_by_case.png (300 dpi)

CLI:  python scripts/run_rq2_eval.py [--eval-dir DIR] [--metrics-dir DIR] [--figures-dir DIR]
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from memory_manager import ChromaMemoryManager, importance_from_turn, load_memory_config

CODE_DIR = Path(__file__).resolve().parent.parent          # Code/
EVAL_DIR = CODE_DIR / "data" / "rq2_eval"
METRICS_DIR = CODE_DIR / "results" / "metrics"
FIGURES_DIR = CODE_DIR / "results" / "figures"

CONDITIONS = {"weighted": (0.7, 0.3), "similarity_only": (1.0, 0.0)}      # (w_similarity, w_importance)
CASE_TYPES = ("charged_target", "calm_target_charged_distractor", "calm_target")
CALM_CASES = ("calm_target", "calm_target_charged_distractor")
P_NEUTRAL = {"calm": 1.0, "mild": 0.5, "strong": 0.1}
FULL_RANKING_TOP_K = 5        # every one of a user's facts that passes the cutoff
AGENT_TOP_K = 3               # base.memory.top_k: what hit@3 and distractor-in-top-3 measure
N_RESAMPLES = 10_000
SEED = 42
GUARDRAIL_MAX_DROP = 0.10
README_MIN_SIMILARITY = 0.15  # values the README rule was written for; the run stops if the config differs
README_BASE_IMPORTANCE = 0.3
README_TOP_K = 3
EMBEDDER = "chromadb DefaultEmbeddingFunction (all-MiniLM-L6-v2, ONNX), ChromaMemoryManager default"


# ---------------------------------------------------------------------------------------------
# Rule pieces (pure)
# ---------------------------------------------------------------------------------------------

def intensity_probs(intensity) -> dict:
    """README mapping: p_neutral by intensity, the rest split evenly over happy / angry / sad."""
    p_n = P_NEUTRAL[intensity]
    rest = (1.0 - p_n) / 3.0
    return {"neutral": p_n, "happy": rest, "angry": rest, "sad": rest}


def ranking_row(hits, id_to_fact, relevant_fact_id, distractor_fact_id=None) -> dict:
    """Read one ranking (retrieve_relevant output, best first) for one query.
    rank is 1-based, None if the relevant fact was cut; distractor_in_top3 is None when there is no distractor."""
    order = [id_to_fact[h["memory_id"]] for h in hits]
    rank = order.index(relevant_fact_id) + 1 if relevant_fact_id in order else None
    row = {
        "rank": rank,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "hit_at_1": int(rank == 1),
        "hit_at_3": int(rank is not None and rank <= AGENT_TOP_K),
        "n_returned": len(order),
        "top1_fact_id": order[0] if order else "",
        "distractor_rank": None,
        "distractor_in_top3": None,
    }
    if distractor_fact_id:
        d_rank = order.index(distractor_fact_id) + 1 if distractor_fact_id in order else None
        row["distractor_rank"] = d_rank
        row["distractor_in_top3"] = int(d_rank is not None and d_rank <= AGENT_TOP_K)
    return row


def paired_bootstrap(rr_a, rr_b, n_resamples=N_RESAMPLES, seed=SEED) -> dict:
    """Paired bootstrap of mean(rr_a - rr_b): {mean_diff, ci_low, ci_high, n, n_resamples, seed}."""
    d = np.asarray(rr_a, dtype=np.float64) - np.asarray(rr_b, dtype=np.float64)
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(n), "n_resamples": int(n_resamples), "seed": int(seed)}


def verdict(ci_low, ci_high) -> str:
    if ci_low > 0:
        return "helps"
    if ci_high < 0:
        return "hurts"
    return "no detectable difference"


def guardrail(hits_similarity_only, hits_weighted, max_drop=GUARDRAIL_MAX_DROP) -> dict:
    """Calm-target hit@1 drop = hit@1(similarity-only) - hit@1(weighted), compared on counts so that exactly
    max_drop is not flagged by float rounding."""
    s, w = np.asarray(hits_similarity_only, dtype=int), np.asarray(hits_weighted, dtype=int)
    n = len(s)
    drop_count = int(s.sum() - w.sum())
    flagged = bool(n and drop_count > max_drop * n + 1e-9)
    return {"n": n, "hit_at_1_similarity_only": s.sum() / n if n else float("nan"),
            "hit_at_1_weighted": w.sum() / n if n else float("nan"),
            "drop": drop_count / n if n else float("nan"), "max_drop": max_drop, "flagged": flagged}


# ---------------------------------------------------------------------------------------------
# Running retrieval
# ---------------------------------------------------------------------------------------------

def retrieve_with(mm, user_id, query, top_k, **overrides) -> list:
    """The real mm.retrieve_relevant with some cfg keys temporarily overridden (restored afterwards)."""
    saved = mm.cfg
    mm.cfg = {**saved, **overrides}
    try:
        return mm.retrieve_relevant(user_id, query, top_k=top_k)
    finally:
        mm.cfg = saved


def store_user_facts(mm, user_facts: pd.DataFrame, cfg_memory) -> tuple:
    """Store one user's facts in fact_id order. Returns ({memory_id: fact_id}, {fact_id: importance})."""
    id_to_fact, importance = {}, {}
    for turn_id, f in enumerate(user_facts.sort_values("fact_id").itertuples(index=False), start=1):
        imp, _ = importance_from_turn(intensity_probs(f.intensity), False, cfg_memory)
        memory_id = mm.store(f.user_id, f.fact, turn_id, {}, imp)
        if memory_id in id_to_fact:
            raise RuntimeError(f"{f.fact_id} was merged into {id_to_fact[memory_id]} as a duplicate; check facts.csv")
        id_to_fact[memory_id] = f.fact_id
        importance[f.fact_id] = imp
    return id_to_fact, importance


def evaluate(facts: pd.DataFrame, queries: pd.DataFrame, cfg_memory: dict, store_root, embedding_function=None):
    """One row per query x condition. A fresh ChromaMemoryManager per user under store_root (closed after use)."""
    intensity = dict(zip(facts["fact_id"], facts["intensity"]))
    rows = []
    for user_id in sorted(queries["user_id"].unique()):
        mm = ChromaMemoryManager(Path(store_root) / user_id, cfg_memory, embedding_function=embedding_function)
        try:
            id_to_fact, importance = store_user_facts(mm, facts[facts["user_id"] == user_id], cfg_memory)
            for q in queries[queries["user_id"] == user_id].sort_values("query_id").itertuples(index=False):
                every = retrieve_with(mm, user_id, q.query, FULL_RANKING_TOP_K, min_similarity=-1.0)
                sim_of = {id_to_fact[h["memory_id"]]: h["similarity"] for h in every}
                for cond, (w_sim, w_imp) in CONDITIONS.items():
                    hits = retrieve_with(mm, user_id, q.query, FULL_RANKING_TOP_K,
                                         w_similarity=w_sim, w_importance=w_imp)
                    r = ranking_row(hits, id_to_fact, q.relevant_fact_id, q.distractor_fact_id or None)
                    rows.append({
                        "user_id": user_id, "query_id": q.query_id, "case_type": q.case_type, "condition": cond,
                        "query": q.query, "relevant_fact_id": q.relevant_fact_id,
                        "relevant_intensity": intensity[q.relevant_fact_id],
                        "relevant_similarity": sim_of[q.relevant_fact_id],
                        "relevant_importance": importance[q.relevant_fact_id],
                        "distractor_fact_id": q.distractor_fact_id, **r,
                        "top1_intensity": intensity.get(r["top1_fact_id"], ""),
                    })
        finally:
            mm.close()
    out = pd.DataFrame(rows)
    for col in ("rank", "distractor_rank", "distractor_in_top3"):       # ints with gaps, not floats
        out[col] = pd.array([None if v is None or pd.isna(v) else int(v) for v in out[col]], dtype="Int64")
    return out


# ---------------------------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------------------------

def summarize(per_query: pd.DataFrame) -> pd.DataFrame:
    """condition x case_type ("all" first): n, MRR, hit@1, hit@3, n_cut, distractor rows / rate."""
    out = []
    for cond in CONDITIONS:
        for ct in ("all",) + CASE_TYPES:
            d = per_query[per_query["condition"] == cond]
            if ct != "all":
                d = d[d["case_type"] == ct]
            dis = d["distractor_in_top3"].dropna().astype(float)
            out.append({
                "condition": cond, "case_type": ct, "n": len(d),
                "mrr": d["reciprocal_rank"].mean(), "hit_at_1": d["hit_at_1"].mean(), "hit_at_3": d["hit_at_3"].mean(),
                "n_relevant_cut": int(d["rank"].isna().sum()),
                "n_with_distractor": len(dis),
                "distractor_in_top3_rate": dis.mean() if len(dis) else float("nan"),
            })
    return pd.DataFrame(out)


def paired_arrays(per_query: pd.DataFrame, column, case_types=None) -> tuple:
    """(weighted values, similarity-only values) aligned by query_id."""
    d = per_query if case_types is None else per_query[per_query["case_type"].isin(case_types)]
    wide = d.pivot(index="query_id", columns="condition", values=column).sort_index()
    return wide["weighted"].to_numpy(), wide["similarity_only"].to_numpy()


def changed_top1(per_query: pd.DataFrame) -> pd.DataFrame:
    wide = per_query.pivot(index="query_id", columns="condition",
                           values=["top1_fact_id", "top1_intensity"]).sort_index()
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    base = per_query[per_query["condition"] == "weighted"].set_index("query_id")[
        ["user_id", "case_type", "query", "relevant_fact_id"]]
    wide = base.join(wide)
    return wide[wide["top1_fact_id_weighted"] != wide["top1_fact_id_similarity_only"]].reset_index()


def plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ("all",) + CASE_TYPES
    names = {"all": "All", "charged_target": "Charged target",
             "calm_target_charged_distractor": "Calm target,\ncharged distractor", "calm_target": "Calm target"}
    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (cond, color, label) in enumerate((("weighted", "#1f77b4", "Weighted (0.7 sim + 0.3 importance)"),
                                              ("similarity_only", "#ff7f0e", "Similarity only"))):
        s = summary[summary["condition"] == cond].set_index("case_type").loc[list(groups)]
        bars = ax.bar(x + (i - 0.5) * width, s["mrr"], width, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in s["mrr"]], fontsize=9, padding=2)
    n = summary[summary["condition"] == "weighted"].set_index("case_type")["n"]
    ax.set_xticks(x, [f"{names[g]}\n(n={int(n[g])})" for g in groups], fontsize=10)
    ax.set_ylabel("MRR of the relevant fact", fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("RQ2: weighted vs similarity-only retrieval (data/rq2_eval)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _chromadb_version() -> str:
    try:
        return version("chromadb")
    except PackageNotFoundError:
        return "not installed"


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RQ2: weighted vs similarity-only retrieval on data/rq2_eval/ (Task F R2).")
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    ap.add_argument("--metrics-dir", default=str(METRICS_DIR))
    ap.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = ap.parse_args(argv)
    metrics_dir, figures_dir = Path(args.metrics_dir), Path(args.figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cfg_memory = load_memory_config()["memory"]
    for key, want in (("min_similarity", README_MIN_SIMILARITY), ("base_importance", README_BASE_IMPORTANCE),
                      ("top_k", README_TOP_K)):
        if float(cfg_memory[key]) != want:
            raise SystemExit(f"gpu_config.yaml memory.{key} = {cfg_memory[key]}, but the RQ2 rule was fixed for {want}; "
                             "stopping without running.")

    facts = pd.read_csv(Path(args.eval_dir) / "facts.csv", dtype=str, keep_default_na=False)
    queries = pd.read_csv(Path(args.eval_dir) / "queries.csv", dtype=str, keep_default_na=False)
    with tempfile.TemporaryDirectory(prefix="rq2_store_", ignore_cleanup_errors=True) as tmp:
        per_query = evaluate(facts, queries, cfg_memory, tmp)
    per_query.to_csv(metrics_dir / "rq2_per_query.csv", index=False)
    summary = summarize(per_query)
    summary.to_csv(metrics_dir / "rq2_summary.csv", index=False)

    boot = paired_bootstrap(*paired_arrays(per_query, "reciprocal_rank"))
    boot["verdict"] = verdict(boot["ci_low"], boot["ci_high"])
    boot_by_case = {}
    for ct in CASE_TYPES:
        b = paired_bootstrap(*paired_arrays(per_query, "reciprocal_rank", [ct]))
        b["verdict_information_only"] = verdict(b["ci_low"], b["ci_high"])
        boot_by_case[ct] = b
    w1, s1 = paired_arrays(per_query, "hit_at_1", CALM_CASES)
    guard = guardrail(s1, w1)
    flips = changed_top1(per_query)

    result = {
        "rule": "data/rq2_eval/README.md, 'Decision rule (fixed before running)'",
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chromadb_version": _chromadb_version(), "embedder": EMBEDDER,
        "config_used": {
            "conditions": {c: {"w_similarity": w[0], "w_importance": w[1]} for c, w in CONDITIONS.items()},
            "min_similarity": float(cfg_memory["min_similarity"]), "base_importance": float(cfg_memory["base_importance"]),
            "agent_top_k": AGENT_TOP_K, "ranking_top_k": FULL_RANKING_TOP_K,
            "intensity_probs": {k: intensity_probs(k) for k in P_NEUTRAL},
            "importance_by_intensity": {k: importance_from_turn(intensity_probs(k), False, cfg_memory)[0] for k in P_NEUTRAL},
            "n_resamples": N_RESAMPLES, "seed": SEED, "guardrail_max_drop": GUARDRAIL_MAX_DROP,
        },
        "n_users": int(facts["user_id"].nunique()), "n_facts": int(len(facts)), "n_queries": int(len(queries)),
        "summary": summary.to_dict(orient="records"),
        "mrr_difference_weighted_minus_similarity_only": boot,
        "mrr_difference_by_case_type": boot_by_case,
        "guardrail_calm_target_hit_at_1": guard,
        "queries_with_different_top1": flips.to_dict(orient="records"),
        "torch_imported": "torch" in sys.modules,
    }
    with open(metrics_dir / "rq2_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)
    plot(summary, figures_dir / "rq2_mrr_by_case.png")

    print(f"RQ2 on {result['n_users']} users, {result['n_facts']} facts, {result['n_queries']} queries "
          f"(min_similarity {result['config_used']['min_similarity']}, chromadb {result['chromadb_version']})")
    print(f"{'condition':16s} {'case_type':31s} {'n':>3s} {'MRR':>7s} {'hit@1':>7s} {'hit@3':>7s} {'cut':>4s} {'distr@3':>8s}")
    for r in summary.itertuples(index=False):
        dis = f"{r.distractor_in_top3_rate:.3f}" if r.n_with_distractor else "-"
        print(f"{r.condition:16s} {r.case_type:31s} {r.n:3d} {r.mrr:7.4f} {r.hit_at_1:7.4f} {r.hit_at_3:7.4f} "
              f"{r.n_relevant_cut:4d} {dis:>8s}")
    print(f"MRR(weighted) - MRR(similarity-only) = {boot['mean_diff']:+.4f}, 95% CI [{boot['ci_low']:+.4f}, "
          f"{boot['ci_high']:+.4f}] -> {boot['verdict'].upper()}")
    for ct, b in boot_by_case.items():
        print(f"  {ct:31s} {b['mean_diff']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] (information only)")
    print(f"Guardrail (calm targets, n={guard['n']}): hit@1 similarity-only {guard['hit_at_1_similarity_only']:.4f}, "
          f"weighted {guard['hit_at_1_weighted']:.4f}, drop {guard['drop']:+.4f} -> "
          f"{'FLAGGED (weighting costs calm facts)' if guard['flagged'] else 'not flagged'}")
    print(f"Queries where the two conditions rank a different fact first: {len(flips)}")
    text = dict(zip(facts["fact_id"], facts["fact"]))
    for r in flips.itertuples(index=False):
        print(f"  {r.query_id} [{r.case_type}] {r.query!r}  (relevant {r.relevant_fact_id})")
        print(f"      weighted        -> {r.top1_fact_id_weighted} ({r.top1_intensity_weighted}): "
              f"{text.get(r.top1_fact_id_weighted, '')}")
        print(f"      similarity-only -> {r.top1_fact_id_similarity_only} ({r.top1_intensity_similarity_only}): "
              f"{text.get(r.top1_fact_id_similarity_only, '')}")
    print(f"torch imported: {result['torch_imported']}")
    print(f"Wrote {metrics_dir / 'rq2_per_query.csv'}, rq2_summary.csv, rq2_summary.json, "
          f"{figures_dir / 'rq2_mrr_by_case.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
