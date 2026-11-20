"""Task D step D3b: the threshold rule in scripts/choose_min_similarity.py, on small hand-made arrays (no embedder)."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from choose_min_similarity import (
    candidate_cutoffs,
    choose_cutoff,
    recall_at_k,
    recall_frr,
    split_report,
    threshold_table,
    top_k_mask,
)

CODE_DIR = Path(__file__).resolve().parent.parent

# 2 positives, 10 negatives. Negatives recalled: 10 up to cutoff 0.10, 2 for 0.11-0.20, 1 for 0.21-0.50, 0 above.
SIMS = np.array([0.90, 0.40] + [0.50, 0.20] + [0.10] * 8)
LABELS = np.array([1, 1] + [0] * 10)


def test_candidate_cutoffs():
    c = candidate_cutoffs()
    assert len(c) == 101 and c[0] == 0.0 and c[-1] == 1.0
    assert 0.1 in c and 0.21 in c and np.all(np.diff(c) > 0)


def test_lowest_qualifying_cutoff_is_picked():
    assert choose_cutoff(SIMS, LABELS) == 0.21                       # 0.20 still has FRR 2/10
    assert choose_cutoff(SIMS, LABELS, max_frr=0.05) == 0.51         # first cutoff with no negative recalled
    assert choose_cutoff(SIMS, LABELS, max_frr=1.0) == 0.0


def test_frr_exactly_at_the_limit_qualifies():
    assert recall_frr(SIMS, LABELS, 0.21)[1] == 0.1                  # 1/10
    assert choose_cutoff(SIMS, LABELS, max_frr=0.10) == 0.21
    assert choose_cutoff(SIMS, LABELS, max_frr=0.0999) == 0.51       # just below the limit it no longer qualifies


def test_positives_do_not_affect_the_choice():
    other_pos = SIMS.copy()
    other_pos[:2] = [0.01, 0.99]
    assert choose_cutoff(other_pos, LABELS) == choose_cutoff(SIMS, LABELS)


def test_no_qualifying_cutoff_returns_none():
    sims = np.array([0.9, 1.0, 1.0, 1.0])                            # negatives at 1.0 are recalled at every cutoff
    assert choose_cutoff(sims, np.array([1, 0, 0, 0])) is None
    assert choose_cutoff(np.array([0.5, 0.6]), np.array([1, 1])) is None   # no negatives: FRR undefined


def test_recall_and_frr_arithmetic():
    assert recall_frr(SIMS, LABELS, 0.30) == (1.0, 0.1, 2, 2, 1, 10)
    assert recall_frr(SIMS, LABELS, 0.45) == (0.5, 0.1, 1, 2, 1, 10)
    assert recall_frr(SIMS, LABELS, 0.15) == (1.0, 0.2, 2, 2, 2, 10)
    assert recall_frr(SIMS, LABELS, 0.95) == (0.0, 0.0, 0, 2, 0, 10)
    assert recall_frr(SIMS, LABELS, 0.40)[:2] == (1.0, 0.1)          # similarity == cutoff counts as recalled


def test_threshold_table_matches_recall_frr():
    t = threshold_table(SIMS, LABELS)
    assert len(t) == 101
    row = t.loc[np.isclose(t["cutoff"], 0.21)].iloc[0]
    assert (row["recall"], row["frr"], row["recalled_neg"], row["n_neg"]) == (1.0, 0.1, 1, 10)
    assert np.all(np.diff(t["frr"]) <= 0) and np.all(np.diff(t["recall"]) <= 0)


def test_recall_at_3_needs_both_cutoff_and_top_3():
    q = ["qa"] * 5 + ["qb"] * 4
    f = ["f1", "f2", "f3", "f4", "f5"] + ["f1", "f2", "f3", "f4"]
    s = [0.90, 0.80, 0.70, 0.60, 0.65] + [0.50, 0.50, 0.50, 0.50]
    y = [1, 0, 0, 1, 0] + [0, 0, 0, 1]
    # qa ranks: f1 .90, f2 .80, f3 .70, f5 .65, f4 .60 -> f4 is 5th; qb all tied -> top 3 by fact_id = f1, f2, f3
    assert top_k_mask(q, f, s, 3).tolist() == [True, True, True, False, False] + [True, True, True, False]
    assert recall_at_k(q, f, s, y, cutoff=0.40) == pytest.approx(1 / 3)   # only qa:f1 (f4 not top-3, qb:f4 tie-out)
    assert recall_frr(s, y, 0.40)[0] == 1.0                                 # plain recall counts all three
    assert recall_at_k(q, f, s, y, cutoff=0.95) == 0.0                      # qa:f1 is top-3 but below the cutoff
    assert np.isnan(recall_at_k(q, f, s, [0] * 9, cutoff=0.4))


def test_split_report_case_type_counts():
    pairs = pd.DataFrame({
        "query_id":  ["p1", "p1", "d1", "d1", "w1", "w1", "n1", "n1"],
        "fact_id":   ["f1", "f2"] * 4,
        "case_type": ["paraphrase"] * 2 + ["direct"] * 2 + ["word_trap"] * 2 + ["none"] * 2,
        "label":     [1, 0, 1, 0, 0, 0, 0, 0],
        "similarity": [0.60, 0.10, 0.80, 0.20, 0.50, 0.10, 0.05, 0.30],
    })
    r = split_report(pairs, 0.40)
    assert (r["n_queries"], r["n_pairs"], r["n_positives"], r["n_negatives"]) == (4, 8, 2, 6)
    assert (r["recall"], r["recalled_negatives"]) == (1.0, 1) and r["frr"] == pytest.approx(1 / 6)
    assert (r["paraphrase_recall"], r["direct_recall"]) == (1.0, 1.0)
    assert (r["word_trap_queries_with_false_recall"], r["word_trap_queries"]) == (1, 1)
    assert (r["none_queries_with_recall"], r["none_queries"]) == (0, 1)
    assert r["recall_at_3"] == 1.0
    r2 = split_report(pairs, 0.25)
    assert r2["none_queries_with_recall"] == 1 and r2["frr"] == pytest.approx(2 / 6)
    r3 = split_report(pairs, 0.70)
    assert (r3["paraphrase_recall"], r3["direct_recall"], r3["word_trap_queries_with_false_recall"]) == (0.0, 1.0, 0)


def test_script_imports_no_torch_or_gpu_module():
    tree = ast.parse((CODE_DIR / "scripts" / "choose_min_similarity.py").read_text(encoding="utf-8"))
    gpu_modules = {p.stem for p in (CODE_DIR / "gpu").glob("*.py")} | {"gpu"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "torch" not in imported and not imported & gpu_modules, imported
