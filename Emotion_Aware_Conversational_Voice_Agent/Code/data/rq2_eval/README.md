# RQ2 memory scenario set (Task F step R1)

**Status: labels and decision rule approved 2026-11-22, before any similarity or retrieval score was computed.**

A small scripted set for RQ2: does ranking remembered facts by
`score = w_similarity * similarity + w_importance * importance` (paper Eq. `eq:retrieval`, alpha = 0.7) recall
the right fact better or worse than similarity alone? Labels and the decision rule below were written before any
similarity or retrieval score was computed on this set. This folder is separate from `data/memory_eval/`
(the D3 set used to choose `min_similarity`); no sentence is shared with it.

## Files

| File | Columns | What it is |
|---|---|---|
| `facts.csv` | `user_id, fact_id, fact, intensity, topic` | 15 imaginary users (`u01`-`u15`) x 5 stored facts = 75 facts, written the way `scripts/fact_extractor.py` writes them: one sentence starting "The user ...", third person, at most 25 words, no names or digits. |
| `queries.csv` | `user_id, query_id, query, relevant_fact_id, case_type, distractor_fact_id` | 3 things each user says in a later session = 45 queries, written as short, casual, first-person spoken turns. |

`intensity` is `calm`, `mild` or `strong`: how emotionally the user said the fact when it was stored.
`topic` is one of `study, work, family, pet, hobbies, plans, money, friends` (`pet` includes a pet's health;
`money` is money worries stated generally, no amounts).

## Layout per user

Every user has the same shape: 2 strong, 2 calm, 1 mild fact.

| fact | intensity | role |
|---|---|---|
| `_f1` | strong | target of the `charged_target` query |
| `_f2` | calm | target of the `calm_target_charged_distractor` query |
| `_f3` | strong | the distractor for that query: **same topic as `_f2`** |
| `_f4` | calm | target of the `calm_target` query: **no strong fact of that user has this topic** |
| `_f5` | mild | filler on another topic |

Topics differ across users, so the charged targets cover all eight topics.

## Case types (15 queries each, one per user)

| case_type | Meaning |
|---|---|
| `charged_target` | The query is about the user's strong fact `_f1`. |
| `calm_target_charged_distractor` | The query is about the calm fact `_f2`; the same user has a strong fact `_f3` on the same topic (`distractor_fact_id`). |
| `calm_target` | The query is about the calm fact `_f4`; no strong fact of that user shares its topic. `distractor_fact_id` is empty. |

## How the labels were decided

- **Relevance is decided by topic only, never by intensity.** `relevant_fact_id` is the one fact the query is
  specifically about. A distractor shares the broad `topic` but the query is not about it (e.g. u04: practising
  scales on the keyboard is about the calm piano-practice fact, not the strong piano-recital fact).
- "Nearby topic" is made concrete by the `topic` column: a distractor has the same `topic` as its target.
- Queries are mostly paraphrases: written to share no content word with their relevant fact
  (`memory_manager.content_words`, i.e. after lowercasing, removing punctuation and stopwords; no stemming,
  so "game" and "games" count as different words). The test allows at most one.
- No two facts of one user are near-duplicates (normalized text equal or word overlap >= 0.8), so
  `store()` keeps all five as separate records.

## Limitations

- **Intensity is scripted**, not produced by the emotion model. In the live agent, importance comes from the
  SER probabilities of the turn the fact was said in (`importance_from_turn`), and the T3 model misreads emotion
  on new voices (my_voice_v3 12/32). This set tests the ranking formula assuming the emotion reading is right;
  it does not test whether the agent measures intensity correctly.
- Every user has the same 2 strong / 2 calm / 1 mild shape, and the three case types are equally weighted
  (15 each). The overall MRR depends on that mix: by design, weighting should favour `charged_target` and may
  hurt `calm_target_charged_distractor`. Per-case-type results are reported for this reason.
- Written by one author (Claude Code) and reviewed by one person; text only (no ASR errors); strong facts also
  use emotional words, so the embedding can partly see intensity in the text itself.

## Checks

`tests/test_rq2_eval_set.py` (no embeddings): 15 users x 5 facts, 45 queries, one of each case type per user,
every relevant / distractor id exists and belongs to the same user, distractors are strong and share the target's
topic, `calm_*` targets are calm and `charged_target` targets are strong, `calm_target` targets share no topic with
a strong fact of that user, at most one shared content word between a query and its relevant fact, no near-duplicate
facts within a user, and no sentence overlaps `data/memory_eval/`.

## Decision rule (fixed before running)

Written before any similarity or retrieval score was computed on this set. To be implemented in a later step.

**Conditions** (everything identical except the two weights):

| | `w_similarity` | `w_importance` |
|---|---|---|
| weighted | 0.7 | 0.3 |
| similarity-only | 1.0 | 0.0 |

- Both: `min_similarity` 0.15, `top_k` 3, `base_importance` 0.3, Chroma backend (`ChromaMemoryManager`, MiniLM
  `DefaultEmbeddingFunction`, exact cosine), as in `base.memory` of gpu/gpu_config.yaml.
- **Fresh store per user**: a new, empty Chroma folder for each user; that user's 5 facts are stored in
  `fact_id` order (`turn_id` = 1..5, empty `mood_snapshot`). Both conditions query the same store, so
  similarities are identical across conditions and only the ranking weights differ. Score ties are broken as in
  `retrieve_relevant` (newest first).
- **Importance** from `memory_manager.importance_from_turn(probs, abstained=False, cfg)` with probs built from
  `intensity`; the rest of the probability is split evenly over happy / angry / sad:

  | intensity | p_neutral | p_happy = p_angry = p_sad | importance (= 0.3 + 0.7 x (1 - p_neutral)) |
  |---|---|---|---|
  | calm | 1.0 | 0 | 0.30 |
  | mild | 0.5 | 1/6 | 0.65 |
  | strong | 0.1 | 0.3 | 0.93 |

**Ranking read for each query**: all of that user's facts with similarity >= 0.15, ordered by score
(`retrieve_relevant(user, query, top_k=5)`, i.e. the full list after the cutoff). `top_k` 3 is the agent's
setting and is what hit@3 and distractor-in-top-3 measure. The cutoff is the same in both conditions, so a
relevant fact below 0.15 scores 0 in both.

**Primary metric:** MRR of the relevant fact. Reciprocal rank = 1 / (rank of the relevant fact in that list);
0 if it is not returned (below the cutoff).

**Secondary metrics:** hit@1 (relevant fact ranked first), hit@3 (relevant fact in the top 3), and
distractor-in-top-3 rate (share of `calm_target_charged_distractor` queries whose `distractor_fact_id` is in
the top 3). MRR, hit@1 and hit@3 are reported overall (45 queries) and per case_type (15 each); the
distractor rate over its 15 queries.

**Verdict** on all 45 queries: paired bootstrap of d = MRR(weighted) - MRR(similarity-only).
Per query, d_i = RR_weighted - RR_similarity_only; `rng = numpy.random.default_rng(42)`;
`idx = rng.integers(0, 45, size=(10000, 45))`; each resample's mean of d_i; 95% CI = 2.5th and 97.5th percentiles
(`numpy.percentile`, default linear interpolation).
- lower bound > 0 → **weighting helps**
- upper bound < 0 → **weighting hurts**
- otherwise → **no detectable difference**

Per-case-type CIs are computed the same way within each type (15 queries, `size=(10000, 15)`, a fresh
`default_rng(42)` for each type), for information only; they do not change the verdict.

**Guardrail (reported, does not change the verdict):** on the 30 calm-target queries
(`calm_target` + `calm_target_charged_distractor`), if hit@1(similarity-only) - hit@1(weighted) > 0.10, it is
flagged as "weighting costs calm facts".

**Reported either way**, whatever the verdict: all metrics for both conditions, the CI, the guardrail, and the
per-query ranks.

## Provenance

Scenarios and labels drafted by Claude Code, reviewed and approved by Shan on 2026-11-22 with no label changes.
Accepted as written at review: MRR reads the full ranking after the cutoff (`top_k=5`, ranks 4-5 count 1/4 and
1/5); the weaker distractor pairs (u11, u12, u15) and the very indirect paraphrases (u02_q3, u12_q3, u14_q2) stay.
