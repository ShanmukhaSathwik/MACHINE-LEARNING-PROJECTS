# Memory retrieval labelled pair set (Task D step D3)

A small hand-labelled set for choosing `memory.min_similarity` (gpu/gpu_config.yaml) for the Chroma memory
backend. Labels were fixed before any query-fact similarity was computed.

## Files

| File | Columns | What it is |
|---|---|---|
| `facts.csv` | `fact_id, fact` | 20 stored facts about one imaginary student, written the way `scripts/fact_extractor.py` writes them: one sentence starting "The user ...", third person, at most 25 words, no names, addresses or numbers. Includes deliberate near-neighbours (e.g. a statistics exam vs a driving test vs a history essay; guitar vs watercolours vs running; working at a coffee shop vs drinking coffee). |
| `queries.csv` | `query_id, query, relevant_fact_ids, case_type, split` | 40 things the user might say, written as short, casual, first-person spoken turns. |

## How a label is read

For each query, every fact in `relevant_fact_ids` (semicolon-separated) **should be recalled**.
Every other fact in `facts.csv` **should not be recalled** for that query. An empty `relevant_fact_ids` means
no fact should be recalled. So each query gives 20 labels (one per fact), 800 query-fact pairs in total.

## Case types (10 queries each)

| case_type | Meaning |
|---|---|
| `paraphrase` | Same topic as its relevant fact, but shares **no content word** with it (checked with `memory_manager.content_words`, i.e. after lowercasing, removing punctuation and stopwords). Word overlap (the JSON backend) cannot find these; meaning match should. |
| `direct` | Mentions the topic plainly; shared words allowed. |
| `word_trap` | Shares a content word with at least one fact but is about something else (e.g. "coffee table books" vs the coffee facts). `relevant_fact_ids` is empty. |
| `none` | Small talk or unrelated; shares no content word with any fact. `relevant_fact_ids` is empty. |

## Split

`split` is `tune` (used to choose the threshold) or `test` (used once, to report how the chosen threshold does).
Assigned with a fixed seed, balanced within each case type: one `random.Random(42)`; for each case type in the
order `paraphrase, direct, word_trap, none`, that type's `query_id`s are sorted, shuffled with that generator,
and the first half become `tune`, the rest `test` (5 + 5 per type). The rule is `assign_splits` in
`tests/test_memory_eval_set.py`, which also checks that the file matches it.

## Checks

`tests/test_memory_eval_set.py`: ids unique, every `relevant_fact_id` exists, facts look like extractor output,
`case_type` / `split` values valid, split equals the seeded assignment, paraphrase queries share no content word
with their relevant facts, word-trap queries share a content word with at least one fact, none queries have no
relevant fact and share no content word with any fact.

## Provenance

Labels drafted by Claude Code, reviewed and approved by Shan on 2026-11-19.
Review change: q09 ("I keep missing my family and my old bedroom.") also recalls f08 (calls their mother weekly).
