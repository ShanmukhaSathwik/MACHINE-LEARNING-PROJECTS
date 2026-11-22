# Emotion-Aware Conversational Voice Agent — Build Roadmap

**Course**: DTSC 5082 · UNT · Fall 2026
**Team**: 06

---

## Completed

- **Phase 1 — Baseline classifiers** (RF 0.625, XGBoost 0.719 in current env)
  - Feature extraction: 39 features × 576 clips, speaker-independent split (16/4/4 actors)
  - Random Forest + XGBoost, confusion matrices, feature importance
  - Per-clip probability export (`clip_probabilities.csv`)
- **MoodManager v1** — EMA-smoothed mood state (γ=0.7, persistence 2 turns)
  - Real-data experiment: 12-turn scripted conversation, 1000 runs
  - Reactive switches 4.45±1.74 vs MoodManager 1.36±0.62
- **Research poster** — presented Oct 6
- **GPU demo build, steps 1–10 (code, 2026-11-11 → 11-13)** — `gpu/` device, SER, MoodManagerV2, safety, JSON memory, ASR, LLM, CSM voice, live agent, bench (f0c530b … 66688df); see "GPU demo build" below
- **GPU sessions 1–2 (2026-11-12, 2026-11-13)** — T3 training + Whisper / Qwen / CSM on GPU (182d6f5); live agent end to end, CSM spoke 15/15 turns (41f5ccd)
- **T1–T3 HuBERT fine-tune** — T1 RAVDESS + CREMA-D manifest (ed4ef2c), T2 script (b204e91), T3 run `gpu_24gb_20261112_1500`: val macro-F1 0.777, test macro-F1 CREMA-D 0.745 / RAVDESS 0.676; my_voice_v3 12/32, FAIL against the 22/32 rule → not approved for the demo (05eff07)
- **Task B — JSON memory scaffold (2026-11-12)** — 521e061; now the fallback behind Task D
- **Task D — RAG memory on Chroma (2026-11-19 → 11-22)** — D1–D5, `min_similarity` 0.15, `memory.backend: "chroma"` (fcd9e95)

---

## Build Sequence (what's next, in order)

> P1–P9 = the nine design principles in the workflow doc, Section 0.6 (added 2026-11-05).
>
> **Next (2026-11-22):** retrain the emotion model on the student recordings → **F** (evaluation, including the RQ2 two-session test with `w_importance` 0 vs 0.3) → **A2.5** (shuffle-order placebo) → **H** (web frontend).

### A. Thin end-to-end agent — DONE (2026-10-31)
Wire the minimum viable pipeline printed to the terminal:
1. Whisper ASR (tiny.en) → text
2. XGBoost probabilities → emotion label + distress score
3. MoodManager → tone mode (DEFAULT / EMPATHETIC / CALMING / POSITIVE_ENCOURAGING)
4. `build_system_prompt(tone, transcript)` → prompt
5. Gemini (or template fallback) → reply text
6. Print to screen (no TTS yet)

**Goal**: Prove the loop works. No memory, no streaming, no audio output.

**Result** (`scripts/agent.py --script`, 12-turn label script, test speaker 16):
- MoodManager: 1 mode switch (DEFAULT → EMPATHETIC on turn 7, ignoring the one-off angry spike on turn 3) vs reactive baseline (`--no-mood`): 4 switches
- Average turn latency ~2.5 s (LLM ~1.3–3 s, ASR ~0.45 s, emotion ~0.15 s); 5.1 s on a re-run where gemini-3.6-flash was rate-limited (429) on 10/12 turns
- Shipped as built: 2 modes only (DEFAULT / EMPATHETIC, MoodManager v1); `build_system_prompt(mode, perceived_emotion, memory_facts)`; tone driven by mood state only, never the per-turn label
- **Confound**: the Gemini fallback chain (3.6-flash → 3.5-flash-lite → 3.1-flash-lite) can change model mid-conversation, so tone differences may partly come from the model. Every attempt is logged per turn; pin one model for the Task F evaluation
- RAVDESS transcripts are lexically neutral ("Kids are talking by the door"), so reply content is uninformative; tone is the only signal in scripted runs

### A2. Guardrails in the current agent (A2.1–A2.4 done; only A2.5 open)
Design principles from the workflow doc, Section 0.6. All CPU, all on existing code.
1. **Inspectable state** (P1) — DONE (2026-11-05): add `MoodManager.explain()`; every turn in the session JSONL logs probs, confidence, abstain flag, mood score, turns_above, mode, strategy, reason, exact system prompt, LLM model, TTS tags
   - Shipped: `explain()` one-line reason; JSONL adds `confidence`, `mood_params`, `reason`, `system_prompt` (null with `--no-llm`) and always-present placeholders `abstained` / `check_in` / `strategy` / `tts_tags`. Re-run matches Task A (MoodManager 1 switch on turn 7, reactive 4)
2. **Abstain** (P2): confidence = top-1 prob; if < τ_conf (start 0.5, tune on val) → mood holds state, no mode switch, logged `abstained`; after 2+ abstains in a non-neutral mood → `check_in` flag. Report coverage vs accuracy (targets the ~53% happy→distress bias)
   - **Shipped / finding** (2026-11-05): `scripts/abstain_curve.py`; pre-set rule = highest val accuracy among taus with val coverage ≥ 0.70 → **τ_conf = 0.82** (shipped as the first default; replaced by Rule B's 0.72 below; `--tau 0` disables). Abstain holds score / turns_above / mode; `check_in` after 2+ consecutive abstains in a non-DEFAULT mood (logged only until A2.3). Outputs: `results/metrics/abstain_curve.csv`, `results/figures/abstain_coverage.png`

     | τ | split | coverage | accuracy | happy→distress (kept happy) |
     |---|---|---|---|---|
     | 0 (no abstain) | val | 1.000 | 0.719 | 0.438 |
     | 0 (no abstain) | test | 1.000 | 0.719 | 0.531 |
     | 0.82 | val | 0.729 | 0.843 | 0.227 |
     | 0.82 | test | 0.646 | 0.855 | 0.500 |

   - Accuracy gain holds on test, but the happy→distress bias barely changes there (0.531 → 0.500, with only 44% of happy test clips kept); the val gain (0.438 → 0.227) didn't carry over
   - `--script --tau 0` reproduces the A2.1 run exactly; with τ = 0.82 the script abstains on turns 1, 2, 4, 5, 8 and still switches once, on turn 7
   - My-voice run (`data/my_voice` 01–10): abstained on 6/10 turns, 5 of which were correct predictions; mode never reached EMPATHETIC through the sad/angry stretch (without abstain it switches on turn 7). Confidence on out-of-domain audio runs low, so 0.82 is too strict there
   - Abstain can't catch confident mistakes: `03_angry` was read as happy at confidence 0.91 and passed through
   - **Rule B** (recorded 2026-11-05, before computing it): tau = highest val accuracy among taus with val coverage >= 0.85. Tested only on a fresh my-voice set (`data/my_voice_v2`), never on `data/my_voice` 01–10
     → **τ = 0.72**: val coverage 0.854 / accuracy 0.768 / happy→distress 0.346; test coverage 0.771 / accuracy 0.784 / happy→distress 0.500 (`abstain_curve.py --min-coverage 0.85`)
   - **my_voice_v2 result** (15 fresh clips, labels fixed before recording; `agent.py --no-llm data/my_voice_v2` at τ 0 / 0.72 / 0.82):
     - Plain accuracy 5/15 (happy 0/5, sad 0/5, angry 5/5); 13/15 predicted angry (06_sad and 08_sad read as happy)
     - First EMPATHETIC switch: τ 0 turn 4, τ 0.72 turn 7, τ 0.82 turn 11; all end EMPATHETIC (no recovery)
     - τ 0.72: 8 abstains, 6 of them wrong predictions; `check_in` fired on turns 9–10
     - **Decision**: τ_conf = 0.72 is the agent default (`--tau` still overrides). Abstain delays false switches but can't stop high-confidence errors (sad→angry 0.99, happy→angry 0.95)
     - **Open question**: `my_voice` (Nov 1) was 8/10 correct, `my_voice_v2` is 5/15 → audio check next
     - Audio check (`check_audio.py`): v2 ~13 dB louder than RAVDESS and louder/shorter than v1; on the top features all three emotions sit past RAVDESS angry → model learned 'loud = angry'; workflow Section 2 RMS normalization was never implemented
     - Pitch features drift up to 19 Hz vs the manifest (old Python 3.14 env); not in top 10
     - Loudness test (`test_loudness.py`): scaling each clip to RAVDESS speech RMS (−34.7 dBFS) drops v2 angry predictions 13 → 2 but accuracy stays 5/15 (errors move to happy); v1 drops 8/10 → 6/10. Conclusion: missing RMS normalization causes the angry bias, but XGBoost on the 39 features doesn't recognize emotion in my voice. Per-clip scaling also erases real loudness differences; fix belongs in Task C, not a separate XGBoost retrain
3. **Code decides, LLM speaks** (P3) — DONE (2026-11-11, 7cd03f9; MoodManagerV2 with all 4 strategies in 1aff92f): MoodManager outputs a strategy (VALIDATE / CALM / ENCOURAGE / NEUTRAL); `build_system_prompt(strategy, facts)` — drop `perceived_emotion` from the prompt. MoodManager v1 (single distress axis) can only give VALIDATE / NEUTRAL; MoodManager v2 (one EMA per class: sad→VALIDATE, angry→CALM, happy→ENCOURAGE) unlocks the other two
4. **Crisis check + consent** (P9) — DONE (2026-11-12, `scripts/safety.py`, 51842e0: consent record/revoke + crisis phrase override, 988 safe reply): `safety.check_crisis(transcript)` right after ASR; on match skip the LLM, return a fixed safe response (988 in the US), no memory stored from that turn
5. **Shuffle-order placebo** (P7) — **open**: rerun `plot_mood_real.py` (1000 conversations) with turn order shuffled; compare MoodManager vs reactive, original vs shuffled; report either way

### C. Bi-LSTM + CNN classifiers (moved before B, 2026-11-07: A2.2 audio/loudness checks showed XGBoost doesn't transfer to my voice)
- CNN on mel-spectrograms (3-class)
- Bi-LSTM on feature sequences (3-class)
- Extra model: frozen pretrained speech encoder (WavLM-base or wav2vec2-base) embeddings + small classifier, alongside CNN and Bi-LSTM
- Same preprocessing in training and in the agent (one shared function)
- Random gain augmentation in training (±12 dB) + noise/pitch/speed augmentation
- CREMA-D for more speakers if time allows
- Compare against XGBoost 0.719 baseline in same environment
- Update model_comparison.csv with all models
- Out-of-domain test on a fresh my-voice set (`data/my_voice_v3`), never used for tuning
- Rerun Rule B for τ_conf with whichever model wins

### GPU demo build (`gpu/`, settings in `gpu/gpu_config.yaml`)
Steps as numbered in the `gpu/*.py` docstrings: 1 `device.py`, 2 `ser_gpu.py`, 6 `asr_gpu.py`, 7 `llm_gpu.py`, 8a/8b `tts_gpu.py`, 9a `live_agent_gpu.py`; training T2/T3 `finetune_ser.py`.

**GPU session 1 (2026-11-12) — measured**
- **Training T3 — DONE**: run `gpu_24gb_20261112_1500`, `superb/hubert-large-superb-er`, 4 classes; early-stopped at epoch 7, best epoch 4, val macro-F1 0.777
  - Test: CREMA-D acc 0.750 / macro-F1 0.745; RAVDESS acc 0.715 / macro-F1 0.676
  - Main weakness: sad read as neutral (top error on both corpora: 61 CREMA-D and 17 RAVDESS test clips)
  - Per-clip results: `results/metrics/ser_finetune_test_gpu_24gb_20261112_1500.csv`
  - Fix in 182d6f5: the model-settings dict reused the name `overrides` (the CLI override list) and crashed with hubert-large; now `model_kwargs`, with a regression test in `tests/test_finetune_ser.py`
- **Step 6 on GPU**: Whisper large-v3-turbo, WER 0.049 on 24 RAVDESS clips (all word errors / all reference words; the skipped clip counts as all words deleted), ~0.08 s per clip
  - The −45 dBFS silence gate (`asr.silence_dbfs`) skipped one quiet sad clip (`Actor_20/03-01-04-01-01-02-20.wav`)
  - To do: measure the silence-gate threshold
  - `results/metrics/asr_ravdess_whisper-large-v3-turbo.csv`
- **Step 7 on GPU**: Qwen3-8B 4-bit; all four strategies + fact extraction answered by Qwen (no fallback); first sentence in ~0.3–1.1 s
  - Prompt fixes (6474ab2): CALM gave several steps instead of one; NEUTRAL invented a fact. **Closed in GPU session 2** (rerun passed, see below)
  - `results/metrics/llm_demo_qwen_Qwen3-8B.csv`
- **Step 8a — DONE**: Expresso speaker ex02; picks saved in `gpu/voice_refs.json`: VALIDATE `ex02_default_00001_default1`; CALM and NEUTRAL `ex02_default_00002`; ENCOURAGE `ex02_happy_00002`
- **Not yet done**: the T3 model has not been tested on `data/my_voice_v3`, so it is **not yet approved for the demo**

**Step 8b — code done, not yet run on the GPU (2026-11-13)**
- `tts_gpu.py`: `CSMVoice` (reference clip per strategy; crisis → NEUTRAL), `split_sentences`, `strip_tags` (`tts.tags_enabled: false`), `speak_stream` (next sentence generated while the previous one plays), `speak_with_fallback`; CLI `--speak` / `--fallback-test`
- `scripts/fallback_tts.py`: pyttsx3, else the text is printed; no torch
- CPU tests: `tests/test_tts_speak.py` (stub CSM, no audio)
- To do on the GPU: `python gpu/tts_gpu.py --speak "..." --strategy VALIDATE --preset gpu_24gb` for each strategy → `results/metrics/tts_speak_csm.csv` (time_to_first_audio, real-time factor); listen for gaps between sentences

**Step 9a — code done, not yet run on the GPU (2026-11-13)**
- `gpu/live_agent_gpu.py`: `LiveAgent.run_turn(user_id, audio)` in a fixed order (consent → ASR → crisis → emotion → abstain → MoodManagerV2 → memory → prompt → LLM → TTS → fact extraction → JSONL log); all parts injected; no torch at import
- Abstain threshold = `brain.tau_conf` in `gpu_config.yaml`, still **0.0 (abstain off)** until it is re-chosen for the new emotion model
- LLM → TTS pipelined: CSM starts on the first sentence (`tts_gpu.speak_iter_with_fallback`)
- Per-user mood / history / memory / log file; `delete_user_data(user_id)` reuses `delete_all_my_data`
- CPU tests: `tests/test_live_agent.py` (stubs)
- Not wired yet: the "delete all my data" voice command promised in `safety.CONSENT_TEXT`; microphone / push-to-talk (9b)

**Step 9b — code done, GPU and real-mic runs pending (2026-11-13)**
- Emotion model = T3 (`ser.checkpoint` → run `gpu_24gb_20261112_1500/best`); label order checked against `best/id2label.json` (a mismatch stops); missing / failed checkpoint → pretrained fallback with one warning; every log line has `ser_model` {name, fallback}
- **tau_conf = 0.64** (Rule B, fixed before computing: max val accuracy with val coverage ≥ 0.85; RAVDESS + CREMA-D val only). Val probs from `gpu/ser_split_probs.py` on CPU through the agent's own `SERModel` path; no-abstain val accuracy 0.78111 = the run's own best-epoch val_acc
  - Val at 0.64: coverage 0.851 (766/900), accuracy 0.837 (no abstain: 0.781)
  - Test, information only: coverage 0.825 (697/845), accuracy 0.801 (no abstain: 0.748); sad coverage 0.813, happy 0.706
  - `results/metrics/ser_t3_val_probs.csv`, `ser_t3_test_probs.csv`, `abstain_curve_t3.csv`
- Silence gate (`scripts/measure_silence_gate.py`), speech side measured on 4014 train clips: quietest pass level −60.0 dBFS (RAVDESS sad, actor 09), 1st percentile −49.1, median −29.0; the current −45 gate blocks 90/4014. **Room noise not recorded yet** → gate unchanged
- Push-to-talk `--live` (cap = `ser.max_seconds`), delete-all by voice (spoken "yes" only; neither turn logged or stored), `--script` rehearsal mode
- CPU tests: `tests/test_live_agent_9b.py` (stubs)

**Step 10 — code done, GPU pending (2026-11-13)**
- `gpu/bench_gpu.py`: runs `gpu/bench_turns.yaml` through `LiveAgent` via the scripted mode; per turn the agent's own stage times, time to first audio, turn total, voice path, LLM backend, `ser_model` + fallback, abstained; peak GPU memory after load and after the run ("n/a" without CUDA); one warm-up turn discarded, then `bench.repeats` passes (median and min–max); fallbacks flagged; bench users / memory / logs kept in `gpu/outputs/bench/`
- `gpu/bench_turns.yaml`: first test-split clip per corpus × class (no my_voice), plus a crisis text case and a delete request cancelled by "No."
- `gpu/SESSION2_CHECKLIST.md`: home prep (room noise with enhancements off, gate only if its check passes, USB copies), GPU tests in order, results home, cleanup
- Only a `--stub` plumbing run so far (no models; its timings are not results). CPU tests: `tests/test_bench_gpu.py`
- To do on the GPU: `python gpu/bench_gpu.py --preset gpu_24gb` → `results/metrics/bench_gpu_24gb_<time>.csv`

**GPU session 2 (2026-11-13) — measured**
Sources: `session2_notes.txt`, `session2_terminal.txt`, `results/metrics/llm_demo_qwen_Qwen3-8B_20261113_100929.csv`; agent logs (7 `session_*.jsonl`, 16 turns) and 2 demo videos kept outside git (USB `session2_export.zip` + `live demo/`)
- **Models loaded on the RTX 5000 Ada (32 GB)**: T3 SER 1.19 GB, Whisper large-v3-turbo 1.52 GB, Qwen3-8B 4-bit 5.68 GB, CSM-1B 6.64 GB
- **Qwen prompt-fix rerun — PASSED**: CALM gives one step, NEUTRAL invents no fact; all 4 strategies + fact extraction answered by Qwen (no fallback); first sentence 0.36–0.80 s
- **Live agent (9a/9b) end to end**: T3 loaded with no fallback on all 16 turns (2 RAVDESS wav turns, 14 push-to-talk turns, users `shan_test` / `shan_demo` / `shan_noabstain`)
  - First RAVDESS turn fell back to text voice (`No module named 'sounddevice'`); after installing it, 15/15 turns spoke with CSM
  - CSM turns: time to first audio median 6.55 s (3.91–14.02), turn total median 25.1 s (18.1–44.5); CSM takes 16.5–41.3 s of each turn. Push-to-talk turns (`shan_noabstain`): ASR median 0.54 s, LLM median 0.79 s
  - With tau_conf 0.64, both scored push-to-talk turns abstained (confidence 0.48 and 0.63)
- **Abstain-off run** (`tau_conf` set to 0.0 on the GPU machine only, never committed; 12 turns): **strategy stayed NEUTRAL on every turn**
  - Per-turn emotion: happy 6, sad 5, neutral 1, angry 0 (the angry work story was read as happy, 0.73)
  - Mood never held ≥ 0.50 for 2 turns: max sad 0.43, happy 0.47, angry 0.01. MoodManagerV2 worked as designed; the alternating happy/sad perception is the limit (matches my_voice_v3 12/32)
- **Not run**: CSM `--speak` per strategy, crisis voice check, `bench_gpu.py`, room-noise silence gate
- **Known issues**: live agent plays the CSM voice but does not save the wav; test memories (`shan_test`, `shan_demo`, `shan_noabstain`) may still be on the professor's machine

### B. MemoryManager — JSON scaffold — DONE (2026-11-12)
- **Done in 521e061**: `scripts/memory_manager.py` (JSON, scored retrieval, forget / delete-all) + `scripts/fact_extractor.py` (user turns only)
- **Superseded as default by D4** (`memory.backend: "chroma"`); the JSON backend is kept as the fallback
- Original plan:
- Conversation buffer (last N turns)
- User-fact store (name, preferences) with emotion-gated write
- JSON file backend — no vector DB yet
- Every fact gets a `memory_id`, `source: "user_turn"` and `turn_id`; methods `list_user_facts()` and `delete(memory_id)` from day one (P4)
- **Perspective boundary** (P5): `extract_fact()` sees only the user's transcript, never Gemini's reply; unit test: a fact invented in a reply is never stored
- ABSTAIN turns don't raise `importance_score` (P2)

### D. RAG upgrade (MemoryManager v2) — DONE (2026-11-20)
- **D1 done (2026-11-19):** chromadb 1.5.9 + onnxruntime 1.32.0 in .venv312 (no pinned package changed). `scripts/check_embedder.py`: Chroma's DefaultEmbeddingFunction (all-MiniLM-L6-v2, ONNX, CPU) → 384-dim, cosine A–B 0.2991 vs A–C 0.0682, first call 10.986 s with the one-time 79.3 MB download (0.637 s with model cached), second call 0.411–0.462 s; torch not imported → PASS. `tests/test_embedder.py` added; test_embedder + test_memory + test_live_agent: 68 passed
- **D2 done (2026-11-19):** `ChromaMemoryManager` next to the JSON `MemoryManager` in `scripts/memory_manager.py` (shared `BaseMemoryManager.delete_all_my_data`; `make_memory_manager` picks by `memory.backend`, still `"json"`; CLI `--backend json|chroma`). New config keys `backend`, `vector_store_dir` (`vector_store/`, already git-ignored), `min_similarity` (placeholder 0.0 until D3). Chroma: cosine collection "memories", every user fact scored, similarity < min_similarity dropped before scoring. Real MiniLM test: query "I'm stressed about Monday." → exam fact similarity 0.2991 (score 0.3594) ranks above pizza fact 0.0682 (score 0.1977). `tests/test_memory_chroma.py` 20 passed; with test_memory + test_fact_extractor + test_live_agent + test_embedder: 126 passed. Agent still uses JSON (wiring is D4)
- **D3 done (2026-11-20):** labelled set `data/memory_eval/` (40 queries x 20 facts = 800 pairs; tune 13 positives / 387 negatives, test 10 / 390), labels approved before any similarity was computed; exact cosine scoring in `ChromaMemoryManager`. Rule fixed and committed first (f529829): `min_similarity` = lowest cutoff (0.00-1.00, step 0.01) with tune false-recall rate <= 0.10. `scripts/choose_min_similarity.py` → **min_similarity = 0.15** (tune FRR at 0.14 was 43/387 = 0.111). Tune: recall 0.8462 (11/13), FRR 0.0982 (38/387). Test: recall 1.0000 (10/10), FRR 0.1077 (42/390). Reported only (added in D4): queries with a negative fact among their top 3 and >= 0.15: tune 13/20 (0.65), test 16/20 (0.80). Outputs: `results/metrics/memory_pairs.csv`, `memory_threshold.csv`, `memory_threshold_summary.json`, `results/figures/memory_threshold.png`. `backend` still "json"
- **D4 done (2026-11-20):** `memory.backend: "chroma"`. `live_agent_gpu.build_memory` (used by the live agent and `bench_gpu.build_bench_agent`, bench store inside its bench folder) loads the MiniLM embedder once at startup and prints backend + min_similarity; if chroma fails it warns and falls back to JSON. Every turn log line records `memory_backend` (chroma / json / json-fallback) and `min_similarity`. Prompt: remembered facts carry a guard ("use only if it is about what they just said, otherwise do not mention it at all"). Two-session demo (`fact_extractor.py --demo`), session-2 query sharing no content word with the stored fact: chroma recalled the exam fact (similarity 0.575, score 0.686); JSON recalled nothing. smoke_cpu, 3 scripted turns (temporary store): turn 3 paraphrase recalled the turn-1 fact (similarity 0.4301). Requirements: chromadb + deps in `gpu/requirements_gpu.txt`, 8 used packages pinned in requirements.txt, fsspec pin 2026.7.0; `pip check` clean. Full suite: 567 passed
- **D5 done (2026-11-22):** no JSON memories exist at home (`data/memories.json` absent), so no migration was needed. The test memories on the professor's GPU machine (`shan_test`, `shan_demo`, `shan_noabstain`) are deleted in the next GPU session instead
- [x] Replace JSON fact store with Chroma/FAISS → Chroma (D2 `ChromaMemoryManager`; default backend in D4)
- [x] MiniLM embeddings + emotion-weighted retrieval → D1 (MiniLM ONNX embedder), D2 (score = w_similarity × similarity + w_importance × importance, importance from emotion intensity), D3 (`min_similarity` 0.15). The weighting was built and tested but is **switched off** since RQ2 R2: the live agent retrieves by similarity only (`w_importance` 0; see F, R1/R2)
- [x] Integrate into agent loop → D4 (`live_agent_gpu.build_memory`, also used by `bench_gpu`)
- [x] `delete(memory_id)` wired to Chroma; unit test: a deleted fact is never retrieved again (P4) → D2 (`tests/test_memory_chroma.py::test_deleted_fact_never_retrieved_again_even_after_reopening`)
- [x] Log similarity / importance / final score for every retrieved memory (P1) → D2 (same retrieval return keys as JSON) + D4 (turn log adds `memory_backend` and `min_similarity`)

### E. TTS — CSM-1B on the professor's GPU machine (Colab dropped Nov 2026)
- **Built**: `gpu/tts_gpu.py` running base CSM-1B with Expresso speaker ex02, one reference clip per strategy (`gpu/voice_refs.json`, step 8a); pyttsx3 fallback via `scripts/fallback_tts.py` (253cdec)
- **Measured**: spoke 15/15 turns with CSM in GPU session 2 (after `sounddevice` was installed; see GPU session 2 above)
- **No direct emotion → voice path** (P6): TTS tags come only from the MoodManager strategy, never the per-turn SER output; `voice/` does not import the emotion predictor
- Open:
  - [ ] Save the CSM wavs in the live agent (it plays them but does not save them)
  - [ ] LoRA fine-tune CSM on Expresso (after the demo)
  - [ ] Decide the demo voice host

### F. Evaluation & demo
- **R1/R2 — RQ2 memory test, done (2026-11-22):** weighted retrieval (0.7 similarity + 0.3 importance) vs similarity-only (1.0 / 0.0) on `data/rq2_eval/` (15 users x 5 facts, 45 queries; labels and rule approved and committed first in 1e7b92d). Real MiniLM, Chroma 1.5.9, `min_similarity` 0.15, fresh store per user, importance from scripted intensity (calm 0.30 / mild 0.65 / strong 0.93). `scripts/run_rq2_eval.py`, `tests/test_run_rq2_eval.py`; outputs `results/metrics/rq2_per_query.csv`, `rq2_summary.csv`, `rq2_summary.json`, `results/figures/rq2_mrr_by_case.png`
  - **Verdict: weighting HURTS.** MRR weighted 0.8759 vs similarity-only 0.9778; difference −0.1019, 95% paired-bootstrap CI [−0.1741, −0.0370] (10,000 resamples, seed 42). hit@1 0.7778 vs 0.9556; hit@3 0.9778 vs 1.0000; no relevant fact cut by the threshold in either condition
  - Per case type (information only): charged_target +0.0333 [+0.0000, +0.1000]; calm_target_charged_distractor −0.2722 [−0.4111, −0.1333]; calm_target −0.0667 [−0.1667, +0.0000]
  - **Guardrail flagged**: calm-target hit@1 0.9667 → 0.6667 (drop 0.3000 > 0.10). 10/45 queries get a different top-1 fact; in 9 of them the weighted top-1 is a strong or mild fact over the right calm one, in 1 (u03_q1) weighting fixes a charged target
  - Limitation: intensity is scripted, not from the emotion model
  - **Decision (2026-11-22):** live agent uses similarity-only retrieval (`w_similarity` 1.0, `w_importance` 0 in gpu_config.yaml); importance is still computed, stored with each fact and logged per retrieved memory, and the spike flag still shapes the fact-extraction prompt. Paper reports RQ2 as a negative result for additive weighting
- End-to-end evaluation: emotion accuracy, response quality, mood stability
- Prepare demo script and recording
- Pin one LLM model for every evaluation run (Task A fallback-chain confound)
- **Mishandle suite** (P8): 5–10 scripted scenarios in `tests/` (missed distress, mood lost after topic change, agreeing with something harmful, false empathy on a happy user, recalling a deleted fact, crisis phrase); pass criteria written first; plain Gemini vs full system, report the first failing turn
- Include the shuffle-order placebo (A2.5) and the abstain coverage curve (A2.2) in the results

### G. Repo restructure
- Clean folder layout per `Modular Project Structure.pdf`
- Final README, docstrings, reproducibility check

### H. Web frontend (final demo)
- **Consent screen** before the mic turns on: what is recorded, stored, how to delete (P9)
- **"What I remember about you"** page: list facts with a delete button each → `delete(memory_id)` (P4)
- **"Delete all my data"** button → `reset_user_memory(user_id)` + session logs (P9)
- **"Why did it answer this way?"** debug panel showing the per-turn state from A2.1 (P1)

---

## Environment

- Python 3.12.10, CPU-only torch 2.15.0
- Venv: `(emotion-agent-312)` in `Code/.venv312/`
- Activate (PowerShell, from `Code/`): `.\.venv312\Scripts\Activate.ps1`
- Pinned deps: `requirements.txt` (unpinned originals in `requirements.in`)
- The original Python 3.14.5 venv (`Code/.venv/`) is kept as a backup and is no longer used
