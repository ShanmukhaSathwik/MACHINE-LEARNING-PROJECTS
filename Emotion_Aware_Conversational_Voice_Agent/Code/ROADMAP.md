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

---

## Build Sequence (what's next, in order)

> P1–P9 = the nine design principles in the workflow doc, Section 0.6 (added 2026-11-05).

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

### A2. Guardrails in the current agent (next — before B)
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
3. **Code decides, LLM speaks** (P3): MoodManager outputs a strategy (VALIDATE / CALM / ENCOURAGE / NEUTRAL); `build_system_prompt(strategy, facts)` — drop `perceived_emotion` from the prompt. MoodManager v1 (single distress axis) can only give VALIDATE / NEUTRAL; MoodManager v2 (one EMA per class: sad→VALIDATE, angry→CALM, happy→ENCOURAGE) unlocks the other two
4. **Crisis check** (P9): `safety.check_crisis(transcript)` right after ASR; on match skip the LLM, return a fixed safe response (988 in the US), no memory stored from that turn
5. **Shuffle-order placebo** (P7): rerun `plot_mood_real.py` (1000 conversations) with turn order shuffled; compare MoodManager vs reactive, original vs shuffled; report either way

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
Steps as numbered in the `gpu/*.py` docstrings: 1 `device.py`, 2 `ser_gpu.py`, 6 `asr_gpu.py`, 7 `llm_gpu.py`, 8a `tts_gpu.py`; training T2/T3 `finetune_ser.py`.

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
  - Prompt fixes to do: CALM gives several steps instead of one; NEUTRAL invented a fact
  - `results/metrics/llm_demo_qwen_Qwen3-8B.csv`
- **Step 8a — DONE**: Expresso speaker ex02; picks saved in `gpu/voice_refs.json`: VALIDATE `ex02_default_00001_default1`; CALM and NEUTRAL `ex02_default_00002`; ENCOURAGE `ex02_happy_00002`
- **Not yet done**: the T3 model has not been tested on `data/my_voice_v3`, so it is **not yet approved for the demo**

### B. MemoryManager — JSON scaffold
- Conversation buffer (last N turns)
- User-fact store (name, preferences) with emotion-gated write
- JSON file backend — no vector DB yet
- Every fact gets a `memory_id`, `source: "user_turn"` and `turn_id`; methods `list_user_facts()` and `delete(memory_id)` from day one (P4)
- **Perspective boundary** (P5): `extract_fact()` sees only the user's transcript, never Gemini's reply; unit test: a fact invented in a reply is never stored
- ABSTAIN turns don't raise `importance_score` (P2)

### D. RAG upgrade (MemoryManager v2)
- Replace JSON fact store with Chroma/FAISS
- MiniLM embeddings + emotion-weighted retrieval
- Integrate into agent loop
- `delete(memory_id)` wired to Chroma; unit test: a deleted fact is never retrieved again (P4)
- Log similarity / importance / final score for every retrieved memory (P1)

### E. TTS — CSM-1B on Colab
- Google Colab notebook running Sesame CSM-1B
- REST endpoint the agent calls for audio output
- Tone-conditioned voice (map mood mode → speaker style)
- **No direct emotion → voice path** (P6): TTS tags come only from the MoodManager strategy, never the per-turn SER output; `voice/` does not import the emotion predictor

### F. Evaluation & demo
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
