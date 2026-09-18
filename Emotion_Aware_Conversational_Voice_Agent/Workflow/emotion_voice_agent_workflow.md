# Emotion-Aware Conversational Voice Agent — Final Workflow

An end-to-end plan for a system that (1) detects a speaker's emotion from voice, (2) tracks it as a smooth, evolving **mood state**, (3) **remembers durable facts** about the user across sessions, and (4) adapts both the wording and the *sound* of its spoken replies. Runs locally on a normal PC, with free Google Colab GPU used only for the voice model.

---

## 0. System at a Glance

**Runtime loop (per user turn):**

```
User speaks
   │
   ▼
[ASR: Whisper] ──► transcript text ─────────────────────────┐
   │                                                         │
   ▼                                                         │
[Audio pre-processing] ─► [Feature extraction] ─► [Emotion model: small Bi-LSTM]
                                                             │
                                          emotion probabilities  P(over classes)
                                                             │
                        ┌────────────────────────────────────┤
                        ▼                                     ▼
                 [MoodManager]                        (emotion signal also
             short-term mood state                     informs what to remember)
             → behavior profile                                │
                        │                                      ▼
                        │                              [MemoryManager]
                        │                   long-term facts (Chroma/FAISS RAG)
                        │                                      │
                        ▼                                      ▼
             ┌───────────────────────────────────────────────────────┐
             │  build_system_prompt(profile, facts)                   │
             │  MoodManager → TONE    +    Memory → FACTS              │
             └───────────────────────────────────────────────────────┘
                        │
                        ▼
             [Response generator: LLM (Gemini)]  ── template fallback
                        │
                        ▼
             [CSM-1B TTS on Colab]  ──► spoken audio
                        │
                        ▼
             Play audio + log interaction + maybe store new fact
```

**Four modules — this separation is the story you tell in the paper:**

1. **Emotion recognition** — perception (small Bi-LSTM). Standard-ish.
2. **MoodManager** — *short-term emotional* state. Decays. Controls **HOW** the agent speaks (tone). *Core contribution.*
3. **MemoryManager** — *long-term factual* memory. Persists. Controls **WHAT** the agent can reference (content). *Supporting contribution.*
4. **Naturalistic output** — LLM reply + CSM-1B voice, both steered by the two layers above.

> **The one idea that ties it together:** the same detected emotion that drives MoodManager also *votes on what gets remembered* — emotionally charged moments are stored; flat small-talk is not. Perception feeds both the mood and the memory. At retrieval time, emotional importance is blended with semantic similarity so the agent preferentially recalls what *emotionally mattered*.

---

## 0.5 AI Agent Layer — Architecture, State, and Orchestration

### The Agent Layer

The "agent" is not any single component — it is the **orchestration** of perception, state, and decision-making between raw audio input and spoken output. The LLM (Gemini) and TTS (CSM-1B) are **tools the agent uses**, not the agent itself. Swap Gemini for GPT or Llama — the agent behaves the same because the intelligence lives in the layers above.

```
                   ┌─────────────────────────────────────┐
                   │          AI AGENT LAYER              │
                   │                                      │
Audio input   →    │  Perception  (what emotion?)         │
                   │      ↓                               │
                   │  MoodManager (how are they feeling   │
                   │              over time?)              │
                   │      ↓                               │
                   │  MemoryManager (what do I know       │
                   │                about them?)           │
                   │      ↓                               │
                   │  Decision (what should I say          │
                   │           and how should I say it?)   │
                   │                                      │
                   └─────────────────────────────────────┘
                                    ↓
Output        →    spoken reply with appropriate tone
```

**Three properties that make this an agent, not a chatbot:**

1. **It perceives** — it detects emotion from voice, not just text. A regular chatbot reads words; this agent hears *how* you said them.
2. **It has state** — it maintains two layers of memory (mood + facts) that carry forward across turns and sessions. A regular chatbot treats every turn as independent.
3. **It acts autonomously** — it decides when to shift tone (MoodManager threshold), what's worth remembering (emotion-spike gating), which memories to recall (RAG), and how to speak (prosody). Nobody tells it "be empathetic now."

**Mapping to human abilities:**

| Human ability | Agent equivalent | Component |
|---|---|---|
| Hearing words | Speech-to-text | Whisper ASR |
| Reading someone's emotional tone | Emotion classification | Bi-LSTM |
| Sensing their mood building over time | Temporal mood tracking | MoodManager |
| Remembering important things about them | Long-term semantic memory | MemoryManager + RAG |
| Deciding what to say and how | Prompt construction | `build_system_prompt()` |
| Speaking with the right tone of voice | Expressive speech synthesis | CSM-1B TTS |

### Conversation State — Three Tiers

The agent maintains a **three-tier conversation state**. Each tier has a different lifespan and controls a different aspect of the response.

```
┌───────────────────────────────────────────────────────┐
│                CONVERSATION STATE                      │
│                                                        │
│  TIER 1: TURN STATE (dies after each turn)             │
│  ──────────────────────────────                        │
│  • Current transcript (Whisper output)                 │
│  • Current emotion probabilities (Bi-LSTM output)      │
│  • Current audio features (MFCCs)                      │
│  • Timestamp                                           │
│                                                        │
│  TIER 2: MOOD STATE (decays over turns within session) │
│  ──────────────────────────────                        │
│  • mood_scores for each emotion axis                   │
│  • Current behavior profile / mode                     │
│  • Turn counter for persistence gating                 │
│  • Mood history (for trajectory visualization)         │
│                                                        │
│  TIER 3: MEMORY STATE (persists forever across sessions│)
│  ──────────────────────────────                        │
│  • All stored facts in Chroma/FAISS vector DB          │
│  • Embeddings + importance scores per memory           │
│  • User ID                                             │
│                                                        │
└───────────────────────────────────────────────────────┘
```

**How the tiers differ:**

| Property | Tier 1: Turn | Tier 2: Mood | Tier 3: Memory |
|---|---|---|---|
| Lifespan | One turn | One session (decays between turns) | Forever (across sessions) |
| What it holds | Raw perception output | Smoothed emotional trajectory | Durable facts about the user |
| What it controls | Input to mood + memory | **HOW** the agent responds (tone) | **WHAT** the agent references (content) |
| On restart | Gone | Resets to neutral | Intact on disk |
| Example | `P(anxious)=0.70` | `anxious_score=0.72 → EMPATHETIC mode` | `"User gets anxious before exams"` |

**Cross-session behavior:** when a user returns the next day, Tier 1 and Tier 2 are fresh (the agent doesn't assume they're still upset), but Tier 3 is intact (the agent remembers who they are and what matters to them). This is why the user hears "How did that exam go?" instead of "Hi, how can I help?"

**Example — 4-turn state evolution:**

| Turn | User says | Tier 1 (emotion) | Tier 2 (mood) | Tier 3 (memory) | Agent mode |
|---|---|---|---|---|---|
| 1 | "I have an exam Monday" | anxious: 0.40 | anxious: 0.32 | stores: "exam on Monday" | DEFAULT |
| 2 | "I haven't studied at all" | anxious: 0.60 | anxious: 0.51 | stores: "exam is data structures" | DEFAULT |
| 3 | "I'm going to fail" | anxious: 0.70 | anxious: 0.72 (above threshold, turn 1 of N) | stores: "feels they will fail" (importance: 0.85 — emotional spike) | DEFAULT (persistence not met) |
| 4 | "I don't know where to start" | anxious: 0.65 | anxious: 0.68 (above threshold, turn 2 of N → switches) | no new fact | **EMPATHETIC** |

No single turn triggered empathy. The **pattern across turns** did — that's conversation state in action.

### Orchestrator — Tool Calling Pattern

The agent uses a **deterministic orchestrator** that calls tools in a fixed order every turn. The LLM does not decide which tools to call — the orchestrator does.

```
┌──────────────────────────────────────────────────────┐
│              ORCHESTRATOR (run_turn)                   │
│                                                        │
│  1. calls → transcribe(audio)          [Perception]    │
│  2. calls → predict_emotion(audio)     [Perception]    │
│  3. calls → mood.update(probs)         [Mood Tool]     │
│  4. calls → mood.get_profile()         [Mood Tool]     │
│  5. calls → memory.retrieve(query)     [Memory Tool]   │
│  6. calls → build_system_prompt()      [Decision]      │
│  7. calls → gemini.generate(prompt)    [Generation]    │
│  8. calls → extract_fact(transcript)   [Memory Tool]   │
│  9. calls → memory.store(fact)         [Memory Tool]   │
│ 10. calls → tts.speak(reply, prosody)  [Output Tool]   │
│                                                        │
└──────────────────────────────────────────────────────┘
```

**Tool inventory:**

| Category | Tool | What it does | When called |
|---|---|---|---|
| Perception | `transcribe(audio)` | Whisper ASR → text | Every turn, first |
| Perception | `predict_emotion(audio)` | Bi-LSTM → probability vector | Every turn, parallel with transcription |
| Mood | `mood.update(probs)` | Feeds emotion into EMA decay | Every turn |
| Mood | `mood.get_current_profile()` | Returns behavior mode + tone settings | Before every reply |
| Mood | `mood.get_mood_vector()` | Returns raw scores (for visualization/logging) | For demo display |
| Memory | `memory.retrieve_relevant(query, top_k)` | RAG semantic search of user facts | Before every reply |
| Memory | `memory.store_user_fact(user_id, fact)` | Saves new fact to vector DB | When extract_fact returns something |
| Memory | `memory.reset_user_memory(user_id)` | Wipes all facts for a user | On explicit user request (ethics) |
| Extraction | `extract_fact(transcript, emotion)` | LLM micro-call: "durable fact or NONE?" | After every turn |
| Generation | `gemini.generate(system_prompt, user_msg)` | LLM generates reply text | Every turn |
| Output | `tts.speak(text, prosody_profile)` | CSM-1B → spoken audio | Every turn, last |

**Why a deterministic orchestrator, not agentic tool calling:**
- **Reliable** — same flow every turn, no forgotten tool calls
- **Demo-safe** — no surprises during a live presentation
- **Evaluable** — easy to ablate (turn off MoodManager, compare; turn off Memory, compare)
- **Debuggable** — you know exactly what ran and in what order

The paper can note that Gemini supports native function calling and the architecture could be extended to let the LLM autonomously decide when to store and retrieve (agentic mode), but for reproducibility the deterministic pipeline is used.

### Multi-User Handling

The system runs on a single laptop but supports multiple test users with **isolated state**.

**User selection at startup:**
```
Welcome! Who am I talking to?
> Enter your name: shan
> "Welcome back, Shan! I remember you."
```

**State isolation per user:**

| State tier | Isolation method |
|---|---|
| Tier 1 (turn) | Always fresh — no isolation needed |
| Tier 2 (mood) | Fresh per session — no isolation needed |
| Tier 3 (memory) | Keyed by `user_id` — each user's facts stored and retrieved separately |

**RAG filtering:** every memory record includes a `user_id` field. At retrieval time, the vector DB filters by `user_id` so Shan's memories never leak into Sreelakshmi's responses.

**For the user study:** each participant gets a unique ID; `reset_user_memory(id)` wipes their data between conditions (MoodManager ON vs OFF) to prevent cross-condition contamination.

**Not in scope (future work):** voice-based speaker identification, login/authentication, automatic speaker diarization. A manual name prompt is sufficient for an academic prototype.

### Real-Time Communication — Streaming & Transport

The system lives in three places (laptop, Colab, Gemini API) that must communicate every turn. This subsection specifies how data flows between them and how streaming minimizes perceived latency.

**Where each component runs:**

| Location | Components | Network? |
|---|---|---|
| Laptop (local) | Mic capture, Whisper, Bi-LSTM, MoodManager, MemoryManager (Chroma), `build_system_prompt()`, audio playback | No — Python function calls |
| Gemini API (cloud) | Reply generation, fact extraction | Yes — REST API over internet |
| Google Colab (cloud GPU) | CSM-1B TTS | Yes — ngrok tunnel or WebSocket |

**Per-turn communication flow and latency budget:**

```
User stops speaking
    │
    ▼  LOCAL (~1.5s)
Whisper + Bi-LSTM (parallel via ThreadPoolExecutor)
    │
    ▼  LOCAL (~0.3s)
MoodManager.update + get_profile
MemoryManager.retrieve_relevant (MiniLM embed + Chroma search)
build_system_prompt()
    │
    ▼  NETWORK CALL 1 — Gemini API, streamed (~0.5s to first sentence)
Gemini generates reply, streamed token by token
    │
    ├── Sentence 1 ready → send to Colab TTS immediately
    ├── Sentence 2 ready → queue for Colab TTS
    └── Sentence 3 ready → queue for Colab TTS
    │
    ▼  NETWORK CALL 2 — Colab TTS via WebSocket (~2-3s for first chunk)
CSM-1B generates audio per sentence
    │
    ▼  LOCAL (instant)
Play audio + log interaction
    │
    ▼  NETWORK CALL 3 — Gemini API, async (~0.5-1s, runs in parallel)
extract_fact() → store to memory if durable
```

**Streaming architecture (sentence-level pipelining):**

The key optimization: Gemini streams its reply token by token. Instead of waiting for the full reply, buffer tokens into **complete sentences** and dispatch each sentence to TTS as soon as it's ready. The user hears the first sentence while subsequent sentences are still being generated and synthesized.

```
Gemini streaming:     [sentence 1] ──► [sentence 2] ──► [sentence 3]
                           │                 │                │
                     (dispatched             (dispatched      (dispatched
                      immediately)            while 1 plays)   while 2 plays)
                           │                 │                │
Colab TTS:           [audio 1] ─────► [audio 2] ─────► [audio 3]
                           │                 │                │
User hears:          ▶ plays 1        ▶ plays 2        ▶ plays 3
```

**Latency comparison:**

| Architecture | Time to first audio | Total turn time |
|---|---|---|
| REST + no streaming (baseline) | ~8s | ~8–12s |
| Gemini streaming + REST TTS | ~5s | ~7–9s |
| Gemini streaming + WebSocket TTS | ~3s | ~5–7s |
| Gemini streaming + WebSocket + audio chunk streaming | ~2–3s | ~4–6s |

**Transport choices:**

| Link | Technology | Why |
|---|---|---|
| Laptop → Gemini API | REST with `stream=True` | Native Gemini SDK support, one-flag change |
| Laptop ↔ Colab TTS | WebSocket over ngrok tunnel | Persistent connection avoids per-turn reconnection overhead; enables future audio-chunk streaming |
| Laptop internal | Python function calls | No network needed; all local components run in the same process |

**Sentence-buffering strategy:**

Gemini streams word-by-word fragments. Sending one word to TTS produces choppy audio. Buffer until a sentence boundary (`.` `!` `?` `...`), then dispatch the complete sentence to TTS. This produces natural-sounding speech without waiting for the full reply.

**Parallelism opportunities:**

| What runs in parallel | Why it's safe |
|---|---|
| Whisper ASR ∥ Bi-LSTM emotion prediction | Both read audio independently |
| Colab TTS generation ∥ `extract_fact()` + memory storage | Storing a fact doesn't block speech output |
| Sentence N TTS ∥ Gemini generating sentence N+1 | Pipelining — each sentence is independent |

**Fallback on network failure:**

| Failure | Impact | Fallback |
|---|---|---|
| Gemini API down | Can't generate LLM reply | Template fallback (`generate_reply_template`) |
| Colab disconnects | Can't generate speech | Text-only reply (print to screen) or local `pyttsx3` for basic TTS |
| Internet fully gone | Both Gemini + Colab fail | Template reply + local `pyttsx3` — degraded but functional |
| Chroma/local tools fail | Can't retrieve memories | Empty facts list — agent responds without personal context |

The system **never crashes** — it degrades gracefully. This is critical for a live demo.

**Implementation priority:**

| Technology | Difficulty | Impact | Priority |
|---|---|---|---|
| Gemini `stream=True` | Easy — one flag | Big — perceived latency halves | **Must have** |
| Sentence buffering + dispatch | Medium — ~30 lines | Big — enables sentence-level pipelining | **Must have** (if streaming) |
| WebSocket for Colab TTS | Medium — async code both sides | Medium — saves ~300ms/turn, persistent connection | **Should have** |
| Parallel Whisper ∥ Bi-LSTM | Easy — `ThreadPoolExecutor` | Small — saves ~0.1s | **Nice to have** |
| Audio chunk streaming from CSM-1B | Hard — model may not support it | Big — but requires model-level changes | **Future work** |
| Streaming ASR (real-time Whisper) | Hard — needs `faster-whisper` | Medium — transcribes as user speaks | **Future work** |

---

## 1. Locked Decisions

These were the open choices; here is what this project commits to.

| Decision | Choice | Why |
|---|---|---|
| Where it runs | **Local PC for ~80%; free Colab only for CSM-1B** | Only TTS needs a GPU; everything else is CPU-light |
| Primary dataset | **RAVDESS** to start, **CREMA-D** if time allows | RAVDESS = free, direct download, filename labels, no license wait |
| Emotion model | **Small Bi-LSTM** (hidden ~128, 1 layer) | Matches the small dataset; avoids overfitting; trains on CPU |
| Emotion taxonomy | 4–5 classes: happy, sad, angry, neutral, (+ fearful → "anxious") | Covers the mood axes; RAVDESS has `fearful` as the anxiety stand-in |
| Mood layer | **MoodManager** (EMA of emotion, decay + persistence) | Smooth, memory-ful personality — the novelty |
| Memory layer | **MemoryManager** (Chroma/FAISS RAG + MiniLM embeddings; JSON as initial scaffolding) | Semantic retrieval of relevant facts; emotion-weighted importance scoring; local, CPU-only |
| Response generator | **LLM (Google Gemini, free tier)** + template fallback | Most natural conversation; template kept as safe backup + comparison |
| Voice | **CSM-1B** (Apache-2.0) on Colab GPU | Open-source, citable, free |
| Demo mode | **Offline (safe) + Live (impressive)**, both ready | Play a clip *or* speak into the mic and show mood curves shift live |

**Taxonomy note:** keep the **classifier label set** (what the model predicts) separate from the **mood axis set** (what MoodManager tracks), with an explicit mapping. `Anxious` has no dataset labels, so derive it from `fearful` for the classifier, or treat it as a mood-only axis.

---

## 2. Data & Pre-processing

**Dataset — RAVDESS.** Free, direct download, ~1,440 clips, clean acted speech; the emotion is encoded in the filename, so labeling is trivial. Move to **CREMA-D** (free, 91 speakers) later if you want stronger, speaker-independent results.

Pipeline:
- **Load** to mono, 16 kHz (`librosa.load(path, sr=16000, mono=True)`), float32.
- **Trim** silence (`librosa.effects.trim`, top_db ≈ 25–30).
- **Denoise** lightly (`noisereduce`) — don't over-clean, prosody *is* the signal.
- **Normalize** with RMS (preserves emotional dynamics better than peak).
- **Segment:** use dataset utterance boundaries for training; VAD (Silero/`webrtcvad`) for the live agent.
- **Split speaker-independently** (group by speaker), 70/15/15, fixed seeds saved to disk.

**Deliverable:** a manifest CSV — `utt_id, speaker_id, wav_path, label, split, duration`.

---

## 3. Feature Extraction

- **For the Bi-LSTM (sequence):** MFCCs (13–40 + Δ + ΔΔ) and/or log-mel spectrogram (64–128 bins). Frame: 25 ms window, 10 ms hop. Shape `[T, D]` per clip. **Cache to disk** — recomputing is slow.
- **For the classical baseline (one vector):** prosodic functionals — F0 stats, RMS energy stats, voicing ratio, speaking rate. `openSMILE` (eGeMAPS) gives a research-standard vector in one call and is very citable.

**Deliverable:** cached feature tensors + a loader returning `(sequence_features, prosodic_vector, label)`.

---

## 4. Modeling

**A) Baseline (classical).** Random Forest / XGBoost on the prosodic functional vector. Cheap, interpretable floor; also gives feature-importance insight. Proves deep learning earns its complexity.

**B) CNN baseline (middle comparison).**
```
Input [B, T, D] (same MFCCs as Bi-LSTM)
  → Conv1D (64 filters, kernel 3) → ReLU → Dropout 0.3
  → Conv1D (64 filters, kernel 3) → ReLU → Dropout 0.3
  → Global Average Pooling (collapse time axis into one vector)
  → FC → softmax over emotion classes
```
- Same data pipeline, same MFCCs, same splits as the Bi-LSTM — only the model changes.
- Catches **local patterns** (a pitch spike, an energy burst) but **loses long-range temporal order** because pooling collapses the sequence.
- Proves that learned features beat hand-crafted ones (A → B gap), but temporal modeling still matters (B → C gap).
- ~30 lines of PyTorch, trains on CPU in minutes.

**C) Small Bi-LSTM (primary).**
```
Input [B, T, D] (padded + masked)
  → Bi-LSTM, hidden 128, 1 layer, dropout 0.3
  → attention pooling over time
  → FC → softmax over emotion classes
```
- **Loss:** cross-entropy with **class weights** (imbalance).
- **Optimizer:** Adam, lr 1e-3, ReduceLROnPlateau.
- **Batching:** pad per batch, pass lengths (`pack_padded_sequence`) so padding is ignored.
- **Early stopping** on **validation macro-F1** (accuracy lies under imbalance). Checkpoint the best.
- Trains fine on CPU; small model is the *correct* choice for a small dataset, not just a hardware compromise.

**Output:** per turn, a probability vector, e.g. `P(frustrated)=0.80, P(happy)=0.10, P(neutral)=0.10`.

**Deliverable:** a `predict(audio) → prob_vector` function + a results table (per-class P/R/F1, macro-F1) for the three-tier comparison: classical (RF/XGBoost) → CNN → Bi-LSTM, demonstrating that emotion recognition benefits from both learned representations and explicit temporal sequence modeling.

---

## 5. MoodManager — Short-Term Emotional State (core contribution)

Converts noisy per-turn probabilities into a smooth, persistent mood — no brittle per-turn `if/else`.

**State:** `mood_scores = {frustrated, happy, sad, neutral, anxious}` (all start 0).

**Update rule per turn** (an exponential moving average per emotion, `m_t = γ·m_{t-1} + α·p_t`):
```
for e in emotions:
    mood_scores[e] *= decay_rate            # 1) decay old state (γ ≈ 0.8–0.95)
    mood_scores[e] += alpha[e] * p[e]       # 2) add weighted new evidence
    mood_scores[e]  = clamp(-10, 10)        # 3) bound it
```

**Mood → behavior profile** (soft, with an N-turn *persistence* rule to kill flicker):
- frustrated/sad above threshold for ≥ N turns → **EMPATHETIC**
- anxious above threshold → **CALMING**
- happy above threshold → **POSITIVE_ENCOURAGING**
- else → **DEFAULT** (distress modes outrank positive if several fire)

**API:**
```python
mgr.update(prob_vector)
profile = mgr.get_current_profile()
# → {mode, tone, style:{validation, humor, guidance}, tts:{rate, pitch, expressiveness}}
mgr.get_mood_vector()   # continuous state, for the trajectory plot
```

**Deliverable:** a unit-tested `MoodManager` + a plot of mood curves over a scripted conversation (a strong paper figure showing the smoothing work).

---

## 6. MemoryManager — Long-Term Factual Memory with Semantic RAG (new layer)

Persists durable facts about the user across sessions. **Distinct from MoodManager:** mood decays and controls *tone*; memory persists and controls *content*. The final system uses **semantic RAG** (Retrieval-Augmented Generation) for memory storage and retrieval; a plain JSON version is built first as scaffolding, then replaced.

### 6a. JSON Scaffolding (build first, replace later)

Start with a simple `memories.json` to get the pipeline running end to end before adding RAG complexity. Per user:
```json
{ "demo_user": { "facts": [
    "The user gets anxious before exams.",
    "The user has an exam on Monday."
] } }
```

**Core methods:** `get_user_facts(id)`, `add_user_fact(id, fact)`, `reset_user_memory(id)` (keep the reset — it's your ethics-section wipe button).

**Two touchpoints around the response generator:**
- **READ** (before reply): `get_user_facts(id)` → injected into the prompt.
- **WRITE** (after reply): decide if this turn is worth remembering → `add_user_fact`.

**The problem with JSON at scale:** it loads *all* of a user's facts into every prompt. That breaks once a user has dozens of memories — you can't fit them all, and most are irrelevant to the current message. This is why we upgrade to RAG.

### 6b. RAG Upgrade — Semantic Memory (committed final approach)

RAG replaces "load everything" with "**retrieve only the few facts relevant to what the user just said**."

**RAG = Retrieval-Augmented Generation.** Instead of keyword lookup, you search memories by *meaning*:
1. **Embed** each stored fact into a vector (a number list capturing its meaning) using a small sentence-embedding model.
2. **Store** those vectors in a vector database.
3. At reply time, **embed the current user message** and ask the DB for the closest-in-meaning memories. So "I'm stressed about Monday" retrieves "user gets anxious before exams" — *no shared keywords, matched on meaning.* That semantic match is the capability jump, and the best part to write up in the paper.

**The lightweight, local stack:**
- **Vector DB:** **Chroma** or **FAISS** — local, free, `pip install`, runs on CPU. *Not* Pinecone/Milvus (cloud, adds an internet round-trip and an account — save those for real deployment).
- **Embedding model:** a small `sentence-transformers` model (e.g. MiniLM, ~100–400 MB) — tiny next to your TTS/LLM, runs on CPU.

**Structured memory records:**
```
{ id, user_id, timestamp,
  normalized_fact: "User has an exam on 2026-08-28 and feels unprepared.",
  tags: ["exam", "anxiety"],
  mood_snapshot: <mood vector at time of storage>,
  importance_score: <higher if stored during an emotional spike> }
```

**Where it plugs in.** RAG replaces only the memory READ step: instead of `get_user_facts()` returning everything, `retrieve_relevant(user_message)` returns the top-K semantically closest facts, optionally filtered by recency / importance / tag. The prompt builder, MoodManager, and everything else stay identical.

**Retrieval scoring — emotion-weighted semantic search:**
```
final_score = semantic_similarity × 0.7 + importance_score × 0.3
```
The same emotional spike that drives MoodManager sets the memory's `importance_score`, and RAG retrieval weights by it — so the agent preferentially recalls what *emotionally mattered*, not just what's textually similar. That links RAG back to the project's novelty instead of being a generic bolt-on.

### What to store — the filter (applies to both JSON and RAG)

- **Level 1 — keyword rules:** store if the sentence contains "exam/interview/…". Transparent but brittle (misses "viva," false-alarms on "exam in a video game"). Fine for a rigged demo.
- **Level 2 — LLM extractor (recommended):** a tiny separate LLM call — *"output one durable fact worth remembering, or NONE."* Catches synonyms, ignores small talk, phrases cleanly.
- **Level 3 — name in the paper:** deduplication, updating-over-storing ("exam Monday" → "exam went well"), recency/priority.

**Rule of thumb:** store if it's **durable and about the user** (events with a future, stable preferences, relationships, recurring feelings). Skip momentary reactions ("lol", "ok").

**The project-specific twist:** let emotion vote. Store a turn if the LLM flags a durable fact **OR** MoodManager shows a strong emotional spike — so memory fills with the moments that *emotionally mattered*, which is exactly what an emotional companion should remember. And enrich the fact with mood ("The user tends to feel anxious before exams") rather than a bare quote.

**Timing & hardware:** RAG runs on **CPU**, needs **no GPU**, and adds only a *fraction of a second* per turn (embed + search are milliseconds). Because the vector DB is **local**, it adds **no new internet dependency**.

**Privacy:** store summaries, not verbatim quotes; avoid names/addresses; expose `reset_user_memory`.

**Deferred to real deployment (name in the paper as future work):** cloud vector DB (Pinecone/Milvus), a fine-tuned BERT/NER extractor with date normalization, and self-hosted embedding at scale.

**Deliverable:** a `MemoryManager` class with JSON scaffolding first, then Chroma/FAISS-backed RAG with `embed()`, `retrieve_relevant(query, top_k)`, and `extract_fact()` (LLM + emotion-spike override) + the reset function. JSON version kept as offline fallback.

---

## 7. Response Generation — LLM, mood- and memory-steered

Both layers meet in **one merged prompt**:
```python
system_prompt = build_system_prompt(behavior_profile, memory_facts)
# MoodManager → TONE ("soft, validate, no jokes, step-by-step")
# Memory      → FACTS ("they had an exam Monday they were anxious about")
```

**Primary: LLM (Google Gemini, free tier).** Runs from the local PC (no GPU needed), so it won't compete with CSM-1B for Colab memory.
- Isolate the provider call in one `_call_llm()` function → swapping models is a one-function edit.
- **Key handling:** read from an environment variable (`GEMINI_API_KEY`), never hardcode. (You create the account and key yourself.)
- **Fallback:** wrap the call in try/except → on any network/rate-limit error, fall back to the template reply so a live demo never crashes.

**Fallback/comparison: template generator.** Deterministic, offline, free. Keep it — it's your safe demo backup *and* a comparison condition for the paper (does LLM fluency actually beat the template on perceived empathy?).

**The point that must hold:** the LLM only sounds empathetic because `build_system_prompt()` told it to. The model is the mouth; MoodManager + Memory are the brain. If replies go generic, fix the prompt, not the model.

**Deliverable:** `response_generator.py` with `generate_reply_template`, `generate_reply_llm`, shared `build_system_prompt`, and a `USE_LLM` switch. *(Already drafted.)*

---

## 8. TTS Output — CSM-1B on Colab

- Colab GPU runtime; `pip install` CSM deps; clone repo; load 1B checkpoint (cite Apache-2.0).
- **Architecture split:** capture mic + run the light pipeline **locally**; send only the reply *text* to Colab for CSM-1B. Keep the microphone on the machine in the room.

### Two approaches to natural-sounding voice

**Approach A — Audio Context Conditioning (no training, quick start)**

CSM-1B is a base model — it has no fixed voice identity. Pass a **reference audio clip** as context each turn and it mimics that voice. Prepare a few clips per mood (calm voice, empathetic voice, encouraging voice) and pass the mood-matched one as context.

- Works out of the box, no training needed.
- **Limitation:** voice can drift between turns because the base model doesn't have built-in voice consistency. Speaker ID tokens help within a conversation but not across separate generations.
- Use this to get the pipeline running first.

**Approach B — LoRA Fine-tuning on Human Audio (committed final approach)**

Fine-tune CSM-1B using LoRA so it **always** sounds like a consistent, natural human voice — no reference clip needed at inference.

**What you need:**

| Requirement | Details |
|---|---|
| Audio clips | 50–200 clips from one speaker, clean audio, varied emotions |
| Transcripts | Text of what's said in each clip |
| Compute | Free Google Colab GPU (same runtime used for inference) |
| Library | Unsloth (free, has CSM-1B LoRA notebooks ready) |
| Training time | ~1–2 hours on Colab |

**Recommended training data — Elise dataset:**

The Elise dataset (~1,200 samples) embeds emotion tags like `<sigh>`, `<laughs>`, and `<whisper>` into transcripts, triggering expressive audio that matches the emotion. Fine-tuning on this teaches CSM-1B to produce emotional speech natively — sighs when sad, laughs when happy — which ties directly into MoodManager's behavior profiles.

**How emotion tags connect to MoodManager:**

```
MoodManager → EMPATHETIC mode
    │
    ▼
Response generator inserts emotion tags based on mood profile:
    "I hear you, that sounds really tough <sigh>"
    │
    ▼
Fine-tuned CSM-1B → naturally sighs, speaks softly
    │
    ▼
User hears genuine emotional delivery
```

The mood profile determines which tags to inject:

| Mood mode | Tags injected | Vocal effect |
|---|---|---|
| EMPATHETIC | `<sigh>`, `<softly>` | Gentle, understanding tone |
| CALMING | `<slowly>`, `<softly>` | Measured, steady delivery |
| POSITIVE_ENCOURAGING | `<laughs>`, `<brightly>` | Upbeat, warm energy |
| DEFAULT | none | Normal conversational tone |

**Comparison of approaches:**

| | Approach A (audio conditioning) | Approach B (LoRA fine-tuning) |
|---|---|---|
| Voice consistency | Can drift between turns | Stable — always same voice |
| Emotional range | Limited to reference clips provided | Rich — learns from training data |
| Emotion tags (`<sigh>`, `<laughs>`) | Not supported | Supported if trained on tagged data |
| Setup effort | Easy — provide clips | Medium — run training notebook |
| Cost | Free | Free (Colab) |
| Quality | Good | Better |

**Phased implementation:**

| Phase | What | When |
|---|---|---|
| 8a | Audio context conditioning — get pipeline running with mood-matched reference clips | Week 6 |
| 8b | LoRA fine-tuning on Elise dataset — consistent emotional voice | Week 6–7 |
| Optional | Record 10–15 clips of a custom voice and fine-tune for a unique project voice | If time allows |

**Alternative training data sources:**

| Source | Pros | Cons |
|---|---|---|
| Elise dataset (recommended) | Emotion-tagged, professional, ~1,200 samples | Not a custom voice |
| Record yourself | Your voice, full control | Need quiet room, consistency |
| RAVDESS clips (already downloaded) | Acted emotions, multiple speakers | Acted style, not natural conversation |

**Prosody control reality check:** CSM-1B does not expose explicit rate/pitch/expressiveness knobs. The fine-tuning approach with emotion tags replaces the need for those knobs — emotional delivery is learned from the training data rather than parametrically controlled. For any remaining prosody adjustments, use light post-hoc pitch/rate shift (`librosa`/`sox`).

**Deliverable:** a LoRA-fine-tuned CSM-1B checkpoint on Colab that produces consistent, emotionally expressive speech driven by mood-profile-injected tags, with audio context conditioning as fallback.

---

## 9. Evaluation

**A) Model (quantitative):** accuracy, per-class P/R/F1, **macro-F1** (headline), UAR; confusion matrix (spot Angry↔Frustrated, Sad↔Neutral). Baseline vs Bi-LSTM. Train on train, tune on val, touch test once, speaker-independent splits.

**B) MoodManager (system):** feed scripted emotion sequences; show mood curves smoothing noisy input and modes switching without flicker. Ablate decay and persistence.

**C) MemoryManager (system):** show a two-session exam scenario — session 1 stores "anxious about Monday's exam," session 2 the agent recalls it unprompted. Report extractor precision (did it store the right things, skip small talk).

**D) User study (the personality layer):** small pilot, **MoodManager+Memory ON vs a static baseline**. Likert: perceived emotional understanding, appropriateness of personality shift, comfort/safety. This comparison is what makes the study meaningful. Consent + IRB if recording real people.

---

## 10. Demo Plan

- **Offline (safe backup):** run a pre-recorded clip through the full pipeline. Reliable, no surprises.
- **Live (best moment):** speak into the mic across 3–4 turns with *rising frustration*; the audience watches the `frustrated` score climb until the agent flips to **EMPATHETIC**, then speak calmly and watch it decay back. That live mood trajectory *is* your MoodManager — basically impossible to fake with a recording.
- **Memory moment:** reference something from an earlier turn and show the agent recall it — proves the long-term layer live.
- Show the **mood scores updating on screen** each turn.

Have both offline and live ready; do live if the room cooperates, keep offline as backup.

---

## 11. Reproducibility & Repo Structure

```
emotion-voice-agent/
├── data/                 # manifests, split indices (not raw audio)
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── datasets.py            # loaders, padding, masking
│   ├── models/
│   │   ├── baseline.py        # RF/XGB
│   │   ├── cnn.py             # 1D CNN (middle baseline)
│   │   └── lstm.py            # small Bi-LSTM + attention
│   ├── mood_manager.py        # short-term emotional state (core)
│   ├── memory_manager.py      # long-term facts + extractor (new layer)
│   ├── response_generator.py  # LLM + template, merged prompt
│   └── tts_csm.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_train_eval.ipynb
│   ├── 03_colab_demo.ipynb    # CSM-1B inference loop
│   └── 04_tts_finetune.ipynb  # CSM-1B LoRA fine-tuning on Colab
├── configs/                   # YAML: features, model, mood + memory params
├── results/                   # metrics, figures, mood-trajectory plots
├── tests/                     # incl. MoodManager + MemoryManager
├── memories.json              # local long-term store, JSON version (gitignore it)
├── vector_store/              # local Chroma/FAISS index, RAG version (gitignore it)
├── requirements.txt
└── README.md
```
> `memory_manager.py` holds both the JSON scaffolding (Phase 6a) and the Chroma/FAISS RAG store (Phase 6b — the committed final version); build JSON first, swap to RAG once the loop works. JSON is kept as offline fallback.
- Pin versions, set seeds, log configs per run.
- **`.gitignore` your API key, `memories.json`, and `vector_store/`** (keys must never be committed; emotional data stays private).

---

## 12. Timeline (adjust to your term)

| Phase | Work | Rough time |
|---|---|---|
| 1 | Scope, RAVDESS download, speaker splits | Week 1 |
| 2 | Preprocessing + manifest | Week 1–2 |
| 3 | Feature extraction + caching | Week 2 |
| 4 | Baseline → small Bi-LSTM | Week 3 |
| 5 | MoodManager + trajectory plots | Week 4 |
| 6a | **MemoryManager JSON scaffolding + emotion-driven extractor** | Week 4–5 |
| 7 | LLM response gen (Gemini) + template fallback | Week 5 |
| 8a | CSM-1B Colab loop with audio context conditioning | Week 6 |
| 6b | **RAG upgrade (Chroma/FAISS + MiniLM embeddings + emotion-weighted retrieval)** — *after the full loop runs* | Week 6–7 |
| 8b | **CSM-1B LoRA fine-tuning on Elise dataset** — consistent emotional voice with emotion tags | Week 6–7 |
| 9 | Eval (model + system + user pilot) | Week 7 |
| — | Write-up, figures, ablations | Week 7–8 |

Build a **thin end-to-end path early** (even a dummy model) so integration bugs surface before you're time-boxed.

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Speaker leakage inflates accuracy | Speaker-independent splits from day one |
| `Anxious` has no dataset labels | Derive from `fearful`, or keep it a mood-only axis |
| Class imbalance skews the model | Class weights; report macro-F1 & UAR |
| Big LSTM overfits small data | Use the small Bi-LSTM (it's the correct choice, not a compromise) |
| MoodManager flickers between modes | Decay + N-turn persistence |
| Memory stores junk / misses key facts | LLM extractor + emotion-spike override; report extractor precision |
| API key leaks | Env var only; `.gitignore`; never in notebook |
| Live demo internet drops | try/except → template fallback; offline demo ready |
| CSM-1B lacks prosody knobs | LoRA fine-tuning on emotion-tagged data (Elise) replaces parametric knobs; emotion tags in response text trigger learned vocal expressions; audio context conditioning as fallback |
| CSM-1B fine-tuning fails or voice quality is poor | Audio context conditioning (Approach A) works without any training; post-hoc pitch/rate shift (`librosa`/`sox`) for remaining adjustments |
| Voice drifts between turns (base model) | Fine-tuning locks in a consistent voice; if using audio conditioning only, provide the same reference clip every turn |
| Storing sensitive emotional data | Summaries not quotes; no names/addresses; `reset_user_memory` |
| RAG scope-creep stalls the project | Ship JSON memory first; add local Chroma/FAISS RAG only after the full loop works; keep cloud vector DB as future work |

---

### One-line summary
Perception (small Bi-LSTM) feeds **two memories** — a short-term **MoodManager** that decays and sets *tone*, and a long-term **MemoryManager** backed by **local semantic RAG** (Chroma/FAISS + MiniLM embeddings, CPU-only) that persists and supplies *facts* — which merge into one prompt for a Gemini reply spoken by a **LoRA-fine-tuned CSM-1B** (trained on emotion-tagged human speech for natural vocal expression), with the detected emotion itself deciding what's worth remembering and emotion-weighted retrieval ensuring the agent preferentially recalls what *emotionally mattered*, not just what's textually similar.
