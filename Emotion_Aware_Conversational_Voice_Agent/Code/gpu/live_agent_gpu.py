"""
live_agent_gpu.py - the live agent turn loop (GPU demo build, step 9a). No microphone yet (step 9b).

LiveAgent(cfg, asr=, emotion=, llm=, voice=, memory=, logger=, safety=, mood_factory=, extract_fn=)
Every part is injected, so tests run on stubs. This module imports no torch, no model and nothing from gpu/
at import time: load_real_parts() is the only place that does (and AgentVoice imports tts_gpu on first use).

run_turn(user_id, audio) - one fixed, deterministic order; the LLM never picks a tool or a step:
     0 consent    once per session per user (safety.consent_given). No consent -> refused: nothing is run,
                  recorded, logged or stored.
     1 asr        audio -> transcript. Silence-gated by ASR (asr.min_speech_seconds) or no word characters ->
                  the user is asked to repeat (REPEAT_PROMPT); no text is invented, nothing is stored or logged.
     2 safety     safety.check_crisis(transcript). Match -> SAFE_RESPONSE with SAFE_STRATEGY's voice (crisis=True);
                  emotion, mood, memory, LLM and fact extraction are skipped; no memory; the turn IS logged.
     3 emotion    probabilities (4 classes); confidence = top-1, margin = top-1 minus top-2.
     4 abstain    confidence < brain.tau_conf (gpu_config.yaml; the MoodManagerV2 built from it holds the value).
     5 mood       MoodManagerV2.update: on abstain the mood holds, and after brain.check_in_after abstains in a
                  row outside DEFAULT it raises check_in. -> strategy, check_in, reason, tts_tags (strategy only).
     6 memory     memory.retrieve_relevant(user_id, transcript, memory.top_k): this user's facts only.
     7 prompt     response_generator.build_system_prompt(strategy, facts, check_in). The per-turn emotion label and
                  the probabilities never reach the LLM (P3).
     8 llm        llm.stream_reply(...) sentence by sentence (the "llm" stage times the first sentence); if it
                  fails before any sentence, response_generator.generate_reply_template (strategy, check_in).
     9 tts        voice.speak(stream, strategy): the remaining LLM sentences feed CSM as they arrive
                  (tts_gpu.speak_iter_with_fallback); CSM failure -> pyttsx3 -> printed text. Voice code receives
                  the strategy only (P6).
    10 extract    fact_extractor.extract_fact(USER transcript, llm_fn, emotional_spike) - never the reply (P5);
                  stored with source "user_turn" and the turn_id. importance from memory_manager.importance_from_turn
                  (an abstained turn counts as intensity 0, so it does not raise importance).
    11 log        one JSONL line (LOG_FIELDS) in results/logs/session_<user_id>_<stamp>.jsonl
                  (memory_manager.session_log_filename, so delete_all_my_data finds it).

Per-user isolation: mood state, conversation history, consent check, turn counter, memory and log file are all
keyed by user_id. delete_user_data(user_id) = memory_manager.delete_all_my_data (memories, consent record, that
user's session logs) + this session's state for that user.

CLI (one turn from a wav file; asks for consent first if the user has none on record):
    python gpu/live_agent_gpu.py --wav data/my_voice_v2/01_neutral.wav --user sathwik --preset gpu_24gb
    python gpu/live_agent_gpu.py --wav some.wav --user test --preset smoke_cpu --no-play
"""

import argparse
import json
import re
import sys
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# scripts/ only: none of these import torch or anything from gpu/
import fact_extractor  # noqa: E402
import fallback_tts  # noqa: E402
import response_generator  # noqa: E402  (loads Code/.env; never prints it)
import safety as safety_module  # noqa: E402
from memory_manager import (  # noqa: E402
    MemoryManager,
    importance_from_turn,
    load_memory_config,
    session_log_filename,
)
from mood_manager import MoodManagerV2, tts_tags_for  # noqa: E402

STAGES = ("consent", "asr", "safety", "emotion", "abstain", "mood", "memory", "prompt", "llm", "tts", "extract", "log")
REPEAT_PROMPT = "Sorry, I didn't catch that. Could you say it again?"
REPEAT_STRATEGY = "NEUTRAL"
USER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")   # becomes part of a log file name: no path separators

LOG_FIELDS = (
    "user_id", "session_id", "turn_id", "timestamp", "status", "transcript", "crisis", "crisis_phrase",
    "emotion_probs", "confidence", "margin", "abstained", "mood_scores", "turns_above", "mode", "strategy",
    "check_in", "reason", "mood_params", "retrieved_memories", "system_prompt", "llm_backend", "llm_model",
    "llm_fell_back", "llm_error", "reply", "voice_path", "voice_error", "time_to_first_audio", "tts_tags",
    "stored_memory_id", "fact", "timings",
)


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------

def check_user_id(user_id) -> str:
    if not isinstance(user_id, str) or not USER_ID_RE.fullmatch(user_id):
        raise ValueError(f"user_id must be letters, digits, '_', '.' or '-' (not starting with '.', '_' or '-'), "
                         f"got {user_id!r}")
    return user_id


def top1_margin(probs) -> tuple:
    """(confidence = top-1 probability, margin = top-1 minus top-2)."""
    ranked = sorted((float(v) for v in probs.values()), reverse=True)
    return ranked[0], ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)


def has_words(text) -> bool:
    return bool(re.search(r"\w", str(text or "")))


def _json_default(o):
    if hasattr(o, "item"):           # numpy scalar
        return o.item()
    if hasattr(o, "tolist"):         # numpy array
        return o.tolist()
    return str(o)


def _load_audio(audio, sr=None):
    """(array, sr) from a wav path, an (array, sr) tuple, or an array plus sr."""
    if isinstance(audio, (str, Path)):
        import soundfile as sf

        y, sr = sf.read(str(audio), dtype="float32")
        return y, sr
    if isinstance(audio, tuple) and len(audio) == 2:
        return audio
    if sr is None:
        raise ValueError("sr is required when audio is an array")
    return audio, sr


class Stages:
    """Seconds per stage in the order they ran (a dict keeps insertion order)."""

    def __init__(self):
        self.times = {}

    @contextmanager
    def __call__(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] = round(time.perf_counter() - t0, 4)


class TurnLogger:
    """One JSONL file per user per session: <log_dir>/session_<user_id>_<session stamp>.jsonl."""

    def __init__(self, log_dir, session_start=None):
        self.log_dir = Path(log_dir)
        self.session_start = session_start or datetime.now(timezone.utc)
        self.session_id = self.session_start.strftime("%Y%m%dT%H%M%SZ")

    def path_for(self, user_id) -> Path:
        return self.log_dir / session_log_filename(user_id, self.session_start)

    def log(self, user_id, record) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(user_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        return path


class AgentVoice:
    """Voice port: speak(text or iterable of sentences, strategy, crisis=False) -> {"path", "error", "stats", ...}.
    gpu/tts_gpu is imported on first use: a string goes to speak_with_fallback, a stream to speak_iter_with_fallback.
    csm = a loaded tts_gpu.CSMVoice, or None (fallback voice only). Receives the strategy only, never emotion (P6)."""

    def __init__(self, csm=None, play=True):
        self.csm = csm
        self.play = bool(play)

    def speak(self, sentences, strategy, crisis=False) -> dict:
        import tts_gpu

        if isinstance(sentences, str):
            return tts_gpu.speak_with_fallback(sentences, strategy, voice=self.csm, crisis=crisis, play=self.play)
        return tts_gpu.speak_iter_with_fallback(sentences, strategy, voice=self.csm, crisis=crisis, play=self.play)


# ---------------------------------------------------------------------------------------------
# LiveAgent
# ---------------------------------------------------------------------------------------------

class LiveAgent:
    def __init__(self, cfg, *, asr, emotion, llm, voice, memory, logger, safety=safety_module,
                 mood_factory=None, consent_path=None, extract_fn=fact_extractor.extract_fact):
        self.cfg = cfg
        self.asr, self.emotion, self.llm, self.voice = asr, emotion, llm, voice
        self.memory, self.logger, self.safety = memory, logger, safety
        self.mood_factory = mood_factory or (lambda: MoodManagerV2.from_config(cfg["brain"]))
        self.consent_path = consent_path or getattr(safety, "DEFAULT_CONSENT_PATH", None)
        self.extract_fn = extract_fn
        self.top_k = int(cfg["memory"]["top_k"])
        self.history_turns = int(cfg["llm"]["history_turns"])
        self.session_id = getattr(logger, "session_id", None) or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._users = {}

    # -- per-user state --

    def _state(self, user_id) -> dict:
        if user_id not in self._users:
            self._users[user_id] = {"mood": self.mood_factory(), "history": deque(maxlen=self.history_turns),
                                    "turns": 0, "consented": False}
        return self._users[user_id]

    def _consent_ok(self, user_id) -> bool:
        """Checked against the consent file once per session; a refusal is re-checked next turn."""
        st = self._users.get(user_id)
        if st and st["consented"]:
            return True
        if not self.safety.consent_given(user_id, self.consent_path):
            return False
        self._state(user_id)["consented"] = True
        return True

    def delete_user_data(self, user_id) -> dict:
        """Memories, consent record and session logs (memory_manager.delete_all_my_data), plus this session's mood,
        history and consent state for the user. Returns delete_all_my_data's report."""
        check_user_id(user_id)
        report = self.memory.delete_all_my_data(user_id, self.logger.log_dir, self.consent_path)
        self._users.pop(user_id, None)
        return report

    # -- speaking --

    def _speak(self, text_or_stream, strategy, crisis=False) -> dict:
        """voice.speak, and if the voice itself raises (it should not), the fallback voice directly."""
        if isinstance(text_or_stream, str):
            text, stream = text_or_stream, None
        else:
            spoken = []
            stream = (spoken.append(s) or s for s in text_or_stream)
        try:
            rec = self.voice.speak(text_or_stream if stream is None else stream, strategy, crisis=crisis)
            return dict(rec or {})
        except Exception as e:  # noqa: BLE001
            if stream is not None:
                for _ in stream:                 # drain: each item lands in `spoken` once
                    pass
                text = " ".join(spoken)
            return {"path": fallback_tts.speak(text), "error": f"voice crashed: {type(e).__name__}: {e}",
                    "stats": None}

    # -- reply --

    def _llm_fn(self):
        return self.llm.as_llm_fn() if hasattr(self.llm, "as_llm_fn") else self.llm.complete

    def _reply_sentences(self, system_prompt, history, transcript, strategy, check_in, info):
        """LLM sentences; the template if the LLM fails (or says nothing) before its first sentence. Never raises."""
        t0 = time.perf_counter()
        got = False
        try:
            for s in self.llm.stream_reply(system_prompt, history, transcript, strategy):
                got = True
                info["sentences"].append(s)
                yield s
            last = getattr(self.llm, "last_reply", None) or {}
            info.update(backend=last.get("backend"), model=last.get("model"), fell_back=bool(last.get("fell_back")))
            if last.get("errors"):
                info["error"] = "; ".join(str(e.get("error", e)) for e in last["errors"])
        except Exception as e:  # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            if got:                          # part of the reply is already out: stop, do not start over
                info["fell_back"] = True
        if not got:
            text = response_generator.generate_reply_template(strategy, transcript, check_in)
            info.update(backend="template", model=None, fell_back=True)
            info["sentences"].append(text)
            yield text
        info["seconds"] = round(time.perf_counter() - t0, 3)

    # -- the turn --

    def _record(self, user_id, turn_id, transcript, status) -> dict:
        rec = {k: None for k in LOG_FIELDS}
        rec.update(user_id=user_id, session_id=self.session_id, turn_id=turn_id, status=status,
                   timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"), transcript=transcript,
                   crisis=False, abstained=False, check_in=False, retrieved_memories=[], tts_tags=[])
        return rec

    def _voice_fields(self, rec, voice_rec):
        stats = voice_rec.get("stats") or {}
        rec.update(voice_path=voice_rec.get("path"), voice_error=voice_rec.get("error"),
                   time_to_first_audio=stats.get("time_to_first_audio") if isinstance(stats, dict) else None)

    def _finish(self, rec, stages, t_start) -> dict:
        rec["timings"] = {**stages.times, "total": round(time.perf_counter() - t_start, 4)}   # "log" itself not in it
        with stages("log"):
            path = self.logger.log(rec["user_id"], {k: rec[k] for k in LOG_FIELDS})
        rec["log_file"] = str(path) if path else None
        return rec

    def run_turn(self, user_id, audio, sr=None) -> dict:
        check_user_id(user_id)
        t_start = time.perf_counter()
        stages = Stages()

        # 0. consent
        with stages("consent"):
            consented = self._consent_ok(user_id)
        if not consented:
            return {"status": "refused", "user_id": user_id,
                    "reason": "no consent on record: nothing was recorded or stored"}
        st = self._state(user_id)

        # 1. ASR
        with stages("asr"):
            y, sr = _load_audio(audio, sr)
            asr_out = self.asr.transcribe(y, sr=sr)
            transcript = " ".join(str(asr_out.get("text") or "").split())
        if asr_out.get("skipped") or not has_words(transcript):
            voice_rec = self._speak(REPEAT_PROMPT, REPEAT_STRATEGY)
            return {"status": "repeat", "user_id": user_id, "reply": REPEAT_PROMPT,
                    "asr_reason": asr_out.get("reason") or "no words in the transcript",
                    "voice_path": voice_rec.get("path"), "timings": dict(stages.times)}

        st["turns"] += 1
        turn_id = f"{self.session_id}-{st['turns']}"

        # 2. crisis check
        with stages("safety"):
            crisis = self.safety.check_crisis(transcript)
        if crisis.get("matched"):
            rec = self._record(user_id, turn_id, transcript, "crisis")
            strategy = self.safety.SAFE_STRATEGY
            rec.update(crisis=True, crisis_phrase=crisis.get("phrase"), strategy=strategy,
                       reason=f"crisis phrase {crisis.get('phrase')!r} matched: fixed safe response, "
                              "emotion / mood / memory / LLM skipped",
                       reply=self.safety.SAFE_RESPONSE, tts_tags=tts_tags_for(strategy))
            with stages("tts"):
                voice_rec = self._speak(self.safety.SAFE_RESPONSE, strategy, crisis=True)
            self._voice_fields(rec, voice_rec)
            return self._finish(rec, stages, t_start)      # no memory, no history

        rec = self._record(user_id, turn_id, transcript, "ok")

        # 3. emotion
        with stages("emotion"):
            emo = self.emotion.predict_array(y, sr)
            probs = {c: float(v) for c, v in emo["probs"].items()}
            confidence, margin = top1_margin(probs)

        # 4. abstain gate (threshold: brain.tau_conf, held by the MoodManagerV2 built from the config)
        mood = st["mood"]
        with stages("abstain"):
            abstained = confidence < float(mood.tau_conf)

        # 5. mood (holds on abstain; check_in after brain.check_in_after abstains outside DEFAULT)
        with stages("mood"):
            profile = mood.update(probs, confidence)
            if bool(mood.abstained) != abstained:
                raise RuntimeError("abstain gate and MoodManagerV2 disagree; they must use the same tau_conf")
        vector = mood.get_mood_vector()
        strategy, check_in = profile["strategy"], profile["check_in"]
        rec.update(emotion_probs=probs, confidence=confidence, margin=margin, abstained=abstained,
                   mood_scores=vector, turns_above=vector.get("turns_on_target"), mode=profile["mode"],
                   strategy=strategy, check_in=check_in, reason=profile["reason"], tts_tags=profile["tts_tags"],
                   mood_params={k: getattr(mood, k, None) for k in MoodManagerV2.CONFIG_KEYS})

        # 6. memory (this user's facts only)
        with stages("memory"):
            hits = self.memory.retrieve_relevant(user_id, transcript, self.top_k)
        rec["retrieved_memories"] = [{k: h[k] for k in ("memory_id", "similarity", "importance", "score")}
                                     for h in hits]

        # 7. prompt: strategy, facts, check_in - never the emotion label or probabilities
        with stages("prompt"):
            system_prompt = response_generator.build_system_prompt(strategy, [h["fact"] for h in hits] or None,
                                                                    check_in)
        rec["system_prompt"] = system_prompt

        # 8. LLM (first sentence) -> 9. TTS (the rest of the stream feeds the voice as it arrives)
        info = {"sentences": [], "backend": None, "model": None, "fell_back": False, "error": None, "seconds": None}
        with stages("llm"):
            sentences = self._reply_sentences(system_prompt, list(st["history"]), transcript, strategy, check_in,
                                              info)
            first = next(sentences)                     # never StopIteration: the template stands in
        with stages("tts"):
            voice_rec = self._speak(chain([first], sentences), strategy)
            for _ in sentences:                         # a voice that stopped early: still finish the reply text
                pass
        reply = " ".join(info["sentences"])
        rec.update(reply=reply, llm_backend=info["backend"], llm_model=info["model"],
                   llm_fell_back=info["fell_back"], llm_error=info["error"])
        self._voice_fields(rec, voice_rec)

        # 10. fact extraction from the USER transcript only
        with stages("extract"):
            importance, spike = importance_from_turn(probs, abstained, self.cfg["memory"])
            fact = self.extract_fn(transcript, self._llm_fn(), emotional_spike=spike)
            if fact:
                snapshot = {"strategy": strategy, "mode": profile["mode"], "abstained": abstained, "mood": vector}
                rec["stored_memory_id"] = self.memory.store(user_id, fact, turn_id, snapshot, importance)
                rec["fact"] = fact

        st["history"].append((transcript, reply))
        return self._finish(rec, stages, t_start)


# ---------------------------------------------------------------------------------------------
# Real components (the only place torch and the models are imported)
# ---------------------------------------------------------------------------------------------

def load_real_parts(preset=None, play=True):
    """(cfg, parts) for LiveAgent(cfg, **parts): ASR, SER, LLM and CSM loaded and warmed up once.
    The CSM voice is loaded only when tts.backend is csm; if it fails to load, the agent uses the fallback voice."""
    from device import setup
    import asr_gpu
    import llm_gpu
    import ser_gpu

    cfg, device, dtype = setup(preset)
    asr = asr_gpu.ASR(cfg, device, dtype).load().warm_up()
    emotion = ser_gpu.SERModel(cfg, device, dtype)            # loads + warms up in __init__
    llm = llm_gpu.LLM(cfg, device, dtype).load().warm_up()
    csm = None
    if cfg["tts"].get("backend") == "csm":
        try:
            import tts_gpu

            csm = tts_gpu.CSMVoice.from_preset(preset)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: CSM voice not loaded ({type(e).__name__}: {e}); using the fallback voice.",
                  file=sys.stderr)
    mconf = load_memory_config()
    parts = {"asr": asr, "emotion": emotion, "llm": llm, "voice": AgentVoice(csm, play=play),
             "memory": MemoryManager(mconf["memory_file"], cfg["memory"]), "logger": TurnLogger(mconf["log_dir"])}
    return cfg, parts


def _print_turn(r):
    print(f"\nstatus     : {r['status']}")
    if r["status"] == "refused":
        print(f"reason     : {r['reason']}")
        return
    if r["status"] == "repeat":
        print(f"asr        : {r['asr_reason']} -> asked to repeat ({r['voice_path']})")
        return
    print(f"transcript : {r['transcript']}")
    if r["crisis"]:
        print(f"CRISIS     : phrase {r['crisis_phrase']!r} -> safe response, NEUTRAL voice")
    else:
        probs = "  ".join(f"{k}={v:.2f}" for k, v in r["emotion_probs"].items())
        print(f"emotion    : {probs}  confidence={r['confidence']:.2f}  margin={r['margin']:.2f}"
              + ("  ABSTAINED" if r["abstained"] else ""))
        print(f"mood       : mode={r['mode']}  strategy={r['strategy']}" + ("  CHECK_IN" if r["check_in"] else ""))
        print(f"memories   : {len(r['retrieved_memories'])} retrieved")
        print(f"llm        : {r['llm_backend']} ({r['llm_model']})" + (f"  error={r['llm_error']}" if r["llm_error"] else ""))
    print(f"reason     : {r['reason']}")
    print(f"reply      : {r['reply']}")
    print(f"voice      : {r['voice_path']}  time_to_first_audio={r['time_to_first_audio']}"
          + (f"  error={r['voice_error']}" if r["voice_error"] else ""))
    if r.get("fact"):
        print(f"stored     : {r['fact']}  ({r['stored_memory_id']})")
    print("timings    : " + "  ".join(f"{k} {v:.3f}s" for k, v in r["timings"].items()))
    print(f"log        : {r['log_file']}")


def _main():
    ap = argparse.ArgumentParser(description="Live agent: one turn from a wav file (GPU demo build, step 9a).")
    ap.add_argument("--wav", required=True, help="16 kHz (or any rate) wav file with one user turn")
    ap.add_argument("--user", required=True, help="user id (letters, digits, '_', '.', '-')")
    ap.add_argument("--preset", help="preset name (default: $AGENT_PRESET, else smoke_cpu)")
    ap.add_argument("--no-play", action="store_true", help="do not play audio (CSM audio is still generated)")
    args = ap.parse_args()
    try:
        check_user_id(args.user)
    except ValueError as e:
        ap.error(str(e))
    if not Path(args.wav).is_file():
        ap.error(f"wav file not found: {args.wav}")

    # Consent before anything is loaded or recorded.
    if not safety_module.consent_given(args.user) and not safety_module.ask_consent(args.user):
        print("No consent: nothing was run, recorded or stored.")
        return 1

    t0 = time.perf_counter()
    try:
        cfg, parts = load_real_parts(args.preset, play=not args.no_play)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"preset {cfg['preset']}: parts loaded in {time.perf_counter() - t0:.1f}s  "
          f"(tau_conf {cfg['brain']['tau_conf']}, check_in_after {cfg['brain']['check_in_after']})")
    agent = LiveAgent(cfg, **parts)
    _print_turn(agent.run_turn(args.user, args.wav))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
