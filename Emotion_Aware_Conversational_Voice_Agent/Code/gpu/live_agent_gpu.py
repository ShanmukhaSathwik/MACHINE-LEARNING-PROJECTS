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

Step 9b:
    Emotion model = ser.checkpoint (T3) through ser_gpu.load_ser_with_fallback; the pretrained ser.model only if the
    checkpoint is missing or fails to load (one warning). Every log line records ser_model {name, fallback}.
    Delete all my data by voice: safety.DELETE_PHRASE -> "say yes to confirm" -> next turn: yes deletes
    (delete_user_data), anything else cancels. Neither turn is logged, stored, added to history or fed to the mood.
    A crisis phrase still wins over everything (and cancels a pending deletion).
    Push-to-talk (--live): Enter starts, Enter stops (sounddevice, mic's own rate -> 16 kHz mono like
    scripts/record.py), capped at ser.max_seconds (the emotion model reads no more than that). q quits.
    Scripted mode (--script): several wav files as one conversation for one user, debug view per turn.

CLI (consent is asked once, before anything is loaded and before the mic opens):
    python gpu/live_agent_gpu.py --wav data/my_voice_v2/01_neutral.wav --user sathwik --preset gpu_24gb
    python gpu/live_agent_gpu.py --live --user sathwik --preset gpu_24gb
    python gpu/live_agent_gpu.py --script rehearsal.yaml --preset gpu_24gb [--user NAME] [--no-play]
    python gpu/live_agent_gpu.py --wav some.wav --user test --preset smoke_cpu --no-play
--script file: YAML {user: NAME (optional), turns: [{wav: PATH, label: LABEL (optional)}, ...]} or a list of
those, or a text/CSV file with one "PATH[,LABEL]" per line (# comments, an optional "wav,label" header).
Relative paths are taken from Code/, else from the script file's folder. Labels are printed, never given to the agent.
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
MIC_TARGET_SR = 16000          # same as scripts/record.py TARGET_SR (a test checks)
CODE_DIR = HERE.parent

LOG_FIELDS = (
    "user_id", "session_id", "turn_id", "timestamp", "status", "transcript", "crisis", "crisis_phrase",
    "emotion_probs", "confidence", "margin", "abstained", "mood_scores", "turns_above", "mode", "strategy",
    "check_in", "reason", "mood_params", "retrieved_memories", "system_prompt", "llm_backend", "llm_model",
    "llm_fell_back", "llm_error", "reply", "voice_path", "voice_error", "time_to_first_audio", "tts_tags",
    "stored_memory_id", "fact", "timings", "ser_model",
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
                 mood_factory=None, consent_path=None, extract_fn=fact_extractor.extract_fact, ser_model=None):
        self.cfg = cfg
        # {name, fallback} of the emotion model actually loaded (ser_gpu.load_ser_with_fallback); in every log line
        self.ser_model = ({"name": ser_model.get("name"), "fallback": bool(ser_model.get("fallback"))}
                          if ser_model else None)
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
                                    "turns": 0, "consented": False, "pending_delete": False}
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

    def _delete_by_voice(self, user_id, st, transcript):
        """safety.DELETE_PHRASE -> ask for a spoken yes; the NEXT turn confirms (yes) or cancels (anything else).
        Returns the turn's result for these turns, None for an ordinary turn. Nothing here is logged or stored."""
        wants = getattr(self.safety, "wants_delete_all", safety_module.wants_delete_all)
        is_yes = getattr(self.safety, "is_spoken_yes", safety_module.is_spoken_yes)
        text = lambda name: getattr(self.safety, name, getattr(safety_module, name))  # noqa: E731
        if st.get("pending_delete"):
            st["pending_delete"] = False
            if is_yes(transcript):
                report = self.delete_user_data(user_id)
                voice_rec = self._speak(text("DELETE_DONE"), REPEAT_STRATEGY)
                return {"status": "deleted", "user_id": user_id, "transcript": transcript, "report": report,
                        "reply": text("DELETE_DONE"), "voice_path": voice_rec.get("path")}
            voice_rec = self._speak(text("DELETE_CANCELLED"), REPEAT_STRATEGY)
            return {"status": "delete_cancelled", "user_id": user_id, "transcript": transcript,
                    "reply": text("DELETE_CANCELLED"), "voice_path": voice_rec.get("path")}
        if wants(transcript):
            st["pending_delete"] = True
            voice_rec = self._speak(text("DELETE_CONFIRM_PROMPT"), REPEAT_STRATEGY)
            return {"status": "delete_confirm", "user_id": user_id, "transcript": transcript,
                    "reply": text("DELETE_CONFIRM_PROMPT"), "voice_path": voice_rec.get("path")}
        return None

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
                   crisis=False, abstained=False, check_in=False, retrieved_memories=[], tts_tags=[],
                   ser_model=self.ser_model)
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

        # 2. crisis check (first, always: it also cancels a pending deletion)
        with stages("safety"):
            crisis = self.safety.check_crisis(transcript)
        if crisis.get("matched"):
            st["pending_delete"] = False
        else:
            handled = self._delete_by_voice(user_id, st, transcript)
            if handled is not None:
                return handled                              # not logged, not stored, mood untouched

        st["turns"] += 1
        turn_id = f"{self.session_id}-{st['turns']}"
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
    SER = the ser.checkpoint (T3) if it loads and its label order checks out, else the pretrained ser.model (one
    warning; parts["ser_model"] says which). A label-order mismatch stops here (ser_gpu.LabelOrderError).
    The CSM voice is loaded only when tts.backend is csm; if it fails to load, the agent uses the fallback voice."""
    from device import setup
    import asr_gpu
    import llm_gpu
    import ser_gpu

    cfg, device, dtype = setup(preset)
    asr = asr_gpu.ASR(cfg, device, dtype).load().warm_up()
    emotion, ser_info = ser_gpu.load_ser_with_fallback(cfg, device, dtype,
                                                       warn=lambda m: print(m, file=sys.stderr))
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
             "memory": MemoryManager(mconf["memory_file"], cfg["memory"]), "logger": TurnLogger(mconf["log_dir"]),
             "ser_model": ser_info}
    return cfg, parts


# ---------------------------------------------------------------------------------------------
# Debug view, push-to-talk, scripted conversations
# ---------------------------------------------------------------------------------------------

def debug_view(r) -> str:
    """Short per-turn view: transcript, abstained, confidence, mood scores, strategy, check_in, reason, voice path,
    time to first audio."""
    status = r.get("status")
    if status == "refused":
        return f"[refused] {r.get('reason')}"
    if status == "repeat":
        return f"[repeat] {r.get('asr_reason')} -> asked to repeat   voice {r.get('voice_path')}"
    if status in ("delete_confirm", "deleted", "delete_cancelled"):
        extra = ""
        if status == "deleted":
            rep = r.get("report") or {}
            extra = (f"   memories removed {rep.get('memories_removed')}, logs removed "
                     f"{len(rep.get('logs_removed') or [])}, consent revoked {rep.get('consent_revoked')}")
        return (f"[{status}] heard: {r.get('transcript')!r}   (not logged, not stored)   "
                f"voice {r.get('voice_path')}{extra}")
    lines = [f"transcript : {r.get('transcript')}"]
    if r.get("crisis"):
        lines.append(f"CRISIS     : {r.get('crisis_phrase')!r} -> safe response, NEUTRAL voice")
    else:
        conf = r.get("confidence")
        mood = r.get("mood_scores") or {}
        scores = "  ".join(f"{c} {mood[c]:.2f}" for c in ("neutral", "happy", "angry", "sad") if c in mood)
        lines.append(f"emotion    : confidence {conf:.2f}  " + ("ABSTAINED" if r.get("abstained") else "kept")
                     if conf is not None else "emotion    : n/a")
        lines.append(f"mood       : {scores}")
        lines.append(f"strategy   : {r.get('strategy')}" + ("   CHECK_IN" if r.get("check_in") else ""))
    lines.append(f"reason     : {r.get('reason')}")
    ttfa = r.get("time_to_first_audio")
    lines.append(f"voice      : {r.get('voice_path')}   time to first audio "
                 + (f"{ttfa:.2f} s" if isinstance(ttfa, (int, float)) else "n/a"))
    return "\n".join(lines)


def chunks_to_16k(chunks, mic_sr, max_seconds):
    """Mic blocks -> one float32 mono array at 16 kHz, cut to max_seconds; the conversion scripts/record.py does
    (record at the mic's own rate, flatten, librosa.resample to 16 kHz)."""
    import numpy as np

    y = (np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])
         if chunks else np.zeros(0, dtype=np.float32))
    y = y[: int(max_seconds * mic_sr)]
    if y.size and int(mic_sr) != MIC_TARGET_SR:
        import librosa

        y = librosa.resample(y, orig_sr=int(mic_sr), target_sr=MIC_TARGET_SR)
    return np.ascontiguousarray(y, dtype=np.float32)


def record_push_to_talk(max_seconds, input_fn=input, sd=None, print_fn=print):
    """Mic is open only between the two Enter presses (the first one is the caller's). Stops by itself at
    max_seconds; the user still presses Enter to go on. Returns 16 kHz mono float32."""
    import threading

    if sd is None:
        import sounddevice as sd
    mic_sr = int(sd.query_devices(kind="input")["default_samplerate"])
    cap = int(max_seconds * mic_sr)
    chunks, got, capped = [], [0], threading.Event()

    def callback(indata, frames, time_info, status):
        if got[0] >= cap:
            capped.set()
            raise sd.CallbackStop
        chunks.append(indata.copy())
        got[0] += len(indata)

    with sd.InputStream(samplerate=mic_sr, channels=1, dtype="float32", callback=callback):
        input_fn(f"  recording (max {max_seconds:g} s) ... press Enter to stop ")
    if capped.is_set():
        print_fn(f"  (stopped at the {max_seconds:g} s cap)")
    return chunks_to_16k(chunks, mic_sr, max_seconds)


def live_loop(agent, user_id, max_seconds, input_fn=input, record_fn=record_push_to_talk, print_fn=print) -> int:
    """Enter = talk (Enter again = stop), q = quit. Ends after a deletion (consent is gone) or a refusal."""
    while True:
        cmd = input_fn("\n[Enter] talk   [q] quit > ").strip().lower()
        if cmd == "q":
            return 0
        y = record_fn(max_seconds, input_fn=input_fn)
        r = agent.run_turn(user_id, (y, MIC_TARGET_SR))
        print_fn(debug_view(r))
        if r["status"] == "deleted":
            print_fn("Your data was deleted and your consent withdrawn. Ending the session.")
            return 0
        if r["status"] == "refused":
            return 1


def parse_script(path):
    """(user or None, [{"wav": Path, "label": str or None}]) from a YAML or a PATH[,LABEL] text/CSV file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    user, turns = None, []
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            user, items = data.get("user"), data.get("turns")
        else:
            items = data
        if not isinstance(items, list) or not items:
            raise ValueError(f"{path}: expected a list of turns (or {{user, turns: [...]}})")
        for it in items:
            if isinstance(it, str):
                turns.append({"wav": it, "label": None})
            elif isinstance(it, dict) and it.get("wav"):
                turns.append({"wav": str(it["wav"]), "label": it.get("label")})
            else:
                raise ValueError(f"{path}: a turn needs a wav path, got {it!r}")
    else:
        for n, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if not turns and parts[0].lower() in ("wav", "path", "wav_path"):
                continue                                         # header
            turns.append({"wav": parts[0], "label": (parts[1] or None) if len(parts) > 1 else None})
        if not turns:
            raise ValueError(f"{path}: no wav paths found")
    resolved, missing = [], []
    for t in turns:
        p = Path(t["wav"])
        if not p.is_absolute():
            p = CODE_DIR / p if (CODE_DIR / p).is_file() else path.parent / p
        (resolved if p.is_file() else missing).append({"wav": p, "label": t["label"]})
    if missing:
        raise ValueError(f"{path}: wav file(s) not found: {[str(m['wav']) for m in missing]}")
    return (str(user) if user else None), resolved


def run_script(agent, user_id, turns, print_fn=print) -> list:
    """All turns as one conversation for user_id; the label (if any) is printed next to the view, never used."""
    results = []
    for i, t in enumerate(turns, 1):
        label = f"   label: {t['label']} (not given to the agent)" if t.get("label") else ""
        print_fn(f"\n--- turn {i}/{len(turns)}: {Path(t['wav']).name}{label}")
        r = agent.run_turn(user_id, t["wav"])
        print_fn(debug_view(r))
        results.append(r)
        if r["status"] in ("deleted", "refused"):
            break
    return results


def _print_turn(r):
    print(f"\nstatus     : {r['status']}")
    if r["status"] not in ("ok", "crisis"):
        print(debug_view(r))
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
    print(f"ser model  : {r.get('ser_model')}")
    print("timings    : " + "  ".join(f"{k} {v:.3f}s" for k, v in r["timings"].items()))
    print(f"log        : {r['log_file']}")


def _main():
    ap = argparse.ArgumentParser(description="Live agent (GPU demo build, steps 9a/9b).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--wav", help="one turn from a wav file")
    mode.add_argument("--live", action="store_true", help="push-to-talk with the default microphone")
    mode.add_argument("--script", help="several wav files as one conversation (YAML or PATH[,LABEL] lines)")
    ap.add_argument("--user", help="user id (letters, digits, '_', '.', '-'); --script may name it instead")
    ap.add_argument("--preset", help="preset name (default: $AGENT_PRESET, else smoke_cpu)")
    ap.add_argument("--no-play", action="store_true", help="do not play audio (CSM audio is still generated)")
    args = ap.parse_args()

    turns = None
    user = args.user
    if args.script:
        try:
            script_user, turns = parse_script(args.script)
        except (OSError, ValueError) as e:
            ap.error(str(e))
        user = user or script_user
    if not user:
        ap.error("--user is required (or name the user in the --script file)")
    try:
        check_user_id(user)
    except ValueError as e:
        ap.error(str(e))
    if args.wav and not Path(args.wav).is_file():
        ap.error(f"wav file not found: {args.wav}")

    # Consent once, before anything is loaded and before the microphone opens.
    if not safety_module.consent_given(user) and not safety_module.ask_consent(user):
        print("No consent: nothing was run, recorded or stored.")
        return 1

    t0 = time.perf_counter()
    try:
        cfg, parts = load_real_parts(args.preset, play=not args.no_play)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    ser = parts.get("ser_model") or {}
    print(f"preset {cfg['preset']}: parts loaded in {time.perf_counter() - t0:.1f}s  "
          f"(tau_conf {cfg['brain']['tau_conf']}, check_in_after {cfg['brain']['check_in_after']}, "
          f"emotion model {ser.get('name')}{' [FALLBACK]' if ser.get('fallback') else ''})")
    agent = LiveAgent(cfg, **parts)
    if args.wav:
        _print_turn(agent.run_turn(user, args.wav))
        return 0
    if args.script:
        run_script(agent, user, turns)
        return 0
    return live_loop(agent, user, float(cfg["ser"]["max_seconds"]))


if __name__ == "__main__":
    sys.exit(_main())
