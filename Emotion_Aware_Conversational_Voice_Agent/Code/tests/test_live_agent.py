"""
test_live_agent.py - step 9a of the GPU demo build (gpu/live_agent_gpu.py LiveAgent, plus
tts_gpu.speak_iter_with_fallback). CPU only: every part is a stub that records into one shared event list;
memory, consent, mood and the crisis check are the real scripts/ code on temporary files. No model, no network.
"""
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import fact_extractor
import fallback_tts
import live_agent_gpu
import response_generator
import safety
from device import load_config
from live_agent_gpu import LOG_FIELDS, REPEAT_PROMPT, STAGES, AgentVoice, LiveAgent, TurnLogger
from memory_manager import MemoryManager
from mood_manager import CHECK_IN_AFTER, MoodManagerV2

CODE_DIR = Path(__file__).resolve().parent.parent
AUDIO = (np.zeros(1600, dtype=np.float32), 16000)
CALMISH = {"neutral": 0.70, "happy": 0.10, "angry": 0.10, "sad": 0.10}
SAD = {"neutral": 0.10, "happy": 0.05, "angry": 0.05, "sad": 0.80}
UNSURE = {"neutral": 0.30, "happy": 0.20, "angry": 0.10, "sad": 0.40}     # top-1 0.40
REPLY = ("That sounds like a lot.", "What part feels heaviest right now?")


# ---------------------------------------------------------------- stubs

class StubASR:
    def __init__(self, events):
        self.events, self.text, self.skipped = events, "", False

    def transcribe(self, y, sr=None):
        self.events.append("asr")
        if self.skipped:
            return {"text": "", "skipped": True, "reason": "silence"}
        return {"text": self.text, "skipped": False, "reason": None}


class StubEmotion:
    def __init__(self, events):
        self.events, self.probs = events, CALMISH

    def predict_array(self, y, sr):
        self.events.append("emotion")
        return {"probs": dict(self.probs), "label": "would-be-label"}


class StubLLM:
    def __init__(self, events, reply=REPLY, fail=None):
        self.events, self.reply, self.fail = events, list(reply), fail
        self.fact = "NONE"
        self.calls, self.complete_prompts = [], []
        self.last_reply = None

    def stream_reply(self, system_prompt, history, user_text, strategy):
        self.events.append("llm")
        self.calls.append({"system_prompt": system_prompt, "history": list(history), "user_text": user_text,
                           "strategy": strategy})
        self.last_reply = None
        for i, s in enumerate(self.reply):
            if self.fail == "start" or (self.fail == "mid" and i == 1):
                raise RuntimeError("qwen out of memory (stub)")
            yield s
        self.last_reply = {"backend": "qwen", "model": "Qwen/Qwen3-8B", "fell_back": False, "errors": []}

    def complete(self, prompt):
        self.events.append("llm_fn")
        self.complete_prompts.append(prompt)
        return self.fact


class StubVoice:
    def __init__(self, events, fail=False):
        self.events, self.fail, self.spoken = events, fail, []

    def speak(self, sentences, strategy, crisis=False):
        self.events.append("tts")
        if self.fail:
            raise RuntimeError("audio device gone (stub)")
        text = sentences if isinstance(sentences, str) else " ".join(sentences)
        self.spoken.append({"text": text, "strategy": strategy, "crisis": crisis})
        return {"path": "csm", "error": None, "stats": {"time_to_first_audio": 0.5}}


class StubSafety:
    SAFE_RESPONSE = safety.SAFE_RESPONSE
    SAFE_STRATEGY = safety.SAFE_STRATEGY

    def __init__(self, events):
        self.events = events

    def consent_given(self, user_id, path):
        self.events.append("consent")
        return safety.consent_given(user_id, path)

    def check_crisis(self, transcript):
        self.events.append("safety")
        return safety.check_crisis(transcript)


class RecMemory(MemoryManager):
    events = None

    def retrieve_relevant(self, user_id, query, top_k=None):
        self.events.append("memory")
        return super().retrieve_relevant(user_id, query, top_k)

    def store(self, *a, **k):
        self.events.append("store")
        return super().store(*a, **k)


class RecMood(MoodManagerV2):
    events = None

    def update(self, probs, confidence=None):
        self.events.append("mood")
        return super().update(probs, confidence)


class RecLogger(TurnLogger):
    events = None

    def log(self, user_id, record):
        self.events.append("log")
        return super().log(user_id, record)


class World:
    """A LiveAgent on stubs + real memory / consent / mood / crisis check on temporary files."""

    def __init__(self, tmp_path, tau=None, consent=("alice",), llm_fail=None, voice_fail=False):
        self.cfg = load_config("gpu_24gb")
        self.events = []
        self.consent_path = tmp_path / "consent.json"
        for user in consent:
            safety.record_consent(user, self.consent_path)
        brain = dict(self.cfg["brain"])
        if tau is not None:
            brain["tau_conf"] = tau                    # tests only: exercise the gate with a non-zero threshold

        def mood_factory():
            m = RecMood.from_config(brain)
            m.events = self.events
            return m

        self.memory = RecMemory(tmp_path / "memories.json", self.cfg["memory"])
        self.memory.events = self.events
        self.logger = RecLogger(tmp_path / "logs")
        self.logger.events = self.events
        self.asr, self.emotion = StubASR(self.events), StubEmotion(self.events)
        self.llm, self.voice = StubLLM(self.events, fail=llm_fail), StubVoice(self.events, fail=voice_fail)
        self.extract_calls = []
        self.agent = LiveAgent(self.cfg, asr=self.asr, emotion=self.emotion, llm=self.llm, voice=self.voice,
                               memory=self.memory, logger=self.logger, safety=StubSafety(self.events),
                               mood_factory=mood_factory, consent_path=self.consent_path, extract_fn=self.extract)

    def extract(self, user_transcript, llm_fn, emotional_spike=False):
        self.events.append("extract")
        self.extract_calls.append({"args": (user_transcript,), "spike": emotional_spike})
        return fact_extractor.extract_fact(user_transcript, llm_fn, emotional_spike=emotional_spike)

    def turn(self, text, probs=CALMISH, user="alice", fact="NONE"):
        self.asr.text, self.emotion.probs, self.llm.fact = text, probs, fact
        return self.agent.run_turn(user, AUDIO)

    def log_lines(self, user="alice"):
        path = self.logger.path_for(user)
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


# ---------------------------------------------------------------- stage order

def test_stage_order_is_exactly_the_spec(world, monkeypatch):
    real = response_generator.build_system_prompt

    def prompt(*a, **k):
        world.events.append("prompt")
        return real(*a, **k)

    monkeypatch.setattr(response_generator, "build_system_prompt", prompt)
    r = world.turn("I have so much homework this week.", fact="The user has a lot of homework this week.")
    assert r["status"] == "ok"
    assert world.events == ["consent", "asr", "safety", "emotion", "mood", "memory", "prompt", "llm", "tts",
                            "extract", "llm_fn", "store", "log"]
    assert list(r["timings"]) == list(STAGES[:-1]) + ["total"]       # consent ... extract, then total
    assert STAGES == ("consent", "asr", "safety", "emotion", "abstain", "mood", "memory", "prompt", "llm", "tts",
                      "extract", "log")


def test_consent_is_checked_once_per_session(world):
    world.turn("First thing today.")
    world.turn("Second thing today.")
    assert world.events.count("consent") == 1


# ---------------------------------------------------------------- consent

def test_no_consent_means_nothing_happens(tmp_path):
    w = World(tmp_path, consent=())
    r = w.turn("My exam is on Monday.", fact="The user has an exam on Monday.")
    assert r["status"] == "refused"
    assert w.events == ["consent"]                                     # no ASR, nothing after
    assert not (tmp_path / "logs").exists() and not (tmp_path / "memories.json").exists()
    assert w.voice.spoken == [] and w.llm.calls == []
    safety.record_consent("alice", w.consent_path)                     # consent given later -> the next turn runs
    assert w.turn("My exam is on Monday.")["status"] == "ok"


# ---------------------------------------------------------------- ASR

@pytest.mark.parametrize("text, skipped", [("", False), ("  ...  ", False), ("anything", True)])
def test_empty_or_silent_turn_asks_to_repeat_and_stores_nothing(tmp_path, text, skipped):
    w = World(tmp_path)
    w.asr.skipped = skipped
    r = w.turn(text, fact="The user said something.")
    assert r["status"] == "repeat" and r["reply"] == REPEAT_PROMPT
    assert w.events == ["consent", "asr", "tts"]
    assert w.voice.spoken == [{"text": REPEAT_PROMPT, "strategy": "NEUTRAL", "crisis": False}]
    assert not (tmp_path / "logs").exists() and not (tmp_path / "memories.json").exists()
    assert w.agent._users["alice"]["turns"] == 0 and w.agent._users["alice"]["mood"].n_updates == 0


# ---------------------------------------------------------------- crisis

def test_crisis_overrides_everything_and_stores_nothing(world):
    world.turn("I have an exam on Monday.", fact="The user has an exam on Monday.")     # one normal turn first
    facts_before = world.memory.list_user_facts("alice")
    mood_before = world.agent._users["alice"]["mood"].get_mood_vector()
    history_before = list(world.agent._users["alice"]["history"])
    world.events.clear()

    r = world.turn("Honestly I want to kill myself before that exam.", probs=SAD,
                   fact="The user is stressed about an exam.")
    assert r["status"] == "crisis" and r["crisis"] is True and r["crisis_phrase"] == "kill myself"
    assert world.events == ["asr", "safety", "tts", "log"]          # no emotion, mood, memory, LLM, extraction
    assert world.voice.spoken[-1] == {"text": safety.SAFE_RESPONSE, "strategy": "NEUTRAL", "crisis": True}
    assert r["reply"] == safety.SAFE_RESPONSE and r["strategy"] == "NEUTRAL" and r["tts_tags"] == []
    assert world.memory.list_user_facts("alice") == facts_before
    assert world.agent._users["alice"]["mood"].get_mood_vector() == mood_before
    assert list(world.agent._users["alice"]["history"]) == history_before          # not fed to the LLM later
    line = world.log_lines()[-1]                                     # still logged
    assert line["crisis"] is True and line["system_prompt"] is None and line["emotion_probs"] is None
    assert line["stored_memory_id"] is None and line["llm_backend"] is None


# ---------------------------------------------------------------- abstain

def test_abstain_threshold_and_check_in_count_come_from_config(world):
    mood = world.agent._state("bob")["mood"]
    assert mood.tau_conf == world.cfg["brain"]["tau_conf"]
    assert mood.check_in_after == world.cfg["brain"]["check_in_after"]
    default = LiveAgent(world.cfg, asr=None, emotion=None, llm=None, voice=None, memory=None, logger=world.logger)
    assert default.mood_factory().tau_conf == world.cfg["brain"]["tau_conf"]


def test_abstain_holds_mood_and_does_not_raise_importance(tmp_path):
    w = World(tmp_path, tau=0.5)
    w.turn("Rough morning at work.", probs=SAD)
    held = w.agent._users["alice"]["mood"].get_mood_vector()
    r = w.turn("I keep thinking about my grandmother's garden.", probs=UNSURE,
               fact="The user often thinks about their grandmother's garden.")
    assert r["abstained"] is True and r["confidence"] == pytest.approx(0.40)
    assert "abstained" in r["reason"]
    after = w.agent._users["alice"]["mood"].get_mood_vector()
    assert {c: after[c] for c in ("neutral", "happy", "angry", "sad")} == \
        {c: held[c] for c in ("neutral", "happy", "angry", "sad")}     # mood held
    assert w.extract_calls[-1]["spike"] is False
    stored = [m for m in w.memory.list_user_facts("alice") if m["memory_id"] == r["stored_memory_id"]][0]
    assert stored["importance"] == pytest.approx(w.cfg["memory"]["base_importance"])
    assert w.log_lines()[-1]["abstained"] is True


def test_check_in_after_configured_abstains_outside_default(tmp_path):
    w = World(tmp_path, tau=0.5)
    for _ in range(6):
        r = w.turn("Everything went wrong again today.", probs=SAD)
    assert r["mode"] != "DEFAULT"
    n = w.cfg["brain"]["check_in_after"]
    results = [w.turn("Hmm, I don't know.", probs=UNSURE) for _ in range(n)]
    assert [x["check_in"] for x in results] == [False] * (n - 1) + [True]
    assert response_generator.CHECK_IN_INSTRUCTION in w.llm.calls[-1]["system_prompt"]
    assert response_generator.CHECK_IN_INSTRUCTION not in w.llm.calls[-2]["system_prompt"]


# ---------------------------------------------------------------- P3: what the LLM gets

def test_prompt_contains_no_emotion_label_or_probabilities(world):
    world.turn("I have an exam on Monday.", fact="The user has an exam on Monday.")
    for _ in range(4):
        r = world.turn("My exam on Monday is stressing me out.", probs=SAD)
    call = world.llm.calls[-1]
    facts = [world.memory.list_user_facts("alice")[0]["fact"]]
    assert call["system_prompt"] == response_generator.build_system_prompt(r["strategy"], facts, r["check_in"])
    assert call["user_text"] == "My exam on Monday is stressing me out."
    blob = json.dumps(call)
    for word in ("sad", "angry", "happy", "neutral", "would-be-label", "probab", "confidence", "emotion"):
        assert not re.search(rf"\b{word}", blob), word
    for v in list(SAD.values()) + list(CALMISH.values()):
        assert f"{v:.2f}" not in blob and repr(v) not in blob
    assert all(set(world.voice.spoken[i]) == {"text", "strategy", "crisis"} for i in range(len(world.voice.spoken)))


# ---------------------------------------------------------------- log line

def test_log_line_has_every_required_field(world):
    world.turn("I have an exam on Monday.", fact="The user has an exam on Monday.")
    r = world.turn("Monday's exam is coming fast.")
    line = world.log_lines()[-1]
    assert set(line) == set(LOG_FIELDS)
    assert re.fullmatch(r"session_alice_\d{8}T\d{6}Z\.jsonl", world.logger.path_for("alice").name)
    assert line["user_id"] == "alice" and line["turn_id"] == r["turn_id"] and line["timestamp"]
    assert line["transcript"] == "Monday's exam is coming fast." and line["crisis"] is False
    assert line["emotion_probs"] == CALMISH
    assert line["confidence"] == pytest.approx(0.70) and line["margin"] == pytest.approx(0.60)
    assert line["abstained"] is False and isinstance(line["turns_above"], int)
    assert {"neutral", "happy", "angry", "sad"} <= set(line["mood_scores"])
    for key in ("mode", "strategy", "reason", "system_prompt", "llm_backend", "llm_model", "voice_path"):
        assert line[key], key
    assert line["llm_backend"] == "qwen" and line["voice_path"] == "csm" and line["time_to_first_audio"] == 0.5
    assert isinstance(line["check_in"], bool) and isinstance(line["tts_tags"], list)
    assert line["mood_params"]["tau_conf"] == world.cfg["brain"]["tau_conf"]
    assert line["retrieved_memories"] and set(line["retrieved_memories"][0]) == \
        {"memory_id", "similarity", "importance", "score"}
    assert set(line["timings"]) == set(STAGES[:-1]) | {"total"}
    assert line["reply"] == " ".join(REPLY)


# ---------------------------------------------------------------- P5: extraction

def test_extractor_only_receives_the_user_transcript(world):
    r = world.turn("My sister's wedding is next month.", fact="The user's sister is getting married next month.")
    assert world.extract_calls[-1]["args"] == ("My sister's wedding is next month.",)
    prompt = world.llm.complete_prompts[-1]
    assert "My sister's wedding is next month." in prompt
    assert all(s not in prompt for s in REPLY)
    stored = world.memory.list_user_facts("alice")[0]
    assert stored["source"] == "user_turn" and stored["turn_id"] == r["turn_id"]
    assert stored["memory_id"] == r["stored_memory_id"]


# ---------------------------------------------------------------- isolation and deletion

def test_users_cannot_see_each_others_memories(tmp_path):
    w = World(tmp_path, consent=("alice", "bob"))
    w.turn("I have an exam on Monday.", fact="The user has an exam on Monday.", user="alice")
    r = w.turn("Monday has an exam for me too.", user="bob")
    assert r["retrieved_memories"] == []
    assert "exam on Monday" not in w.llm.calls[-1]["system_prompt"]
    assert w.agent._users["bob"]["mood"] is not w.agent._users["alice"]["mood"]
    assert w.logger.path_for("alice") != w.logger.path_for("bob")
    assert [line["user_id"] for line in w.log_lines("bob")] == ["bob"]
    assert w.llm.calls[-1]["history"] == []                            # alice's turns are not bob's history


def test_deleted_fact_is_never_retrieved_again(world):
    r = world.turn("I have an exam on Monday.", fact="The user has an exam on Monday.")
    assert world.turn("Thinking about Monday's exam.")["retrieved_memories"]
    assert world.memory.delete("alice", r["stored_memory_id"])
    r2 = world.turn("Thinking about Monday's exam.")
    assert r2["retrieved_memories"] == []
    assert "exam on Monday" not in world.llm.calls[-1]["system_prompt"]


def test_delete_user_data_wipes_memories_logs_and_session_state(tmp_path):
    w = World(tmp_path, consent=("alice", "bob"))
    w.turn("I have an exam on Monday.", fact="The user has an exam on Monday.", user="alice")
    w.turn("I love hiking on weekends.", fact="The user loves hiking on weekends.", user="bob")
    assert w.logger.path_for("alice").exists()
    report = w.agent.delete_user_data("alice")
    assert report["memories_removed"] == 1 and report["consent_revoked"] is True
    assert report["logs_removed"] == [w.logger.path_for("alice").name]
    assert not w.logger.path_for("alice").exists() and "alice" not in w.agent._users
    assert w.memory.retrieve_relevant("alice", "exam Monday") == []
    assert len(w.memory.list_user_facts("bob")) == 1 and w.logger.path_for("bob").exists()   # bob untouched
    assert w.turn("Thinking about Monday's exam.", user="alice")["status"] == "refused"      # consent revoked


@pytest.mark.parametrize("bad", ["", "../etc", "a/b", "a\\b", ".hidden", "_x", None, 3])
def test_user_id_must_be_safe_for_a_file_name(world, bad):
    with pytest.raises(ValueError):
        world.agent.run_turn(bad, AUDIO)


# ---------------------------------------------------------------- LLM and voice failures

def test_llm_failure_falls_back_to_the_template(tmp_path):
    w = World(tmp_path, llm_fail="start")
    r = w.turn("Today was a long day.")
    expected = response_generator.generate_reply_template(r["strategy"], "Today was a long day.", r["check_in"])
    assert r["reply"] == expected and w.voice.spoken[-1]["text"] == expected
    assert r["llm_backend"] == "template" and r["llm_fell_back"] is True and "out of memory" in r["llm_error"]
    assert w.log_lines()[-1]["llm_backend"] == "template"


def test_llm_failure_mid_reply_keeps_what_was_said(tmp_path):
    w = World(tmp_path, llm_fail="mid")
    r = w.turn("Today was a long day.")
    assert r["reply"] == REPLY[0] and r["llm_fell_back"] is True and r["llm_error"]


def test_voice_failure_falls_back_without_crashing(tmp_path, monkeypatch):
    w = World(tmp_path, voice_fail=True)
    said = []
    monkeypatch.setattr(fallback_tts, "speak", lambda text: said.append(text) or "pyttsx3")
    r = w.turn("Today was a long day.", fact="The user had a long day today.")
    assert r["status"] == "ok" and r["voice_path"] == "pyttsx3" and "audio device gone" in r["voice_error"]
    assert said == [" ".join(REPLY)]
    assert r["stored_memory_id"] and w.log_lines()[-1]["voice_path"] == "pyttsx3"


# ---------------------------------------------------------------- AgentVoice / speak_iter_with_fallback

SR = 24000


class StubTTS:
    def __init__(self, order, fail_on=None):
        self.sr, self.order, self.fail_on = SR, order, fail_on

    def synth(self, text, ref_text, ref_audio_24k):
        self.order.append(("synth", text))
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("CUDA error (stub)")
        return {"audio": np.full(2400, 0.1, dtype=np.float32), "sr": SR}


def csm_voice(tmp_path, order, fail_on=None):
    import tts_gpu

    refs = {}
    for s in tts_gpu.STRATEGIES:
        sf.write(str(tmp_path / f"{s}.wav"), np.full(4800, 0.05, dtype=np.float32), SR, subtype="PCM_16")
        refs[s] = {"file": f"{s}.wav", "text": f"{s} words", "speaker": "ex02", "style": "default",
                   "source_id": s, "seconds": 0.2}
    return tts_gpu.CSMVoice(StubTTS(order, fail_on), refs, tmp_path)


def stream(order, sentences):
    for s in sentences:
        order.append(("pull", s))
        yield s


def test_stream_voice_starts_before_the_reply_is_finished(tmp_path):
    import tts_gpu

    order = []
    v = csm_voice(tmp_path, order)
    rec = tts_gpu.speak_iter_with_fallback(stream(order, ["First one here.", "Second one here."]), "VALIDATE",
                                           voice=v, play=False)
    assert rec["path"] == "csm" and rec["csm_sentences"] == 2
    assert order == [("pull", "First one here."), ("synth", "First one here."),
                     ("pull", "Second one here."), ("synth", "Second one here.")]


def test_stream_voice_failure_speaks_unspoken_part_and_rest_of_stream(tmp_path, monkeypatch):
    import tts_gpu

    order, said = [], []
    monkeypatch.setattr(fallback_tts, "speak", lambda text: said.append(text) or "pyttsx3")
    v = csm_voice(tmp_path, order, fail_on="Second")
    rec = tts_gpu.speak_iter_with_fallback(stream(order, ["First one here.", "Second <sigh> one.", "Third one."]),
                                           "CALM", voice=v, play=False)
    assert rec["path"] == "pyttsx3" and rec["csm_sentences"] == 1 and "CUDA error" in rec["error"]
    assert said == ["Second one. Third one."]                       # tags stripped, nothing repeated or lost


def test_stream_voice_without_csm_and_empty_stream(monkeypatch):
    import tts_gpu

    said = []
    monkeypatch.setattr(fallback_tts, "speak", lambda text: said.append(text) or "text")
    rec = tts_gpu.speak_iter_with_fallback(iter(["Hi there.", "Bye."]), "NEUTRAL", voice=None)
    assert rec["path"] == "text" and rec["error"] == "no CSMVoice loaded" and said == ["Hi there. Bye."]

    class OkVoice:
        last_stream = None

        def speak_stream(self, sentences, strategy, crisis=False, play=True):
            yield from ()

    assert tts_gpu.speak_iter_with_fallback(iter([]), "NEUTRAL", voice=OkVoice())["path"] == "none"


def test_agent_voice_routes_text_and_streams_and_forces_neutral_on_crisis(tmp_path, monkeypatch):
    order = []
    voice = AgentVoice(csm_voice(tmp_path, order), play=False)
    assert voice.speak("You matter. Please call someone.", "ENCOURAGE", crisis=True)["path"] == "csm"
    assert voice.csm.last_stream["ref_strategy"] == "NEUTRAL"
    assert voice.speak(iter(["One here.", "Two here."]), "CALM")["path"] == "csm"
    assert voice.csm.last_stream["ref_strategy"] == "CALM"
    fake = types.SimpleNamespace(init=lambda: (_ for _ in ()).throw(RuntimeError("no SAPI5 (stub)")))
    monkeypatch.setitem(sys.modules, "pyttsx3", fake)
    assert AgentVoice(None).speak(iter(["Hello."]), "NEUTRAL")["path"] == "text"


# ---------------------------------------------------------------- imports

def test_module_import_loads_no_torch_and_no_gpu_module():
    code = (f"import sys; sys.path.insert(0, {str(CODE_DIR / 'gpu')!r}); import live_agent_gpu; "
            "bad = [m for m in ('torch', 'transformers', 'device', 'asr_gpu', 'ser_gpu', 'llm_gpu', 'tts_gpu') "
            "if m in sys.modules]; print(bad)")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_brain_scripts_used_by_the_agent_import_no_torch_or_gpu():
    gpu_modules = {p.stem for p in (CODE_DIR / "gpu").glob("*.py")}
    for name in ("safety", "memory_manager", "fact_extractor", "mood_manager", "response_generator", "fallback_tts"):
        text = (CODE_DIR / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        imported = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", text, flags=re.M))
        assert "torch" not in imported and not imported & gpu_modules, (name, imported)


def test_check_in_constant_matches_config():
    assert load_config("gpu_24gb")["brain"]["check_in_after"] == CHECK_IN_AFTER


def test_cli_rejects_a_bad_user_id(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["live_agent_gpu.py", "--wav", "x.wav", "--user", "../evil"])
    with pytest.raises(SystemExit):
        live_agent_gpu._main()
