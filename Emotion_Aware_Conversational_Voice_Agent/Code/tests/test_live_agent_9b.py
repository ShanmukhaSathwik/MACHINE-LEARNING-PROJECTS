"""
test_live_agent_9b.py - step 9b: T3 checkpoint + fallback (gpu/ser_gpu.py), Rule B tau selection
(scripts/abstain_curve_t3.py), silence-gate measurement (scripts/measure_silence_gate.py), delete-all by voice,
push-to-talk, scripted mode (gpu/live_agent_gpu.py). CPU only, stubs, no model, no microphone, no network.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import abstain_curve_t3 as act
import live_agent_gpu
import measure_silence_gate as msg
import safety
import ser_gpu
from device import list_presets, load_config
from live_agent_gpu import (
    LOG_FIELDS,
    MIC_TARGET_SR,
    chunks_to_16k,
    debug_view,
    live_loop,
    parse_script,
    record_push_to_talk,
    run_script,
)
from test_live_agent import CALMISH, SAD, World

CODE_DIR = Path(__file__).resolve().parent.parent
REAL_OUTPUTS_DIR = __import__("os").environ.get("GPU_OUTPUTS_DIR")   # read at import, before conftest clears it
RUN = "gpu_24gb_20261112_1500"
T3_ID2LABEL = {"0": {"class": "neutral", "model_label": "neu"}, "1": {"class": "happy", "model_label": "hap"},
               "2": {"class": "angry", "model_label": "ang"}, "3": {"class": "sad", "model_label": "sad"}}
MODEL_ID2LABEL = {0: "neu", 1: "hap", 2: "ang", 3: "sad"}


# ================================================================ 1. T3 checkpoint

def test_checkpoint_key_in_every_preset_and_path_resolution(tmp_path, monkeypatch):
    for preset in list_presets():
        assert load_config(preset)["ser"]["checkpoint"] == f"gpu/outputs/ser_finetune/{RUN}/best"
    cfg = load_config("gpu_24gb")
    monkeypatch.setenv("GPU_OUTPUTS_DIR", str(tmp_path))
    assert ser_gpu.resolve_checkpoint(cfg) == tmp_path / "ser_finetune" / RUN / "best"
    monkeypatch.delenv("GPU_OUTPUTS_DIR")
    assert ser_gpu.resolve_checkpoint(cfg) == CODE_DIR / "gpu" / "outputs" / "ser_finetune" / RUN / "best"
    assert ser_gpu.resolve_checkpoint({"ser": {"model": "x"}}) is None


def test_label_order_check():
    assert ser_gpu.check_label_order(T3_ID2LABEL, MODEL_ID2LABEL) == {0: "neutral", 1: "happy", 2: "angry", 3: "sad"}
    assert ser_gpu.check_label_order(T3_ID2LABEL, {str(k): v for k, v in MODEL_ID2LABEL.items()})   # str keys ok
    with pytest.raises(ser_gpu.LabelOrderError, match="differs"):
        ser_gpu.check_label_order(T3_ID2LABEL, {0: "hap", 1: "neu", 2: "ang", 3: "sad"})
    with pytest.raises(ser_gpu.LabelOrderError, match="indices"):
        ser_gpu.check_label_order(T3_ID2LABEL, {0: "neu", 1: "hap", 2: "ang"})
    wrong_class = {**T3_ID2LABEL, "0": {"class": "happy", "model_label": "neu"}}
    with pytest.raises(ser_gpu.LabelOrderError, match="classes"):
        ser_gpu.check_label_order(wrong_class, MODEL_ID2LABEL)


def test_real_t3_id2label_json_if_present(monkeypatch):
    """Read-only look at the real run folder (conftest hides GPU_OUTPUTS_DIR; the value seen at import is used)."""
    if REAL_OUTPUTS_DIR:
        monkeypatch.setenv("GPU_OUTPUTS_DIR", REAL_OUTPUTS_DIR)
    real = ser_gpu.resolve_checkpoint(load_config("gpu_24gb")) / "id2label.json"
    if not real.is_file():
        pytest.skip(f"T3 run folder not on this machine ({real.parent})")
    ser_gpu.check_label_order(json.loads(real.read_text(encoding="utf-8")), MODEL_ID2LABEL)
    config = json.loads((real.parent / "config.json").read_text(encoding="utf-8"))
    ser_gpu.check_label_order(json.loads(real.read_text(encoding="utf-8")), config["id2label"])


class StubSER:
    """Records the model_name it was built with; id2label as given."""
    built = []

    def __init__(self, cfg, device, dtype, model_name=None, id2label=None, fail=False):
        StubSER.built.append(model_name)
        if fail:
            raise OSError("safetensors header is corrupt (stub)")
        self.model = type("M", (), {"config": type("C", (), {"id2label": id2label or MODEL_ID2LABEL})()})()


def make_ckpt(tmp_path, id2label_json=T3_ID2LABEL):
    ckpt = tmp_path / "ser_finetune" / RUN / "best"
    ckpt.mkdir(parents=True)
    if id2label_json is not None:
        (ckpt / "id2label.json").write_text(json.dumps(id2label_json), encoding="utf-8")
    return ckpt


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_OUTPUTS_DIR", str(tmp_path))
    StubSER.built = []
    return load_config("gpu_24gb")


def test_checkpoint_loads_without_fallback(tmp_path, cfg_home):
    ckpt = make_ckpt(tmp_path)
    warnings = []
    model, info = ser_gpu.load_ser_with_fallback(cfg_home, "cpu", None, model_cls=StubSER, warn=warnings.append)
    assert info == {"name": str(ckpt), "fallback": False, "checkpoint": str(ckpt), "reason": None}
    assert StubSER.built == [str(ckpt)] and warnings == []


@pytest.mark.parametrize("case", ["missing_folder", "missing_json", "load_fails"])
def test_fallback_to_pretrained_with_one_warning(tmp_path, cfg_home, case):
    if case == "missing_json":
        make_ckpt(tmp_path, id2label_json=None)
    elif case == "load_fails":
        make_ckpt(tmp_path)
    warnings = []
    cls = (lambda *a, **k: StubSER(*a, fail=k.get("model_name", "").endswith("best"), **k)) \
        if case == "load_fails" else StubSER
    model, info = ser_gpu.load_ser_with_fallback(cfg_home, "cpu", None, model_cls=cls, warn=warnings.append)
    assert info["fallback"] is True and info["name"] == cfg_home["ser"]["model"]
    assert StubSER.built[-1] == cfg_home["ser"]["model"]
    assert len(warnings) == 1 and "fallback" in warnings[0] and cfg_home["ser"]["model"] in warnings[0]


def test_label_mismatch_stops_instead_of_falling_back(tmp_path, cfg_home):
    make_ckpt(tmp_path)
    cls = lambda *a, **k: StubSER(*a, id2label={0: "sad", 1: "hap", 2: "ang", 3: "neu"}, **k)  # noqa: E731
    with pytest.raises(ser_gpu.LabelOrderError):
        ser_gpu.load_ser_with_fallback(cfg_home, "cpu", None, model_cls=cls, warn=lambda m: None)
    assert cfg_home["ser"]["model"] not in StubSER.built


@pytest.mark.parametrize("info", [{"name": "C:/x/best", "fallback": False, "reason": None},
                                  {"name": "superb/hubert-large-superb-er", "fallback": True, "reason": "missing"}])
def test_ser_model_is_in_every_log_line(tmp_path, info):
    w = World(tmp_path)
    w.agent.ser_model = {"name": info["name"], "fallback": info["fallback"]}
    w.turn("Just a normal day.")
    w.turn("I want to kill myself.")                                    # crisis lines too
    lines = w.log_lines()
    assert "ser_model" in LOG_FIELDS and len(lines) == 2
    assert all(line["ser_model"] == {"name": info["name"], "fallback": info["fallback"]} for line in lines)


def test_live_agent_keeps_only_name_and_fallback(tmp_path):
    w = World(tmp_path)
    agent = live_agent_gpu.LiveAgent(w.cfg, asr=None, emotion=None, llm=None, voice=None, memory=None,
                                     logger=w.logger, ser_model={"name": "n", "fallback": True, "reason": "r",
                                                                 "checkpoint": "c"})
    assert agent.ser_model == {"name": "n", "fallback": True}


# ================================================================ 2. Rule B tau selection

def made_up(n_per_conf):
    """{confidence: (n correct, n wrong)} -> a probs frame (top-1 = confidence)."""
    rows = []
    for conf, (ok, bad) in n_per_conf.items():
        rest = (1 - conf) / 3
        p = {"neutral": conf, "happy": rest, "angry": rest, "sad": rest}
        rows += [{"true_label": "neutral", **{f"p_{c}": v for c, v in p.items()}}] * ok
        rows += [{"true_label": "sad", **{f"p_{c}": v for c, v in p.items()}}] * bad
    return pd.DataFrame(rows)


def test_rule_b_picks_max_val_accuracy_with_coverage_at_least_085():
    # 100 clips: 10 at 0.30 (all wrong), 5 at 0.50 (all wrong), 85 at 0.90 (80 right)
    val = made_up({0.30: (0, 10), 0.50: (0, 5), 0.90: (80, 5)})
    cur = act.curve(val, taus=[0.0, 0.30, 0.40, 0.60, 0.95])
    tau, row = act.choose_tau(cur)
    # 0.30 keeps 100 (0.80); 0.40 keeps 90 (80/90 = 0.889); 0.60 keeps 85 = exactly 0.85 coverage (80/85 = 0.941);
    # 0.95 keeps 0 (not eligible) -> 0.60
    assert tau == 0.60 and row["val_n_kept"] == 85 and row["val_accuracy"] == pytest.approx(80 / 85)


def test_rule_b_ties_go_to_the_smallest_tau_and_coverage_below_085_is_never_eligible():
    val = made_up({0.30: (0, 5), 0.50: (95, 0)})
    cur = act.curve(val, taus=[0.0, 0.40, 0.45, 0.60])
    tau, _ = act.choose_tau(cur)            # 0.40 and 0.45 both keep 95/100 at accuracy 1.0 -> the smaller
    assert tau == 0.40
    with pytest.raises(ValueError, match="no threshold"):
        act.choose_tau(act.curve(made_up({0.30: (0, 50), 0.50: (50, 0)}), taus=[0.0, 0.40]))


def test_rule_b_never_looks_at_test():
    val = made_up({0.30: (0, 10), 0.50: (5, 0), 0.90: (80, 5)})
    test_a = made_up({0.30: (40, 0), 0.90: (0, 60)})
    test_b = made_up({0.30: (0, 40), 0.90: (60, 0)})
    taus = [0.0, 0.30, 0.40, 0.60, 0.95]
    assert act.choose_tau(act.curve(val, test_a, taus))[0] == act.choose_tau(act.curve(val, test_b, taus))[0] \
        == act.choose_tau(act.curve(val, None, taus))[0]


def test_tau_grid_matches_the_a22_grid():
    import abstain_curve

    grid = act.tau_grid()
    assert grid[0] == 0.0 and grid[1] == 0.25 and grid[-1] == max(abstain_curve.TAUS)
    assert np.allclose(np.diff(grid[1:]), np.diff(abstain_curve.TAUS)[0])
    assert act.MIN_COVERAGE == 0.85 and act.CLASSES == ser_gpu.CLASSES


def test_curve_cli_refuses_non_val_rows(tmp_path):
    bad = made_up({0.5: (3, 1)}).assign(split="test")
    bad.to_csv(tmp_path / "v.csv", index=False)
    with pytest.raises(SystemExit, match="validation clips only"):
        act.main(["--val", str(tmp_path / "v.csv")])


def test_config_tau_conf_is_from_t3_rule_b():
    text = (CODE_DIR / "gpu" / "gpu_config.yaml").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("tau_conf:"))
    assert "Rule B" in line and RUN in line


# ================================================================ 3. silence gate

def test_gate_framing_matches_asr():
    import asr_gpu

    assert (msg.FRAME_SECONDS, msg.HOP_SECONDS, msg.SAMPLE_RATE) == \
        (asr_gpu.FRAME_SECONDS, asr_gpu.HOP_SECONDS, asr_gpu.SAMPLE_RATE)


def test_pass_level_agrees_with_the_asr_gate():
    import asr_gpu

    sr = 16000
    t = np.arange(sr) / sr
    y = np.concatenate([0.001 * np.sin(2 * np.pi * 200 * t), 0.05 * np.sin(2 * np.pi * 200 * t[: sr // 2])])
    min_speech = load_config("gpu_24gb")["asr"]["min_speech_seconds"]
    k = msg.frames_needed(min_speech)
    level = msg.pass_level(msg.frame_dbfs(y), k)
    assert asr_gpu.speech_seconds(y, sr, level - 0.01) >= min_speech       # just below: passes
    assert asr_gpu.speech_seconds(y, sr, level + 0.01) < min_speech        # just above: blocked
    assert msg.pass_level(np.zeros(3), k) == float("-inf")


def test_choose_gate():
    g = msg.choose_gate(-40.0, -60.0)
    assert g == {"gate": -50.0, "margin": 10.0, "gap": 20.0}
    assert msg.choose_gate(-60.0, -55.0)["gate"] is None


def test_missing_room_noise_prints_how_to_record_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(msg, "OUT_CSV", tmp_path / "silence_gate.csv")
    assert msg.main(["--noise", str(tmp_path / "nope.wav")]) == 0
    out = capsys.readouterr().out
    assert msg.RECORD_CMD in out and "python scripts/record.py room_noise --seconds 10" in out
    assert not (tmp_path / "silence_gate.csv").exists()
    with pytest.raises(SystemExit, match="test-only"):
        msg.main(["--noise", "data/my_voice_v3/x.wav"])


def test_gate_csv_with_stub_speech(tmp_path, monkeypatch, capsys):
    sr = 16000
    sf.write(str(tmp_path / "noise.wav"), (0.0005 * np.random.default_rng(0).standard_normal(sr)).astype(np.float32), sr)
    monkeypatch.setattr(msg, "OUT_CSV", tmp_path / "silence_gate.csv")
    monkeypatch.setattr(msg, "measure_speech", lambda k: pd.DataFrame(
        [{"utt_id": "a", "corpus": "ravdess", "label": "sad", "dbfs": -38.0},
         {"utt_id": "b", "corpus": "crema_d", "label": "happy", "dbfs": -20.0}]))
    assert msg.main(["--noise", str(tmp_path / "noise.wav")]) == 0
    rows = pd.read_csv(tmp_path / "silence_gate.csv")
    noise = rows.loc[rows.measure == "room_noise", "dbfs"].item()
    gate = rows.loc[rows.measure == "gate_midpoint", "dbfs"].item()
    assert gate == pytest.approx((noise + -38.0) / 2, abs=0.01)
    assert rows.loc[rows.measure == "current_silence_dbfs", "dbfs"].item() == \
        load_config("gpu_24gb")["asr"]["silence_dbfs"]                     # default untouched
    assert "NOT changed" in capsys.readouterr().out


# ================================================================ 4. push-to-talk

def test_mic_target_rate_matches_record_py():
    import record

    assert MIC_TARGET_SR == record.TARGET_SR == 16000


def test_chunks_to_16k_caps_and_resamples():
    blocks = [np.full((4800, 1), 0.1, dtype=np.float32) for _ in range(30)]       # 3 s at 48 kHz
    y = chunks_to_16k(blocks, 48000, max_seconds=2.0)
    assert y.dtype == np.float32 and y.ndim == 1 and abs(y.size - 32000) <= 2
    assert chunks_to_16k([], 48000, 2.0).size == 0
    assert chunks_to_16k([np.ones(160, np.float32)], 16000, 1.0).size == 160      # 16 kHz mic: no resample


class FakeSD:
    """sounddevice stand-in: InputStream feeds `blocks` to the callback when entered."""

    class CallbackStop(Exception):
        pass

    def __init__(self, blocks, rate=48000):
        self.blocks, self.rate, self.opened = blocks, rate, 0

    def query_devices(self, kind=None):
        return {"default_samplerate": float(self.rate)}

    def InputStream(self, samplerate, channels, dtype, callback):
        sd = self

        class Stream:
            def __enter__(self):
                sd.opened += 1
                for b in sd.blocks:
                    try:
                        callback(b, len(b), None, None)
                    except sd.CallbackStop:
                        break
                return self

            def __exit__(self, *a):
                return False

        assert channels == 1 and dtype == "float32" and samplerate == sd.rate
        return Stream()


def test_push_to_talk_records_until_enter_and_stops_at_the_cap():
    sd = FakeSD([np.full((4800, 1), 0.2, dtype=np.float32) for _ in range(50)])   # 5 s offered
    printed = []
    y = record_push_to_talk(2.0, input_fn=lambda prompt: "", sd=sd, print_fn=printed.append)
    assert sd.opened == 1 and abs(y.size - 2.0 * MIC_TARGET_SR) <= 4800 // 3 + 2
    assert any("cap" in p for p in printed)


def test_live_loop_runs_turns_until_q_and_prints_the_debug_view(tmp_path):
    w = World(tmp_path)
    inputs = iter(["", "", "q"])
    texts = iter(["First thing today.", "Second thing today."])
    out = []

    def record_fn(max_seconds, input_fn=None):
        w.asr.text = next(texts)
        assert max_seconds == float(w.cfg["ser"]["max_seconds"])
        return np.zeros(1600, dtype=np.float32)

    code = live_loop(w.agent, "alice", float(w.cfg["ser"]["max_seconds"]), input_fn=lambda p: next(inputs),
                     record_fn=record_fn, print_fn=out.append)
    assert code == 0 and len(w.log_lines()) == 2
    assert sum("transcript :" in o for o in out) == 2


def test_live_cli_asks_consent_before_loading_or_opening_the_mic(monkeypatch, capsys):
    monkeypatch.setattr(live_agent_gpu.safety_module, "consent_given", lambda user, *a: False)
    monkeypatch.setattr(live_agent_gpu.safety_module, "ask_consent", lambda user, *a: False)
    monkeypatch.setattr(live_agent_gpu, "load_real_parts", lambda *a, **k: pytest.fail("models loaded"))
    monkeypatch.setattr(live_agent_gpu, "record_push_to_talk", lambda *a, **k: pytest.fail("mic opened"))
    monkeypatch.setattr(sys, "argv", ["live_agent_gpu.py", "--live", "--user", "alice"])
    assert live_agent_gpu._main() == 1
    assert "No consent" in capsys.readouterr().out


def test_debug_view_shows_the_required_items(tmp_path):
    w = World(tmp_path)
    r = w.turn("Today was fine.")
    view = debug_view(r)
    for item in ("Today was fine.", "confidence 0.70", "kept", "neutral", "sad", "strategy", "reason", "csm",
                 "time to first audio 0.50 s"):
        assert item in view, item
    assert "ABSTAINED" in debug_view({**r, "abstained": True}) and "CHECK_IN" in debug_view({**r, "check_in": True})


# ================================================================ 5. delete all my data, by voice

def test_delete_phrase_is_the_one_consent_promises():
    assert f'"{safety.DELETE_PHRASE}"' in safety.CONSENT_TEXT.replace("\n", " ")
    assert safety.wants_delete_all("Please, DELETE all my data now.")
    assert not safety.wants_delete_all("delete all my old photos")
    assert [safety.is_spoken_yes(t) for t in ("Yes.", "yes, yes", "YES!")] == [True] * 3
    assert not any(safety.is_spoken_yes(t) for t in ("yeah", "yes but wait", "no", "", None, "yes no"))


def _with_memory(w):
    w.turn("I have an exam on Monday.", fact="The user has an exam on Monday.")
    assert len(w.memory.list_user_facts("alice")) == 1 and len(w.log_lines()) == 1


def test_delete_by_voice_confirmed(tmp_path):
    w = World(tmp_path)
    _with_memory(w)
    w.events.clear()
    r1 = w.turn("Okay, delete all my data please.", probs=SAD, fact="The user wants their data deleted.")
    assert r1["status"] == "delete_confirm" and w.voice.spoken[-1]["text"] == safety.DELETE_CONFIRM_PROMPT
    assert w.events == ["asr", "safety", "tts"]                       # no emotion / mood / memory / LLM / log
    assert len(w.log_lines()) == 1 and len(w.memory.list_user_facts("alice")) == 1
    r2 = w.turn("Yes.")
    assert r2["status"] == "deleted" and r2["report"]["memories_removed"] == 1
    assert w.voice.spoken[-1]["text"] == safety.DELETE_DONE
    assert w.memory.list_user_facts("alice") == [] and not w.logger.path_for("alice").exists()
    assert not safety.consent_given("alice", w.consent_path)
    assert w.turn("Hello again.")["status"] == "refused"


@pytest.mark.parametrize("answer", ["No.", "yeah", "Wait, actually my exam moved to Tuesday."])
def test_delete_by_voice_anything_but_yes_cancels_and_nothing_is_stored(tmp_path, answer):
    w = World(tmp_path)
    _with_memory(w)
    mood_before = w.agent._users["alice"]["mood"].get_mood_vector()
    turns_before = w.agent._users["alice"]["turns"]
    assert w.turn("delete all my data")["status"] == "delete_confirm"
    r = w.turn(answer, fact="The user's exam moved to Tuesday.")
    assert r["status"] == "delete_cancelled" and w.voice.spoken[-1]["text"] == safety.DELETE_CANCELLED
    assert len(w.memory.list_user_facts("alice")) == 1 and len(w.log_lines()) == 1     # neither turn logged
    assert w.agent._users["alice"]["mood"].get_mood_vector() == mood_before
    assert w.agent._users["alice"]["turns"] == turns_before
    assert all("delete all my data" not in u for u, _ in w.agent._users["alice"]["history"])
    r3 = w.turn("Back to normal now.")                                 # the conversation carries on
    assert r3["status"] == "ok" and len(w.log_lines()) == 2
    assert all("delete" not in line["transcript"].lower() for line in w.log_lines())


def test_crisis_wins_over_a_pending_deletion(tmp_path):
    w = World(tmp_path)
    _with_memory(w)
    w.turn("delete all my data")
    r = w.turn("I want to kill myself.")
    assert r["status"] == "crisis" and w.agent._users["alice"]["pending_delete"] is False
    assert w.turn("Yes.")["status"] == "ok"                            # a later "yes" deletes nothing
    assert len(w.memory.list_user_facts("alice")) == 1


# ================================================================ 6. scripted mode

def _wavs(folder, names):
    for n in names:
        sf.write(str(folder / n), np.zeros(1600, dtype=np.float32), 16000)


def test_parse_script_yaml_dict_list_and_csv(tmp_path):
    _wavs(tmp_path, ["a.wav", "b.wav"])
    (tmp_path / "s.yaml").write_text("user: rehearsal\nturns:\n  - wav: a.wav\n    label: sad\n  - b.wav\n",
                                     encoding="utf-8")
    user, turns = parse_script(tmp_path / "s.yaml")
    assert user == "rehearsal" and [t["wav"].name for t in turns] == ["a.wav", "b.wav"]
    assert [t["label"] for t in turns] == ["sad", None]
    (tmp_path / "l.yml").write_text("- a.wav\n- {wav: b.wav, label: happy}\n", encoding="utf-8")
    assert parse_script(tmp_path / "l.yml")[0] is None
    (tmp_path / "s.csv").write_text("wav,label\n# comment\na.wav,angry\n\nb.wav\n", encoding="utf-8")
    user, turns = parse_script(tmp_path / "s.csv")
    assert user is None and [(t["wav"].name, t["label"]) for t in turns] == [("a.wav", "angry"), ("b.wav", None)]
    (tmp_path / "s.txt").write_text(str(tmp_path / "a.wav") + "\n", encoding="utf-8")    # absolute path
    assert parse_script(tmp_path / "s.txt")[1][0]["wav"] == tmp_path / "a.wav"


def test_parse_script_refuses_missing_files_and_empty_scripts(tmp_path):
    (tmp_path / "s.csv").write_text("missing.wav,sad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not found"):
        parse_script(tmp_path / "s.csv")
    (tmp_path / "e.txt").write_text("# nothing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no wav"):
        parse_script(tmp_path / "e.txt")


def test_run_script_is_one_conversation_and_labels_never_reach_the_agent(tmp_path, monkeypatch):
    _wavs(tmp_path, ["a.wav", "b.wav", "c.wav"])
    (tmp_path / "s.csv").write_text("a.wav,sad\nb.wav,sad\nc.wav,happy\n", encoding="utf-8")
    _, turns = parse_script(tmp_path / "s.csv")
    w = World(tmp_path)
    texts = iter(["Rough day.", "Still rough.", "Better now."])
    seen = []
    real = w.agent.run_turn

    def run_turn(user_id, audio, sr=None):
        seen.append((user_id, audio))
        w.asr.text = next(texts)
        return real(user_id, audio, sr)

    monkeypatch.setattr(w.agent, "run_turn", run_turn)
    out = []
    results = run_script(w.agent, "alice", turns, print_fn=out.append)
    assert [r["status"] for r in results] == ["ok"] * 3
    assert [s[0] for s in seen] == ["alice"] * 3 and all(isinstance(s[1], Path) for s in seen)
    assert [line["turn_id"].split("-")[-1] for line in w.log_lines()] == ["1", "2", "3"]
    assert w.llm.calls[-1]["history"][0][0] == "Rough day."           # earlier turns are this turn's history
    assert any("label: sad (not given to the agent)" in o for o in out)
    assert all("sad" not in json.dumps(c) for c in w.llm.calls)       # the label never reaches the LLM


# ================================================================ scripts/ stay torch-free

@pytest.mark.parametrize("module", ["abstain_curve_t3", "measure_silence_gate", "safety"])
def test_new_scripts_import_no_torch_or_gpu(module):
    code = (f"import sys; sys.path.insert(0, {str(CODE_DIR / 'scripts')!r}); import {module}; "
            "print([m for m in ('torch', 'device', 'ser_gpu', 'asr_gpu', 'live_agent_gpu') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
