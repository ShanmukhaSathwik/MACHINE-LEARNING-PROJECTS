"""
test_mood_manager_v2.py - demo step 3: MoodManagerV2 + reactive_strategy_v2 (scripts/mood_manager.py).

Hand calculation used below (gamma 0.7, m starts all neutral):
  a class fed p every turn reaches  m_t = p * (1 - 0.7**t)
  p = 0.80:  t1 0.240  t2 0.408  t3 0.5256  t4 0.6079   -> first >= 0.5 at turn 3 (target), switch at turn 4 (persistence 2)
  p = 0.30:  t1 0.090  t2 0.153  t3 0.1971  t4 0.2280  t5 0.2496  t6 0.2647
             sad + angry (0.3 each) = 2x that: t5 0.4992 (< 0.5), t6 0.5294  -> target at turn 6, switch at turn 7
"""
import pytest

from mood_manager import (MoodManager, MoodManagerV2, V2_CLASSES, reactive_strategy_v2, strategy_for,
                          tts_tags_for)


def dist(neutral=0.0, happy=0.0, angry=0.0, sad=0.0):
    return {"neutral": neutral, "happy": happy, "angry": angry, "sad": sad}


NEUTRAL = dist(neutral=0.80, happy=0.10, angry=0.05, sad=0.05)
SAD = dist(neutral=0.10, happy=0.05, angry=0.05, sad=0.80)
ANGRY = dist(neutral=0.10, happy=0.05, angry=0.80, sad=0.05)
HAPPY = dist(neutral=0.10, happy=0.80, angry=0.05, sad=0.05)


def run(mm, seq, confidence=1.0):
    return [mm.update(p, confidence)["strategy"] for p in seq]


def first_turn(strategies, value):
    return strategies.index(value) + 1


def test_starts_default_and_m_sums_to_one():
    mm = MoodManagerV2()
    assert mm.mode == "DEFAULT" and mm.strategy == "NEUTRAL"
    assert mm.m == {"neutral": 1.0, "happy": 0.0, "angry": 0.0, "sad": 0.0}
    # dist(...) sums to 0.9995: accepted (within 1e-3) and rescaled, so m still sums to exactly 1
    for p in [NEUTRAL, SAD, ANGRY, HAPPY, dist(0.25, 0.25, 0.25, 0.2495), SAD, NEUTRAL] * 5:
        mm.update(p, 1.0)
        assert abs(sum(mm.m.values()) - 1.0) < 1e-9


@pytest.mark.parametrize("probs,strategy", [(SAD, "VALIDATE"), (ANGRY, "CALM"), (HAPPY, "ENCOURAGE")])
def test_sustained_emotion_switches_at_hand_calculated_turn(probs, strategy):
    mm = MoodManagerV2()
    out = run(mm, [probs] * 6)
    # m for the fed class: 0.240, 0.408, 0.5256 (target, 1/2), 0.6079 (2/2 -> switch)
    assert out[:3] == ["NEUTRAL"] * 3
    assert first_turn(out, strategy) == 4
    assert out[3:] == [strategy] * 3


def test_sustained_neutral_stays_neutral():
    assert run(MoodManagerV2(), [NEUTRAL] * 10) == ["NEUTRAL"] * 10


def test_hand_calculated_m_values():
    mm = MoodManagerV2()
    for t, expected in enumerate([0.240, 0.408, 0.5256, 0.6079], 1):
        mm.update(SAD, 1.0)
        assert mm.m["sad"] == pytest.approx(expected, abs=1e-4), t


def test_one_off_angry_spike_causes_no_switch():
    mm = MoodManagerV2()
    out, peak_angry, targets = [], 0.0, []
    for p in [NEUTRAL, NEUTRAL, ANGRY, NEUTRAL, NEUTRAL, NEUTRAL]:
        out.append(mm.update(p, 1.0)["strategy"])
        peak_angry, targets = max(peak_angry, mm.m["angry"]), targets + [mm.target]
    assert out == ["NEUTRAL"] * 6 and set(targets) == {"DEFAULT"}
    # hand calc: m_angry 0.015, 0.0255, then the spike 0.7*0.0255 + 0.3*0.8 = 0.2579, far below 0.5
    assert peak_angry == pytest.approx(0.2579, abs=1e-4)


def test_pooled_distress_and_larger_wins():
    mixed = dist(neutral=0.40, angry=0.30, sad=0.30)
    mm = MoodManagerV2()
    out = run(mm, [mixed] * 8)
    assert max(mm.m["sad"], mm.m["angry"]) < 0.5                     # neither class alone reaches 0.5
    assert out[:6] == ["NEUTRAL"] * 6                                 # pooled 0.4992 at t5; target at t6 (1/2)
    assert first_turn(out, "VALIDATE") == 7 and out[7] == "VALIDATE"
    assert "pooled distress" in mm.explain()
    # both sad and angry above a 0.4 threshold: the larger one wins
    mm_sad = MoodManagerV2(gamma=0.0, threshold=0.4, persistence=1)   # gamma 0: m = this turn's probs
    assert mm_sad.update(dist(angry=0.45, sad=0.55), 1.0)["strategy"] == "VALIDATE"
    mm_ang = MoodManagerV2(gamma=0.0, threshold=0.4, persistence=1)
    assert mm_ang.update(dist(angry=0.55, sad=0.45), 1.0)["strategy"] == "CALM"


def test_recovery_after_persistence_turns():
    for persistence in (2, 3):
        mm = MoodManagerV2(persistence=persistence)
        run(mm, [SAD] * 6)
        assert mm.strategy == "VALIDATE"
        strategies, targets, reasons = [], [], []
        for _ in range(12):
            prof = mm.update(NEUTRAL, 1.0)
            strategies.append(prof["strategy"])
            targets.append(mm.target)
            reasons.append(prof["reason"])
        k = targets.index("DEFAULT")                                  # first neutral-target turn
        assert strategies[k:k + persistence - 1] == ["VALIDATE"] * (persistence - 1)
        assert strategies[k + persistence - 1] == "NEUTRAL"           # back after exactly `persistence` turns
        assert reasons[k].startswith("target NEUTRAL") and f"1/{persistence} turns, holding VALIDATE" in reasons[k]
        assert "(switched from VALIDATE)" in reasons[k + persistence - 1]


def test_abstain_holds_state_and_raises_check_in():
    mm = MoodManagerV2(tau_conf=0.5, check_in_after=2)
    run(mm, [SAD] * 5)
    assert mm.strategy == "VALIDATE"
    before = (dict(mm.m), mm.mode, mm.target, mm.turns_on_target)
    p1 = mm.update(ANGRY, confidence=0.3)
    assert (dict(mm.m), mm.mode, mm.target, mm.turns_on_target) == before
    assert mm.abstained and mm.consecutive_abstains == 1 and p1["check_in"] is False
    assert mm.explain().startswith("abstained (confidence 0.30 < 0.50), holding VALIDATE")
    p2 = mm.update(ANGRY, confidence=0.3)
    assert p2["check_in"] is True and mm.consecutive_abstains == 2
    assert (dict(mm.m), mm.mode) == (before[0], before[1])
    p3 = mm.update(SAD, confidence=0.9)                               # a normal turn resets
    assert p3["check_in"] is False and mm.consecutive_abstains == 0 and not mm.abstained
    # in DEFAULT, abstains never raise check_in
    mm0 = MoodManagerV2(tau_conf=0.5)
    assert [mm0.update(NEUTRAL, 0.1)["check_in"] for _ in range(3)] == [False, False, False]


def test_profile_keys_match_v1_and_tts_tags():
    v1_keys = set(MoodManager().get_current_profile())
    mm = MoodManagerV2()
    assert set(mm.get_current_profile()) == v1_keys
    for p in [NEUTRAL, SAD, SAD, SAD, ANGRY, ANGRY, ANGRY, HAPPY, HAPPY, HAPPY, HAPPY]:
        prof = mm.update(p, 1.0)
        assert set(prof) == v1_keys
        assert prof["strategy"] == strategy_for(prof["mode"])
        assert prof["tts_tags"] == tts_tags_for(prof["strategy"])
    vec = mm.get_mood_vector()
    assert set(vec) == set(V2_CLASSES) | {"target", "turns_on_target", "consecutive_abstains"}


@pytest.mark.parametrize("bad", [
    {"neutral": 0.5, "happy": 0.5, "angry": 0.0},                    # missing class
    dist(neutral=0.9, happy=0.3, angry=0.2, sad=0.1),                 # sum 1.5
    dist(neutral=0.6, happy=0.2, angry=0.1, sad=0.0),                 # sum 0.9
    dist(neutral=1.1, happy=-0.1),                                    # out of [0, 1]
    {**dist(neutral=1.0), "fearful": 0.0},                            # unknown class
])
def test_bad_input_raises(bad):
    with pytest.raises(ValueError):
        MoodManagerV2().update(bad, 1.0)
    with pytest.raises(ValueError):
        reactive_strategy_v2(bad)


def test_from_config_reads_brain_block():
    from device import load_config   # tests may read the config; mood_manager itself imports nothing from gpu/

    brain = load_config("smoke_cpu")["brain"]
    mm = MoodManagerV2.from_config(brain)
    assert (mm.gamma, mm.threshold, mm.persistence, mm.check_in_after, mm.tau_conf) == \
        (brain["gamma"], brain["threshold"], brain["persistence"], brain["check_in_after"], brain["tau_conf"])
    custom = MoodManagerV2.from_config({"gamma": 0.5, "threshold": 0.6, "persistence": 3,
                                        "check_in_after": 4, "tau_conf": 0.4})
    assert (custom.gamma, custom.threshold, custom.persistence, custom.check_in_after, custom.tau_conf) == \
        (0.5, 0.6, 3, 4, 0.4)


def test_reactive_strategy_v2_priority_rules():
    assert reactive_strategy_v2(dist(neutral=0.4, sad=0.6)) == "VALIDATE"                   # (a) sad
    assert reactive_strategy_v2(dist(neutral=0.4, angry=0.6)) == "CALM"                     # (a) angry
    assert reactive_strategy_v2(dist(neutral=0.4, angry=0.3, sad=0.3)) == "VALIDATE"        # (b) pooled
    assert reactive_strategy_v2(dist(neutral=0.4, happy=0.6)) == "ENCOURAGE"                # (c) happy
    assert reactive_strategy_v2(dist(neutral=0.4, happy=0.45, angry=0.15)) == "NEUTRAL"     # (d)
    assert reactive_strategy_v2(dist(happy=0.5, angry=0.25, sad=0.25)) == "VALIDATE"        # (b) before (c)
    assert reactive_strategy_v2(dist(angry=0.45, sad=0.55), threshold=0.4) == "VALIDATE"    # larger wins
    assert reactive_strategy_v2(dist(angry=0.55, sad=0.45), threshold=0.4) == "CALM"
    # single turn, no memory: one spike is enough (unlike MoodManagerV2)
    assert reactive_strategy_v2(ANGRY) == "CALM" and MoodManagerV2().update(ANGRY, 1.0)["strategy"] == "NEUTRAL"
