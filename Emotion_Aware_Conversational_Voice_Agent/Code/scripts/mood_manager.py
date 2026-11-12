"""
mood_manager.py
Minimal MoodManager: normalized EMA mood score + threshold + N-turn persistence.
MoodManagerV2 (demo step 3): EMA over all 4 class probabilities -> DEFAULT / EMPATHETIC / CALMING /
POSITIVE_ENCOURAGING, with one persistence rule for entering and leaving a mode.
Goes in src/mood/mood_manager.py

Demo:  python scripts/mood_manager.py --demo-v2
"""

# A2.2: chosen by scripts/abstain_curve.py --min-coverage 0.85 (Rule B: max val accuracy with
# val coverage >= 0.85). Rule A (coverage >= 0.70) gave 0.82; see ROADMAP A2.2 for both results.
TAU_CONF = 0.72
CHECK_IN_AFTER = 2   # consecutive abstains in a non-DEFAULT mood before raising check_in

# A2.3 (P3, code decides, LLM speaks): the mood mode picks a response strategy; the LLM only phrases it.
# v1 produces DEFAULT / EMPATHETIC only; CALMING / POSITIVE_ENCOURAGING are kept for MoodManager v2.
STRATEGY_FOR_MODE = {
    "DEFAULT": "NEUTRAL",
    "EMPATHETIC": "VALIDATE",
    "CALMING": "CALM",
    "POSITIVE_ENCOURAGING": "ENCOURAGE",
}
# TTS style tags come from the strategy only, never from the per-turn emotion prediction (P6).
TTS_TAGS = {
    "VALIDATE": ["<sigh>", "<softly>"],
    "CALM": ["<slowly>", "<softly>"],
    "ENCOURAGE": ["<laughs>", "<brightly>"],
    "NEUTRAL": [],
}


def strategy_for(mode: str) -> str:
    return STRATEGY_FOR_MODE[mode]


def tts_tags_for(strategy: str) -> list:
    return list(TTS_TAGS[strategy])


class MoodManager:
    def __init__(self, gamma=0.7, threshold=0.5, persistence=2):
        self.gamma = gamma              # decay rate (how much the old mood is kept)
        self.threshold = threshold      # score needed to count as "distressed"
        self.persistence = persistence  # consecutive turns above threshold before switching
        self.score = 0.0
        self.turns_above = 0
        self.mode = "DEFAULT"
        # Inspection state for explain(); written by update(), never read by its math
        self.last_p = None
        self.prev_mode = None
        self.prev_turns_above = None
        self.last_confidence = None
        self.tau_conf = None
        # Abstain state (A2.2)
        self.abstained = False
        self.consecutive_abstains = 0
        self.check_in = False

    def update(self, p_distress: float, confidence=None, tau_conf=TAU_CONF) -> str:
        self.last_p = p_distress
        self.last_confidence = confidence
        self.tau_conf = tau_conf
        self.prev_mode = self.mode
        self.prev_turns_above = self.turns_above

        # 0) abstain: classifier not confident -> hold score, turns_above and mode
        self.abstained = confidence is not None and confidence < tau_conf
        self.consecutive_abstains = self.consecutive_abstains + 1 if self.abstained else 0
        self.check_in = self.consecutive_abstains >= CHECK_IN_AFTER and self.mode != "DEFAULT"
        if self.abstained:
            return self.mode

        # 1) EMA update: m_t = gamma * m_{t-1} + (1 - gamma) * p_t
        self.score = self.gamma * self.score + (1 - self.gamma) * p_distress

        # 2) persistence gating
        if self.score >= self.threshold:
            self.turns_above += 1
            if self.turns_above >= self.persistence:
                self.mode = "EMPATHETIC"
        else:
            self.turns_above = 0
            self.mode = "DEFAULT"
        return self.mode

    def explain(self) -> str:
        """One-line reason for the mode chosen by the last update()."""
        if self.last_p is None:
            return "no update yet -> DEFAULT"
        if self.abstained:
            reason = (f"confidence {self.last_confidence:.2f} < tau {self.tau_conf:.2f} -> ABSTAIN, "
                      f"mood held at {self.score:.2f} ({self.mode})")
            if self.consecutive_abstains > 1:
                reason += f", {self.consecutive_abstains} abstains in a row"
            if self.check_in:
                reason += " -> CHECK_IN"
            return reason
        thr = f"threshold {self.threshold:.2f}"
        if self.score >= self.threshold:
            n, k = self.turns_above, self.persistence
            turns = f"for {n}/{k} turns" if n <= k else f"for {n} turns (>= {k})"
            reason = f"score {self.score:.2f} >= {thr} {turns} -> {self.mode}"
            if n < k:
                reason += f" (needs {k} consecutive turns)"
        else:
            reason = f"score {self.score:.2f} < {thr} -> {self.mode}"
            if self.prev_turns_above:
                reason += " (turns_above reset)"
        if self.mode != self.prev_mode:
            reason += f" [switched from {self.prev_mode}]"
        return reason

    @property
    def strategy(self) -> str:
        return strategy_for(self.mode)

    def get_current_profile(self) -> dict:
        """Everything downstream (prompt, TTS, log) needs from the mood state, decided in code."""
        return {
            "mode": self.mode,
            "strategy": self.strategy,
            "check_in": self.check_in,
            "reason": self.explain(),
            "tts_tags": tts_tags_for(self.strategy),
        }


def reactive_mode(p_distress: float, threshold=0.5) -> str:
    """Baseline: react to each turn in isolation, no memory of previous turns."""
    return "EMPATHETIC" if p_distress >= threshold else "DEFAULT"


# MoodManagerV2 (demo step 3)
# Smooths all 4 class probabilities instead of a single p_distress, so it can tell sadness (VALIDATE),
# anger (CALM) and happiness (ENCOURAGE) apart. Code decides the strategy; the LLM only phrases it (P3).

V2_CLASSES = ["neutral", "happy", "angry", "sad"]
_V2_SUM_TOL = 1e-3


def _validate_probs(probs) -> dict:
    """The 4 class probabilities as floats, rescaled to sum to exactly 1. ValueError on bad input."""
    if not isinstance(probs, dict):
        raise ValueError(f"probs must be a dict of {V2_CLASSES}, got {type(probs).__name__}")
    missing = [c for c in V2_CLASSES if c not in probs]
    extra = [c for c in probs if c not in V2_CLASSES]
    if missing or extra:
        raise ValueError(f"probs must have exactly {V2_CLASSES}; missing {missing}, unexpected {extra}")
    p = {c: float(probs[c]) for c in V2_CLASSES}
    bad = {c: v for c, v in p.items() if not 0.0 <= v <= 1.0}   # also catches NaN
    if bad:
        raise ValueError(f"probabilities must be in [0, 1]: {bad}")
    total = sum(p.values())
    if abs(total - 1.0) > _V2_SUM_TOL:
        raise ValueError(f"probabilities must sum to 1 (within {_V2_SUM_TOL}), got {total:.4f}")
    return {c: v / total for c, v in p.items()}   # exact sum 1, so the smoothed state stays a distribution


def _target_v2(v, threshold):
    """Target mode for a 4-class distribution v, in priority order. Returns (mode, rule, detail).
    (a) sad or angry >= threshold: the larger of the two (a tie goes to sad -> EMPATHETIC, the gentler choice)
    (b) sad + angry >= threshold: EMPATHETIC (pooled distress, as in v1)
    (c) happy >= threshold: POSITIVE_ENCOURAGING
    (d) otherwise DEFAULT"""
    sad, angry, happy = v["sad"], v["angry"], v["happy"]
    if sad >= threshold or angry >= threshold:
        if sad >= angry:
            return "EMPATHETIC", "sad", f"sad {sad:.2f}"
        return "CALMING", "angry", f"angry {angry:.2f}"
    if sad + angry >= threshold:
        return "EMPATHETIC", "pooled", f"pooled distress {sad + angry:.2f} (sad {sad:.2f} + angry {angry:.2f})"
    if happy >= threshold:
        return "POSITIVE_ENCOURAGING", "happy", f"happy {happy:.2f}"
    return "DEFAULT", "none", f"distress {sad + angry:.2f}, happy {happy:.2f}, both < {threshold:.2f}"


def reactive_strategy_v2(probs, threshold=0.5) -> str:
    """Baseline: the v2 priority rules on this turn's probabilities alone (no smoothing, no persistence)."""
    mode, _, _ = _target_v2(_validate_probs(probs), threshold)
    return strategy_for(mode)


class MoodManagerV2:
    """Smoothed 4-class mood state -> mode -> strategy.

    Each normal turn: m_c = gamma * m_c + (1 - gamma) * p_c for every class (m starts all neutral).
    The target mode comes from m (see _target_v2). The mode switches only when the same target has held for
    `persistence` consecutive non-abstain turns and differs from the current mode: one rule for entering and leaving.
    Abstain (confidence < tau_conf): m, target and its turn count stay unchanged; after `check_in_after`
    abstains in a row outside DEFAULT, check_in is raised. A normal turn resets the abstain count."""

    CONFIG_KEYS = ("gamma", "threshold", "persistence", "check_in_after", "tau_conf")

    def __init__(self, gamma=0.7, threshold=0.5, persistence=2, check_in_after=2, tau_conf=0.0):
        self.gamma = gamma                    # how much of the previous mood carries into the next turn
        self.threshold = threshold            # smoothed probability needed for a class (or pooled distress) to count
        self.persistence = persistence        # consecutive turns a new target must hold before the mode switches
        self.check_in_after = check_in_after  # abstains in a row (outside DEFAULT) before check_in
        self.tau_conf = tau_conf              # confidence below this -> abstain (0.0 = never)
        self.m = {c: (1.0 if c == "neutral" else 0.0) for c in V2_CLASSES}
        self.mode = "DEFAULT"
        self.target = "DEFAULT"
        self.turns_on_target = 0
        self.consecutive_abstains = 0
        self.abstained = False
        self.check_in = False
        # Inspection state for explain(); written by update(), never read by its math
        self.n_updates = 0
        self.prev_mode = None
        self.last_confidence = None
        self.rule, self.detail = "none", ""

    @classmethod
    def from_config(cls, brain: dict):
        """Build from gpu_config.yaml's brain block, passed in as a plain dict."""
        return cls(**{k: brain[k] for k in cls.CONFIG_KEYS})

    def update(self, probs: dict, confidence=None) -> dict:
        """One turn. probs: {neutral, happy, angry, sad}. confidence None = never abstain. Returns the profile."""
        p = _validate_probs(probs)
        self.n_updates += 1
        self.prev_mode = self.mode
        self.last_confidence = confidence

        self.abstained = confidence is not None and confidence < self.tau_conf
        if self.abstained:
            self.consecutive_abstains += 1
            self.check_in = self.consecutive_abstains >= self.check_in_after and self.mode != "DEFAULT"
            return self.get_current_profile()
        self.consecutive_abstains = 0
        self.check_in = False

        for c in V2_CLASSES:
            self.m[c] = self.gamma * self.m[c] + (1 - self.gamma) * p[c]
        target, self.rule, self.detail = _target_v2(self.m, self.threshold)
        self.turns_on_target = self.turns_on_target + 1 if target == self.target else 1
        self.target = target
        if self.target != self.mode and self.turns_on_target >= self.persistence:
            self.mode = self.target
        return self.get_current_profile()

    @property
    def strategy(self) -> str:
        return strategy_for(self.mode)

    def explain(self) -> str:
        """One plain line on why the agent uses its current strategy."""
        if self.n_updates == 0:
            return f"no update yet -> {self.strategy}"
        if self.abstained:
            reason = (f"abstained (confidence {self.last_confidence:.2f} < {self.tau_conf:.2f}), "
                      f"holding {self.strategy}")
            if self.consecutive_abstains > 1:
                reason += f", {self.consecutive_abstains} abstains in a row"
            return reason + (" -> CHECK_IN" if self.check_in else "")
        target_strategy = strategy_for(self.target)
        if self.target != self.mode:
            return (f"target {target_strategy} ({self.detail}) {self.turns_on_target}/{self.persistence} turns, "
                    f"holding {self.strategy}")
        if self.rule in ("sad", "angry", "happy"):
            reason = f"{self.detail} >= {self.threshold:.2f} for {self.turns_on_target} turns -> {self.strategy}"
        else:   # pooled distress, or nothing above threshold
            reason = f"{self.detail} -> {self.strategy}"
        if self.mode != self.prev_mode:
            reason += f" (switched from {strategy_for(self.prev_mode)})"
        return reason

    def get_current_profile(self) -> dict:
        """Same keys as MoodManager.get_current_profile(): {mode, strategy, check_in, reason, tts_tags}."""
        return {
            "mode": self.mode,
            "strategy": self.strategy,
            "check_in": self.check_in,
            "reason": self.explain(),
            "tts_tags": tts_tags_for(self.strategy),
        }

    def get_mood_vector(self) -> dict:
        """Smoothed state for logging."""
        return {**{c: self.m[c] for c in V2_CLASSES}, "target": self.target,
                "turns_on_target": self.turns_on_target, "consecutive_abstains": self.consecutive_abstains}


def _demo_v2():
    """12-turn synthetic trace: neutral, neutral, one angry spike, neutral, sad x4, neutral x4."""
    neutral = {"neutral": 0.80, "happy": 0.10, "angry": 0.05, "sad": 0.05}
    angry = {"neutral": 0.10, "happy": 0.05, "angry": 0.80, "sad": 0.05}
    sad = {"neutral": 0.10, "happy": 0.05, "angry": 0.05, "sad": 0.80}
    trace = ([("neutral", neutral)] * 2 + [("ANGRY", angry), ("neutral", neutral)]
             + [("SAD", sad)] * 4 + [("neutral", neutral)] * 4)
    mm = MoodManagerV2()
    print(f"MoodManagerV2 demo: gamma {mm.gamma}, threshold {mm.threshold}, persistence {mm.persistence}\n")
    print(f"{'turn':>4} {'input':8s} {'p(n/h/a/s)':19s} {'m(n/h/a/s)':23s} {'target':21s} {'mode':21s} strategy")
    for t, (name, p) in enumerate(trace, 1):
        prof = mm.update(p, confidence=1.0)
        pv = "/".join(f"{p[c]:.2f}" for c in V2_CLASSES)
        mv = "/".join(f"{mm.m[c]:.3f}" for c in V2_CLASSES)
        print(f"{t:>4} {name:8s} {pv:19s} {mv:23s} {mm.target:21s} {prof['mode']:21s} {prof['strategy']}")
        print(f"{'':14}{prof['reason']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoodManager v1/v2.")
    parser.add_argument("--demo-v2", action="store_true", help="print a 12-turn synthetic MoodManagerV2 trace")
    args = parser.parse_args()
    if args.demo_v2:
        _demo_v2()
    else:
        parser.print_help()