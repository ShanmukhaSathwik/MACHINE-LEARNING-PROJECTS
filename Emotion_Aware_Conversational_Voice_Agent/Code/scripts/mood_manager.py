"""
mood_manager.py
Minimal MoodManager: normalized EMA mood score + threshold + N-turn persistence.
Goes in src/mood/mood_manager.py
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