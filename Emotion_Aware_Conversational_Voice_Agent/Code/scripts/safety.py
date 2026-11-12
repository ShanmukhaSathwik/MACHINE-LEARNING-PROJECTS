"""
safety.py
Demo step 4: consent + crisis check. Brain only: no models, no torch, nothing from gpu/.

Consent
    CONSENT_TEXT is shown once per user. Consent is stored in Code/data/consent.json as
    {user_id: ISO-8601 UTC timestamp} and nothing else (no transcripts, no audio).
    consent.json is runtime data written by this code; it is git-ignored and never committed.
    A corrupt consent.json raises ConsentFileError; it is never silently replaced.

Crisis check
    check_crisis(transcript) looks for suicide/self-harm phrases (CRISIS_PHRASES) as whole words in
    the normalized transcript.

    (a) LIMITATION: a phrase list is a minimum safety floor, not a crisis classifier. It only sees
        the words of one transcript: it misses indirect wording, sarcasm, ASR errors and anything
        conveyed by tone, and it can fire on harmless sentences. It errs toward triggering.

    (b) CONTRACT for the orchestrator when check_crisis(...)["matched"] is True:
        - skip the emotion model, the mood update, memory retrieval, the LLM and fact extraction;
        - speak SAFE_RESPONSE with SAFE_STRATEGY (the neutral voice, no TTS style tags);
        - store no memory from that turn;
        - log the turn as a crisis override, including the matched phrase.

Demo:  python scripts/safety.py --demo
"""

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------------------------

CONSENT_TEXT = """\
Before we start, here is what this voice agent does with your data.

- What is recorded: your voice, one turn at a time, and only while you are talking.
- What is stored: short facts you tell the agent, plus session logs. They stay on this computer only.
- How to delete it: say "delete all my data" at any time to remove the facts and logs stored about you.
- This agent is not a counsellor and not a crisis service. If you are in danger or thinking about
  harming yourself, please contact a crisis line or emergency services.

Do you agree? Type yes to continue."""

# Resolved from this file's location (Code/scripts/safety.py -> Code/data/consent.json), not the cwd.
DEFAULT_CONSENT_PATH = Path(__file__).resolve().parent.parent / "data" / "consent.json"

_YES_ANSWERS = ("yes", "y")


class ConsentFileError(RuntimeError):
    """consent.json exists but cannot be read as {user_id: timestamp}. Fix or remove it by hand."""


def _load_consent(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ConsentFileError(
            f"Consent file {path} is corrupt ({e}). It was not changed; fix or remove it by hand."
        ) from e
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ConsentFileError(
            f"Consent file {path} is not a {{user_id: timestamp}} object. "
            "It was not changed; fix or remove it by hand."
        )
    return data


def _save_consent(path, data: dict) -> None:
    """Atomic write (temp file in the same folder + os.replace), so a crash never leaves a half file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _check_user_id(user_id) -> None:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError(f"user_id must be a non-empty string, got {user_id!r}")


def consent_given(user_id, path=DEFAULT_CONSENT_PATH) -> bool:
    """True if user_id has a consent entry. A missing or empty file means no consent."""
    _check_user_id(user_id)
    return user_id in _load_consent(path)


def record_consent(user_id, path=DEFAULT_CONSENT_PATH) -> None:
    """Add {user_id: ISO-8601 UTC timestamp}, keeping other users' entries. Stores nothing else."""
    _check_user_id(user_id)
    data = _load_consent(path)
    data[user_id] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save_consent(path, data)


def revoke_consent(user_id, path=DEFAULT_CONSENT_PATH) -> bool:
    """Remove only user_id's entry. True if one was removed. Called by step 5's "delete all my data"."""
    _check_user_id(user_id)
    data = _load_consent(path)
    if user_id not in data:
        return False
    del data[user_id]
    _save_consent(path, data)
    return True


def ask_consent(user_id, path=DEFAULT_CONSENT_PATH, input_fn=input, print_fn=print) -> bool:
    """Show CONSENT_TEXT; True (and recorded) only for "yes"/"y", case and surrounding spaces ignored."""
    _check_user_id(user_id)
    print_fn(CONSENT_TEXT)
    try:
        answer = input_fn("> ")
    except EOFError:
        answer = ""
    if (answer or "").strip().lower() in _YES_ANSWERS:
        record_consent(user_id, path)
        return True
    return False


# ---------------------------------------------------------------------------------------------
# Crisis check
# ---------------------------------------------------------------------------------------------

# Applied word by word after apostrophes are removed ("don't" -> "dont" -> "do not").
# Words that are also real words are left alone on purpose ("were", "ill", "id", "its", "well").
_CONTRACTIONS = {
    "im": "i am",
    "ive": "i have",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "cant": "can not",
    "cannot": "can not",
    "wont": "will not",
    "wouldnt": "would not",
    "couldnt": "could not",
    "shouldnt": "should not",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "havent": "have not",
    "hasnt": "has not",
    "youre": "you are",
    "theyre": "they are",
    "thats": "that is",
    "wanna": "want to",
    "gonna": "going to",
}

_CURLY_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"})


def normalize(text) -> str:
    """Lowercase; curly apostrophes -> straight -> removed; hyphens and all other punctuation -> spaces;
    spaces collapsed; common contractions unified ("im" -> "i am", "dont" -> "do not", ...)."""
    if not text:
        return ""
    t = str(text).lower().translate(_CURLY_APOSTROPHES)
    t = t.replace("'", "")
    t = re.sub(r"[^\w\s]|_", " ", t)          # hyphens and every other punctuation mark -> space
    words = t.split()                          # also collapses runs of whitespace
    return " ".join(_CONTRACTIONS.get(w, w) for w in words)


# Suicide / self-harm phrases, written in normalized form and matched as WHOLE WORDS on the
# normalized transcript, so "I want to KILL MYSELF." matches "kill myself" ("kill my self" is listed
# too, for ASR splits and "my-self"). Bare words that cause false alarms ("die", "dying", "dead", "kill")
# are deliberately absent: "this exam is killing me", "my phone died" and "I'm dead tired" must not
# trigger. Negation is NOT handled: "I don't want to kill myself" still matches (errs toward triggering).
CRISIS_PHRASES = (
    # killing / ending one's life
    "kill myself",
    "kill my self",
    "killing myself",
    "end my life",
    "end my own life",
    "ending my life",
    "take my own life",
    "take my life",
    "end it all",
    "ending it all",
    # wanting to be dead
    "want to die",
    "wish i was dead",
    "wish i were dead",
    "want to be dead",
    "better off dead",
    "suicide",
    "suicidal",
    # self-harm
    "hurt myself",
    "hurting myself",
    "harm myself",
    "harming myself",
    "self harm",
    "self harming",
    "selfharm",
    "cut myself",
    "cutting myself",
    # not wanting to live
    "do not want to live",
    "do not want to be alive",
    "no reason to live",
    "not worth living",
    "not want to be here anymore",
    "no point in living",
)

# Normalized once at import so a phrase written with punctuation still compares like a transcript.
_NORMALIZED_PHRASES = tuple((p, normalize(p)) for p in CRISIS_PHRASES)

SAFE_RESPONSE = (
    "I'm really glad you told me. I'm not able to help with this the way a person can, "
    "but you don't have to go through it alone. In the US you can call or text 988 to reach "
    "the Suicide and Crisis Lifeline, any time. If you're in immediate danger, please call 911. "
    "Is there someone you trust you could reach out to right now?"
)
SAFE_STRATEGY = "NEUTRAL"   # crisis replies always use the neutral voice


def check_crisis(transcript) -> dict:
    """{"matched": bool, "phrase": str or None}. Empty or None transcript -> not matched.

    Whole-word phrase match on normalize(transcript). Errs toward triggering: negated sentences
    such as "I don't want to kill myself" still match, because a missed crisis costs far more than
    a gentle safe response. This is a safety floor, not a classifier (see the module docstring).
    """
    padded = f" {normalize(transcript)} "
    if not padded.strip():
        return {"matched": False, "phrase": None}
    for phrase, norm in _NORMALIZED_PHRASES:
        if f" {norm} " in padded:
            return {"matched": True, "phrase": phrase}
    return {"matched": False, "phrase": None}


# ---------------------------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------------------------

DEMO_SENTENCES = (
    # should match
    "I want to KILL MYSELF.",
    "i’m thinking about self-harm",
    "Im   suicidal",
    "I don't want to live anymore",
    "I don't want to kill myself",
    # should not match
    "this exam is killing me",
    "I'm dying to see that movie",
    "my phone died",
    "kill the process",
    "I could die of embarrassment",
    "that joke killed me",
    "this traffic is killing me",
    "I'm dead tired",
    "",
)


def _demo() -> None:
    # ascii() so the curly apostrophe prints as ’ on any console encoding
    width = max(len(ascii(s)) for s in DEMO_SENTENCES)
    for s in DEMO_SENTENCES:
        r = check_crisis(s)
        status = "MATCHED " if r["matched"] else "no match"
        phrase = r["phrase"] if r["phrase"] else "-"
        print(f"{ascii(s):<{width}}  {status}  phrase={phrase}")
    print(f"\nSAFE_STRATEGY = {SAFE_STRATEGY}\nSAFE_RESPONSE = {SAFE_RESPONSE}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Consent + crisis check (demo step 4).")
    ap.add_argument("--demo", action="store_true", help="run check_crisis on the test sentences")
    args = ap.parse_args(argv)
    if args.demo:
        _demo()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
