"""
fact_extractor.py
Demo step 5: turn ONE user transcript into at most one durable fact ("The user ...") for memory_manager.py.
Brain only: no torch, no models, nothing imported from gpu/. The LLM is passed in as llm_fn(prompt) -> str.

  - build_extraction_prompt(user_transcript, emotional_spike=False) -> prompt (no probabilities, no emotion labels)
  - parse_extraction(raw) -> fact or None   (NONE / empty / too long -> None; quotes and bullets stripped)
  - extract_fact(user_transcript, llm_fn, emotional_spike=False) -> fact or None
      Takes ONLY the user's transcript: the agent's reply is never a source of facts.
      A crisis transcript (safety.check_crisis) returns None without calling llm_fn (safety contract: no memory
      from a crisis turn). Any llm_fn error returns None.

Demo:  python scripts/fact_extractor.py --demo [--backend json|chroma]   (stub LLM, temporary store; two sessions,
           session 2 shares no content word with the stored fact; default backend = memory.backend)
       python scripts/fact_extractor.py --live-gemini   (needs GEMINI_API_KEY; never writes the real memory file)
"""

import argparse
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from safety import check_crisis

TRANSCRIPT_MARKER = "What the user said:"
_BULLET = re.compile(r"^\s*(?:[-*•>]+|\d+[.)])\s*")
_QUOTES = "\"'`“”‘’"
_PREFIX = re.compile(r"^(?:fact|answer|memory)\s*:\s*", re.IGNORECASE)


@lru_cache(maxsize=1)
def default_max_fact_words() -> int:
    from memory_manager import load_memory_config
    return int(load_memory_config()["memory"]["max_fact_words"])


def build_extraction_prompt(user_transcript, emotional_spike=False, max_fact_words=None) -> str:
    n = int(max_fact_words or default_max_fact_words())
    if emotional_spike:
        feeling = ("- The user sounded strongly emotional while saying this. If they say what the feeling is about, "
                   "include how they feel about it in the fact.\n")
    else:
        feeling = "- Only include a feeling if the user names it and it keeps coming back.\n"
    return (
        "You decide whether something a user said to a voice assistant contains ONE durable fact about the user "
        "that is worth remembering in future conversations.\n\n"
        "Durable means: an upcoming event (something that will happen later), a stable preference, a relationship, "
        "or a feeling that keeps coming back. Small talk, greetings, questions to the assistant and momentary "
        "reactions are not durable.\n\n"
        "Rules:\n"
        f"- Write the fact in the third person, starting with \"The user\", as one sentence of at most {n} words.\n"
        "- Summarize in your own words; do not quote the user.\n"
        "- Do not include other people's names, addresses, or any numbers. Say \"a friend\" or \"their sister\" "
        "instead of a name.\n"
        f"{feeling}"
        "- If there is no durable fact, reply with exactly NONE.\n"
        "- Reply with the fact or NONE and nothing else.\n\n"
        f"{TRANSCRIPT_MARKER}\n\"\"\"{str(user_transcript).strip()}\"\"\""
    )


def parse_extraction(raw, max_fact_words=None):
    """The LLM's answer -> one fact, or None for NONE / empty / more than max_fact_words words."""
    if not isinstance(raw, str):
        return None
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    line = lines[0]
    for _ in range(3):                                 # e.g. '- "Fact: The user ..."'
        line = _PREFIX.sub("", _BULLET.sub("", line)).strip().strip(_QUOTES).strip()
    if not line or line.rstrip(".!").strip().lower() == "none":
        return None
    if len(line.split()) > int(max_fact_words or default_max_fact_words()):
        return None
    return line


def extract_fact(user_transcript, llm_fn, emotional_spike=False):
    """One durable fact about the user from their own words, or None. Never raises on an llm_fn error."""
    if not user_transcript or not str(user_transcript).strip():
        return None
    if check_crisis(user_transcript)["matched"]:
        return None                                    # safety contract: store no memory from a crisis turn
    try:
        raw = llm_fn(build_extraction_prompt(user_transcript, emotional_spike))
    except Exception:  # noqa: BLE001 - any LLM failure just means no fact this turn
        return None
    return parse_extraction(raw)


# Demo


def _transcript_from_prompt(prompt) -> str:
    return prompt.rsplit(TRANSCRIPT_MARKER, 1)[-1].strip().strip('"').lower()


def stub_llm(prompt) -> str:
    """Deterministic stand-in for the LLM, keyed on the transcript only."""
    said = _transcript_from_prompt(prompt)
    if "exam" in said:
        return "The user has a big exam on Monday and is nervous about it."
    if "hiking" in said:
        return "- \"The user enjoys hiking on weekends.\""
    return "NONE"


SESSION2_QUERY = "I can't stop worrying about the test this week."   # shares no content word with the exam fact


def demo(backend=None) -> int:
    """Two sessions on a TEMPORARY store (never the real data/ or vector_store/). backend: "json" or "chroma"
    (default: memory.backend in gpu_config.yaml). Session 2 opens a new manager on the same store and asks something
    that shares no content word with the session-1 exam fact (checked with memory_manager.content_words)."""
    from memory_manager import content_words, importance_from_turn, load_memory_config, make_memory_manager

    mconf = load_memory_config()
    cfg = dict(mconf["memory"])
    backend = backend or cfg.get("backend", "json")
    cfg["backend"] = backend
    user = "demo_user"
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return stub_llm(prompt)

    with tempfile.TemporaryDirectory() as tmp:
        conf = {"memory": cfg, "memory_file": Path(tmp) / "memories.json",
                "vector_store_dir": Path(tmp) / "vector_store", "log_dir": Path(tmp) / "logs"}
        store_path = conf["vector_store_dir"] if backend == "chroma" else conf["memory_file"]
        print(f"backend {backend}; memory store (temporary): {store_path}"
              + (f"; min_similarity {float(cfg['min_similarity']):.2f}" if backend == "chroma" else "") + "\n")

        print("=== Session 1 ===")
        session1 = [
            ("Hi! How are you doing today?", {"neutral": 0.85, "happy": 0.1, "angry": 0.02, "sad": 0.03}, False),
            ("I go hiking most weekends, it clears my head.", {"neutral": 0.7, "happy": 0.25, "angry": 0.0, "sad": 0.05}, False),
            ("I have my exam on Monday and I'm honestly freaking out.", {"neutral": 0.08, "happy": 0.02, "angry": 0.2, "sad": 0.7}, False),
        ]
        exam_id, exam_fact = None, None
        with make_memory_manager(conf) as mm:
            for turn_id, (text, probs, abstained) in enumerate(session1, 1):
                importance, spike = importance_from_turn(probs, abstained, cfg)
                fact = extract_fact(text, llm, emotional_spike=spike)
                print(f"turn {turn_id}: {text!r}")
                print(f"   importance {importance:.2f}  spike {spike}  -> fact: {fact or 'NONE'}")
                if fact:
                    mid = mm.store(user, fact, turn_id, {"probs": probs, "abstained": abstained}, importance)
                    if "exam" in fact:
                        exam_id, exam_fact = mid, fact
            print(f"stored in session 1: {[r['fact'] for r in mm.list_user_facts(user)]}")

        print(f"\n=== Session 2 (new {backend} manager, same store) ===")
        query = SESSION2_QUERY
        shared = content_words(query) & content_words(exam_fact)
        print(f"user says: {query!r}")
        print(f"   content words: query {sorted(content_words(query))}  vs  stored exam fact "
              f"{sorted(content_words(exam_fact))}  -> shared {sorted(shared) or 'none'}  "
              f"[{'PASS: no shared content word' if not shared else 'FAIL: words shared'}]")
        if shared:
            return 1
        with make_memory_manager(conf) as mm2:
            print(f"   fact from this turn: {extract_fact(query, llm) or 'NONE'}")
            hits = mm2.retrieve_relevant(user, query)
            print(f"   retrieved {len(hits)} (score = {cfg['w_similarity']} x similarity + {cfg['w_importance']} x importance):")
            for h in hits:
                print(f"     similarity {h['similarity']:.3f}  importance {h['importance']:.3f}  "
                      f"score {h['score']:.3f}  {h['fact']}")
            print(f"   exam fact recalled: {any(h['memory_id'] == exam_id for h in hits)}")
            print(f"   other user sees: {mm2.retrieve_relevant('someone_else', query)}")

            print("\n=== Forget the exam memory ===")
            print(f"   delete({exam_id[:8]}...) -> {mm2.delete(user, exam_id)}")
        with make_memory_manager(conf) as mm3:
            hits = mm3.retrieve_relevant(user, query)
            print(f"   retrieve {query!r} after reopening -> {[h['fact'] for h in hits]}")
            print(f"   remaining facts: {[r['fact'] for r in mm3.list_user_facts(user)]}")

        print("\n=== Crisis turn ===")
        n_before = len(calls)
        crisis = "I don't want to be alive anymore"
        print(f"   {crisis!r} -> fact {extract_fact(crisis, llm)}  (LLM calls made: {len(calls) - n_before})")
    print(f"\ntemporary store removed: {not store_path.exists()}")
    return 0


LIVE_SENTENCES = [
    ("I'm starting a new job at a hospital next month.", "durable"),
    ("I've been vegetarian for years, so please don't suggest meat dishes.", "durable"),
    ("My sister is my best friend, we talk every Sunday.", "durable"),
    ("Haha, that's pretty funny.", "small talk"),
    ("What's the weather like today?", "small talk"),
    ("Okay, thanks.", "small talk"),
]
LIVE_SYSTEM = "Follow the instructions in the user's message exactly. Answer with a single line."


def live_gemini() -> int:
    import os

    import response_generator   # loads Code/.env

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set (Code/.env); skipping --live-gemini.")
        return 0
    print("Live Gemini via response_generator._call_llm (no memory file is written)\n")
    for text, kind in LIVE_SENTENCES:
        used = {}

        def llm(prompt):
            reply, model = response_generator._call_llm(LIVE_SYSTEM, prompt)
            used["model"], used["raw"] = model, reply
            return reply

        fact = extract_fact(text, llm)
        print(f"[{kind:10s}] {text!r}")
        print(f"   -> {fact or 'NONE'}   (model: {used.get('model', 'error: no answer')})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract one durable fact about the user (demo step 5).")
    ap.add_argument("--demo", action="store_true", help="two-session exam scenario with a stub LLM")
    ap.add_argument("--live-gemini", action="store_true", help="6 sentences through Gemini (needs GEMINI_API_KEY)")
    ap.add_argument("--backend", choices=("json", "chroma"),
                    help="memory backend for --demo (default: memory.backend in gpu_config.yaml); temporary store")
    args = ap.parse_args(argv)
    if args.demo:
        return demo(args.backend)
    if args.live_gemini:
        return live_gemini()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
