"""
test_response_generator.py
1) system prompts  2) template path  3) forced LLM failure -> template  4) real Gemini (if key set)
Never prints the API key.
"""
import os
import time

import response_generator as rg

TRANSCRIPT = "I have an exam tomorrow and I haven't studied at all."
STRATEGIES = rg.STRATEGIES   # VALIDATE / CALM / ENCOURAGE / NEUTRAL


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


section("1) build_system_prompt")
for strategy, check_in in [(s, False) for s in STRATEGIES] + [("VALIDATE", True)]:
    print(f"\n--- {strategy} (check_in={check_in}) ---")
    print(rg.build_system_prompt(strategy, check_in=check_in))

section("2) Template path (use_llm=False)")
for strategy in STRATEGIES:
    out = rg.generate_reply(strategy, TRANSCRIPT, use_llm=False)
    print(f"{strategy:10s} source={out['source']:8s} error={out['error']}  text={out['text']!r}")
    assert out["source"] == "template" and out["error"] is None and out["system_prompt"] is None

section("3) Forced failure (invalid key injected into this call only)")
out = rg.generate_reply("VALIDATE", TRANSCRIPT, use_llm=True, api_key="INVALID-test-key")
print(f"source={out['source']}  model={out['model']}  error={out['error']}")
print(f"text={out['text']!r}")
ok = (out["source"] == "template" and out["error"] is not None and out["model"] is None
      and out["system_prompt"] == rg.build_system_prompt("VALIDATE"))
print("PASS" if ok else "FAIL", "- fell back to template with error set, prompt still returned")
assert ok

section("4) Real Gemini")
if not os.environ.get("GEMINI_API_KEY"):
    print("SKIPPED — GEMINI_API_KEY not set")
else:
    print("model chain:", " -> ".join(rg.GEMINI_MODELS))
    print(f"per-model timeout {rg.PER_MODEL_TIMEOUT_S}s, total budget {rg.TOTAL_BUDGET_S}s")
    for strategy in ["NEUTRAL", "VALIDATE"]:
        t0 = time.perf_counter()
        out = rg.generate_reply(strategy, TRANSCRIPT)
        dt = time.perf_counter() - t0
        print(f"\n{strategy} source={out['source']} model={out['model']} latency={dt:.2f}s error={out['error']}")
        print(f"  {out['text']}")
