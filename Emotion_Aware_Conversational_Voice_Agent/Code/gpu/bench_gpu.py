"""
bench_gpu.py - benchmark the live agent on a fixed conversation (step 10).

Runs gpu/bench_turns.yaml (bench.turns_file) through LiveAgent with the existing scripted mode
(live_agent_gpu.run_script). Every number comes from the agent's own per-stage timings (run_turn's "timings"),
the voice's own time_to_first_audio (tts_gpu stats), or a wall clock around run_turn. Nothing is estimated.

Per turn:  asr, emotion, abstain, mood, memory, llm (= first LLM sentence), voice first audio, turn first audio,
           turn total (wall), voice path (csm/pyttsx3/text), LLM backend + model, ser_model + fallback, abstained.
  turn first audio = every stage that ran before the "tts" stage (consent ... llm, in the agent's timing order)
                     + the voice's time_to_first_audio. n/a when the voice did not report one (pyttsx3 / text).
Runs:      one warm-up turn (the first bench turn, own user) is run and DISCARDED, then the whole conversation runs
           bench.repeats times (gpu_config.yaml), each pass with a fresh user so passes are comparable.
           Per turn: median and min-max over the passes.
GPU:       peak torch.cuda memory after all models are loaded and after the run; "n/a" without CUDA.
Fallbacks: flagged per turn and in the summary: T3 -> pretrained (ser_model.fallback), LLM backend other than
           llm.backend (Qwen -> Gemini / template) or a model switch inside it, voice other than csm when
           tts.backend is csm (CSM -> pyttsx3 / text).

Isolation: bench users, memories, consent and session logs live in gpu/outputs/bench/<run>/ ($GPU_OUTPUTS_DIR when
set); the real data/memories.json, data/consent.json and results/logs are never touched. Audio is not played
unless --play (pyttsx3 always speaks if it is used).

This module imports no torch at import time (models load inside main via live_agent_gpu.load_real_parts).

Usage (from Code/):
    python gpu/bench_gpu.py --preset gpu_24gb
    python gpu/bench_gpu.py --preset smoke_cpu                  # small models / Gemini / pyttsx3, at home
    python gpu/bench_gpu.py --preset smoke_cpu --stub           # no models, no network: checks the plumbing only
Output: results/metrics/bench_<preset>_<YYYYmmdd_HHMMSS>.csv; a _2, _3 ... suffix if that name exists.
        Nothing is overwritten. With --stub the CSV (bench_<preset>-stub_...) stays in the bench folder instead,
        so stub timings never sit next to real results.
"""

import argparse
import csv
import datetime
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
SCRIPTS_DIR = CODE_DIR / "scripts"
for p in (str(HERE), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import live_agent_gpu as la  # noqa: E402  (torch-free at import)
import safety  # noqa: E402
from memory_manager import MemoryManager  # noqa: E402

PRE_TTS_ORDER = ("consent", "asr", "safety", "emotion", "abstain", "mood", "memory", "prompt", "llm")
NUMERIC = ("asr_s", "emotion_s", "abstain_s", "mood_s", "memory_s", "llm_first_sentence_s", "voice_first_audio_s",
           "turn_first_audio_s", "turn_total_s")
FIELDS = ("row_type", "pass", "turn", "label", "wav", "text_override", "status", *NUMERIC, "voice_path",
          "llm_backend", "llm_model", "ser_model", "ser_fallback", "abstained", "fallback_flags",
          "gpu_peak_after_load_gb", "gpu_peak_after_run_gb", "preset")


# ---------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------------------------

def turn_first_audio(timings, voice_ttfa):
    """Stages that ran before "tts" (in the order the agent timed them) + the voice's own time_to_first_audio."""
    if not timings or "tts" not in timings or not isinstance(voice_ttfa, (int, float)):
        return None
    before = []
    for k in timings:
        if k == "tts":
            break
        if k != "total":
            before.append(timings[k])
    return round(sum(before) + float(voice_ttfa), 4)


def fallback_flags(row, expect) -> list:
    """Which fallbacks this turn used. expect = {"llm_backend": cfg llm.backend, "voice": cfg tts.backend}."""
    flags = []
    if row.get("ser_fallback") is True:
        flags.append("SER: T3 -> pretrained")
    backend = row.get("llm_backend")
    if backend and expect.get("llm_backend") and backend != expect["llm_backend"]:
        flags.append(f"LLM: {expect['llm_backend']} -> {backend}")
    elif backend and row.get("llm_fell_back"):
        flags.append(f"LLM: model switch inside {backend}")
    voice = row.get("voice_path")
    if expect.get("voice") == "csm" and voice and voice not in ("csm", "none"):
        flags.append(f"voice: csm -> {voice}")
    return flags


def turn_row(r, turn_no, turn, text, ser_info, expect) -> dict:
    """One CSV row from run_turn's result (plus the bench's wall time in r["wall_s"])."""
    t = r.get("timings") or {}
    ttfa = r.get("time_to_first_audio")
    row = {
        "turn": turn_no, "label": turn.get("label"), "wav": Path(turn["wav"]).name, "text_override": text or "",
        "status": r.get("status"),
        "asr_s": t.get("asr"), "emotion_s": t.get("emotion"), "abstain_s": t.get("abstain"), "mood_s": t.get("mood"),
        "memory_s": t.get("memory"), "llm_first_sentence_s": t.get("llm"),
        "voice_first_audio_s": ttfa if isinstance(ttfa, (int, float)) else None,
        "turn_first_audio_s": turn_first_audio(t, ttfa), "turn_total_s": r.get("wall_s"),
        "voice_path": r.get("voice_path"), "llm_backend": r.get("llm_backend"), "llm_model": r.get("llm_model"),
        "llm_fell_back": r.get("llm_fell_back"),
        "ser_model": (ser_info or {}).get("name"), "ser_fallback": (ser_info or {}).get("fallback"),
        "abstained": r.get("abstained"),
    }
    row["fallback_flags"] = "; ".join(fallback_flags(row, expect))
    return row


def summarize(rows) -> list:
    """Per turn, over row_type == "run" rows only (the warm-up is excluded): median, min, max of every NUMERIC
    field, ignoring missing values. Returns [{"turn", "label", "status", "n", "<field>": {"median","min","max"}}]."""
    by_turn = {}
    for r in rows:
        if r.get("row_type") == "run":
            by_turn.setdefault(r["turn"], []).append(r)
    out = []
    for turn in sorted(by_turn):
        rs = by_turn[turn]
        s = {"turn": turn, "label": rs[0].get("label"), "status": rs[0].get("status"), "n": len(rs)}
        for f in NUMERIC:
            vals = [r[f] for r in rs if isinstance(r.get(f), (int, float))]
            s[f] = ({"median": round(statistics.median(vals), 4), "min": round(min(vals), 4),
                     "max": round(max(vals), 4)} if vals else None)
        s["voice_paths"] = sorted({str(r.get("voice_path")) for r in rs})
        s["llm_backends"] = sorted({str(r.get("llm_backend")) for r in rs if r.get("llm_backend")})
        s["flags"] = sorted({f for r in rs for f in str(r.get("fallback_flags") or "").split("; ") if f})
        out.append(s)
    return out


def new_bench_path(preset, when, out_dir) -> Path:
    """results/metrics/bench_<preset>_<YYYYmmdd_HHMMSS>.csv; _2, _3 ... if taken. Never an existing file."""
    out_dir = Path(out_dir)
    stem = f"bench_{preset}_{when:%Y%m%d_%H%M%S}"
    path, n = out_dir / f"{stem}.csv", 2
    while path.exists():
        path, n = out_dir / f"{stem}_{n}.csv", n + 1
    return path


def peak_gpu_gb(torch_mod=None):
    """Peak torch.cuda memory (GB) so far, or "n/a" without CUDA (or without torch). Imports torch only here."""
    try:
        torch = torch_mod if torch_mod is not None else __import__("torch")
    except Exception:  # noqa: BLE001
        return "n/a"
    try:
        if not torch.cuda.is_available():
            return "n/a"
        return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    except Exception:  # noqa: BLE001
        return "n/a"


def load_bench_turns(path):
    """(turns from live_agent_gpu.parse_script, [text override or None per turn], in the same order)."""
    import yaml

    _, turns = la.parse_script(path)
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = data.get("turns") if isinstance(data, dict) else data
    texts = [it.get("text") if isinstance(it, dict) else None for it in items]
    if any(part.startswith("my_voice") for t in turns for part in Path(t["wav"]).parts):
        raise ValueError("bench turns must not use my_voice clips (my_voice, my_voice_v2, my_voice_v3, ...)")
    return turns, texts


# ---------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------

class TextOverrideASR:
    """Runs the real ASR (its time is measured), then replaces the transcript when a text override is set."""

    def __init__(self, asr):
        self.asr, self.override = asr, None

    def transcribe(self, y, sr=None):
        out = dict(self.asr.transcribe(y, sr=sr))
        if self.override is not None:
            out.update(text=self.override, skipped=False, reason=None)
        return out


class BenchAgent:
    """What run_script drives: LiveAgent.run_turn with a wall clock around it and the next text override set."""

    def __init__(self, agent, asr):
        self.agent, self.asr, self._texts = agent, asr, []

    def queue(self, texts):
        self._texts = list(texts)

    def run_turn(self, user_id, audio, sr=None):
        self.asr.override = self._texts.pop(0) if self._texts else None
        t0 = time.perf_counter()
        r = dict(self.agent.run_turn(user_id, audio, sr))
        r["wall_s"] = round(time.perf_counter() - t0, 4)
        return r


def build_bench_agent(cfg, parts, bench_dir):
    """LiveAgent on the given parts, with memory / consent / logs in bench_dir (never the real files)."""
    bench_dir = Path(bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)
    asr = TextOverrideASR(parts["asr"])
    p = {**parts, "asr": asr, "memory": MemoryManager(bench_dir / "memories.json", cfg["memory"]),
         "logger": la.TurnLogger(bench_dir / "logs")}
    consent = bench_dir / "consent.json"
    return BenchAgent(la.LiveAgent(cfg, consent_path=consent, **p), asr), consent


def run_bench(bench, turns, texts, repeats, consent_path, ser_info, expect, print_fn=print) -> list:
    """Warm-up turn (discarded), then `repeats` passes of the whole conversation, each with a fresh user."""
    if int(repeats) < 1:
        raise ValueError("bench.repeats must be at least 1")
    users = ["bench_warmup"] + [f"bench_p{p}" for p in range(1, int(repeats) + 1)]
    for u in users:
        safety.record_consent(u, consent_path)
    rows = []
    print_fn("\n=== warm-up turn (discarded) ===")
    bench.queue(texts[:1])
    for r in la.run_script(bench, users[0], turns[:1], print_fn=print_fn):
        rows.append({"row_type": "warmup", "pass": 0, **turn_row(r, 1, turns[0], texts[0], ser_info, expect)})
    for p, user in enumerate(users[1:], 1):
        print_fn(f"\n=== pass {p}/{repeats} (user {user}) ===")
        bench.queue(texts)
        results = la.run_script(bench, user, turns, print_fn=print_fn)
        if len(results) != len(turns):
            print_fn(f"WARNING: pass {p} stopped after {len(results)} of {len(turns)} turns")
        for i, r in enumerate(results):
            rows.append({"row_type": "run", "pass": p, **turn_row(r, i + 1, turns[i], texts[i], ser_info, expect)})
    return rows


def _fmt(stat):
    return "n/a" if not stat else f"{stat['median']:.3f} [{stat['min']:.3f}-{stat['max']:.3f}]"


def print_summary(summary, rows, gpu_load, gpu_run, print_fn=print):
    print_fn("\n=== summary: median [min-max] seconds over the passes (warm-up excluded) ===")
    print_fn(f"{'#':>2} {'label':14s} {'status':16s} {'asr':>21s} {'emotion':>21s} {'llm 1st sent.':>21s} "
             f"{'voice 1st audio':>21s} {'turn 1st audio':>21s} {'turn total':>21s}  voice / llm")
    for s in summary:
        print_fn(f"{s['turn']:>2} {str(s['label'])[:14]:14s} {str(s['status'])[:16]:16s} {_fmt(s['asr_s']):>21s} "
                 f"{_fmt(s['emotion_s']):>21s} {_fmt(s['llm_first_sentence_s']):>21s} "
                 f"{_fmt(s['voice_first_audio_s']):>21s} {_fmt(s['turn_first_audio_s']):>21s} "
                 f"{_fmt(s['turn_total_s']):>21s}  {','.join(s['voice_paths'])} / {','.join(s['llm_backends']) or '-'}")
    print_fn(f"\nGPU peak memory: after load {gpu_load} GB, after run {gpu_run} GB" if gpu_load != "n/a"
             else "\nGPU peak memory: n/a (no CUDA)")
    flagged = [(r["pass"], r["turn"], r["fallback_flags"]) for r in rows if r.get("fallback_flags")]
    if flagged:
        print_fn(f"\n!!! FALLBACK USED in {len(flagged)} turn(s):")
        for p, t, f in flagged:
            print_fn(f"    pass {p} turn {t}: {f}")
    else:
        print_fn("\nNo fallback used (T3, configured LLM backend and configured voice on every turn).")
    abst = [r for r in rows if r["row_type"] == "run" and r.get("abstained")]
    print_fn(f"Abstained turns: {len(abst)} of {sum(r['row_type'] == 'run' for r in rows)}")


def write_csv(path, rows, summary, gpu_load, gpu_run, preset) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"gpu_peak_after_load_gb": gpu_load, "gpu_peak_after_run_gb": gpu_run, "preset": preset}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **extra})
        for s in summary:
            for stat in ("median", "min", "max"):
                w.writerow({"row_type": stat, "turn": s["turn"], "label": s["label"], "status": s["status"],
                            **{f: (s[f][stat] if s[f] else None) for f in NUMERIC},
                            "voice_path": ",".join(s["voice_paths"]), "llm_backend": ",".join(s["llm_backends"]),
                            "fallback_flags": "; ".join(s["flags"]), **extra})
    return path


# ---------------------------------------------------------------------------------------------
# Stub parts (--stub: no models, no network; checks the bench plumbing only, its timings mean nothing)
# ---------------------------------------------------------------------------------------------

class _StubASR:
    def transcribe(self, y, sr=None):
        return {"text": "This is a bench stub turn.", "skipped": False, "reason": None}


class _StubEmotion:
    def predict_array(self, y, sr):
        return {"probs": {"neutral": 0.25, "happy": 0.25, "angry": 0.25, "sad": 0.25}}


class _StubLLM:
    last_reply = None

    def stream_reply(self, system_prompt, history, user_text, strategy):
        yield "Stub reply."
        self.last_reply = {"backend": "stub", "model": "stub", "fell_back": False, "errors": []}

    def complete(self, prompt):
        return "NONE"


class _StubVoice:
    def speak(self, sentences, strategy, crisis=False):
        for _ in ([sentences] if isinstance(sentences, str) else sentences):
            pass
        return {"path": "text", "error": None, "stats": None}


def stub_parts():
    return ({"asr": _StubASR(), "emotion": _StubEmotion(), "llm": _StubLLM(), "voice": _StubVoice()},
            {"name": "stub", "fallback": False}, {"llm_backend": "stub", "voice": "stub"})


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark the live agent on a fixed conversation (step 10).")
    ap.add_argument("--preset", help="preset name (default: $AGENT_PRESET, else smoke_cpu)")
    ap.add_argument("--turns", help="bench file (default bench.turns_file in gpu_config.yaml)")
    ap.add_argument("--stub", action="store_true", help="stub parts: no models, no network (plumbing check only)")
    ap.add_argument("--play", action="store_true", help="play CSM audio (default: generate only)")
    ap.add_argument("--quiet", action="store_true", help="print only the summary, not every turn's debug view")
    args = ap.parse_args(argv)

    from device import code_path, load_config

    cfg = load_config(args.preset)
    turns, texts = load_bench_turns(code_path(args.turns or cfg["bench"]["turns_file"]))
    repeats = int(cfg["bench"]["repeats"])
    when = datetime.datetime.now()
    label = cfg["preset"] + ("-stub" if args.stub else "")
    bench_dir = code_path(f"gpu/outputs/bench/bench_{label}_{when:%Y%m%d_%H%M%S}")
    say = (lambda *a, **k: None) if args.quiet else print

    if args.stub:
        base, ser_info, expect = stub_parts()
        parts = {**base, "ser_model": ser_info}
    else:
        try:
            cfg, parts = la.load_real_parts(args.preset, play=args.play)
        except (ValueError, RuntimeError, OSError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        ser_info = parts.get("ser_model") or {}
        expect = {"llm_backend": cfg["llm"]["backend"], "voice": cfg["tts"].get("backend")}
    gpu_load = peak_gpu_gb() if not args.stub else "n/a"
    print(f"preset {label}  turns {len(turns)} x {repeats} passes (+1 warm-up)  emotion model {ser_info.get('name')}"
          f"{' [FALLBACK]' if ser_info.get('fallback') else ''}  expected llm {expect['llm_backend']}, "
          f"voice {expect['voice']}  bench data: {bench_dir}")

    parts = {k: v for k, v in parts.items() if k in ("asr", "emotion", "llm", "voice", "ser_model")}
    bench, consent = build_bench_agent(cfg, parts, bench_dir)
    rows = run_bench(bench, turns, texts, repeats, consent, ser_info, expect, print_fn=say)
    gpu_run = peak_gpu_gb() if not args.stub else "n/a"
    summary = summarize(rows)
    print_summary(summary, rows, gpu_load, gpu_run)
    # A --stub CSV holds stub timings: it stays in the (git-ignored) bench folder, never in results/metrics.
    out_dir = bench_dir if args.stub else code_path("results/metrics")
    path = write_csv(new_bench_path(label, when, out_dir), rows, summary, gpu_load, gpu_run, label)
    print(f"\nsaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
