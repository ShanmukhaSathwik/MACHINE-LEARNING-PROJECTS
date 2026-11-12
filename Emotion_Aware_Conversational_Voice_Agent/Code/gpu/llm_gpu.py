"""
llm_gpu.py - the reply LLM for the GPU demo build.

Step 7 of the GPU demo build. GPU-first: Qwen/Qwen3-8B in 4 bits (bitsandbytes nf4) on the gpu_* presets;
smoke_cpu uses Gemini through scripts/response_generator.py. Replies stream out sentence by sentence, cleaned
for speech, so TTS can start on the first sentence.

P3 (code decides, LLM speaks): this module never receives emotion probabilities or labels. The caller builds
the system prompt (response_generator.build_system_prompt(strategy, facts, check_in)); `strategy` is used here
ONLY to pick the fixed template reply when every backend fails. It does not import ser_gpu, mood_manager or
emotion_predictor.

Reused from scripts/response_generator.py (unchanged):
  - _call_llm(system_prompt, user_message, history, models=...)  the Gemini call (model chain, timeouts)
  - generate_reply_template(strategy, transcript)                 the per-strategy fixed reply
  - build_system_prompt(strategy, facts, check_in)                (CLI demo only)

Pure helpers (no model, unit-tested in tests/test_llm_gpu.py):
  - SentenceBuffer      feed(text) -> finished sentences; flush() -> the rest
  - ThinkFilter         drops <think>...</think> from a stream, even when a tag is split across chunks
  - clean_for_speech()  strips thinking, markdown, emoji, stage directions and <tags> (voice tags come from MoodManager, P6)
  - build_messages()    system, last history_turns (user, agent) pairs, user
  - chat_template_kwargs(cfg)  always enable_thinking=False

LLM(cfg, device, dtype, backend=None, model_name=None, quantization=None)
    load() once, warm_up() once, then stream_reply / generate_reply / complete (as_llm_fn for fact_extractor).
    Fallback: the configured backend, then llm.fallback in order; the template always answers.
    llm.allow_fallback false (evaluation runs): a failure raises instead.

Usage (works from any folder):
    python gpu/llm_gpu.py --preset smoke_cpu --demo
    python gpu/llm_gpu.py --preset smoke_cpu --backend qwen --model Qwen/Qwen3-0.6B --quantization none --demo
    python gpu/llm_gpu.py --preset gpu_16gb --demo

Output of --demo: results/metrics/llm_demo_<backend>_<model-short>.csv
(a timestamp is added to the name if that file already exists; nothing is overwritten).
"""

import argparse
import csv
import datetime
import re
import sys
import threading
import time
from pathlib import Path

from device import StageTimer, code_path, setup

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import response_generator  # noqa: E402  (scripts/; loads Code/.env)

BACKENDS = ("qwen", "gemini", "template")
QUANTIZATIONS = ("4bit", "none")
MIN_SENTENCE_WORDS = 3
ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "st", "vs", "e.g", "i.e", "prof", "jr", "sr"}
WARMUP_TOKENS = 8


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------

class SentenceBuffer:
    """Collects streamed text and hands back finished sentences.

    A sentence ends at . ! ? (followed by whitespace) or at a newline. Not a split: a decimal ("3.5"),
    the abbreviations in ABBREVIATIONS ("Dr.", "e.g."), or a terminator at the very end of the buffer
    (it may still become "3.5" or "..." when more text arrives). A finished piece under 3 words is held
    and merged into the next sentence.
    """

    _END = re.compile(r"[.!?]+[\"')\]]*(?=\s)|\n+")

    def __init__(self, min_words=MIN_SENTENCE_WORDS):
        self.min_words = min_words
        self.buf = ""
        self.pending = ""

    def _is_abbreviation(self, text_before, terminator) -> bool:
        if terminator.strip("\"')]") != ".":
            return False
        tokens = text_before.split()
        last = tokens[-1].lstrip("(\"'[").lower() if tokens else ""
        return last in ABBREVIATIONS

    def _accept(self, piece) -> list:
        piece = " ".join(piece.split())
        if not piece:
            return []
        combined = f"{self.pending} {piece}".strip()
        if len(combined.split()) < self.min_words:
            self.pending = combined
            return []
        self.pending = ""
        return [combined]

    def feed(self, text) -> list:
        self.buf += text or ""
        out, start, pos = [], 0, 0
        while True:
            m = self._END.search(self.buf, pos)
            if not m:
                break
            if m.group().startswith("\n"):
                out += self._accept(self.buf[start:m.start()])
            else:
                if self._is_abbreviation(self.buf[start:m.start()], m.group()):
                    pos = m.end()
                    continue
                out += self._accept(self.buf[start:m.end()])
            start = pos = m.end()
        self.buf = self.buf[start:]
        return out

    def flush(self) -> list:
        """Whatever is left (even under 3 words) as one last sentence; the buffer is then empty."""
        rest = " ".join(f"{self.pending} {self.buf}".split())
        self.buf, self.pending = "", ""
        return [rest] if rest else []


class ThinkFilter:
    """Removes <think>...</think> from streamed text, holding back a tag that is split across chunks."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf, self.in_think = "", False

    @staticmethod
    def _partial_tail(text, tag) -> int:
        for n in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:n]):
                return n
        return 0

    def feed(self, text) -> str:
        self.buf += text or ""
        out = []
        while True:
            if self.in_think:
                i = self.buf.find(self.CLOSE)
                if i < 0:
                    self.buf = self.buf[len(self.buf) - self._partial_tail(self.buf, self.CLOSE):]
                    break
                self.buf, self.in_think = self.buf[i + len(self.CLOSE):], False
            else:
                i = self.buf.find(self.OPEN)
                if i < 0:
                    hold = self._partial_tail(self.buf, self.OPEN)
                    out.append(self.buf[:len(self.buf) - hold])
                    self.buf = self.buf[len(self.buf) - hold:]
                    break
                out.append(self.buf[:i])
                self.buf, self.in_think = self.buf[i + len(self.OPEN):], True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.in_think else self.buf
        self.buf, self.in_think = "", False
        return rest


_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U0001F1E6-\U0001F1FF☀-➿⬀-⯿️‍⃣]")


def clean_for_speech(text) -> str:
    """Text a TTS voice can read: no <think> blocks, markdown, emoji, stage directions (*sighs*, (laughs),
    [pause]) or <tags> such as <sigh> (voice tags come only from MoodManager's strategy, P6)."""
    t = str(text or "")
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<think>.*", " ", t, flags=re.DOTALL | re.IGNORECASE)          # unclosed thinking: drop the rest
    t = re.sub(r"</think>", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"</?[A-Za-z][\w\- ]{0,30}/?>", " ", t)                           # <sigh>, <softly>, </b>
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)                                 # headings
    t = re.sub(r"(?m)^\s*[-*+•]\s+", "", t)                                # bullets
    t = re.sub(r"(?m)^\s*\d+[.)]\s+", "", t)                                    # numbered-list prefixes
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)                                      # **bold** keeps its words
    t = re.sub(r"__(.+?)__", r"\1", t)
    t = re.sub(r"\*[^*\n]+\*", " ", t)                                          # *stage direction*
    t = re.sub(r"\([^()]*\)", " ", t)                                           # (laughs)
    t = re.sub(r"\[[^\[\]]*\]", " ", t)                                         # [pause]
    t = re.sub(r"[*_#`]", "", t)
    t = _EMOJI.sub("", t)
    t = " ".join(t.split())
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    return t.strip()


def build_messages(system_prompt, history, user_text, history_turns) -> list:
    """[system?] + the last history_turns (user, agent) pairs + the user turn, as chat messages."""
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    pairs = list(history or [])[-history_turns:] if history_turns > 0 else []
    for user_said, agent_said in pairs:
        messages.append({"role": "user", "content": user_said})
        messages.append({"role": "assistant", "content": agent_said})
    messages.append({"role": "user", "content": user_text})
    return messages


def chat_template_kwargs(cfg) -> dict:
    """Arguments for tokenizer.apply_chat_template. enable_thinking is always False (llm.thinking must be false)."""
    if cfg["llm"].get("thinking", False):
        raise ValueError("llm.thinking must be false: a live voice reply cannot wait for a thinking step")
    return {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}


def model_short(name) -> str:
    return str(name or "default").rstrip("/\\").replace("\\", "/").split("/")[-1]


def _err(name, e) -> dict:
    return {"backend": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------

class LLM:
    """Reply LLM with streaming, fallback and a one-shot complete() for fact extraction."""

    def __init__(self, cfg, device, dtype, backend=None, model_name=None, quantization=None):
        import torch

        llm = cfg["llm"]
        self.cfg = cfg
        self.backend = backend or llm["backend"]
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        self.model_name = model_name or llm.get("model")
        self.quantization = quantization or llm.get("quantization") or "none"
        if self.quantization not in QUANTIZATIONS:
            raise ValueError(f"quantization must be one of {QUANTIZATIONS}, got {self.quantization!r}")
        self.device = device
        self.dtype = dtype if device == "cuda" else torch.float32
        self.template_kwargs = chat_template_kwargs(cfg)
        self.history_turns = int(llm.get("history_turns", 6))
        self.fallback = list(llm.get("fallback") or [])
        self.allow_fallback = bool(llm.get("allow_fallback", True))
        self.max_new_tokens = int(llm["max_new_tokens"])
        self.extract_max_new_tokens = int(llm.get("extract_max_new_tokens", 60))
        self.sampling = {k: llm[k] for k in ("temperature", "top_p", "top_k") if llm.get(k) is not None}
        self.tokenizer = self.model = None
        # name -> fn(system_prompt, history, user_text, *, greedy, max_new_tokens) -> (model name, iterator of text)
        self._backends = {"qwen": self._qwen_backend, "gemini": self._gemini_backend}
        self.last_reply = None
        self.last_complete = None

    # -- loading --

    def load(self):
        if self.backend != "qwen":
            return self                                 # gemini / template: no local model
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.model_name:
            raise ValueError("backend qwen needs llm.model (e.g. Qwen/Qwen3-8B) or --model")
        if self.quantization == "4bit" and self.device != "cuda":
            raise ValueError("4-bit (bitsandbytes) needs a CUDA GPU; use --quantization none on the CPU")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.quantization == "4bit":
            bnb = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, quantization_config=bnb, dtype=self.dtype, device_map=self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=self.dtype)
            self.model.to(self.device)
        self.model.eval()
        return self

    def warm_up(self):
        """qwen: one short greedy generation (first call pays kernel / cache setup).
        gemini / template: nothing to warm (every Gemini call opens a fresh client), so no API request is spent."""
        if self.backend == "qwen":
            _, chunks = self._qwen_backend(None, [], "Hi", greedy=True, max_new_tokens=WARMUP_TOKENS)
            "".join(chunks)
        return self

    @classmethod
    def from_preset(cls, preset=None, **overrides):
        """setup(preset) -> loaded, warmed-up LLM. overrides: backend, model_name, quantization."""
        cfg, device, dtype = setup(preset)
        return cls(cfg, device, dtype, **overrides).load().warm_up()

    def model_label(self) -> str:
        if self.backend == "gemini":
            return self.model_name or response_generator.GEMINI_MODELS[0]
        if self.backend == "template":
            return "template"
        return self.model_name or "unset"

    # -- messages --

    def build_messages(self, system_prompt, history, user_text) -> list:
        return build_messages(system_prompt, history, user_text, self.history_turns)

    # -- backends --

    def _qwen_backend(self, system_prompt, history, user_text, *, greedy, max_new_tokens):
        if self.model is None:
            raise RuntimeError("qwen model not loaded (call load(), or it is only a fallback for another backend)")
        from transformers import TextIteratorStreamer

        prompt = self.tokenizer.apply_chat_template(
            self.build_messages(system_prompt, history, user_text), **self.template_kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens,
                   pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if greedy:
            gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            gen.update(do_sample=True, **self.sampling)
        failure = []

        import torch

        def run():
            try:
                with torch.inference_mode():   # thread-local, so it is set inside the generation thread
                    self.model.generate(**gen)
            except BaseException as e:  # noqa: BLE001 - re-raised in the consumer thread
                failure.append(e)
                streamer.end()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        def chunks():
            for piece in streamer:
                yield piece
            thread.join()
            if failure:
                raise failure[0]

        return self.model_name, chunks()

    def _gemini_backend(self, system_prompt, history, user_text, *, greedy, max_new_tokens):
        """response_generator._call_llm. It has no sampling / token-limit arguments (Gemini uses its own
        settings) and keeps at most response_generator.MAX_HISTORY_TURNS pairs itself."""
        models = [self.model_name] if (self.backend == "gemini" and self.model_name) else None
        pairs = list(history or [])[-self.history_turns:] if self.history_turns > 0 else []
        text, model = response_generator._call_llm(system_prompt, user_text, pairs, models=models)
        return model, iter([text])

    def _chain(self, with_template=True) -> list:
        if not self.allow_fallback:
            return [self.backend]
        chain = [self.backend] + [b for b in self.fallback if b != self.backend]
        chain = [b for b in chain if with_template or b != "template"]
        if with_template and "template" not in chain:
            chain.append("template")
        return list(dict.fromkeys(chain))

    # -- replies --

    def stream_reply(self, system_prompt, history, user_text, strategy):
        """Yields clean sentences as they complete. After the last one, self.last_reply holds the same
        dict generate_reply returns. strategy only picks the template if every backend fails."""
        t0 = time.perf_counter()
        info = {"text": "", "sentences": [], "backend": None, "model": None, "fell_back": False,
                "errors": [], "seconds": 0.0, "first_sentence_seconds": None}
        self.last_reply = info
        chain = self._chain()
        for name in chain:
            got_any, model = False, None
            try:
                if name == "template":
                    strat = strategy if strategy in response_generator.TEMPLATES else "NEUTRAL"
                    model, chunks = "template", iter([response_generator.generate_reply_template(strat, user_text)])
                else:
                    fn = self._backends.get(name)
                    if fn is None:
                        raise ValueError(f"unknown backend {name!r}")
                    model, chunks = fn(system_prompt, history, user_text,
                                       greedy=False, max_new_tokens=self.max_new_tokens)
                sentences, think = SentenceBuffer(), ThinkFilter()

                def finished(raw):
                    for s in raw:
                        c = clean_for_speech(s)
                        if c:
                            yield c

                for chunk in chunks:
                    for s in finished(sentences.feed(think.feed(chunk))):
                        got_any = self._record(info, s, t0)
                        yield s
                for s in finished(sentences.feed(think.flush()) + sentences.flush()):
                    got_any = self._record(info, s, t0)
                    yield s
                if not got_any:
                    raise RuntimeError("empty reply after cleaning")
                info.update(backend=name, model=model, fell_back=name != chain[0])
                break
            except Exception as e:  # noqa: BLE001 - every failure is recorded
                info["errors"].append(_err(name, e))
                if got_any:          # part of the reply was already spoken: stop here rather than start over
                    info.update(backend=name, model=model, fell_back=name != chain[0])
                    break
                if not self.allow_fallback:
                    info["seconds"] = round(time.perf_counter() - t0, 3)
                    raise RuntimeError(f"backend {name} failed and allow_fallback is false: {e}") from e
        info["text"] = " ".join(info["sentences"])
        info["seconds"] = round(time.perf_counter() - t0, 3)

    @staticmethod
    def _record(info, sentence, t0) -> bool:
        if info["first_sentence_seconds"] is None:
            info["first_sentence_seconds"] = round(time.perf_counter() - t0, 3)
        info["sentences"].append(sentence)
        return True

    def generate_reply(self, system_prompt, history, user_text, strategy) -> dict:
        """{text, sentences, backend, model, fell_back, errors, seconds, first_sentence_seconds}."""
        for _ in self.stream_reply(system_prompt, history, user_text, strategy):
            pass
        return dict(self.last_reply)

    def complete(self, prompt) -> str:
        """One-shot answer: greedy (qwen), extract_max_new_tokens, no history, no system prompt, thinking off.
        Tries the backend chain without the template (a template cannot extract); raises if all fail."""
        t0 = time.perf_counter()
        info = {"backend": None, "model": None, "errors": [], "seconds": 0.0}
        self.last_complete = info
        for name in self._chain(with_template=False):
            try:
                fn = self._backends.get(name)
                if fn is None:
                    raise ValueError(f"backend {name!r} cannot complete a prompt")
                model, chunks = fn(None, [], prompt, greedy=True, max_new_tokens=self.extract_max_new_tokens)
                think = ThinkFilter()
                text = ("".join(think.feed(c) for c in chunks) + think.flush()).strip()
                if not text:
                    raise RuntimeError("empty answer")
                info.update(backend=name, model=model, seconds=round(time.perf_counter() - t0, 3))
                return text
            except Exception as e:  # noqa: BLE001
                info["errors"].append(_err(name, e))
                if not self.allow_fallback:
                    raise
        info["seconds"] = round(time.perf_counter() - t0, 3)
        raise RuntimeError("no backend could complete the prompt: "
                           + "; ".join(f"{e['backend']}: {e['error']}" for e in info["errors"]))

    def as_llm_fn(self):
        """llm_fn(prompt) -> str for fact_extractor.extract_fact."""
        return self.complete


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

DEMO_FACT = "The user has an exam on Monday."
DEMO_LINES = {
    "VALIDATE": "I bombed my practice test today and I feel like I'm just not good enough.",
    "CALM": "Everything is piling up at once and I can't even think straight right now.",
    "ENCOURAGE": "I finally finished my whole project a day early!",
    "NEUTRAL": "I'm just walking home from class right now.",
}
DEMO_EXTRACT = "My sister's wedding is next month and I'm nervous"
CSV_FIELDS = ["kind", "strategy", "user_text", "reply", "n_sentences", "backend", "model", "fell_back",
              "errors", "first_sentence_seconds", "seconds"]


def _rel(path) -> str:
    try:
        return Path(path).resolve().relative_to(code_path("").resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def new_csv_path(backend, model_name) -> Path:
    out_dir = code_path("results/metrics")
    stem = f"llm_demo_{backend}_{model_short(model_name)}"
    path = out_dir / f"{stem}.csv"
    if path.exists():
        path = out_dir / f"{stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    return path


def _errors_text(errors) -> str:
    return " | ".join(f"{e['backend']}: {e['error']}" for e in errors)


def demo(llm) -> list:
    from fact_extractor import extract_fact

    rows = []
    for strategy, line in DEMO_LINES.items():
        system_prompt = response_generator.build_system_prompt(strategy, [DEMO_FACT], check_in=False)
        print(f"\n[{strategy}] user: {line!r}")
        for sentence in llm.stream_reply(system_prompt, [], line, strategy):
            print(f"   > {sentence}", flush=True)
        r = llm.last_reply
        print(f"   backend {r['backend']}  model {r['model']}  fell_back {r['fell_back']}  "
              f"first sentence {r['first_sentence_seconds']}s  total {r['seconds']}s")
        if r["errors"]:
            print(f"   errors: {_errors_text(r['errors'])}")
        rows.append({"kind": "reply", "strategy": strategy, "user_text": line, "reply": r["text"],
                     "n_sentences": len(r["sentences"]), "backend": r["backend"], "model": r["model"],
                     "fell_back": r["fell_back"], "errors": _errors_text(r["errors"]),
                     "first_sentence_seconds": r["first_sentence_seconds"], "seconds": r["seconds"]})

    print(f"\n[fact extraction] user: {DEMO_EXTRACT!r}")
    t0 = time.perf_counter()
    fact = extract_fact(DEMO_EXTRACT, llm.as_llm_fn())
    seconds = round(time.perf_counter() - t0, 3)
    c = llm.last_complete or {"backend": None, "model": None, "errors": []}
    print(f"   -> {fact or 'NONE'}")
    print(f"   backend {c['backend']}  model {c['model']}  {seconds}s")
    if c["errors"]:
        print(f"   errors: {_errors_text(c['errors'])}")
    rows.append({"kind": "extract", "strategy": "", "user_text": DEMO_EXTRACT, "reply": fact or "NONE",
                 "n_sentences": "", "backend": c["backend"], "model": c["model"],
                 "fell_back": bool(c["backend"]) and c["backend"] != llm.backend,
                 "errors": _errors_text(c["errors"]), "first_sentence_seconds": "", "seconds": seconds})
    return rows


def _main():
    parser = argparse.ArgumentParser(description="Reply LLM (GPU demo build, step 7).")
    parser.add_argument("--preset", help="preset name (default: $AGENT_PRESET, else smoke_cpu)")
    parser.add_argument("--demo", action="store_true", help="one streamed reply per strategy + one fact extraction")
    parser.add_argument("--backend", choices=BACKENDS, help="override llm.backend")
    parser.add_argument("--model", help="override llm.model (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--quantization", choices=QUANTIZATIONS, help="override llm.quantization")
    args = parser.parse_args()
    if not args.demo:
        parser.print_help()
        return 0

    try:
        cfg, device, dtype = setup(args.preset)
        llm = LLM(cfg, device, dtype, backend=args.backend, model_name=args.model, quantization=args.quantization)
        timer = StageTimer()
        with timer("load"):
            llm.load()
        with timer("warm_up"):
            llm.warm_up()
    except (ValueError, RuntimeError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"preset {cfg['preset']}  device {device}  dtype {str(llm.dtype).replace('torch.', '')}  "
          f"backend {llm.backend}  model {llm.model_label()}  quantization {llm.quantization}")
    print(f"fallback {llm._chain()}  allow_fallback {llm.allow_fallback}  history_turns {llm.history_turns}  "
          f"sampling {llm.sampling}  max_new_tokens {llm.max_new_tokens}")
    print(f"loaded in {timer.times['load']:.1f}s, warm-up {timer.times['warm_up']:.2f}s")

    rows = demo(llm)
    path = new_csv_path(llm.backend, llm.model_label())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRows: {_rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
