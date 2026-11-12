"""
test_llm_gpu.py - step 7 of the GPU demo build (gpu/llm_gpu.py).
Stub backends only, except one integration test that runs when Qwen/Qwen3-0.6B is already in the HF cache
(it never downloads).
"""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import llm_gpu
import response_generator
from device import load_config
from fact_extractor import extract_fact
from llm_gpu import LLM, SentenceBuffer, ThinkFilter, build_messages, chat_template_kwargs, clean_for_speech

CODE_DIR = Path(__file__).resolve().parent.parent


def make_llm(preset="gpu_16gb", **overrides):
    return LLM(load_config(preset), "cpu", torch.float32, **overrides)


def answer(text, model="stub-model", chunks=None):
    """A stub backend that streams `text` (in `chunks` if given)."""
    calls = []

    def fn(system_prompt, history, user_text, *, greedy, max_new_tokens):
        calls.append({"system_prompt": system_prompt, "history": history, "user_text": user_text,
                      "greedy": greedy, "max_new_tokens": max_new_tokens})
        return model, iter(chunks if chunks is not None else [text])

    fn.calls = calls
    return fn


def failing(message="boom"):
    calls = []

    def fn(*args, **kwargs):
        calls.append(1)
        raise RuntimeError(message)

    fn.calls = calls
    return fn


def feed_all(pieces):
    buf, out = SentenceBuffer(), []
    for p in pieces:
        out += buf.feed(p)
    return out, buf.flush()


# ---------------------------------------------------------------- SentenceBuffer

def test_sentence_buffer_basic_split():
    done, rest = feed_all(["That sounds really hard. I'm here with you! ", "Want to talk about it?"])
    assert done == ["That sounds really hard.", "I'm here with you!"]
    assert rest == ["Want to talk about it?"]


def test_sentence_buffer_streams_across_chunks():
    pieces = ["That so", "unds hard", ". I'm he", "re for you.", " What happened", " today?\n", "Tell me"]
    done, rest = feed_all(pieces)
    assert done == ["That sounds hard.", "I'm here for you.", "What happened today?"]
    assert rest == ["Tell me"]


def test_sentence_buffer_waits_at_end_of_buffer():
    buf = SentenceBuffer()
    assert buf.feed("The exam lasts about 3.") == []        # might become 3.5
    assert buf.feed("5 hours in total. Then") == ["The exam lasts about 3.5 hours in total."]


@pytest.mark.parametrize("text", [
    "It went up by 3.5 points overall today and that's great",
    "You could ask Dr. Patel about it tomorrow morning",
    "Mr. and Mrs. Lee were there with Ms. Park too",
    "Try something small, e.g. a short walk outside",
    "Pick one thing, i.e. the hardest task first",
])
def test_sentence_buffer_no_split_cases(text):
    buf = SentenceBuffer()
    assert buf.feed(text + " ") == []
    assert buf.flush() == [text]


def test_sentence_buffer_ellipsis_in_progress():
    buf = SentenceBuffer()
    assert buf.feed("Well..") == []
    assert buf.feed(". I think you did your best there. ") == ["Well... I think you did your best there."]


def test_sentence_buffer_newlines_end_sentences():
    done, rest = feed_all(["First line of the reply\nSecond line right here\n\n", "Third one now"])
    assert done == ["First line of the reply", "Second line right here"]
    assert rest == ["Third one now"]


def test_sentence_buffer_merges_short_fragments():
    done, rest = feed_all(["Oh. Okay! That makes a lot of sense. Yes. "])
    assert done == ["Oh. Okay! That makes a lot of sense."]
    assert rest == ["Yes."]                                  # flush returns the rest even if short


def test_sentence_buffer_flush_empty_and_resets():
    buf = SentenceBuffer()
    assert buf.flush() == []
    buf.feed("Something left over")
    assert buf.flush() == ["Something left over"]
    assert buf.flush() == []


# ---------------------------------------------------------------- ThinkFilter

def test_think_filter_split_tags():
    f = ThinkFilter()
    out = "".join(f.feed(p) for p in ["Hel", "lo <thi", "nk>secret plan</th", "ink> there friend."]) + f.flush()
    assert out == "Hello  there friend."
    f = ThinkFilter()
    assert f.feed("<think>never closed") + f.flush() == ""


# ---------------------------------------------------------------- clean_for_speech

@pytest.mark.parametrize("raw, expected", [
    ("<think>let me plan this</think>That sounds hard.", "That sounds hard."),
    ("**That** sounds _really_ hard.", "That sounds really hard."),
    ("# Heading\nThat sounds hard.", "Heading That sounds hard."),
    ("- First idea\n- Second idea", "First idea Second idea"),
    ("1. Breathe in.\n2) Breathe out.", "Breathe in. Breathe out."),
    ("Use `this` word.", "Use this word."),
    ("Great job! \U0001F389\U0001F60A", "Great job!"),
    ("*sighs* That sounds hard.", "That sounds hard."),
    ("That sounds hard (smiles warmly).", "That sounds hard."),
    ("[pause] Take a breath.", "Take a breath."),
    ("<sigh> That sounds hard. <softly>I'm here.", "That sounds hard. I'm here."),
    ("That   sounds\n\nhard.", "That sounds hard."),
])
def test_clean_for_speech_removes(raw, expected):
    assert clean_for_speech(raw) == expected


@pytest.mark.parametrize("text", [
    "That sounds really hard, and it makes sense to feel this way.",
    "Your exam is on Monday, so let's plan two short study blocks.",
    "It's 3.5 hours away, but you've got this!",
    "Is 5 < 7? Yes, and 9 > 2.",
])
def test_clean_for_speech_leaves_normal_sentences(text):
    assert clean_for_speech(text) == text


# ---------------------------------------------------------------- messages / chat template

def test_build_messages_keeps_last_pairs():
    history = [(f"user {i}", f"agent {i}") for i in range(10)]
    msgs = build_messages("SYSTEM", history, "now", history_turns=6)
    assert msgs[0] == {"role": "system", "content": "SYSTEM"}
    assert msgs[-1] == {"role": "user", "content": "now"}
    assert len(msgs) == 1 + 2 * 6 + 1
    assert msgs[1] == {"role": "user", "content": "user 4"}
    assert msgs[2] == {"role": "assistant", "content": "agent 4"}
    assert msgs[-2] == {"role": "assistant", "content": "agent 9"}


def test_build_messages_edge_cases():
    assert build_messages(None, None, "hi", 6) == [{"role": "user", "content": "hi"}]
    assert len(build_messages("S", [("a", "b")], "hi", 0)) == 2
    llm = make_llm()
    assert llm.history_turns == 6
    assert llm.build_messages("S", [("a", "b")] * 9, "hi") == build_messages("S", [("a", "b")] * 9, "hi", 6)


@pytest.mark.parametrize("preset", ["smoke_cpu", "gpu_12gb", "gpu_16gb", "gpu_24gb"])
def test_chat_template_kwargs_never_think(preset):
    assert chat_template_kwargs(load_config(preset))["enable_thinking"] is False


def test_thinking_true_is_refused():
    cfg = load_config("gpu_16gb")
    cfg["llm"]["thinking"] = True
    with pytest.raises(ValueError, match="thinking"):
        LLM(cfg, "cpu", torch.float32)


class _Stop(Exception):
    pass


def test_qwen_backend_calls_template_with_enable_thinking_false():
    seen = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"], seen["kwargs"] = messages, kwargs
            raise _Stop

    llm = make_llm()
    llm.tokenizer, llm.model = FakeTokenizer(), SimpleNamespace(device="cpu")
    with pytest.raises(_Stop):
        llm._qwen_backend("SYS", [("a", "b")], "hello", greedy=False, max_new_tokens=5)
    assert seen["kwargs"]["enable_thinking"] is False
    assert seen["kwargs"]["add_generation_prompt"] is True
    assert seen["messages"][0]["role"] == "system" and seen["messages"][-1]["content"] == "hello"


# ---------------------------------------------------------------- streaming + fallback

def test_stream_reply_yields_clean_sentences():
    llm = make_llm()
    llm._backends["qwen"] = answer("", chunks=["<think></think>That sou", "nds hard. *nods* I'm here", " with you."])
    out = list(llm.stream_reply("SYS", [], "I failed", "VALIDATE"))
    assert out == ["That sounds hard.", "I'm here with you."]
    r = llm.last_reply
    assert r["backend"] == "qwen" and r["model"] == "stub-model" and r["fell_back"] is False and r["errors"] == []
    assert r["text"] == "That sounds hard. I'm here with you." and r["sentences"] == out
    assert r["first_sentence_seconds"] is not None and r["seconds"] >= r["first_sentence_seconds"]


def test_qwen_fails_gemini_answers():
    llm = make_llm()
    llm._backends["qwen"] = failing("CUDA out of memory")
    llm._backends["gemini"] = answer("Gemini is here for you today.", model="gemini-x")
    r = llm.generate_reply("SYS", [], "hello", "NEUTRAL")
    assert r["backend"] == "gemini" and r["model"] == "gemini-x" and r["fell_back"] is True
    assert r["errors"] == [{"backend": "qwen", "error": "RuntimeError: CUDA out of memory"}]
    assert r["text"] == "Gemini is here for you today."
    assert set(r) == {"text", "sentences", "backend", "model", "fell_back", "errors", "seconds",
                      "first_sentence_seconds"}


@pytest.mark.parametrize("strategy", ["VALIDATE", "CALM", "ENCOURAGE", "NEUTRAL"])
def test_all_fail_gives_the_strategy_template(strategy):
    llm = make_llm()
    llm._backends["qwen"] = failing("q")
    llm._backends["gemini"] = failing("g")
    r = llm.generate_reply("SYS", [], "hello", strategy)
    assert r["backend"] == "template" and r["fell_back"] is True
    assert [e["backend"] for e in r["errors"]] == ["qwen", "gemini"]
    assert r["text"] == response_generator.generate_reply_template(strategy, "hello")


def test_empty_reply_counts_as_failure():
    llm = make_llm()
    llm._backends["qwen"] = answer("<think>only thinking</think> *smiles*")
    llm._backends["gemini"] = answer("A real answer for you.")
    r = llm.generate_reply("SYS", [], "hello", "NEUTRAL")
    assert r["backend"] == "gemini" and "empty" in r["errors"][0]["error"]


def test_allow_fallback_false_raises():
    cfg = load_config("gpu_16gb")
    cfg["llm"]["allow_fallback"] = False
    llm = LLM(cfg, "cpu", torch.float32)
    gemini = answer("should not be used")
    llm._backends["qwen"], llm._backends["gemini"] = failing("q"), gemini
    with pytest.raises(RuntimeError, match="allow_fallback"):
        llm.generate_reply("SYS", [], "hello", "NEUTRAL")
    assert gemini.calls == []
    assert llm._chain() == ["qwen"]


def test_chain_order_and_template_always_last():
    llm = make_llm()
    assert llm._chain() == ["qwen", "gemini", "template"]
    assert make_llm("smoke_cpu")._chain() == ["gemini", "template"]
    llm.fallback = ["gemini"]
    assert llm._chain() == ["qwen", "gemini", "template"]


def test_qwen_not_loaded_falls_back():
    llm = make_llm()                                        # real qwen backend, never loaded
    llm._backends["gemini"] = answer("Fallback answer is right here.")
    r = llm.generate_reply("SYS", [], "hello", "NEUTRAL")
    assert r["backend"] == "gemini" and "not loaded" in r["errors"][0]["error"]


def test_partial_reply_is_not_restarted():
    def half(*args, **kwargs):
        def gen():
            yield "First full sentence is here. "
            raise RuntimeError("stream broke")
        return "stub", gen()

    llm = make_llm()
    gemini = answer("Should not be spoken at all.")
    llm._backends["qwen"], llm._backends["gemini"] = half, gemini
    r = llm.generate_reply("SYS", [], "hello", "NEUTRAL")
    assert r["sentences"] == ["First full sentence is here."] and r["backend"] == "qwen"
    assert r["errors"][0]["backend"] == "qwen" and gemini.calls == []


def test_sampling_vs_greedy_arguments():
    llm = make_llm()
    stub = answer("Here is a short reply for you.")
    llm._backends["qwen"] = stub
    llm.generate_reply("SYS", [("a", "b")], "hello", "NEUTRAL")
    llm.complete("extract this")
    assert stub.calls[0]["greedy"] is False and stub.calls[0]["max_new_tokens"] == 150
    assert stub.calls[1] == {"system_prompt": None, "history": [], "user_text": "extract this",
                             "greedy": True, "max_new_tokens": 60}
    assert llm.sampling == {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


# ---------------------------------------------------------------- complete / fact_extractor

def test_as_llm_fn_with_extract_fact():
    llm = make_llm()
    llm._backends["qwen"] = answer("<think></think>The user has a sister whose wedding is next month.")
    fact = extract_fact("My sister's wedding is next month and I'm nervous", llm.as_llm_fn())
    assert fact == "The user has a sister whose wedding is next month."
    assert llm.last_complete["backend"] == "qwen"


def test_complete_skips_template_and_extract_fact_survives_total_failure():
    llm = make_llm()
    llm._backends["qwen"], llm._backends["gemini"] = failing("q"), failing("g")
    with pytest.raises(RuntimeError):
        llm.complete("anything")
    assert [e["backend"] for e in llm.last_complete["errors"]] == ["qwen", "gemini"]
    assert extract_fact("My sister's wedding is next month", llm.as_llm_fn()) is None


# ---------------------------------------------------------------- P3

@pytest.mark.parametrize("method", ["generate_reply", "stream_reply"])
def test_no_emotion_parameters(method):
    params = list(inspect.signature(getattr(LLM, method)).parameters)
    assert params == ["self", "system_prompt", "history", "user_text", "strategy"]
    for p in params:
        assert not any(w in p.lower() for w in ("prob", "emotion", "label", "mood", "confidence"))


def test_does_not_import_emotion_modules():
    tree = ast.parse((CODE_DIR / "gpu" / "llm_gpu.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"ser_gpu", "mood_manager", "emotion_predictor"}


# ---------------------------------------------------------------- config

def test_config_presets():
    gpu, cpu = load_config("gpu_16gb")["llm"], load_config("smoke_cpu")["llm"]
    assert gpu["backend"] == "qwen" and gpu["model"] == "Qwen/Qwen3-8B" and gpu["quantization"] == "4bit"
    assert gpu["top_k"] == 20 and gpu["top_p"] == 0.8 and gpu["temperature"] == 0.7
    assert cpu["backend"] == "gemini"
    for llm in (gpu, cpu):
        assert llm["history_turns"] == 6 and llm["fallback"] == ["gemini", "template"]
        assert llm["allow_fallback"] is True and llm["extract_max_new_tokens"] == 60
    for p in ("gpu_12gb", "gpu_24gb"):
        assert load_config(p)["llm"]["model"] == "Qwen/Qwen3-8B"


def test_4bit_on_cpu_is_a_clear_error():
    llm = make_llm()                                        # gpu_16gb values on a CPU device
    with pytest.raises(ValueError, match="CUDA"):
        llm.load()


def test_new_csv_path_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_gpu, "code_path", lambda rel: tmp_path / rel)
    (tmp_path / "results" / "metrics").mkdir(parents=True)
    first = llm_gpu.new_csv_path("qwen", "Qwen/Qwen3-0.6B")
    assert first.name == "llm_demo_qwen_Qwen3-0.6B.csv"
    first.write_text("x")
    assert llm_gpu.new_csv_path("qwen", "Qwen/Qwen3-0.6B") != first


# ---------------------------------------------------------------- integration (no downloads)

def _qwen06_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001
        return False
    return all(isinstance(try_to_load_from_cache("Qwen/Qwen3-0.6B", f), str)
               for f in ("config.json", "tokenizer_config.json", "model.safetensors"))


@pytest.mark.skipif(not _qwen06_cached(), reason="needs Qwen/Qwen3-0.6B in the HF cache")
def test_integration_qwen3_06b_cpu():
    llm = LLM(load_config("smoke_cpu"), "cpu", torch.float32,
              backend="qwen", model_name="Qwen/Qwen3-0.6B", quantization="none").load()
    llm.allow_fallback = False                               # the answer must come from Qwen itself
    llm.max_new_tokens = 40
    system_prompt = response_generator.build_system_prompt("NEUTRAL", None, check_in=False)
    r = llm.generate_reply(system_prompt, [], "I'm walking home from class.", "NEUTRAL")
    assert r["backend"] == "qwen" and r["text"]
    assert "<think>" not in r["text"] and "</think>" not in r["text"]
