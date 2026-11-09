"""
Offline smoke test. Builds two tiny random Qwen2-style models whose output
layers have different widths (as the real Qwen2.5 0.5B and 7B do), then checks
the engine, the model tiers, the saved options and conversations, and the web
page. It needs no internet and downloads nothing.

Run from the project folder:
    python tests/smoke_test.py
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers import models as tk_models
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import engine as engine_mod  # noqa: E402
import models as model_mod  # noqa: E402
import store as store_mod  # noqa: E402
import ui as ui_mod  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SYSTEM = "You are helpful."


# ------------------------------------------------------------
# Tiny stand-in models
# ------------------------------------------------------------

def make_tokenizer(path):
    tok = Tokenizer(tk_models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(["the quick brown fox jumps over the lazy dog " * 20,
                             "hello world how are you today " * 20], trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)
    return len(fast)


def make_model(path, vocab_size, hidden, layers, seed):
    torch.manual_seed(seed)
    Qwen2ForCausalLM(Qwen2Config(
        vocab_size=vocab_size, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=False,
        eos_token_id=2, pad_token_id=0, bos_token_id=None)).save_pretrained(path)


def last(stream):
    """The final (text, stats) pair of an engine.stream() call."""
    result = None
    for result in stream:
        pass
    return result


# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

def check_tiers():
    def hw(gb, kind):
        return model_mod.Hardware(torch.device(kind), gb, "test")

    expected = [(100, "cuda", "xl"), (96, "mps", "xl"), (64, "mps", "l"), (24, "mps", "m"),
                (16, "mps", "s"), (8, "mps", "xs"), (128, "cpu", "s"), (8, "cpu", "xs")]
    for gb, kind, key in expected:
        assert model_mod.choose_tier(hw(gb, kind)).key == key, (gb, kind, key)
    assert model_mod.choose_tier(hw(8, "mps"), forced="m").key == "m"
    try:
        model_mod.choose_tier(hw(8, "mps"), forced="nope")
        raise AssertionError("unknown tier accepted")
    except SystemExit:
        pass
    assert model_mod.tier_table(hw(24, "mps")).count("auto-selected") == 1
    print("1. model tiers are chosen by memory; CPU-only computers stay small")


def check_generation(engine, target_dir, device):
    history = [{"role": "user", "content": "hello world how are you today"}]
    run = lambda temperature, speculative, max_new=40: last(  # noqa: E731
        engine.stream(history, SYSTEM, max_new, temperature, speculative))

    base_text, base_stats = run(0.0, False)
    spec_text, spec_stats = run(0.0, True)
    assert not base_stats.speculative and spec_stats.speculative, "speculative path fell back unexpectedly"
    assert base_text == spec_text, f"greedy outputs differ:\n{base_text!r}\n{spec_text!r}"
    print("2. greedy output is identical with and without speculative decoding")

    # A draft identical to the target is always accepted, which exercises the
    # "all tokens accepted" path. The answer must still match the baseline.
    same = engine_mod.Engine(target_dir, target_dir, device, torch.float32)
    same_text, same_stats = last(same.stream(history, SYSTEM, 40, 0.0, True))
    assert same_stats.speculative and same_text == base_text
    print("3. an identical draft (full acceptance) matches the baseline")

    torch.manual_seed(0)
    assert run(0.7, True)[1].speculative
    print("4. sampled speculative decoding runs")

    # Hitting the length limit is reported; finishing on its own is not.
    _, cut = run(0.0, False, max_new=5)
    assert cut.tokens == 5 and cut.truncated
    assert "cut off at the length limit" in ui_mod.stats_line(cut)
    assert "tokens/s" in ui_mod.stats_line(cut)
    cut.truncated = False
    assert "cut off" not in ui_mod.stats_line(cut)
    real_eos = engine.eos_ids
    engine.eos_ids = set(range(engine.target.config.vocab_size))   # pretend every token ends an answer
    assert not run(0.0, False, max_new=5)[1].truncated
    engine.eos_ids = real_eos
    print("5. cut-off answers are flagged; finished ones are not")

    cut.truncated = True
    cleaned = ui_mod.clean_history([{"role": "user", "content": "hi"},
                                    {"role": "assistant", "content": "partial" + ui_mod.stats_line(cut)}])
    assert cleaned[1]["content"] == "partial", cleaned
    print("6. the speed line and cut-off note are stripped from history")


def check_trimming(engine):
    engine_mod.MAX_PROMPT_TOKENS = 60
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "the quick brown fox " * 5}
               for i in range(9)]
    ids = engine.build_input(history, "")
    text = engine.tokenizer.decode(ids[0])
    assert ids.shape[1] < 400 and text.startswith("<|im_start|>user"), text[:60]
    assert text.count("<|im_start|>user") == text.count("<|im_start|>assistant")   # whole pairs only
    print("10. long histories lose whole oldest question-and-answer pairs")


def check_store(root):
    path = os.path.join(root, "home")
    store = store_mod.Store(path)
    defaults = store_mod.DEFAULT_SETTINGS
    assert store.load_settings() == defaults and not defaults["save_chats"]
    store.save_settings({**defaults, "max_new": 2048, "system_prompt": "I teach C++."})
    again = store_mod.Store(path).load_settings()
    assert again["max_new"] == 2048 and again["system_prompt"] == "I teach C++."
    store.settings_file.write_text("{ not json")
    assert store.load_settings() == defaults                       # damaged file: defaults
    store.settings_file.write_text('{"max_new": "lots", "save_chats": 1}')
    assert store.load_settings() == defaults                       # wrong types ignored
    print("7. options are remembered; damaged or odd settings files fall back to defaults")

    assert store.list_chats() == []                                # no chats folder yet
    msgs = [{"role": "user", "content": "Write 3 C++ problems!\nThanks"},
            {"role": "assistant", "content": "1. Loops\n2. Pointers"}]
    name = store.save_chat(None, msgs, "model-x")
    assert name.endswith("_write-3-c-problems.json"), name
    msgs.append({"role": "user", "content": "more"})
    assert store.save_chat(name, msgs, "model-x") == name          # updates in place
    assert store.save_chat(None, msgs[:1], "m").startswith(name[:16])  # a new chat gets a new file
    assert store.load_chat(name) == msgs
    assert (store.chat_dir / name.replace(".json", ".md")).exists()
    labels = [label for label, _ in store.list_chats()]
    assert len(labels) == 2 and "Write 3 C++ problems!" in labels[0] + labels[1], labels
    for bad in ("../settings.json", "/etc/passwd", "x/../y.json", "nope.json", "", None):
        assert store.load_chat(bad) is None, bad
    (store.chat_dir / "broken.json").write_text("{ nope")
    (store.chat_dir / "odd.json").write_text('{"title": 1, "created": "x", "messages": []}')
    assert len(store.list_chats()) == 2                            # damaged files are skipped
    print("8. conversations save, update, list and reopen; unsafe or damaged files are refused")
    return store


def check_web(engine, store):
    from gradio_client import Client

    ui = ui_mod.build_ui(engine, "test", store)
    ui.queue().launch(server_name="127.0.0.1", prevent_thread_lock=True, quiet=True, share=False)
    try:
        client = Client(ui.local_url, verbose=False)
        before = len(store.list_chats())
        reply = client.predict("hello world", [], SYSTEM, 64, 0.0, True, False, api_name="/chat")
        assert reply[0][-1]["role"] == "assistant" and "tokens/s" in reply[0][-1]["content"], reply
        assert len(store.list_chats()) == before, "saved a conversation although saving was off"
        client.predict("hello world", [], SYSTEM, 64, 0.0, True, True, api_name="/chat")
        assert len(store.list_chats()) == before + 1, "did not save when saving was on"
    finally:
        ui.close()
    print("9. the web page answers end to end and saves only when the option is on")


def check_logo(root):
    ui_mod.ASSETS = Path(root) / "assets"
    assert "<img" not in ui_mod.header_html("Title")
    assert "Title &lt;b&gt;" in ui_mod.header_html("Title <b>")
    ui_mod.ASSETS.mkdir()
    (ui_mod.ASSETS / "logo.png").write_bytes(b"\x89PNG fake")
    assert "data:image/png;base64," in ui_mod.header_html("Title")
    print("11. the logo appears only when assets/logo.png exists")


def main():
    check_tiers()

    root = tempfile.mkdtemp()
    target_dir, draft_dir = os.path.join(root, "target"), os.path.join(root, "draft")
    vocab = make_tokenizer(target_dir)
    make_tokenizer(draft_dir)
    # Different padded widths, like Qwen2.5 0.5B (151936) and 7B (152064)
    make_model(target_dir, vocab + 16, 64, 4, seed=1)
    make_model(draft_dir, vocab + 8, 32, 2, seed=2)

    device = torch.device("cpu")
    engine = engine_mod.Engine(target_dir, draft_dir, device, torch.float32)

    check_generation(engine, target_dir, device)
    store = check_store(root)
    check_web(engine, store)
    check_trimming(engine)
    check_logo(root)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
