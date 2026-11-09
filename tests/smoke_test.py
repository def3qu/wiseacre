"""
Offline smoke test. Builds two tiny random Qwen2-style models whose output
layers have different widths (as the real Qwen2.5 0.5B and 7B do), then checks:

  1. greedy answers are identical with and without speculative decoding
  2. sampled answers run with speculative decoding
  3. the chat history cleaner strips the speed line
  4. the web interface builds

It needs no internet and downloads nothing. Run from the project folder:
    python tests/smoke_test.py
"""

import os
import sys
import tempfile

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import wiseacre  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def make_tokenizer(path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = ["the quick brown fox jumps over the lazy dog " * 20, "hello world how are you today " * 20]
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)
    return len(fast)


def make_model(path, vocab_size, hidden, layers, seed):
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab_size, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=False,
        eos_token_id=2, pad_token_id=0, bos_token_id=None)
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(path)


def main():
    root = tempfile.mkdtemp()
    target_dir, draft_dir = os.path.join(root, "target"), os.path.join(root, "draft")
    vocab = make_tokenizer(target_dir)
    make_tokenizer(draft_dir)
    # Different padded widths, like Qwen2.5 0.5B (151936) and 7B (152064)
    make_model(target_dir, vocab + 16, 64, 4, seed=1)
    make_model(draft_dir, vocab + 8, 32, 2, seed=2)

    device = torch.device("cpu")
    engine = wiseacre.Engine(target_dir, draft_dir, device, torch.float32)
    history = [{"role": "user", "content": "hello world how are you today"}]

    def run(temperature, speculative):
        final = None
        for text, stats in engine.stream(history, "You are helpful.", 40, temperature, speculative):
            final = (text, stats)
        return final

    base_text, base_stats = run(0.0, False)
    spec_text, spec_stats = run(0.0, True)
    assert base_stats and not base_stats["speculative"]
    assert spec_stats and spec_stats["speculative"], "speculative path fell back unexpectedly"
    assert base_text == spec_text, f"greedy outputs differ:\n{base_text!r}\n{spec_text!r}"
    print("1. greedy output identical with and without speculative decoding:", len(base_text), "chars")

    # A draft identical to the target is always accepted, which exercises the
    # "all tokens accepted" path. The answer must still match the baseline.
    same = wiseacre.Engine(target_dir, target_dir, device, torch.float32)
    same_text = None
    for text, stats in same.stream(history, "You are helpful.", 40, 0.0, True):
        same_text = (text, stats)
    assert same_text[1]["speculative"] and same_text[0] == base_text
    print("1b. identical draft (full acceptance) matches baseline")

    torch.manual_seed(0)
    sampled_text, sampled_stats = run(0.7, True)
    assert sampled_stats and sampled_stats["speculative"]
    print("2. sampled speculative decoding ran:", sampled_stats["tokens"], "tokens")

    line = wiseacre.stats_line(spec_stats)
    cleaned = wiseacre.clean_history([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "answer text" + line},
    ])
    assert cleaned[1]["content"] == "answer text", cleaned
    print("3. speed line stripped from history")

    ui = wiseacre.build_ui(engine, "test")
    assert ui is not None
    print("4. web interface builds")

    # Talk to the running web app the way a browser would
    from gradio_client import Client
    ui.queue().launch(server_name="127.0.0.1", prevent_thread_lock=True, quiet=True, share=False)
    try:
        client = Client(ui.local_url, verbose=False)
        reply = client.predict("hello world", "You are helpful.", 64, 0.0, True, api_name="/chat")
        assert isinstance(reply, str) and "tokens/s" in reply, reply
        print("4b. web app answered end to end:", repr(reply[-60:]))
    finally:
        ui.close()

    # Long history is trimmed from the oldest turns
    wiseacre.MAX_PROMPT_TOKENS = 60
    long_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "the quick brown fox " * 5}
                    for i in range(9)]
    ids = engine.build_input(long_history, "")
    assert ids.shape[1] <= 400
    print("5. long histories are trimmed:", ids.shape[1], "tokens")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
