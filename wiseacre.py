#!/usr/bin/env python3
"""
Wiseacre: a private, offline-capable AI chat that runs on your own computer.

It picks a Qwen model that fits your hardware, downloads it the first time,
and opens a chat page in your browser. Nothing you type leaves your machine.

When the model is large enough to benefit, a small "draft" model is used for
speculative decoding, which makes answers appear faster without changing them.

Usage:
    python wiseacre.py                  # auto-pick a model, open the browser
    python wiseacre.py --list           # show the model tiers and what fits
    python wiseacre.py --tier m         # force a tier
    python wiseacre.py --no-speculative # turn speculative decoding off
    python wiseacre.py --target PATH_OR_HF_ID --draft PATH_OR_HF_ID
"""

import argparse
import os
import re
import sys
import threading
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


# ------------------------------------------------------------
# Model tiers
# ------------------------------------------------------------
# All models are Qwen2.5 Instruct. They share one tokenizer, which speculative
# decoding requires, and none of them needs a login or license click-through.
# Edit this table to try other models (for example the Qwen3 family).
#
# min_gb  : memory needed (GPU memory on NVIDIA, otherwise system RAM)
# size_gb : approximate download and in-memory size at 16-bit precision
# license : Apache-2.0 models are free for any use. The 3B model uses the Qwen
#           Research License (non-commercial use), and a notice is printed.

TIERS = [
    {"key": "xl", "min_gb": 90, "target": "Qwen/Qwen2.5-32B-Instruct", "draft": "Qwen/Qwen2.5-0.5B-Instruct", "size_gb": 66, "license": "Apache-2.0"},
    {"key": "l",  "min_gb": 40, "target": "Qwen/Qwen2.5-14B-Instruct", "draft": "Qwen/Qwen2.5-0.5B-Instruct", "size_gb": 30, "license": "Apache-2.0"},
    {"key": "m",  "min_gb": 20, "target": "Qwen/Qwen2.5-7B-Instruct",  "draft": "Qwen/Qwen2.5-0.5B-Instruct", "size_gb": 15, "license": "Apache-2.0"},
    {"key": "s",  "min_gb": 12, "target": "Qwen/Qwen2.5-3B-Instruct",  "draft": "Qwen/Qwen2.5-0.5B-Instruct", "size_gb": 6,  "license": "Qwen-Research"},
    {"key": "xs", "min_gb": 0,  "target": "Qwen/Qwen2.5-1.5B-Instruct", "draft": None,                        "size_gb": 3,  "license": "Apache-2.0"},
]

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MAX_PROMPT_TOKENS = 6000      # older turns are dropped past this, to bound memory use
DEFAULT_K = 5                 # tokens the draft model proposes per round
STATS_MARK = "⏱"         # stopwatch; marks the speed line added under answers


# ------------------------------------------------------------
# Hardware
# ------------------------------------------------------------

def detect_hardware():
    """Return the best device and how much memory the models can use."""
    ram_gb = psutil.virtual_memory().total / 1e9
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {"device": torch.device("cuda"), "memory_gb": props.total_memory / 1e9,
                "name": f"NVIDIA {props.name}"}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"device": torch.device("mps"), "memory_gb": ram_gb, "name": "Apple Silicon GPU"}
    return {"device": torch.device("cpu"), "memory_gb": ram_gb, "name": "CPU only"}


def pick_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def choose_tier(hw, forced=None):
    if forced:
        for tier in TIERS:
            if tier["key"] == forced:
                return tier
        raise SystemExit(f"Unknown tier '{forced}'. Choose from: {', '.join(t['key'] for t in TIERS)}")
    usable = hw["memory_gb"]
    candidates = [t for t in TIERS if usable >= t["min_gb"]]
    if hw["device"].type == "cpu":
        # Large models are painfully slow without a GPU.
        candidates = [t for t in candidates if t["key"] in ("s", "xs")] or [TIERS[-1]]
    return candidates[0]


def print_tiers(hw):
    print(f"Detected: {hw['name']}, {hw['memory_gb']:.0f} GB memory\n")
    chosen = choose_tier(hw)
    print(f"{'tier':<6}{'model':<30}{'needs':>8}{'size':>8}  license")
    for t in TIERS:
        mark = "  <- auto-selected" if t is chosen else ""
        print(f"{t['key']:<6}{t['target']:<30}{t['min_gb']:>6} GB{t['size_gb']:>6} GB  {t['license']}{mark}")


# ------------------------------------------------------------
# The chat engine
# ------------------------------------------------------------

def load_model(name, device, dtype):
    kwargs = {}
    if device.type == "cuda":
        kwargs["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs)
    except TypeError:  # older transformers versions call it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kwargs)
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model


class Engine:
    def __init__(self, target_name, draft_name, device, dtype):
        self.device = device
        self.target_name, self.draft_name = target_name, draft_name
        self.tokenizer = AutoTokenizer.from_pretrained(target_name)

        print(f"Loading {target_name} ...", flush=True)
        self.target = load_model(target_name, device, dtype)
        self.draft = None
        if draft_name:
            if AutoTokenizer.from_pretrained(draft_name).get_vocab() != self.tokenizer.get_vocab():
                raise SystemExit("The draft and target models must share a tokenizer.")
            print(f"Loading {draft_name} (draft) ...", flush=True)
            self.draft = load_model(draft_name, device, dtype)
            self._match_draft_to_target()
            self.draft.generation_config.num_assistant_tokens = DEFAULT_K
            self.draft.generation_config.num_assistant_tokens_schedule = "constant"
        self.speculative_ok = self.draft is not None

        eos = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos

    def _match_draft_to_target(self):
        """Make the draft's config agree with the target's.

        Transformers only pairs models for speculative decoding when their
        configs report the same vocabulary size and special-token ids. Qwen2.5
        models share one tokenizer but pad their output layers to different
        widths (0.5B: 151936, 7B: 152064), which would otherwise be rejected as
        "different tokenizers". Padding the draft's output layer to the
        target's width fixes that; the extra entries are unused token ids."""
        width = self.target.config.vocab_size
        if self.draft.config.vocab_size != width:
            self.draft.resize_token_embeddings(width)
        for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
            setattr(self.draft.config, attr, getattr(self.target.config, attr, None))

    # -- prompt building --------------------------------------------------

    def build_input(self, history, system_prompt):
        """Apply the chat template; drop the oldest turns if the prompt gets too long."""
        history = list(history)
        while True:
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt.strip() else []) + history
            encoded = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                return_dict=True, enable_thinking=False,
            )
            ids = encoded["input_ids"]
            if ids.shape[1] <= MAX_PROMPT_TOKENS or len(history) <= 1:
                return ids.to(self.device)
            history.pop(0)

    # -- generation -------------------------------------------------------

    def stream(self, history, system_prompt, max_new, temperature, use_speculative):
        """Yield (text so far, stats dict). Stats are filled in at the end."""
        ids = self.build_input(history, system_prompt)
        attempts = [True, False] if (use_speculative and self.speculative_ok) else [False]

        for speculative in attempts:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            box = {"error": None, "n_new": 0}
            started = time.perf_counter()

            thread = threading.Thread(
                target=self._run, args=(ids, streamer, box, max_new, temperature, speculative), daemon=True)
            thread.start()

            text = ""
            for piece in streamer:
                text += piece
                yield text, None
            thread.join()

            if box["error"] is None:
                elapsed = time.perf_counter() - started
                yield text, {"tokens": box["n_new"], "seconds": elapsed, "speculative": speculative}
                return
            if speculative and not text:
                print(f"Speculative decoding failed ({box['error']}); continuing without it.", flush=True)
                self.speculative_ok = False
                continue
            raise box["error"]

    def _run(self, ids, streamer, box, max_new, temperature, speculative):
        try:
            kwargs = dict(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                streamer=streamer, pad_token_id=self.pad_id,
            )
            if temperature <= 0:
                kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            else:
                kwargs.update(do_sample=True, temperature=temperature, top_p=0.8, top_k=20,
                              repetition_penalty=1.05)
            if speculative:
                kwargs["assistant_model"] = self.draft
            with torch.no_grad():
                out = self.target.generate(**kwargs)
            box["n_new"] = int(out.shape[1] - ids.shape[1])
        except Exception as e:  # reported to the caller; also unblocks the streamer
            box["error"] = e
            streamer.on_finalized_text("", stream_end=True)

    def warm_up(self):
        """One tiny run so first-use GPU setup does not slow the first real answer."""
        history = [{"role": "user", "content": "Hi"}]
        for _ in self.stream(history, "", 4, 0.0, False):
            pass
        if self.speculative_ok:
            for _ in self.stream(history, "", 4, 0.0, True):
                pass


# ------------------------------------------------------------
# Web interface
# ------------------------------------------------------------

def clean_history(history):
    """Gradio keeps our speed line in past answers; remove it before prompting the model."""
    cleaned = []
    for message in history:
        content = message["content"]
        if isinstance(content, list):  # multimodal-style content
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        if message["role"] == "assistant":
            content = re.sub(rf"\n\n\*{STATS_MARK}.*\*\s*$", "", content, flags=re.S)
        cleaned.append({"role": message["role"], "content": content})
    return cleaned


def stats_line(stats):
    rate = stats["tokens"] / stats["seconds"] if stats["seconds"] else 0
    mode = "speculative decoding on" if stats["speculative"] else "speculative decoding off"
    return f"\n\n*{STATS_MARK} {stats['tokens']} tokens in {stats['seconds']:.1f} s ({rate:.1f} tokens/s), {mode}*"


def build_ui(engine, title):
    import gradio as gr

    def respond(message, history, system_prompt, max_new, temperature, use_speculative):
        messages = clean_history(history) + [{"role": "user", "content": message}]
        text = ""
        for text, stats in engine.stream(messages, system_prompt, int(max_new), float(temperature), use_speculative):
            if stats is None:
                yield text
            else:
                yield text + stats_line(stats)

    spec_available = engine.speculative_ok
    extras = [
        gr.Textbox(DEFAULT_SYSTEM_PROMPT, label="Instructions for the assistant", lines=2),
        gr.Slider(64, 2048, value=512, step=64, label="Maximum answer length (tokens)"),
        gr.Slider(0.0, 1.2, value=0.7, step=0.1,
                  label="Creativity (0 = most predictable and repeatable)"),
        gr.Checkbox(spec_available, label="Speculative decoding (faster answers, same quality)",
                    interactive=spec_available),
    ]
    return gr.ChatInterface(
        respond, type="messages", title=title,
        description=("Runs entirely on this computer. Your conversation is not sent anywhere. "
                     "Answers can be wrong, so check anything important."),
        additional_inputs=extras, additional_inputs_accordion=gr.Accordion("Settings", open=False),
        fill_height=True,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wiseacre, a private AI chat that runs on your computer.")
    parser.add_argument("--list", action="store_true", help="show model tiers and exit")
    parser.add_argument("--tier", help="force a tier: " + ", ".join(t["key"] for t in TIERS))
    parser.add_argument("--target", help="custom main model (Hugging Face id or local folder)")
    parser.add_argument("--draft", help="custom draft model for speculative decoding")
    parser.add_argument("--no-speculative", action="store_true", help="do not load the draft model")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--lan", action="store_true",
                        help="let other computers on your network open the chat (off by default)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    # Keep the console readable for non-programmers: show real problems only.
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    hw = detect_hardware()
    if args.list:
        print_tiers(hw)
        return

    if args.target:
        target, draft, license_name = args.target, args.draft, "unknown"
    else:
        tier = choose_tier(hw, args.tier)
        target, draft, license_name = tier["target"], tier["draft"], tier["license"]
        print(f"Detected {hw['name']} with {hw['memory_gb']:.0f} GB. Using tier '{tier['key']}': {target}")
        print(f"First run downloads about {tier['size_gb']} GB, once. After that it works offline.")
        if hw["device"].type == "cpu":
            print("No GPU found, so answers will be slow. A smaller model was chosen.")
    if args.no_speculative:
        draft = None
    if license_name == "Qwen-Research":
        print("Note: this model is released under the Qwen Research License (non-commercial use).")

    engine = Engine(target, draft, hw["device"], pick_dtype(hw["device"]))
    if not args.no_warmup:
        print("Warming up ...", flush=True)
        engine.warm_up()

    short = target.rstrip("/").split("/")[-1]
    ui = build_ui(engine, f"Wiseacre ({short})")
    print("\nOpening the chat in your browser. Press Ctrl+C here to stop.\n")
    ui.queue().launch(server_name="0.0.0.0" if args.lan else "127.0.0.1",
                      server_port=args.port, inbrowser=not args.no_browser, share=False)


if __name__ == "__main__":
    main()
