#!/usr/bin/env python3
"""
Wiseacre: a private, offline-capable AI chat that runs on your own computer.

It picks a Qwen model that fits your hardware, downloads it the first time,
and opens a chat page in your browser. Nothing you type leaves your machine.

When the model is large enough to benefit, a small "draft" model is used for
speculative decoding, which makes answers appear faster without changing them.

Usage:
    python wiseacre.py                   # auto-pick a model, open the browser
    python wiseacre.py --list            # show the model tiers and what fits
    python wiseacre.py --tier m          # force a tier
    python wiseacre.py --no-speculative  # turn speculative decoding off
    python wiseacre.py --target PATH_OR_HF_ID --draft PATH_OR_HF_ID

Files: models.py (which model fits), engine.py (generating answers),
store.py (remembered options and saved chats), ui.py (the web page).
"""

import argparse

from models import choose_tier, detect_hardware, pick_dtype, tier_table, TIERS


def parse_args():
    parser = argparse.ArgumentParser(description="Wiseacre, a private AI chat that runs on your computer.")
    parser.add_argument("--list", action="store_true", help="show model tiers and exit")
    parser.add_argument("--tier", help="force a tier: " + ", ".join(t.key for t in TIERS))
    parser.add_argument("--target", help="custom main model (Hugging Face id or local folder)")
    parser.add_argument("--draft", help="custom draft model for speculative decoding")
    parser.add_argument("--no-speculative", action="store_true", help="do not load the draft model")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--lan", action="store_true",
                        help="let other computers on your network open the chat (off by default)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--data-dir", default="~/Wiseacre",
                        help="folder for remembered options and (if turned on) saved conversations")
    return parser.parse_args()


def choose_models(args, hardware):
    """Work out which main and draft models to load, telling the user what was chosen."""
    if args.target:
        return args.target, (None if args.no_speculative else args.draft)

    tier = choose_tier(hardware, args.tier)
    print(f"Detected {hardware.name} with {hardware.memory_gb:.0f} GB. Using tier '{tier.key}': {tier.target}")
    print(f"First run downloads about {tier.size_gb} GB, once. After that it works offline.")
    if hardware.device.type == "cpu":
        print("No GPU found, so answers will be slow. A smaller model was chosen.")
    if tier.license == "Qwen-Research":
        print("Note: this model is released under the Qwen Research License (non-commercial use).")
    return tier.target, (None if args.no_speculative else tier.draft)


def main():
    args = parse_args()
    hardware = detect_hardware()
    if args.list:
        print(tier_table(hardware))
        return

    # Heavy imports come after the quick paths above, so --help and --list stay fast.
    from transformers.utils import logging as hf_logging
    from engine import Engine
    from store import Store
    from ui import ASSETS, build_ui

    hf_logging.set_verbosity_error()  # keep the console readable: show real problems only

    target, draft = choose_models(args, hardware)
    engine = Engine(target, draft, hardware.device, pick_dtype(hardware.device))
    if not args.no_warmup:
        print("Warming up ...", flush=True)
        engine.warm_up()

    store = Store(args.data_dir)
    title = f"Wiseacre ({target.rstrip('/').split('/')[-1]})"
    logo = ASSETS / "logo.png"

    print(f"\nOptions are remembered in {store.root}. Conversations are saved there only if you turn that on.")
    print("Opening the chat in your browser. Press Ctrl+C here to stop.\n")
    build_ui(engine, title, store).queue().launch(
        server_name="0.0.0.0" if args.lan else "127.0.0.1", server_port=args.port,
        inbrowser=not args.no_browser, share=False, show_api=False,
        favicon_path=str(logo) if logo.exists() else None)


if __name__ == "__main__":
    main()
