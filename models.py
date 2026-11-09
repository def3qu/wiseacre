"""Which model fits this computer: hardware detection and the model tiers.

All models are Qwen2.5 Instruct. They share one tokenizer, which speculative
decoding requires, and none of them needs a login or license click-through.
Edit TIERS to try other models (for example the Qwen3 family).
"""

from dataclasses import dataclass
from typing import Optional

import psutil
import torch


@dataclass(frozen=True)
class Tier:
    key: str
    min_gb: int              # memory needed: GPU memory on NVIDIA, otherwise system RAM
    target: str              # the main model
    draft: Optional[str]     # small helper model for speculative decoding, if it would help
    size_gb: int             # approximate download and in-memory size at 16-bit precision
    license: str             # Apache-2.0 is free for any use; Qwen-Research is non-commercial


_DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"

# Largest first: the first tier that fits the computer is the one chosen.
TIERS = (
    Tier("xl", 90, "Qwen/Qwen2.5-32B-Instruct", _DRAFT, 66, "Apache-2.0"),
    Tier("l",  40, "Qwen/Qwen2.5-14B-Instruct", _DRAFT, 30, "Apache-2.0"),
    Tier("m",  20, "Qwen/Qwen2.5-7B-Instruct",  _DRAFT, 15, "Apache-2.0"),
    Tier("s",  12, "Qwen/Qwen2.5-3B-Instruct",  _DRAFT, 6,  "Qwen-Research"),
    Tier("xs",  0, "Qwen/Qwen2.5-1.5B-Instruct", None,  3,  "Apache-2.0"),
)

# Larger models are painfully slow without a GPU, so CPU-only computers stay small.
CPU_TIERS = ("s", "xs")


@dataclass(frozen=True)
class Hardware:
    device: torch.device
    memory_gb: float
    name: str


def detect_hardware():
    """Find the best device and how much memory the models can use."""
    ram_gb = psutil.virtual_memory().total / 1e9
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return Hardware(torch.device("cuda"), props.total_memory / 1e9, f"NVIDIA {props.name}")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return Hardware(torch.device("mps"), ram_gb, "Apple Silicon GPU")  # memory is shared with the CPU
    return Hardware(torch.device("cpu"), ram_gb, "CPU only")


def pick_dtype(device):
    """The 16-bit number format each kind of device handles best."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def find_tier(key):
    for tier in TIERS:
        if tier.key == key:
            return tier
    raise SystemExit(f"Unknown tier '{key}'. Choose from: {', '.join(t.key for t in TIERS)}")


def choose_tier(hardware, forced=None):
    """The tier the user asked for, or else the largest one that fits."""
    if forced:
        return find_tier(forced)
    fits = [t for t in TIERS if hardware.memory_gb >= t.min_gb]
    if hardware.device.type == "cpu":
        fits = [t for t in fits if t.key in CPU_TIERS]
    return fits[0] if fits else TIERS[-1]


def tier_table(hardware):
    """A printable table of the tiers, marking the one this computer gets."""
    chosen = choose_tier(hardware)
    lines = [f"Detected: {hardware.name}, {hardware.memory_gb:.0f} GB memory", "",
             f"{'tier':<6}{'model':<30}{'needs':>8}{'size':>8}  license"]
    for t in TIERS:
        mark = "  <- auto-selected" if t is chosen else ""
        lines.append(f"{t.key:<6}{t.target:<30}{t.min_gb:>6} GB{t.size_gb:>6} GB  {t.license}{mark}")
    return "\n".join(lines)
