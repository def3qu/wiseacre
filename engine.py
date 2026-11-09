"""Loading the models and generating answers, with optional speculative decoding."""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MAX_PROMPT_TOKENS = 6000   # older turns are dropped past this, to bound memory use
DRAFT_TOKENS = 5           # tokens the draft model proposes per round
# Sampling settings Qwen recommends for chat (used when creativity is above 0)
SAMPLING = dict(top_p=0.8, top_k=20, repetition_penalty=1.05)


@dataclass
class GenStats:
    """How one answer went, reported after its last word."""
    tokens: int
    seconds: float
    speculative: bool
    truncated: bool          # stopped by the length limit rather than finishing

    @property
    def tokens_per_second(self):
        return self.tokens / self.seconds if self.seconds else 0.0


@dataclass
class _Outcome:
    """Filled in by the generation thread for the caller to read afterwards."""
    error: Optional[Exception] = None
    new_tokens: int = 0
    finished: bool = True    # ended with an end-of-answer token


def load_model(name, device, dtype):
    kwargs = {"device_map": {"": 0}} if device.type == "cuda" else {}
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs)
    except TypeError:  # older transformers versions call it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kwargs)
    if device.type != "cuda":
        model = model.to(device)
    return model.eval()


class Engine:
    def __init__(self, target_name, draft_name, device, dtype):
        self.device = device
        self.target_name = target_name
        self.tokenizer = AutoTokenizer.from_pretrained(target_name)

        print(f"Loading {target_name} ...", flush=True)
        self.target = load_model(target_name, device, dtype)
        self.draft = self._load_draft(draft_name, device, dtype) if draft_name else None
        self.speculative_ok = self.draft is not None   # turned off if it ever fails

        eos = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos
        self.eos_ids = self._end_token_ids()

    # -- setup ------------------------------------------------------------

    def _load_draft(self, name, device, dtype):
        if AutoTokenizer.from_pretrained(name).get_vocab() != self.tokenizer.get_vocab():
            raise SystemExit("The draft and target models must share a tokenizer.")
        print(f"Loading {name} (draft) ...", flush=True)
        draft = load_model(name, device, dtype)
        self._match_config(draft)
        draft.generation_config.num_assistant_tokens = DRAFT_TOKENS
        draft.generation_config.num_assistant_tokens_schedule = "constant"
        return draft

    def _match_config(self, draft):
        """Make the draft's config agree with the target's.

        Transformers only pairs models for speculative decoding when their
        configs report the same vocabulary size and special-token ids. Qwen2.5
        models share one tokenizer but pad their output layers to different
        widths (0.5B: 151936, 7B: 152064), which would otherwise be rejected as
        "different tokenizers". Padding the draft's output layer to the
        target's width fixes that; the extra entries are unused token ids."""
        width = self.target.config.vocab_size
        if draft.config.vocab_size != width:
            draft.resize_token_embeddings(width)
        for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
            setattr(draft.config, attr, getattr(self.target.config, attr, None))

    def _end_token_ids(self):
        """Every token id that means "the answer is finished". Used to tell a
        finished answer from one cut off by the length limit."""
        ids = set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        configured = self.target.generation_config.eos_token_id
        if isinstance(configured, int):
            ids.add(configured)
        elif configured:
            ids.update(configured)
        return ids

    # -- prompt building --------------------------------------------------

    def build_input(self, history, system_prompt):
        """Apply the chat template to the conversation. If the prompt is too long,
        drop the oldest question-and-answer pairs, always keeping the last message."""
        history = list(history)
        system = [{"role": "system", "content": system_prompt}] if system_prompt.strip() else []
        while True:
            ids = self.tokenizer.apply_chat_template(
                system + history, add_generation_prompt=True, return_tensors="pt",
                return_dict=True, enable_thinking=False,
            )["input_ids"]
            if ids.shape[1] <= MAX_PROMPT_TOKENS or len(history) <= 1:
                return ids.to(self.device)
            del history[:min(2, len(history) - 1)]

    # -- generation -------------------------------------------------------

    def stream(self, history, system_prompt, max_new, temperature, use_speculative):
        """Yield (text so far, None) as the answer grows, then (final text, GenStats).

        If speculative decoding fails before producing any text, it is switched
        off for the rest of the session and the answer is generated normally."""
        ids = self.build_input(history, system_prompt)
        attempts = [True, False] if (use_speculative and self.speculative_ok) else [False]

        for speculative in attempts:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            outcome = _Outcome()
            started = time.perf_counter()
            thread = threading.Thread(
                target=self._generate, args=(ids, streamer, outcome, max_new, temperature, speculative),
                daemon=True)
            thread.start()

            text = ""
            for piece in streamer:
                text += piece
                yield text, None
            thread.join()

            if outcome.error is None:
                yield text, GenStats(
                    tokens=outcome.new_tokens, seconds=time.perf_counter() - started,
                    speculative=speculative,
                    truncated=outcome.new_tokens >= max_new and not outcome.finished)
                return
            if speculative and not text:
                print(f"Speculative decoding failed ({outcome.error}); continuing without it.", flush=True)
                self.speculative_ok = False
                continue
            raise outcome.error

    def _generate(self, ids, streamer, outcome, max_new, temperature, speculative):
        """Runs in a thread so the streamer can be read while the model works."""
        try:
            kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                          streamer=streamer, pad_token_id=self.pad_id)
            if temperature <= 0:
                kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            else:
                kwargs.update(do_sample=True, temperature=temperature, **SAMPLING)
            if speculative:
                kwargs["assistant_model"] = self.draft
            with torch.no_grad():
                out = self.target.generate(**kwargs)
            outcome.new_tokens = int(out.shape[1] - ids.shape[1])
            outcome.finished = int(out[0, -1]) in self.eos_ids
        except Exception as e:  # handed to the caller; also unblocks the streamer
            outcome.error = e
            streamer.on_finalized_text("", stream_end=True)

    def warm_up(self):
        """One tiny run so first-use GPU setup does not slow the first real answer."""
        history = [{"role": "user", "content": "Hi"}]
        for speculative in (False, True) if self.speculative_ok else (False,):
            for _ in self.stream(history, "", 4, 0.0, speculative):
                pass
