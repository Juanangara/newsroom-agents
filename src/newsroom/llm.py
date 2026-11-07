"""Interchangeable LLM backends. Every agent only depends on `LLM.chat`."""
from __future__ import annotations

import json
import re
from typing import Protocol


class LLM(Protocol):
    def chat(self, system: str, user: str, max_tokens: int = 700) -> str: ...


def extract_json(text: str):
    """Parse the first JSON object/array in a model reply (tolerates ``` fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON in model output: {text[:200]!r}")


def chat_json(llm: LLM, system: str, user: str, retries: int = 2, max_tokens: int = 700):
    """Ask for JSON and retry with the parse error fed back to the model."""
    prompt, last = user, None
    for _ in range(retries + 1):
        reply = llm.chat(system, prompt, max_tokens=max_tokens)
        try:
            return extract_json(reply)
        except ValueError as e:
            last = e
            prompt = f"{user}\n\nYour previous reply was not valid JSON ({e}). Reply with ONLY valid JSON."
    raise last  # type: ignore[misc]


class HFLocalLLM:
    """Local model for Colab's free T4 (4-bit quantised)."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", load_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tok = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"device_map": "auto"}
        if load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def chat(self, system: str, user: str, max_tokens: int = 700) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(
            self.model.device
        )
        out = self.model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
        return self.tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


class ClaudeLLM:
    def __init__(self, model: str = "claude-haiku-5-5", api_key: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, system: str, user: str, max_tokens: int = 700) -> str:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if b.type == "text").strip()


class FakeLLM:
    """Deterministic scripted LLM for tests: replies are popped in order."""

    def __init__(self, replies: list[str]):
        self.replies, self.calls = list(replies), []

    def chat(self, system: str, user: str, max_tokens: int = 700) -> str:
        self.calls.append((system, user))
        return self.replies.pop(0)
