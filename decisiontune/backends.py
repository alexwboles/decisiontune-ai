"""Inference backends: a common interface, a deterministic mock for tests,
and a HuggingFace backend for real models (lazy imports — torch is only
needed when you actually serve or train a real model)."""
from __future__ import annotations

import hashlib
import time
from typing import Dict, List

from .dataset import Question
from .prompts import option_labels, parse_output, render


class Backend:
    name = "base"

    def decide(self, state: str, questions: List[Question]) -> Dict[str, Dict[str, float]]:
        """Return {question_id: {option: probability}}."""
        raise NotImplementedError

    def info(self) -> Dict[str, str]:
        return {"backend": self.name}


class MockBackend(Backend):
    """Deterministic test double. No ML, no downloads.

    mode="first": puts `first_prob` on the first option of every question,
        remainder spread uniformly. Lets eval tests create a backend with a
        known accuracy against any labeled set.
    mode="uniform": flat probabilities (worst-case calibration baseline).
    mode="hash": deterministic pseudo-random probabilities from a hash of
        (state, question id) — stable across runs, looks like a real model.
    `latency_ms` simulates inference time so latency metrics are exercised.
    """
    name = "mock"

    def __init__(self, mode: str = "first", first_prob: float = 0.8, latency_ms: float = 5.0):
        if mode not in ("first", "uniform", "hash"):
            raise ValueError(f"unknown mock mode: {mode}")
        self.mode = mode
        self.first_prob = first_prob
        self.latency_ms = latency_ms

    def _probs(self, q: Question, state: str) -> Dict[str, float]:
        labels = option_labels(q)
        if self.mode == "first":
            rest = (1.0 - self.first_prob) / max(len(labels) - 1, 1)
            return {l: (self.first_prob if i == 0 else rest) for i, l in enumerate(labels)}
        if self.mode == "uniform":
            u = 1.0 / len(labels)
            return {l: u for l in labels}
        # hash mode: deterministic pseudo-random, normalized
        raw = []
        for l in labels:
            h = hashlib.sha256(f"{state}|{q.id}|{l}".encode()).digest()
            raw.append(int.from_bytes(h[:4], "big") % 1000 + 1)
        total = sum(raw)
        return {l: v / total for l, v in zip(labels, raw)}

    def decide(self, state: str, questions: List[Question]) -> Dict[str, Dict[str, float]]:
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        return {q.id: self._probs(q, state) for q in questions}

    def info(self) -> Dict[str, str]:
        return {"backend": self.name, "mode": self.mode,
                "first_prob": str(self.first_prob), "latency_ms": str(self.latency_ms)}


class HFBackend(Backend):
    """Real-model backend: transformers + optional PEFT LoRA adapter.

    model_id: any causal-LM HuggingFace id (default: Clef-flash 9B).
    adapter_path: directory with a PEFT LoRA adapter to load on top.
    Scoring: for each question we score every option as a completion of the
    rendered prompt and softmax the log-probs — the standard way to get
    calibrated-ish probabilities out of a generative model for a closed
    option set. Temperature scaling can be applied via `temperature`.
    """
    name = "hf"

    def __init__(self, model_id: str, adapter_path: str | None = None,
                 temperature: float = 1.0, device: str | None = None,
                 load_in_4bit: bool = False):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.temperature = temperature
        self.device = device
        self.load_in_4bit = load_in_4bit
        self._tok = None
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "HFBackend needs torch and transformers: "
                "pip install torch transformers peft  (see README for the real-model commands)"
            ) from e
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        kwargs: dict = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True
        if self.device:
            kwargs["device_map"] = self.device
        else:
            kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise RuntimeError("LoRA adapter needs peft: pip install peft") from e
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
            self._model = self._model.merge_and_unload()
        self._model.eval()

    def _option_logprob(self, prompt: str, option: str) -> float:
        torch = self._torch
        enc_prompt = self._tok(prompt, return_tensors="pt")
        enc_full = self._tok(prompt + option, return_tensors="pt")
        input_ids = enc_full["input_ids"].to(self._model.device)
        prompt_len = enc_prompt["input_ids"].shape[1]
        with torch.no_grad():
            logits = self._model(input_ids).logits[0]
        # sum log-prob of the option tokens given the prompt
        log_probs = torch.log_softmax(logits, dim=-1)
        opt_ids = input_ids[0][prompt_len:]
        lp = 0.0
        for i, tok in enumerate(opt_ids):
            lp += log_probs[prompt_len - 1 + i, tok].item()
        return lp / max(len(opt_ids), 1)  # length-normalized

    def decide(self, state: str, questions: List[Question]) -> Dict[str, Dict[str, float]]:
        import math
        self._ensure()
        out: Dict[str, Dict[str, float]] = {}
        for q in questions:
            prompt = render(q, state)
            labels = option_labels(q)
            lps = [self._option_logprob(prompt, " " + lab) / max(self.temperature, 1e-6)
                   for lab in labels]
            m = max(lps)
            exps = [math.exp(x - m) for x in lps]
            total = sum(exps)
            out[q.id] = {lab: e / total for lab, e in zip(labels, exps)}
        return out

    def info(self) -> Dict[str, str]:
        return {"backend": self.name, "model_id": self.model_id,
                "adapter_path": self.adapter_path or "",
                "temperature": str(self.temperature)}
