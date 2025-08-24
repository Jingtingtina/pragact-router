# -*- coding: utf-8 -*-
from typing import List

class LLMClient:
    def mcq_logprobs(self, prompt: str, options: List[str]) -> List[float]:
        raise NotImplementedError
    def mcq_choice(self, prompt: str) -> str:
        raise NotImplementedError

class DummyLLM(LLMClient):
    def mcq_logprobs(self, prompt: str, options: List[str]) -> List[float]:
        import random, math
        xs = [random.random() for _ in options]
        s = sum(xs)
        ps = [x/s for x in xs]
        return [math.log(p) for p in ps]
    def mcq_choice(self, prompt: str) -> str:
        import random
        return random.choice(list("ABCDEF"))


# --- OpenRouter wrapper appended ---
class OpenRouterLLM(LLMClient):
    """
    OpenRouter (OpenAI-compatible) chat/completions with logprobs.
    Needs env var: OPENROUTER_API_KEY
    """
    def __init__(self, model: str):
        import os
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY in your environment.")

    def _chat(self, prompt: str, *, logprobs: bool, top_logprobs: int = 6, max_tokens: int = 1):
        import requests
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Answer with a single capital letter (A-F) only."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0
        }
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def mcq_choice(self, prompt: str) -> str:
        data = self._chat(prompt, logprobs=False, max_tokens=1)
        out = data["choices"][0]["message"]["content"].strip()
        return out[:1].upper()

    
    def mcq_logprobs(self, prompt: str, options):
        import math, re
        data = self._chat(prompt, logprobs=True, top_logprobs=50, max_tokens=1)
        try:
            content = data["choices"][0]["logprobs"]["content"]
        except Exception:
            return [math.log(1.0/len(options))]*len(options)

        # Use the first non-whitespace token's top_logprobs if available
        tl = []
        for tok in content:
            tokstr = tok.get("token") or tok.get("decoded") or ""
            if tokstr.strip() != "":
                tl = tok.get("top_logprobs") or []
                break
        if not tl and content:
            tl = content[0].get("top_logprobs") or []

        # Accept variations: "A", " A", "A.", "A:", "A)" and provider's "decoded"
        def lp_for(letter):
            best = None
            patterns = {
                letter, f" {letter}", f"{letter}.", f"{letter}:", f"{letter})",
                f" {letter}.", f" {letter}:", f" {letter})"
            }
            for x in tl:
                tok = x.get("token") or ""
                dec = x.get("decoded") or ""
                if tok in patterns or dec in patterns:
                    best = x["logprob"] if (best is None or x["logprob"] > best) else best
            return best if best is not None else -20.0

        letters = ["A","B","C","D","E","F"]
        return [lp_for(L) for L in letters]
