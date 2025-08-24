from typing import List

class LLMClient:
    def mcq_logprobs(self, prompt: str, options: List[str]) -> List[float]:
        raise NotImplementedError
    def mcq_choice(self, prompt: str) -> str:
        raise NotImplementedError
    def mcq_label(self, prompt: str, labels: List[str]) -> str:
        raise NotImplementedError

class DummyLLM(LLMClient):
    """Local dummy model for plumbing tests (no API)."""
    def mcq_logprobs(self, prompt: str, options: List[str]) -> List[float]:
        import random, math
        xs = [random.random() for _ in options]
        s = sum(xs)
        ps = [x/s for x in xs]
        return [math.log(max(p, 1e-12)) for p in ps]
    def mcq_choice(self, prompt: str) -> str:
        import random
        return random.choice(list("ABCDEF"))
    def mcq_label(self, prompt: str, labels: List[str]) -> str:
        import random
        return random.choice(labels)

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

    def _chat(self, prompt: str, *, logprobs: bool, top_logprobs: int = 50, max_tokens: int = 1, system: str = None):
        import requests
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        system_msg = system or "Answer with a single capital letter (A-F) only."
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
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
        import re
        data = self._chat(prompt, logprobs=False, max_tokens=1, system="Answer with a single capital letter (A-F) only.")
        out = data["choices"][0]["message"]["content"]
        txt = (out or "").strip().upper()
        m = re.search(r'\b([A-F])\b', txt)
        if m:
            return m.group(1)
        m2 = re.search(r'([A-F])', txt)
        return m2.group(1) if m2 else "A"

    def mcq_label(self, prompt: str, labels: List[str]) -> str:
        allowed = ", ".join(labels)
        sysmsg = f"Return EXACTLY one label from this set, no extra text: {allowed}"
        data = self._chat(prompt + "\nReturn exactly ONE label word from the list.", logprobs=False, max_tokens=8, system=sysmsg)
        out = (data["choices"][0]["message"]["content"] or "").strip()
       
        out = out.strip(' "\'\n\r\t：:，,。.;；')
        # Exact match first
        for lab in labels:
            if out == lab:
                return lab
        # Then substring match
        for lab in labels:
            if lab in out:
                return lab
        # Last resort: first label
        return labels[0]

    # --- logprob path (letters A-F) for ablations ---
    def mcq_logprobs(self, prompt: str, options):
        import math
        data = self._chat(prompt, logprobs=True, top_logprobs=50, max_tokens=1, system="Answer with a single capital letter (A-F) only.")
        try:
            content = data["choices"][0]["logprobs"]["content"]
        except Exception:
            return [math.log(1.0/len(options))]*len(options)

        # choose first non-whitespace token's top_logprobs
        tl = []
        for tok in content:
            tokstr = (tok.get("token") or tok.get("decoded") or "")
            if tokstr.strip() != "":
                tl = tok.get("top_logprobs") or []
                break
        if not tl and content:
            tl = content[0].get("top_logprobs") or []

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
