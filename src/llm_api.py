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
