# -*- coding: utf-8 -*-
import json
from typing import List, Tuple
from src.llm_api import LLMClient

class PragActProbe:
    def __init__(self, templates_path: str, labels_path: str, use_logprobs: bool = True):
        self.tpl = json.load(open(templates_path, "r", encoding="utf-8"))
        self.labels_doc = json.load(open(labels_path, "r", encoding="utf-8"))
        # options text can be EN or ZH
        self.options_text = self.tpl.get("options_en") or self.tpl.get("options_zh")
        self.keys = self.tpl["options_key_order"]  # ["statement", ...]
        self.templates = self.tpl["mcq_templates"]
        self.calib_filler = self.tpl["calibration_template"]
        self.use_logprobs = use_logprobs
        self.priors = None  # filled by calibrate_priors()

    def calibrate_priors(self, llm: LLMClient):
        import numpy as np
        logs = []
        for t in self.templates:
            prompt = t.format(text=self.calib_filler)
            z = llm.mcq_logprobs(prompt, self.options_text)
            logs.append(z)
        self.priors = list(np.mean(logs, axis=0))

    def score(self, text: str, llm: LLMClient) -> Tuple[int, float, List[float]]:
        import numpy as np
        if self.use_logprobs:
            all_scores = []
            for t in self.templates:
                prompt = t.format(text=text)
                z = llm.mcq_logprobs(prompt, self.options_text)
                if self.priors is not None:
                    z = [zi - pi for zi, pi in zip(z, self.priors)]
                all_scores.append(z)
            sbar = np.mean(all_scores, axis=0)
            idx = int(np.argmax(sbar))
            top = float(sbar[idx])
            second = float(np.partition(sbar, -2)[-2])
            margin = top - second
            return idx, margin, list(map(float, sbar))
        else:
            from collections import Counter
            votes = []
            for t in self.templates:
                prompt = t.format(text=text)
                letter = llm.mcq_choice(prompt).strip().upper()[:1]
                votes.append(letter)
            c = Counter(votes)
            letter, _ = c.most_common(1)[0]
            idx = "ABCDEF".index(letter)
            # crude confidence from vote entropy
            import math
            total = sum(c.values())
            probs = [v/total for v in c.values()]
            entropy = -sum(p*math.log(p+1e-12) for p in probs)
            max_entropy = math.log(len(c))
            confidence = 1.0 - (entropy / (max_entropy + 1e-12))
            return idx, confidence, [0.0]*len(self.options_text)
