import re
import os

import json, re
from typing import List, Tuple
from src.llm_api import LLMClient


_Q_EN  = re.compile(r'\?\s*$')
_Q_ZH  = re.compile(r'[？吗呢]\s*$')
_REQ_EN = re.compile(r'^(please|kindly|could you|can you|would you|pls)\b', re.I)
_REQ_ZH = re.compile(r'^(请|麻烦|能否|可以|烦请)')

def fast_cue(text: str):
    """
    High-precision cues only. Return one of:
      "declaration" | "promise" | "expressive" | "question" | "request" | "statement"
    or None if no strong cue. English is checked in lowercase; Chinese left as-is.
    """
    t = text.strip()
    t_en = t.lower()

    # --- hard cues (very high precision) ---
    # Declaration
    if any(kw in t_en for kw in ["we hereby declare", "we hereby announce", "i hereby declare", "hereby declare", "hereby announce"])        or any(kw in t for kw in ["特此宣布","兹宣布","公告如下","特此声明","兹通知","特此公告","兹公告"]):
        return "declaration"

    # Promise
    if "i promise" in t_en or any(kw in t for kw in ["我保证","我承诺","一定会","必将"]):
        return "promise"

    # Expressive
    if any(kw in t_en for kw in ["thanks", "thank you", "sorry", "apologies", "apologise", "apologize"])        or any(kw in t for kw in ["谢谢","多谢","十分感谢","抱歉","很抱歉","对不起"]):
        return "expressive"

    # Question
    if "?" in t        or any(t_en.startswith(w) for w in ["who","what","why","how","when","where","do ","does ","did ","can ","could ","will ","would ","should ","is ","are ","was ","were "])        or any(kw in t for kw in ["吗","呢","是否","能否","请问"]):
        return "question"

    # Request
    if any(kw in t_en for kw in ["please ", "please,", "kindly ", "could you", "would you", "can you", "please help", "please update", "please summarize", "please summarise"])        or any(t.startswith(prefix) for prefix in ["请","麻烦","烦请","请你","请您","劳驾"]):
        return "request"

    # Statement (narrow patterns to avoid stealing Decl/Prom/Expr)
    import re as _re
    if any(_re.search(pat, t_en) for pat in [
        r"\bthere (is|are)\b",
        r"\b(includes|contains)\b",
        r"\bwas updated yesterday\b",
        r"\bis available\b",
    ]) or any(_re.search(pat, t) for pat in [
        r"(是|为).+重要", r"(包含|包括|含有)", r"(已经|已)更新", r"可用", r"中有"
    ]):
        return "statement"

    return None

class PragActProbe:
    def __init__(self, templates_path: str, labels_path: str, use_logprobs: bool = True):
        self.tpl = json.load(open(templates_path, "r", encoding="utf-8"))
        self.labels_doc = json.load(open(labels_path, "r", encoding="utf-8"))
        self.options_text = self.tpl.get("options_en") or self.tpl.get("options_zh")
        self.keys = self.tpl["options_key_order"] 
        self.mcq_templates = self.tpl.get("mcq_templates", [])
        self.word_templates = self.tpl.get("word_templates", self.mcq_templates)
        self.calib_filler = self.tpl.get("calibration_template", "N/A")
        self.use_logprobs = use_logprobs
        self.priors = None 

    def calibrate_priors(self, llm: LLMClient):
        import numpy as np
        logs = []
        for t in self.mcq_templates:
            prompt = t.format(text=self.calib_filler)
            z = llm.mcq_logprobs(prompt, self.options_text)
            logs.append(z)
        self.priors = list(np.mean(logs, axis=0)) if logs else None

    def _tie_break(self, text: str):
        t = text.strip()
        lower = t.lower()
       
        is_zh = any(ch in "".join(self.options_text) for ch in "陈疑请求承表宣")
       
        def idx_of(key_name):
            try:
                k = key_name if key_name in self.keys else {
                    "Statement":"statement","Question":"question","Request":"request",
                    "Promise":"promise","Expressive":"expressive","Declaration":"declaration",
                    "陈述":"statement","疑问":"question","请求":"request","承诺":"promise","表达":"expressive","宣告":"declaration"
                }[key_name]
            except KeyError:
                return None
            try:
                pos = ["statement","question","request","promise","expressive","declaration"].index(k)
                return pos
            except ValueError:
                return None

        if is_zh:
           
            if "？" in t or re.search(r"(吗|呢|什么|为何|为什么).*[？?]$", t):
                return idx_of("question")
            if re.search(r"(^请|请.*?。|麻烦|能否|可以.*?吗)", t):
                return idx_of("request")
            if re.search(r"(保证|承诺)", t):
                return idx_of("promise")
            if re.search(r"(谢谢|多谢|抱歉|对不起)", t):
                return idx_of("expressive")
            if re.search(r"(兹宣布|宣布|特此声明)", t):
                return idx_of("declaration")
        else:
           
            if t.endswith("?") or "?" in t:
                return idx_of("question")
            if any(p in lower for p in ["please", "could you", "would you", "can you", "kindly"]) \
               or re.match(r"^(summarize|open|list|explain|give|write|translate|show|provide|compute|calculate|answer)\b", lower):
                return idx_of("request")
            if any(p in lower for p in ["i promise", "we promise"]) or re.search(r"\b(i|we)\s+will\b", lower):
                return idx_of("promise")
            if any(p in lower for p in ["thanks", "thank you", "sorry", "apologies"]):
                return idx_of("expressive")
            if any(p in lower for p in ["we hereby declare", "i hereby declare", "is hereby declared", "hereby declare"]):
                return idx_of("declaration")
        return None

    def score(self, text: str, llm: LLMClient) -> Tuple[int, float, List[float]]:

        # Optional fast-cue short-circuit (enable via env PRAGACT_FAST_CUES=1)
        use_fast = os.getenv('PRAGACT_FAST_CUES','0').lower() in ('1','true','yes','y','on')
        if use_fast:
            g = fast_cue(text)
            if g in self.keys:
                idx = self.keys.index(g)
                return idx, 1.0, [0.0]*len(self.keys)
        import numpy as np, math
        if self.use_logprobs and self.mcq_templates:
            all_scores = []
            for t in self.mcq_templates:
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
            for t in self.word_templates:
                prompt = t.format(text=text)
                lab = llm.mcq_label(prompt, self.options_text)
                votes.append(lab)
            c = Counter(votes)
            label_text, _ = c.most_common(1)[0]
            idx = self.options_text.index(label_text)
           
            total = sum(c.values())
            probs = [v/total for v in c.values()]
            entropy = -sum(p*math.log(p+1e-12) for p in probs)
            max_entropy = math.log(len(c))
            confidence = 1.0 - (entropy / (max_entropy + 1e-12))
           
            if confidence < 0.30:
                h = self._tie_break(text)
                if h is not None:
                    idx = h
            return idx, float(confidence), [0.0]*len(self.options_text)
