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
    Lightweight cue-based detector for obvious question/request/etc.
    Returns (label, confidence) or None.
    Confidence is 1.0 for strong cues, 0.2 for weak ones.
    """
    import re
    if not text:
        return None
    t = text.strip()
    low = t.lower()

    # --- Declarations (ZH) ---
    if t.startswith(("兹宣布","特此公告","特此通知")):
        return ("declaration", 1.0)

    # --- Promises (EN/ZH) ---
    if any(p in low for p in ("i'll ","i will ","we'll ","we will ","i promise","we promise")):
        return ("promise", 1.0)
    if any(p in t for p in ("我会","我將","我将","我们会","我們會","我保证","我保證","我承诺","我承諾")):
        return ("promise", 1.0)

    # --- Statements (EN/ZH) ---
    if low.startswith(("please note","note that","for your information","fyi")):
        return ("statement", 1.0)
    if t.startswith(("请注意","请知悉")):
        return ("statement", 1.0)
    # generic English statement-y shapes
    if low.startswith("there is ") or low.startswith("there are ") \
       or re.search(r"\bit would be (great|helpful)\b", low) \
       or " looks " in low or " seems " in low:
        return ("statement", 1.0)
    if any(p in t for p in ("看起来","似乎","包含","是")):
        return ("statement", 0.2)

    # --- Request overrides (EN) ---
    if "would you mind" in low:
        return ("request", 1.0)
    NO_PLEASE_TASKS = ("send","share","upload","attach","forward","update","fix","add","remove","delete","create","write","draft")
    if re.search(r"^(could|can|would)\s+you\s+(" + "|".join(NO_PLEASE_TASKS) + r")\b", low):
        return ("request", 1.0)
    WITH_PLEASE_TASKS = NO_PLEASE_TASKS + ("explain","summarize","summarise","list","translate","review")
    if re.search(r"^(could|can|would)\s+you\s+please\s+(" + "|".join(WITH_PLEASE_TASKS) + r")\b", low):
        return ("request", 1.0)

    # --- Chinese special-case: '请问...' = question ---
    if t.startswith(("请问","請問")):
        return ("question", 1.0)

    # --- Requests (ZH), but NOT for '请问...' ---
    if ("帮我" in t or "幫我" in t) or (t.startswith(("请","請","麻烦","麻煩","劳驾","勞駕")) and not t.startswith(("请问","請問"))):
        # '请注意' / '请知悉' handled above as statement
        return ("request", 1.0)

    # --- Generic questions (EN/ZH) ---
    if t.endswith("?") or t.endswith("？") or ("吗？" in t) or ("嗎？" in t):
        return ("question", 1.0)

    # --- EN request fallback ---
    if low.startswith(("please ","kindly ")):
        if low.startswith("please note"):
            return ("statement", 1.0)
        return ("request", 1.0)

    # Imperative (weak) without 'please'
    ACTION_VERBS_EN = ("send","update","summarize","summarise","explain","translate","list","review","draft","write","create")
    if any(low.startswith(v + " ") for v in ACTION_VERBS_EN):
        return ("request", 0.2)

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
        # --- Fast-cue short-circuit (enable via env PRAGACT_FAST_CUES=1) ---
        use_fast = os.getenv("PRAGACT_FAST_CUES","0").lower() in ("1","true","yes","y","on")
        if use_fast:
            fc = fast_cue(text)
            if fc:
                key, conf = fc
                if conf >= 0.95:
                    idx = self.keys.index(key)
                    return idx, conf, {"source":"fast_cue"}

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
