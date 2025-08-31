import re
import os

import json, re
from typing import List, Tuple
from src.llm_api import LLMClient


_Q_EN  = re.compile(r'\?\s*$')
_Q_ZH  = re.compile(r'[？吗呢]\s*$')
_REQ_EN = re.compile(r'^(please|kindly|could you|can you|would you|pls)\b', re.I)
_REQ_ZH = re.compile(r'^(请|麻烦|能否|可以|烦请)')




def fast_cue(t: str):
    """
    Heuristic short-circuit for obvious Q/R plus some Statement shapes (EN/ZH).
    Returns: (label, confidence) or None
    """
    if not t:
        return None
    t = t.strip()
    low = t.lower()

    def any_in(s, seq):
        for x in seq:
            if x in s:
                return True
        return False

    # --- Request overrides (polite interrogatives) ---
    # EN: treat as request even with '?'
    if low.startswith("would you mind"):
        return ("request", 1.0)
    if low.startswith(("could you please", "would you please", "can you please")):
        return ("request", 1.0)
    # ZH: '麻烦…', or explicit '帮我/幫我'
    if t.startswith(("麻烦","麻煩")) or ("帮我" in t or "幫我" in t):
        return ("request", 1.0)

    # --- Strong QUESTIONS ---
    if t.startswith(("请问","請問")):
        return ("question", 1.0)
    if t.endswith("?") or t.endswith("？") or ("吗？" in t) or ("嗎？" in t):
        return ("question", 1.0)

    # --- Strong STATEMENTS (FYI/reminder/heads-up/note) ---
    if low.startswith(("please note","note that","fyi:","as a reminder","heads up:","heads-up:")):
        return ("statement", 1.0)
    # ZH: 请注意/请知悉 are FYI-type statements (not requests)
    if t.startswith(("请注意","請注意","请知悉","請知悉")):
        return ("statement", 1.0)
    # EN 'it would be great/helpful' -> statement
    if any_in(low, ("it would be great","it would be helpful","it'd be great","it'd be helpful")):
        return ("statement", 1.0)

    # --- Strong DECLARATIONS (EN/ZH) ---
    if "it is hereby declared" in low:
        return ("declaration", 1.0)
    if ("兹宣布" in t) or ("特此公告" in t):
        return ("declaration", 1.0)

    # --- Promises (tight) ---
    if any_in(low, ("i promise","i'll ","i will ")):
        return ("promise", 1.0)
    if any_in(t, ("我会","我將","我将","我保证","我保證","我承诺","我承諾")):
        return ("promise", 1.0)

    # --- Extra STATEMENT shapes (EN) with request guard ---
    requestish_en = any_in(low, ("please ","kindly ","would you ","would you mind","could you ","can you "))
    if not requestish_en:
        if (" includes " in low) or (" contains " in low):
            return ("statement", 1.0)
        if (" is available" in low) or (" are available" in low) or low.endswith(" available") or low.endswith(" available."):
            return ("statement", 1.0)
        if any_in(low, (" was updated"," were updated"," has been updated"," have been updated"," updated yesterday")):
            return ("statement", 1.0)
        if (" is important" in low) or (" are important" in low):
            return ("statement", 1.0)

    # --- Extra STATEMENT shapes (ZH) with request guard ---
    requestish_zh = any_in(t, ("请","請","麻烦","麻煩","能否","可以","?","？"))
    if not requestish_zh:
        if any_in(t, ("包含","包括")):
            return ("statement", 1.0)
        if ("可用" in t) or ("是可用" in t):
            return ("statement", 1.0)
        if any_in(t, ("已更新","昨天更新")):
            return ("statement", 1.0)
        if any_in(t, ("很重要","重要")):
            return ("statement", 1.0)

    # --- Requests (EN) ---
    if low.startswith(("please ","kindly ")):
        return ("request", 1.0)
    if "would you mind" in low:
        return ("request", 1.0)
    ACTION_VERBS_EN = ("send","update","summarize","summarise","explain","translate","list","review","draft","write","create","share","provide","upload","forward","include")
    if any(low.startswith(v + " ") for v in ACTION_VERBS_EN):
        return ("request", 0.2)
    if ("please " in low) and any((" " + v + " ") in low for v in ACTION_VERBS_EN):
        return ("request", 1.0)

    # --- Requests (ZH) ---
    if t.startswith(("请","請","劳驾","勞駕")):
        return ("request", 1.0)

    # --- Light STATEMENT cues ---
    if low.startswith(("it seems","it looks")) or (" it seems " in low) or (" it looks " in low) or ("looks like " in low) or ("seems like " in low):
        return ("statement", 0.2)

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
