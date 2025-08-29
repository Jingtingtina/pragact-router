# --- PragAct fast-gate helper (actions) ---
import os
HEAVY_ACTIONS = {"RAG_rerank","RAG_definitional_boost","Prompt_StepByStep","Tool_Execute"}

def _pragact_filter_actions(ctx, actions):
    """
    ctx should contain: pred_label or pred/label, margin, meta (from probe.score)
    - Gate heavy actions unless (fast_cue AND Q/Request AND margin >= tau)
    """
    try:
        meta   = (ctx.get("meta") or {})
        label  = ctx.get("pred_label") or ctx.get("pred") or ctx.get("label")
        margin = float(ctx.get("margin") or 0.0)
    except Exception:
        return actions

    # Only consider fast path
    if meta.get("source") != "fast_cue":
        return actions

    tau = float(os.getenv("PRAGACT_FAST_TAU", "0.30"))
    is_qr = label in {"question", "request", "疑问", "请求"}

    # Allow heavy only if high-confidence Q/Req
    if is_qr and margin >= tau:
        return actions

    # Otherwise strip heavy actions
    try:
        return [a for a in actions if a not in HEAVY_ACTIONS]
    except Exception:
        return actions

class PragActRouter:
    def __init__(self, budget_tokens=120, weights=None):
        self.budget = budget_tokens
        self.weights = weights or {"bm25":0.6, "embed":0.3, "actmatch":0.1}

    def select_actions_by_key(self, key: str, margin: float, threshold: float = 0.25):
        """
        key in {"statement","question","request","promise","expressive","declaration"} (EN) or ZH equivalents.
        margin is the vote-entropy confidence (0..1). threshold ~0.25 ≈ 4/5 votes.
        """
        is_q_or_req = key in {"question","request","疑问","请求"}

        actions = []
        if is_q_or_req:
            
            if margin >= threshold:
                actions = ["Prompt_StepByStep", "RAG_rerank"]
            else:
               
                actions = ["Prompt_StepByStep"]
        else:
           
            actions = ["Prompt_Concise", "RAG_definitional_boost"]
        return actions

def get_actions(key: str, margin: float, threshold: float = 0.25):
    """
    Decide downstream actions from a speech-act label and confidence.

    key: one of {"statement","question","request","promise","expressive","declaration"}
         or ZH equivalents {"陈述","疑问","请求","承诺","表达","宣告"}.
    margin: confidence in [0,1]; higher = more confident.
    threshold: confidence gate for enabling heavier actions (e.g., RAG_rerank).
    """
    zh2en = {
        "陈述": "statement",
        "疑问": "question",
        "请求": "request",
        "承诺": "promise",
        "表达": "expressive",
        "宣告": "declaration",
    }
    k = zh2en.get(key, key).lower()
    if k in {"question", "request"}:
        __pragact_ret = ["Prompt_StepByStep", "RAG_rerank"] if margin >= threshold else ["Prompt_StepByStep"]
        __pragact_ret = _pragact_filter_actions(locals(), __pragact_ret)
        return __pragact_ret
    else:
        __ret_actions = (["Prompt_Concise", "RAG_definitional_boost"])
        __ret_actions = _pragact_filter_actions_locals(locals(), __ret_actions)
        return __ret_actions



# --- PragAct: Gate heavy actions using fast-cue confidence ---

def _pragact_filter_actions_locals(locals_dict, actions):
    """
    Post-filter the planned `actions` using the fast-cue metadata.
    - Only gate when meta.source == 'fast_cue' and label in {Q/Req (EN/ZH)}
    - If margin < tau, drop heavy actions; fallback to ['Prompt_Concise'] if empty.
    """
    meta   = (locals_dict.get("meta") or {})
    margin = locals_dict.get("margin")
    label  = locals_dict.get("pred_label") or locals_dict.get("label") or locals_dict.get("pred") or ""
    tau = float(os.getenv("PRAGACT_FAST_TAU", "0.30"))
    if meta.get("source") == "fast_cue" and label in {"question","request","疑问","请求"}:
        if margin is None or margin < tau:
            slim = [a for a in actions if a not in HEAVY_ACTIONS]
            return slim or ["Prompt_Concise"]
    return actions
