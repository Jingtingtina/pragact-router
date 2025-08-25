
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
        return ["Prompt_StepByStep", "RAG_rerank"] if margin >= threshold else ["Prompt_StepByStep"]
    else:
        return ["Prompt_Concise", "RAG_definitional_boost"]
