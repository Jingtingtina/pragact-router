
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
