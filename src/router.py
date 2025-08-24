# -*- coding: utf-8 -*-
from typing import List, Set

QUESTION_KEYS: Set[str] = {"question"}
REQUEST_KEYS:  Set[str] = {"request"}

class PragActRouter:
    def __init__(self, budget_tokens: int = 120):
        self.budget = budget_tokens

    def select_actions_by_key(self, key: str, margin: float, threshold: float = 0.5) -> List[str]:
        """
        key: one of ["statement","question","request","promise","expressive","declaration"]
        """
        if key in QUESTION_KEYS or key in REQUEST_KEYS:
            return ["RAG_rerank", "Prompt_StepByStep"] if margin >= threshold else ["Prompt_StepByStep"]
        return ["Prompt_Concise", "RAG_definitional_boost"]
