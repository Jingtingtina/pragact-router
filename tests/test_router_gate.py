import os
from src.router import _pragact_filter_actions_locals, HEAVY_ACTIONS

def test_gate_filters_low_confidence_qr():
    os.environ["PRAGACT_FAST_TAU"] = "0.30"
    actions = ["Prompt_StepByStep", "RAG_rerank", "Prompt_Concise"]
    out = _pragact_filter_actions_locals(
        {"pred_label": "question", "margin": 0.10, "meta": {"source": "fast_cue"}},
        actions
    )
    assert all(a not in HEAVY_ACTIONS for a in out)
    assert "Prompt_Concise" in out

def test_gate_keeps_high_confidence_qr():
    os.environ["PRAGACT_FAST_TAU"] = "0.30"
    actions = ["Prompt_StepByStep", "RAG_rerank", "Prompt_Concise"]
    out = _pragact_filter_actions_locals(
        {"pred_label": "request", "margin": 0.40, "meta": {"source": "fast_cue"}},
        actions
    )
    assert out == actions  # no filtering

def test_gate_not_applied_non_qr():
    os.environ["PRAGACT_FAST_TAU"] = "0.30"
    actions = ["Prompt_StepByStep", "Prompt_Concise"]
    out = _pragact_filter_actions_locals(
        {"pred_label": "statement", "margin": 0.05, "meta": {"source": "fast_cue"}},
        actions
    )
    assert out == actions  # non Q/R unaffected

def test_gate_not_applied_without_fast_meta():
    os.environ["PRAGACT_FAST_TAU"] = "0.30"
    actions = ["Prompt_StepByStep", "RAG_rerank"]
    out = _pragact_filter_actions_locals(
        {"pred_label": "question", "margin": 0.01, "meta": {}},
        actions
    )
    assert out == actions  # no fast_cue meta → no gating

def test_gate_fallback_concise_if_all_heavy_removed():
    os.environ["PRAGACT_FAST_TAU"] = "0.30"
    only_heavy = list(HEAVY_ACTIONS)
    out = _pragact_filter_actions_locals(
        {"pred_label": "request", "margin": 0.05, "meta": {"source": "fast_cue"}},
        only_heavy
    )
    assert out == ["Prompt_Concise"]
