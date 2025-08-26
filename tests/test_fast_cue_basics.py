import pytest
from src.probe import fast_cue

CASES = [
    # EN polite request vs question vs notice
    ("Could you please explain the experiment?", "request"),
    ("Could you explain the experiment?", "question"),
    ("Please explain the experiment.", "request"),
    ("Would you mind summarizing the document?", "request"),
    ("Can you tell me what the model does?", "question"),
    ("Please note that we meet at 3pm.", "statement"),

    # ZH polite request / question / notice / declaration
    ("请问你能解释一下实验吗？", "question"),
    ("麻烦你解释一下实验。", "request"),
    ("能否帮我总结一下文档？", "request"),
    ("请注意：我们三点开会。", "statement"),
    ("兹宣布比赛结束。", "declaration"),
]

@pytest.mark.parametrize("text,gold", CASES)
def test_fast_cue_cases(text, gold):
    out = fast_cue(text)
    assert out is not None, f"fast_cue returned None for: {text}"
    label, conf = out
    assert label == gold and conf >= 0.95, f"{text} -> {(label, conf)} expected {gold}"
