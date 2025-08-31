import pytest
from src.probe import fast_cue

CASES = [
    # EN
    ("Could you explain the experiment?", "question"),
    ("Would you mind summarizing the document?", "request"),
    ("Please update the dataset.", "request"),
    ("Please note that the meeting is at 3pm.", "statement"),
    ("It would be great to have the file.", "statement"),
    ("I'll send the report after lunch.", "promise"),
    # ZH
    ("请问现在系统可用吗？", "question"),
    ("能否帮我总结一下文档？", "request"),
    ("请知悉：下午3点开会。", "statement"),
    ("我会尽快给你反馈。", "promise"),
    ("兹宣布比赛结束。", "declaration"),
]

@pytest.mark.parametrize("text,gold", CASES)
def test_fast_cue_labels(text, gold):
    out = fast_cue(text)
    assert out is not None, f"fast_cue returned None for: {text}"
    label, conf = out
    # Your fast_cue returns 1.0 for clear hits and 0.2 for lighter ones,
    # so accept any confident-enough signal (>= 0.2).
    assert label == gold and conf >= 0.2, f"{text} -> {out}, expected {gold}"
