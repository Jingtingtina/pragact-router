import pytest
from src.probe import fast_cue

CASES = [
    ("FYI: the meeting is at 3pm.", "statement", 1.0),
    ("As a reminder, the deadline is Friday.", "statement", 1.0),
    ("Heads up: we will rotate keys tomorrow.", "statement", 1.0),
    ("It seems the server is down.", "statement", 0.2),
    ("It looks like there is a mismatch.", "statement", 0.2),
]

@pytest.mark.parametrize("text,gold,min_conf", CASES)
def test_extra_statement_cues(text, gold, min_conf):
    out = fast_cue(text)
    assert out is not None, f"fast_cue returned None for: {text}"
    label, conf = out
    assert label == gold and conf >= min_conf, f"{text} -> {(label,conf)} expected {gold} >= {min_conf}"
