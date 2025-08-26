import os, yaml, pathlib, pytest
from src.probe import PragActProbe

def mk_probe(cfg_path="configs/default_en.yaml"):
    conf = yaml.safe_load(pathlib.Path(cfg_path).read_text(encoding="utf-8"))
    return PragActProbe(conf["templates_file"], conf["labels_file"], use_logprobs=conf["use_logprobs"])

@pytest.mark.parametrize("cfg,text,expected", [
    ("configs/default_en.yaml", "Could you explain the experiment?", "question"),
    ("configs/default_zh.yaml", "请问现在系统可用吗？", "question"),
])
def test_fast_path_meta_is_set(cfg, text, expected):
    os.environ["PRAGACT_FAST_CUES"] = "1"
    probe = mk_probe(cfg)
    idx, m, meta = probe.score(text, llm=None)
    assert probe.keys[idx] == expected
    assert (meta or {}).get("source") == "fast_cue"
