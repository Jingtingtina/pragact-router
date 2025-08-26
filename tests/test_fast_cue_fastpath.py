import os, yaml, pathlib
from src.probe import PragActProbe

def mk_probe(cfg_path="configs/default_en.yaml"):
    conf = yaml.safe_load(pathlib.Path(cfg_path).read_text(encoding="utf-8"))
    return PragActProbe(conf["templates_file"], conf["labels_file"], use_logprobs=conf["use_logprobs"])

def test_fast_path_meta_is_set():
    os.environ["PRAGACT_FAST_CUES"] = "1"
    probe = mk_probe()
    idx, m, meta = probe.score("Could you explain the experiment?", llm=None)
    assert probe.keys[idx] == "question"
    assert (meta or {}).get("source") == "fast_cue"
