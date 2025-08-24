# -*- coding: utf-8 -*-
import argparse, yaml, json
from src.probe import PragActProbe
from src.llm_api import DummyLLM

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_en.yaml")
    ap.add_argument("--text", default="Could you please open the window?")
    args = ap.parse_args()

    conf = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    probe = PragActProbe(conf["templates_file"], conf["labels_file"], conf["use_logprobs"])
    llm = DummyLLM()
    if conf["use_logprobs"]:
        probe.calibrate_priors(llm)

    idx, margin, scores = probe.score(args.text, llm)
    tpl = json.load(open(conf["templates_file"], "r", encoding="utf-8"))
    options_text = tpl.get("options_en") or tpl.get("options_zh")
    label_text = options_text[idx]
    key = probe.keys[idx]

    print(f"TEXT: {args.text}")
    print(f"PRED: {label_text} (key={key}) | margin={margin:.3f}")
    if scores and max(scores) != 0:
        print("scores:", [round(s, 3) for s in scores])
