# -*- coding: utf-8 -*-
import argparse, yaml, json
from src.probe import PragActProbe
from src.router import PragActRouter
from src.llm_api import DummyLLM

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_en.yaml")
    ap.add_argument("--text", default="请将以下文章分点概述。")
    args = ap.parse_args()

    conf = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    tpl = json.load(open(conf["templates_file"], "r", encoding="utf-8"))

    probe = PragActProbe(conf["templates_file"], conf["labels_file"], conf["use_logprobs"])
    llm = DummyLLM()
    if conf["use_logprobs"]:
        probe.calibrate_priors(llm)

    idx, margin, _ = probe.score(args.text, llm)
    options_text = tpl.get("options_en") or tpl.get("options_zh")
    label_text = options_text[idx]
    key = probe.keys[idx]

    router = PragActRouter(budget_tokens=conf["router"]["budget_tokens"])
    actions = router.select_actions_by_key(key, margin, threshold=conf["probe"]["threshold"])

    print(f"TEXT: {args.text}")
    print(f"ACT : {label_text} (key={key}) | margin={margin:.3f}")
    print(f"ACTIONS -> {actions}")
