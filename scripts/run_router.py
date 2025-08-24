import argparse, yaml, json
from src.probe import PragActProbe
from src.router import PragActRouter
from src.llm_api import DummyLLM, OpenRouterLLM

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_en.yaml")
    ap.add_argument("--text", default="Summarize the following document in bullet points.")
    args = ap.parse_args()

    conf = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    model_name = conf.get("model", "DUMMY")
    llm = DummyLLM() if model_name.upper()=="DUMMY" else OpenRouterLLM(model=model_name)

    probe = PragActProbe(conf["templates_file"], conf["labels_file"], conf["use_logprobs"])
    if conf["use_logprobs"]:
        probe.calibrate_priors(llm)

    idx, margin, _ = probe.score(args.text, llm)
    tpl = json.load(open(conf["templates_file"], "r", encoding="utf-8"))
    options_text = tpl.get("options_en") or tpl.get("options_zh")
    label_text = options_text[idx]
    key = probe.keys[idx]

    router = PragActRouter(budget_tokens=conf["router"]["budget_tokens"])
    actions = router.select_actions_by_key(key, margin, threshold=conf["probe"]["threshold"])

    print(f"TEXT: {args.text}")
    print(f"ACT : {label_text} (key={key}) | margin={margin:.3f}")
    print(f"ACTIONS -> {actions}")
