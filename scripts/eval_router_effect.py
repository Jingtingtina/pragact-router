import os, json, argparse, yaml
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

from src.probe import PragActProbe
from src.llm_api import DummyLLM, OpenRouterLLM

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)


try:
    from src.router import decide_actions
except Exception:
    def decide_actions(key, margin, threshold=0.25):
        is_q_or_req = key in {"question","request","疑问","请求"}
        if is_q_or_req:
            return ["Prompt_StepByStep", "RAG_rerank"] if margin >= threshold else ["Prompt_StepByStep"]
        else:
            return ["Prompt_Concise", "RAG_definitional_boost"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--fast", action="store_true", help="Enable PRAGACT_FAST_CUES=1")
    args = ap.parse_args()

    if args.fast:
        os.environ["PRAGACT_FAST_CUES"] = "1"
    else:
        os.environ.pop("PRAGACT_FAST_CUES", None)

    conf = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    probe = PragActProbe(conf["templates_file"], conf["labels_file"], conf["use_logprobs"])
    model_name = conf.get("model","DUMMY")
    llm = DummyLLM() if model_name.upper()=="DUMMY" else OpenRouterLLM(model=model_name)
    if conf.get("use_logprobs"):
        probe.calibrate_priors(llm)

    keys = probe.keys
    threshold = conf.get("probe",{}).get("threshold", 0.25)

    y_true, y_pred = [], []
    actions_counter = Counter()
    gating_counter = Counter()

    for ex in load_jsonl(args.data):
        idx, margin, _ = probe.score(ex["text"], llm)
        pred_key = keys[idx]
        y_true.append(ex["gold"])
        y_pred.append(pred_key)
        acts = decide_actions(pred_key, margin, threshold)
        actions_counter.update(acts)
        if pred_key in {"question","request","疑问","请求"}:
            gating_counter.update(["high"] if margin >= threshold else ["low"])

    print("\n=== Router effect report ===")
    print(f"FAST_CUES={'ON' if args.fast else 'OFF'} | data={args.data}")
    print("\n-- Classification --")
    print(classification_report(y_true, y_pred, labels=keys, digits=3, zero_division=0))
    print(confusion_matrix(y_true, y_pred, labels=keys))
    print("\n-- Router actions (counts) --")
    for k,v in actions_counter.most_common():
        print(f"{k:26s} {v}")
    if gating_counter:
        tot = sum(gating_counter.values())
        hi = gating_counter.get("high",0)
        lo = gating_counter.get("low",0)
        print("\n-- Gating on Q/Request --")
        print(f"high >= τ: {hi}  | low < τ: {lo}  | total Q/Req: {tot}  | high rate: {hi/(tot or 1):.3f}")

if __name__ == "__main__":
    main()
