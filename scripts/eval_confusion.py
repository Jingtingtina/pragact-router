import argparse, json, yaml
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from src.probe import PragActProbe
from src.llm_api import DummyLLM, OpenRouterLLM

def load_jsonl(p):
    for line in open(p, encoding="utf-8"):
        line=line.strip()
        if line: yield json.loads(line)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    conf = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    probe = PragActProbe(conf["templates_file"], conf["labels_file"], conf["use_logprobs"])
    llm = DummyLLM() if conf.get("model","DUMMY").upper()=="DUMMY" else OpenRouterLLM(model=conf["model"])
    if conf["use_logprobs"]: probe.calibrate_priors(llm)

    keys = probe.keys
    y_true=[]; y_pred=[]
    for ex in load_jsonl(args.data):
        idx, _, _ = probe.score(ex["text"], llm)
        y_pred.append(keys[idx]); y_true.append(ex["gold"])
    print(classification_report(y_true, y_pred, labels=keys, digits=3, zero_division=0))
    print(confusion_matrix(y_true, y_pred, labels=keys))
