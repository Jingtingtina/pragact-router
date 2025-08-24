
import argparse, json, yaml
from src.probe import PragActProbe
from src.llm_api import DummyLLM, OpenRouterLLM

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    conf = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    probe = PragActProbe(conf["templates_file"], conf["labels_file"], conf["use_logprobs"])
    model_name = conf.get("model","DUMMY")
    llm = DummyLLM() if model_name.upper()=="DUMMY" else OpenRouterLLM(model=model_name)
    if conf["use_logprobs"]:
        probe.calibrate_priors(llm)

    keys = probe.keys
    correct = 0; total = 0
    for ex in load_jsonl(args.data):
        idx, margin, _ = probe.score(ex["text"], llm)
        pred = keys[idx]
        ok = int(pred == ex["gold"])
        correct += ok; total += 1
        print(f"[{bool(ok)}] {ex['text']} -> {pred} (m={margin:.3f}) | gold={ex['gold']}")
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.3f}")
