#!/usr/bin/env python3
import json, pathlib, math, argparse
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

ACTS = ["statement","question","request","promise","expressive","declaration"]
WH = ("who","what","when","where","why","how","which","whom","whose")
REQ = ("please","kindly","could you","would you","let me","let us","do ","make ","give ","tell ")

def feats(t: str, lang: str):
    t = t.strip()
    low = t.lower()
    L = len(t)
    return [
        math.log1p(L),
        float(t.endswith("?") or t.endswith("？")),
        float(any(low.startswith(w+" ") for w in WH)),
        float(("请" in t) or any(p in low for p in REQ)),
        float("？" in t),
        float("!" in t or "！" in t),
        float("…" in t or "..." in t),
        float(lang == "zh"),
        float(lang == "en"),
    ]

def _load_jsonl(path):
    return [json.loads(x) for x in open(path, "r", encoding="utf-8") if x.strip()]

def _toXY(path):
    data = _load_jsonl(path)
    X = [feats(d["text"], d.get("lang","en")) for d in data]
    y = [ACTS.index(d["gold_act"]) for d in data]
    return np.array(X), np.array(y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", default="data/acts/heldout_en.jsonl")
    ap.add_argument("--zh", default="data/acts/heldout_zh.jsonl")
    ap.add_argument("--out", default="models/act_probe.joblib")
    ap.add_argument("--report", default="exp/reports/probe_report.txt")
    args = ap.parse_args()

    Xen,yen = _toXY(args.en) if pathlib.Path(args.en).exists() else (np.empty((0,9)), np.empty((0,)))
    Xzh,yzh = _toXY(args.zh) if pathlib.Path(args.zh).exists() else (np.empty((0,9)), np.empty((0,)))
    X = np.vstack([Xen,Xzh]) if Xen.size and Xzh.size else (Xen if Xen.size else Xzh)
    y = np.concatenate([yen,yzh]) if yen.size and yzh.size else (yen if yen.size else yzh)
    if not X.size:
        raise SystemExit("[error] No labeled data. Run merge_and_check + fill TSVs.")

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=300, multi_class="auto"))])
    pipe.fit(X, y)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipe": pipe, "acts": ACTS}, args.out)

    pred = pipe.predict(X)
    rep = classification_report(y, pred, target_names=ACTS, digits=3)
    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    open(args.report, "w").write(rep)
    print("[ok] saved", args.out)
    print(rep)

if __name__ == "__main__":
    main()
