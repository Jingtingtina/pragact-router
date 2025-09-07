#!/usr/bin/env python3
import json, argparse, joblib, numpy as np, math
from train_probe import feats
ap=argparse.ArgumentParser()
ap.add_argument("--in_jsonl", required=True)
ap.add_argument("--out_jsonl", required=True)
ap.add_argument("--model", default="models/act_probe.joblib")
a=ap.parse_args()
obj=joblib.load(a.model); pipe=obj["pipe"]; ACTS=obj["acts"]
def softmax(z): z=np.array(z); z-=z.max(); e=np.exp(z); return (e/e.sum()).tolist()
with open(a.in_jsonl,"r",encoding="utf-8") as f, open(a.out_jsonl,"w",encoding="utf-8") as g:
    for line in f:
        r=json.loads(line); x=np.array([feats(r["text"], r.get("lang","en"))])
        if hasattr(pipe.named_steps["lr"],"decision_function"):
            z=pipe.named_steps["lr"].decision_function(pipe.named_steps["scaler"].transform(x)).ravel()
            p=softmax(z)
        else:
            p=pipe.predict_proba(x)[0].tolist()
        s=sorted(p, reverse=True); margin=float(s[0]-s[1]) if len(s)>=2 else float(s[0])
        r["probe_probs"]={ACTS[i]:float(p[i]) for i in range(len(ACTS))}
        r["probe_top"]=ACTS[int(np.argmax(p))]; r["probe_margin"]=margin
        g.write(json.dumps(r, ensure_ascii=False)+"\n")
print("[ok] wrote", a.out_jsonl)
