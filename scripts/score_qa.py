#!/usr/bin/env python3
import json, argparse, re
def norm(s):
    s=s.lower().strip(); s=re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", s); s=re.sub(r"\s+"," ",s).strip(); return s
def f1(pred,gold):
    p=norm(pred).split(); g=norm(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    common=sum(min(p.count(w), g.count(w)) for w in set(p+g))
    if common==0: return 0.0
    prec=common/len(p); rec=common/len(g); return 2*prec*rec/(prec+rec)
ap=argparse.ArgumentParser()
ap.add_argument("--gold_jsonl", required=True); ap.add_argument("--pred_jsonl", required=True)
ap.add_argument("--mode", choices=["base","heavy"], required=True); ap.add_argument("--out_jsonl", required=True)
a=ap.parse_args()
M={"n":0,"em":0.0,"f1":0.0}
gold={json.loads(l)["id"]:json.loads(l) for l in open(a.gold_jsonl,"r",encoding="utf-8")}
out=open(a.out_jsonl,"w",encoding="utf-8")
for l in open(a.pred_jsonl,"r",encoding="utf-8"):
    r=json.loads(l); gid=r["id"]; pred=r.get(f"pred_{a.mode}",""); ans=gold[gid]["answer"]
    em=1.0 if norm(pred)==norm(ans) else 0.0; F=f1(pred, ans)
    r[f"score_{a.mode}"]={"em":em,"f1":F}; out.write(json.dumps(r, ensure_ascii=False)+"\n")
    M["n"]+=1; M["em"]+=em; M["f1"]+=F
out.close(); print(f"[QA {a.mode}] N={M['n']} EM={M['em']/max(M['n'],1):.3f} F1={M['f1']/max(M['n'],1):.3f}")
