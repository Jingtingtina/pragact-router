#!/usr/bin/env python3
import json, argparse
def lcs(a,b):
    n,m=len(a),len(b); dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            dp[i+1][j+1]=dp[i][j]+1 if a[i]==b[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]
def rouge_l(pred, ref):
    A=pred.strip().split(); B=ref.strip().split()
    if not A or not B: return 0.0
    L=lcs(A,B); prec=L/len(A); rec=L/len(B)
    return (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
ap=argparse.ArgumentParser()
ap.add_argument("--gold_jsonl", required=True); ap.add_argument("--pred_jsonl", required=True)
ap.add_argument("--mode", choices=["base","heavy"], required=True); ap.add_argument("--out_jsonl", required=True)
a=ap.parse_args()
M={"n":0,"rougeL":0.0}
gold={json.loads(l)["id"]:json.loads(l) for l in open(a.gold_jsonl,"r",encoding="utf-8")}
out=open(a.out_jsonl,"w",encoding="utf-8")
for l in open(a.pred_jsonl,"r",encoding="utf-8"):
    r=json.loads(l); gid=r["id"]; pred=r.get(f"pred_{a.mode}",""); ref=gold[gid]["reference"]
    s=rouge_l(pred, ref); r[f"score_{a.mode}"]={"rougeL":s}; out.write(json.dumps(r, ensure_ascii=False)+"\n")
    M["n"]+=1; M["rougeL"]+=s
out.close(); print(f"[INSTR {a.mode}] N={M['n']} ROUGE-L={M['rougeL']/max(M['n'],1):.3f}")
