#!/usr/bin/env python3
import json, argparse

def load_map(p): 
    return {json.loads(l)["id"]:json.loads(l) for l in open(p,"r",encoding="utf-8")}

def ensure_margin(rec):
    if "probe_margin" in rec:
        return float(rec["probe_margin"])
    # reconstruct from probs if needed
    p = rec.get("probe_probs", {})
    vals = sorted([p.get(k,0.0) for k in ["question","request","statement","promise","expressive","declaration"]], reverse=True)
    return float(vals[0]-vals[1]) if len(vals)>=2 else float(vals[0] if vals else 0.0)

ap=argparse.ArgumentParser()
ap.add_argument("--task", choices=["qa","instr"], required=True)
ap.add_argument("--base_scored", required=True)
ap.add_argument("--heavy_scored", required=True)
ap.add_argument("--tau", type=float, required=True, help="use heavy iff margin < tau")
ap.add_argument("--out_jsonl", required=True)
a=ap.parse_args()

base=load_map(a.base_scored); heavy=load_map(a.heavy_scored)
N=0; quality=0.0; tokens=0.0
out=open(a.out_jsonl,"w",encoding="utf-8")
for k,b in base.items():
    h = heavy[k]
    m = ensure_margin(b)
    use = (m < a.tau)
    rec = b.copy(); rec["chosen"] = "heavy" if use else "base"
    if a.task=="qa":
        sb=b.get("score_base",{}).get("f1",0.0); sh=h.get("score_heavy",{}).get("f1",0.0)
        quality += (sh if use else sb)
    else:
        sb=b.get("score_base",{}).get("rougeL",0.0); sh=h.get("score_heavy",{}).get("rougeL",0.0)
        quality += (sh if use else sb)
    tokens += float(h.get("cost_heavy",0.0) or 0.0) if use else 0.0
    out.write(json.dumps(rec, ensure_ascii=False)+"\n"); N+=1
out.close()
print(f"[gate-margin] N={N} avg_quality={quality/max(N,1):.3f} total_tokens={tokens:.1f} tau={a.tau}")
