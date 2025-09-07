#!/usr/bin/env python3
import re, json, glob

def best_from_file(path, tag, key="avg_quality"):
    best=None
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.startswith(tag):
                m=re.search(r"avg_quality=([0-9.]+).*total_tokens=([0-9.]+).*=(\d*\.\d+|\d+)", line)
                if m:
                    q,t,thr = float(m.group(1)), float(m.group(2)), float(m.group(3))
                    if (best is None) or (q>best[0]) or (q==best[0] and t<best[1]):
                        best=(q,t,thr)
    return best

def base_heavy_avg(base_scored, heavy_scored, task):
    # read quickly to compute averages (already printed earlier, but we recompute)
    import json
    b=[json.loads(l) for l in open(base_scored,"r",encoding="utf-8")]
    h=[json.loads(l) for l in open(heavy_scored,"r",encoding="utf-8")]
    if task=="qa":
        qb=sum(x.get("score_base",{}).get("f1",0.0) for x in b)/max(len(b),1)
        qh=sum(x.get("score_heavy",{}).get("f1",0.0) for x in h)/max(len(h),1)
    else:
        qb=sum(x.get("score_base",{}).get("rougeL",0.0) for x in b)/max(len(b),1)
        qh=sum(x.get("score_heavy",{}).get("rougeL",0.0) for x in h)/max(len(h),1)
    tb=0.0
    th=sum(float(x.get("cost_heavy",0.0) or 0.0) for x in h)
    return (qb,tb),(qh,th)

def table(task):
    if task=="qa":
        base,heavy = base_heavy_avg("exp/logs/qa_en.base.scored.jsonl","exp/logs/qa_en.heavy.scored.jsonl","qa")
        pgbi = best_from_file("exp/reports/pcurve_pgbi_qa.txt","[gate]")
        marg = best_from_file("exp/reports/pcurve_margin_qa.txt","[gate-margin]")
    else:
        base,heavy = base_heavy_avg("exp/logs/instr_en.base.scored.jsonl","exp/logs/instr_en.heavy.scored.jsonl","instr")
        pgbi = best_from_file("exp/reports/pcurve_pgbi_instr.txt","[gate]")
        marg = best_from_file("exp/reports/pcurve_margin_instr.txt","[gate-margin]")

    print(f"\n### {task.upper()} summary")
    print("| Method | Quality | Tokens | Note |")
    print("|---|---:|---:|---|")
    print(f"| Base | {base[0]:.3f} | {base[1]:.0f} | — |")
    print(f"| Heavy | {heavy[0]:.3f} | {heavy[1]:.0f} | all heavy |")
    if pgbi: print(f"| PGBI (best λ={pgbi[2]:g}) | {pgbi[0]:.3f} | {pgbi[1]:.0f} | learned gate |")
    if marg: print(f"| Margin (best τ={marg[2]:g}) | {marg[0]:.3f} | {marg[1]:.0f} | uncertainty baseline |")

if __name__=="__main__":
    table("qa"); table("instr")
