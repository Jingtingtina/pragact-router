#!/usr/bin/env python3
import re, matplotlib.pyplot as plt

def load(path, tag):
    xs,ys,ls=[],[],[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"[gate-{tag}]"):
                m=re.search(r"avg_quality=([0-9.]+).*total_tokens=([0-9.]+).*lambda=([0-9.]+)", line)
                if m:
                    q,t,l=map(float,m.groups()); xs.append(t); ys.append(q); ls.append(l)
    pts=sorted(zip(xs,ys,ls))
    return [p[0] for p in pts],[p[1] for p in pts],[p[2] for p in pts]

def plot_all(txt, out):
    tags=[("all","PGBI (all feats)","o"),
          ("noacts","No-Acts","s"),
          ("unc","Uncertainty-only","+"),
          ("acts","Acts-only","^")]
    plt.figure()
    for tag,lab,mark in tags:
        x,y,l = load(txt, tag)
        if x:
            plt.plot(x,y, marker=mark, label=lab)
            for xi,yi,li in zip(x,y,l):
                plt.text(xi,yi,f"{li:g}",fontsize=7)
    plt.xlabel("Tokens spent on heavy")
    plt.ylabel("Task quality")
    plt.title("Instr: VoC ablations")
    plt.grid(True); plt.legend()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print("[ok] saved", out)

if __name__=="__main__":
    plot_all("exp/reports/pcurve_models_instr.txt","paper/figs/pcurve_voc_ablate_instr.png")
