#!/usr/bin/env python3
import re, matplotlib.pyplot as plt

def load_points(path, tag):
    xs,ys,labels=[],[],[]
    for line in open(path,"r",encoding="utf-8"):
        if tag=="pgbi" and line.startswith("[gate]"):
            m = re.search(r"avg_quality=([0-9.]+).*total_tokens=([0-9.]+).*lambda=([0-9.]+)", line)
            if m: q,t,l = map(float, m.groups()); xs.append(t); ys.append(q); labels.append(l)
        if tag=="margin" and line.startswith("[gate-margin]"):
            m = re.search(r"avg_quality=([0-9.]+).*total_tokens=([0-9.]+).*tau=([0-9.]+)", line)
            if m: q,t,l = map(float, m.groups()); xs.append(t); ys.append(q); labels.append(l)
    pts = sorted(zip(xs,ys,labels))
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]

def plot_both(pgbi_txt, margin_txt, png_path, title):
    x1,y1,l1 = load_points(pgbi_txt, "pgbi")
    x2,y2,l2 = load_points(margin_txt, "margin")
    plt.figure()
    if x1: plt.plot(x1,y1, marker="o", label="PGBI (VoC)")
    if x2: plt.plot(x2,y2, marker="s", label="Uncertainty (margin)")
    for xi, yi, li in zip(x1,y1,l1): plt.text(xi, yi, f"{li:g}", fontsize=8)
    for xi, yi, li in zip(x2,y2,l2): plt.text(xi, yi, f"{li:g}", fontsize=8)
    plt.xlabel("Tokens spent on heavy")
    plt.ylabel("Task quality")
    plt.title(title)
    plt.grid(True); plt.legend()
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    print("[ok] saved", png_path)

if __name__=="__main__":
    plot_both("exp/reports/pcurve_pgbi_qa.txt",   "exp/reports/pcurve_margin_qa.txt",   "paper/figs/pcurve_compare_qa.png",   "QA: PGBI vs Uncertainty")
    plot_both("exp/reports/pcurve_pgbi_instr.txt","exp/reports/pcurve_margin_instr.txt","paper/figs/pcurve_compare_instr.png","Instr: PGBI vs Uncertainty")
