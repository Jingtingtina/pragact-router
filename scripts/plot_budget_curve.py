#!/usr/bin/env python3
import re, matplotlib.pyplot as plt

def load_points(path):
    T,Q,L=[],[],[]
    for line in open(path,"r",encoding="utf-8"):
        if line.startswith("[gate]"):
            m = re.search(r"avg_quality=([0-9.]+).*total_tokens=([0-9.]+).*lambda=([0-9.]+)", line)
            if m:
                q,t,l = map(float, m.groups())
                T.append(t); Q.append(q); L.append(l)
    pts = sorted(zip(T,Q,L))
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]

def plot_one(txt_path, png_path, title):
    x,y,l = load_points(txt_path)
    plt.figure()
    plt.plot(x, y, marker="o")
    for xi, yi, li in zip(x,y,l):
        plt.text(xi, yi, f"{li:g}", fontsize=8)
    plt.xlabel("Tokens spent on heavy")
    plt.ylabel("Task quality")
    plt.title(title)
    plt.grid(True)
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    print("[ok] saved", png_path)

if __name__=="__main__":
    plot_one("exp/reports/pcurve_pgbi_qa.txt",   "paper/figs/pcurve_pgbi_qa.png",   "PGBI QA: Quality vs Budget")
    plot_one("exp/reports/pcurve_pgbi_instr.txt","paper/figs/pcurve_pgbi_instr.png","PGBI Instr: Quality vs Budget")
