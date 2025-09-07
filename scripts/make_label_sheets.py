#!/usr/bin/env python3
import csv, pathlib
root = pathlib.Path("data/acts"); root.mkdir(parents=True, exist_ok=True)
for lg in ["en","zh"]:
    p = root/f"label_{lg}.tsv"
    if p.exists(): print(f"[ok] {p} exists"); continue
    with p.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(["id","text","lang","gold_act"])
    print(f"[new] {p}")
print("Fill gold_act with one of: statement,question,request,promise,expressive,declaration")
