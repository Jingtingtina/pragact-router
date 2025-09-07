#!/usr/bin/env python3
import csv, json, argparse
ap=argparse.ArgumentParser()
ap.add_argument("--in_tsv", required=True)
ap.add_argument("--out_jsonl", required=True)
ap.add_argument("--task", choices=["qa","instr"], required=True)
ap.add_argument("--lang", default="en")
a=ap.parse_args()
with open(a.in_tsv,"r",encoding="utf-8") as f, open(a.out_jsonl,"w",encoding="utf-8") as g:
    r=csv.DictReader(f, delimiter="\t")
    for row in r:
        if a.task=="qa":
            rec={"id":row["id"],"task":"qa","question":row["question"],"answer":row["answer"],"lang":a.lang,"text":row["question"]}
        else:
            rec={"id":row["id"],"task":"instr","instruction":row["instruction"],"input":row["input"],"reference":row["reference"],"lang":a.lang,"text":row["input"]}
        g.write(json.dumps(rec, ensure_ascii=False)+"\n")
print("[ok] wrote", a.out_jsonl)
