#!/usr/bin/env python3
import csv, json, pathlib, sys
allowed = {"statement","question","request","promise","expressive","declaration"}
def convert(tsv_path, out_path, lang):
    rows=[]
    with open(tsv_path,"r",encoding="utf-8") as f:
        r=csv.DictReader(f, delimiter="\t")
        for i,row in enumerate(r,1):
            if not row.get("text"): continue
            act=(row.get("gold_act","").strip().lower())
            if act not in allowed: sys.exit(f"[error] {tsv_path} line {i}: '{act}' not in {sorted(allowed)}")
            rows.append({"id":row.get("id") or f"{out_path.stem}-{i:04d}","text":row["text"],"lang":row.get("lang",lang),"gold_act":act})
    with open(out_path,"w",encoding="utf-8") as g:
        for r in rows: g.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"[ok] {out_path} {len(rows)}")
convert("data/acts/label_en.tsv","data/acts/heldout_en.jsonl","en")
convert("data/acts/label_zh.tsv","data/acts/heldout_zh.jsonl","zh")
