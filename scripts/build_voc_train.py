#!/usr/bin/env python3
import json, argparse
ap=argparse.ArgumentParser()
ap.add_argument("--base_scored", required=True)
ap.add_argument("--heavy_scored", required=True)
ap.add_argument("--out_train", required=True)
ap.add_argument("--task", choices=["qa","instr"], required=True)
ap.add_argument("--delta_threshold", type=float, default=0.0)
a=ap.parse_args()
base={json.loads(l)["id"]:json.loads(l) for l in open(a.base_scored,"r",encoding="utf-8")}
heavy={json.loads(l)["id"]:json.loads(l) for l in open(a.heavy_scored,"r",encoding="utf-8")}
out=open(a.out_train,"w",encoding="utf-8")
for k,b in base.items():
    h=heavy[k]
    if a.task=="qa":
        sb=b.get("score_base",{}).get("f1",0.0); sh=h.get("score_heavy",{}).get("f1",0.0)
    else:
        sb=b.get("score_base",{}).get("rougeL",0.0); sh=h.get("score_heavy",{}).get("rougeL",0.0)
    delta=float(sh-sb); gain=int(delta>a.delta_threshold)
    rec={"id":k,"text":b["text"],"lang":b.get("lang","en"),"probe_probs":b.get("probe_probs",{}),
         "cue_bits":{"starts_wh":int(b["text"].lower().startswith(("who","what","when","where","why","how","which"))),
                     "ends_qmark":int(b["text"].strip().endswith(("?","？"))),
                     "imperative":int("please" in b["text"].lower() or "请" in b["text"]),
                     "zh_qmark":int("？" in b["text"]), "zh_request_please":int("请" in b["text"])},
         "cost": float(h.get("cost_heavy",0.0) or 0.0), "delta_score":delta, "gain":gain}
    out.write(json.dumps(rec, ensure_ascii=False)+"\n")
out.close(); print("[ok] wrote", a.out_train)
