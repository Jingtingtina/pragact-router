#!/usr/bin/env python3
import json, argparse, joblib, math, numpy as np

def load_map(p): 
    return {json.loads(l)["id"]:json.loads(l) for l in open(p,"r",encoding="utf-8")}

def features(rec):
    p=rec.get("probe_probs",{})
    def top2(ps):
        xs=sorted([ps.get(k,0.0) for k in ["question","request","statement","promise","expressive","declaration"]], reverse=True)
        return (xs[0]-xs[1]) if len(xs)>=2 else (xs[0] if xs else 0.0)
    def entropy(ps):
        arr=np.array([ps.get(k,0.0) for k in ["question","request","statement","promise","expressive","declaration"]], dtype=float)
        if arr.sum()<=0: return math.log(6)
        arr/=arr.sum(); arr=arr.clip(1e-9,1.0); return float(-(arr*np.log(arr)).sum())
    cues={"starts_wh":int(rec["text"].lower().startswith(("who","what","when","where","why","how","which"))),
          "ends_qmark":int(rec["text"].strip().endswith(("?","？"))),
          "imperative":int("please" in rec["text"].lower() or "请" in rec["text"]),
          "zh_qmark":int("？" in rec["text"]), "zh_request_please":int("请" in rec["text"])}
    return [p.get("question",0.0), p.get("request",0.0), p.get("statement",0.0),
            top2(p), entropy(p), math.log(1+len(rec.get("text",""))),
            float(rec.get("lang","en")=="zh"), float(cues["starts_wh"]), float(cues["ends_qmark"]),
            float(cues["imperative"]), float(cues["zh_qmark"]), float(cues["zh_request_please"])]

def pos_prob(clf, X):
    """Return P(y=1) robustly even if the classifier saw one class."""
    P = clf.predict_proba(X)
    # If two columns, try to map to class '1'
    if P.shape[1] == 2:
        try:
            classes = list(getattr(clf, "classes_", [0,1]))
            idx1 = classes.index(1) if 1 in classes else 1
        except Exception:
            idx1 = 1
        return float(P[0, idx1])
    # Single-column probabilities (e.g., DummyClassifier with one observed class)
    try:
        classes = list(getattr(clf, "classes_", []))
        if classes and classes[0] == 1:
            return float(P[0,0])  # the single prob corresponds to class 1
        else:
            return 0.0            # only class 0 present => P(y=1)=0
    except Exception:
        return 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa","instr"], required=True)
    ap.add_argument("--base_scored", required=True); ap.add_argument("--heavy_scored", required=True)
    ap.add_argument("--voc_model", required=True)
    ap.add_argument("--lambda_", type=float, default=0.002); ap.add_argument("--gain_scale", type=float, default=1.0)
    ap.add_argument("--out_jsonl", required=True)
    a=ap.parse_args()

    base=load_map(a.base_scored); heavy=load_map(a.heavy_scored); clf=joblib.load(a.voc_model)
    N=0; quality=0.0; tokens=0.0; out=open(a.out_jsonl,"w",encoding="utf-8")
    for k,b in base.items():
        h=heavy[k]; X=[features(b)]
        p = pos_prob(clf, X)
        cost = float(h.get("cost_heavy",0.0) or 0.0)
        use = (p*a.gain_scale) >= (a.lambda_*cost)
        rec=b.copy(); rec["chosen"]="heavy" if use else "base"
        if a.task=="qa":
            sb=b.get("score_base",{}).get("f1",0.0); sh=h.get("score_heavy",{}).get("f1",0.0)
            quality+=(sh if use else sb)
        else:
            sb=b.get("score_base",{}).get("rougeL",0.0); sh=h.get("score_heavy",{}).get("rougeL",0.0)
            quality+=(sh if use else sb)
        tokens+=(cost if use else 0.0)
        out.write(json.dumps(rec, ensure_ascii=False)+"\n"); N+=1
    out.close()
    print(f"[gate] N={N} avg_quality={quality/max(N,1):.3f} total_tokens={tokens:.1f} lambda={a.lambda_} gain_scale={a.gain_scale}")

if __name__=="__main__":
    main()
