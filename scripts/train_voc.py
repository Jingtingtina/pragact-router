#!/usr/bin/env python3
import json, argparse, math
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.dummy import DummyClassifier

ACTS = ["question","request","statement","promise","expressive","declaration"]

def top2_margin(p):
    xs = sorted([p.get(a,0.0) for a in ACTS], reverse=True)
    return (xs[0]-xs[1]) if len(xs)>=2 else (xs[0] if xs else 0.0)

def entropy(p):
    arr = np.array([p.get(a,0.0) for a in ACTS], dtype=float)
    if arr.sum()<=0: return math.log(len(ACTS))
    arr = arr/arr.sum()
    arr = np.clip(arr,1e-9,1.0)
    return float(-(arr*np.log(arr)).sum())

# Feature order (12 dims) — keep fixed so gate_blend features match
FEAT_NAMES = [
  "p_q","p_req","p_stmt","margin","entropy","log_len",
  "lang_zh","cue_starts_wh","cue_ends_q","cue_imperative","cue_zh_q","cue_zh_please"
]

def build_vec(rec):
    probs = rec.get("probe_probs",{})
    cues  = rec.get("cue_bits",{})
    return [
        probs.get("question",0.0), probs.get("request",0.0), probs.get("statement",0.0),
        top2_margin(probs), entropy(probs), math.log(1+len(rec.get("text",""))),
        float(rec.get("lang","en")=="zh"),
        float(cues.get("starts_wh",0)), float(cues.get("ends_qmark",0)),
        float(cues.get("imperative",0)), float(cues.get("zh_qmark",0)), float(cues.get("zh_request_please",0))
    ]

def apply_feature_set(x, feature_set):
    x = np.array(x, dtype=float)
    # indices in FEAT_NAMES
    idx = {n:i for i,n in enumerate(FEAT_NAMES)}
    mask = np.ones_like(x, dtype=float)

    if feature_set == "no_acts":
        for n in ["p_q","p_req","p_stmt"]: mask[idx[n]] = 0.0
    elif feature_set == "uncertainty_only":
        keep = {"margin","entropy","log_len"}
        for i,n in enumerate(FEAT_NAMES):
            if n not in keep: mask[i] = 0.0
    elif feature_set == "acts_only":
        keep = {"p_q","p_req","p_stmt"}
        for i,n in enumerate(FEAT_NAMES):
            if n not in keep: mask[i] = 0.0
    else:
        pass  # 'all'
    return (x * mask).tolist()

def Xy(path, feature_set):
    X=[]; y=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            x = build_vec(r)
            x = apply_feature_set(x, feature_set)
            X.append(x); y.append(int(r.get("gain",0)))
    return np.array(X, dtype=float), np.array(y, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="models/voc.joblib")
    ap.add_argument("--feature_set", choices=["all","no_acts","uncertainty_only","acts_only"], default="all")
    args = ap.parse_args()

    Xtr,ytr = Xy(args.train, args.feature_set)
    if Xtr.shape[0] == 0:
        raise SystemExit("[error] No training rows in VoC data.")

    classes, counts = np.unique(ytr, return_counts=True)
    class_counts = {int(c): int(n) for c,n in zip(classes, counts)}
    npos = class_counts.get(1, 0); nneg = class_counts.get(0, 0)
    print(f"[info] set={args.feature_set}  size={len(ytr)}  positives={npos}  negatives={nneg}")

    if len(classes) < 2:
        clf = DummyClassifier(strategy="constant", constant=int(classes[0]))
        clf.fit(Xtr, ytr)
        print("[warn] single-class data; using DummyClassifier.")
    else:
        minc = min(npos, nneg)
        if minc >= 3:
            base = LogisticRegression(max_iter=300, class_weight="balanced")
            clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        elif minc >= 2:
            base = LogisticRegression(max_iter=300, class_weight="balanced")
            clf = CalibratedClassifierCV(base, cv=2, method="sigmoid")
        else:
            clf = LogisticRegression(max_iter=300, class_weight="balanced")
        clf.fit(Xtr, ytr)
        try:
            p = clf.predict_proba(Xtr)[:,1]
            print("[train]", "auc=", roc_auc_score(ytr, p), "brier=", brier_score_loss(ytr, p))
        except Exception:
            pass

    joblib.dump({"clf":clf, "feature_set":args.feature_set, "feat_names":FEAT_NAMES}, args.out)
    print(f"[ok] saved {args.out}")

if __name__ == "__main__":
    main()
