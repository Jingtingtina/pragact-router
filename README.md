

Set `PRAGACT_FAST_CUES=1` to short-circuit likely **question/request** cases using cheap regex cues (EN/ZH).  
This only affects the **margin** (confidence) and may change labels when cues trigger.

Examples:
- `PRAGACT_FAST_CUES=1 python -m scripts.run_probe --config configs/default_en.yaml --text "Please summarize the document."`
- `PRAGACT_FAST_CUES=1 python -m scripts.run_probe --config configs/default_zh.yaml --text "请总结这段文字。"`





## Fast‑Cue Gate (v0.2.x)

The fast‑cue gate short‑circuits obvious **question/request** inputs using cheap EN/ZH patterns.

**Runtime controls**
- `PRAGACT_FAST_CUES` (default `1`): enable the fast gate
- `PRAGACT_FAST_TAU`: confidence threshold for allowing heavy actions  
  - Recommended: **EN `0.30`**, **ZH `0.20`** (based on our sweeps in `results/logs/`)

**Convenience runner**
```bash
# EN
export PRAGACT_FAST_CUES=1
export PRAGACT_FAST_TAU=0.30
./scripts/run_with_gate.sh configs/default_en.yaml data/en/probe_150.jsonl

# ZH
export PRAGACT_FAST_TAU=0.20
./scripts/run_with_gate.sh configs/default_zh.yaml data/zh/probe_150.jsonl

```
