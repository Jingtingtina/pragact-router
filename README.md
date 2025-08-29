

Set `PRAGACT_FAST_CUES=1` to short-circuit likely **question/request** cases using cheap regex cues (EN/ZH).  
This only affects the **margin** (confidence) and may change labels when cues trigger.

Examples:
- `PRAGACT_FAST_CUES=1 python -m scripts.run_probe --config configs/default_en.yaml --text "Please summarize the document."`
- `PRAGACT_FAST_CUES=1 python -m scripts.run_probe --config configs/default_zh.yaml --text "请总结这段文字。"`




We expose a lightweight “fast-cue” gate to short-circuit obvious **questions/requests**.

**Runtime controls**
- `PRAGACT_FAST_CUES` (default `1`): enable fast gate
- `PRAGACT_FAST_TAU`  (EN rec: `0.30`, ZH rec: `0.20`): confidence threshold for short-circuit

**Convenience runner**
```bash

export PRAGACT_FAST_CUES=1
export PRAGACT_FAST_TAU=0.30
./scripts/run_with_gate.sh configs/default_en.yaml data/en/probe_150.jsonl


export PRAGACT_FAST_TAU=0.20
./scripts/run_with_gate.sh configs/default_zh.yaml data/zh/probe_150.jsonl

```
