

Set `PRAGACT_FAST_CUES=1` to short-circuit likely **question/request** cases using cheap regex cues (EN/ZH).  
This only affects the **margin** (confidence) and may change labels when cues trigger.

Examples:
- `PRAGACT_FAST_CUES=1 python -m scripts.run_probe --config configs/default_en.yaml --text "Please summarize the document."`
- `PRAGACT_FAST_CUES=1 python -m scripts.run_probe --config configs/default_zh.yaml --text "请总结这段文字。"`
