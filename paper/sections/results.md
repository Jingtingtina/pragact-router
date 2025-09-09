# Results (Pilot)

**QA (n=20).** Heavy improves average F1 over base (**0.950 vs 0.929**). With \(\lambda=0.010\), **PGBI** reaches the same quality (**0.950**) while spending only **53 tokens** in total, indicating selective investment. The uncertainty baseline stays at base quality while spending zero tokens at the thresholds we swept.

**Instruction (n=20).** Heavy underperforms base (**ROUGE‑L 0.358 vs 0.393**). For \(\lambda\ge 0.010\), **PGBI** shuts off heavy and matches near‑base quality with **0 extra tokens**, avoiding unnecessary compute when the heavier prompt tends to over‑structure or drift in style.

**Ablations.** On instruction following, uncertainty‑only and no‑acts variants behave similarly to the full feature set on this small pilot, while acts‑only is weaker—suggesting uncertainty features carry much of the signal at this size. We expect acts to matter more with additional data (e.g., indirect requests).

**Error analysis.** At \(\lambda=0.010\): QA shows **1 helped** item and no regressions; instruction has **2 helped**, **3 hurt**, **1 wasted**, **2 missed** (see `paper/tables/error_cases_qa.md` and `paper/tables/error_cases_instr.md`). This matches our qualitative goal: spend where predicted value exceeds cost, otherwise do not.
