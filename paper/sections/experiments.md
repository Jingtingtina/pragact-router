# Experimental Setup

**Tasks.** We evaluate two small English tasks curated for this pilot:
- **QA (n=20):** short factual questions with a single reference span.
- **Instruction following (n=20):** short transformations (summarize, rewrite for style, clarify, list/bullets, etc.).

All items are short (1–2 sentences) and manually referenced to keep annotation overhead low.

**Systems.** We compare a single model under two prompting modes:
- **Base:** minimal prompt; produce the answer/transformation only.
- **Heavy:** more compute (deliberation/structure) prompt intended to improve complex items.

**Gate (PGBI).** We train a value‑of‑computation (VoC) classifier on light features per input:
act posteriors (question/request/statement), top‑2 margin, entropy, length, and simple cue bits (WH‑start, question mark, imperative, Chinese markers). The gate invokes heavy iff
\[
\widehat{\Delta U}(h\!\mid\!x) \ge \lambda \, c(h),
\]
where \(c(h)\) is the heavy token cost and \(\lambda\) trades quality for budget.

**Baselines.** (i) **Base** (no heavy), (ii) **Heavy** (always heavy), (iii) **Uncertainty (margin) gate** that routes by top‑2 margin only, and (iv) **Ablations** of VoC features (acts‑only, uncertainty‑only, no‑acts).

**Metrics.** QA: **EM/F1**. Instruction: **ROUGE‑L** against the reference text. **Budget** is the sum of output tokens produced when heavy is invoked (reported by the API). We sweep \(\lambda\in\{0, 0.001, 0.002, 0.005, 0.010, 0.020\}\) and for the margin baseline sweep \(\tau\in\{0.05,0.10,0.15,0.20,0.30\}\).

**Implementation.** Python 3.9; scikit‑learn logistic regression with calibration where class counts permit; all scripts run from terminal and re‑generate metrics, curves, and tables.
