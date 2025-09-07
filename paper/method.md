# Method: Pragmatics-Guided Budgeted Inference (PGBI)

**Goal.** For each input \(x\), decide whether to invoke a more expensive “heavy” action \(h\) (e.g., a chain-of-thought prompt or tool call) under a budget.

**Probe.** A light illocutionary-act probe yields a distribution over acts \(p_\theta(a\!\mid\!x)\) and a margin \(m(x)\) between the top-2 acts.

**Gain estimator.** We learn \(\widehat{\Delta U}(h\!\mid\!x)\approx \mathbb{E}[\text{quality}_{h}-\text{quality}_{\text{base}}\mid x]\) using a small feature vector:
\[
\phi(x)=\big[\,p_\theta(\text{question}\!\mid\!x),\ p_\theta(\text{request}\!\mid\!x),\ p_\theta(\text{statement}\!\mid\!x),\ m(x),\ H(p_\theta),\ \log(1+\lvert x\rvert),\ \text{lang/cues}\,\big].
\]

**Decision rule.** Let \(c(h)\) be the heavy action’s cost (e.g., tokens). For a tradeoff \(\lambda>0\),
\[
\text{invoke }h\ \Longleftrightarrow\ \widehat{\Delta U}(h\!\mid\!x)\ \ge\ \lambda\, c(h).
\]
This reduces to thresholding the instance-wise expected gain; with a single fixed-cost action, greedy thresholding is optimal for any average-budget constraint.

**Why acts help.** Acts (question/request/...) correlate with whether extra computation tends to help. Conditioning the gate on act features lets the policy spend tokens where they yield value, and skip them when they don’t.
