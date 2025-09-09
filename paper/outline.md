
## Error analysis (λ = 0.010)

**QA (n=20).** Heavy helps on a small number of items while base is already strong: we see 1 “helped” case and no “hurt/waste/miss.” PGBI matches heavy’s quality with only 53 tokens total, indicating it learns to spend compute sparingly where it matters. See Table: paper/tables/error_cases_qa.md.

**Instruction (n=20).** Heavy underperforms on average (ROUGE‑L 0.358 vs base 0.393), and we observe 3 “hurt” cases. PGBI avoids unnecessary spend (1 “wasted” heavy at λ=0.010) and misses only 2 potential improvements, keeping total heavy tokens at 0 while matching near‑base quality. See Table: paper/tables/error_cases_instr.md.

**Patterns.** For instruction following, heavy tends to over‑structure or drift in style, lowering ROUGE‑L; for QA, improvements are concentrated on a few items while base remains near‑optimal for the rest. This aligns with our goal: **spend when predicted value exceeds cost; otherwise, don’t.**
