# Error cases (instr)

- N=20; threshold delta=0.02
- Helped: 2  | Hurt: 3  | Wasted (FP): 1  | Missed (FN): 2

## Heavy helped (sh>sb)

| id | Δ | chosen | gold | base | heavy |
|---|---:|---|---|---|---|
| ins-0013 | +0.200 | base | • Goals\n• Timeline\n• Risks | - Goals | - Goals, - Timeline, - Risks |
| ins-0005 | +0.046 | base | This section is verbose and repetitive. | Simplify language for clarity and avoid repetition. | Simplify and eliminate repetition. |

## Heavy hurt (sh<sb)

| id | Δ | chosen | gold | base | heavy |
|---|---:|---|---|---|---|
| ins-0003 | -0.667 | heavy | The meeting is scheduled for 3 p.m. | The scheduled meeting is set for 3:00 p.m. | [ERROR] HTTPSConnectionPool(host='openrouter.ai', port=443): Read timed out. (read timeout=120) |
| ins-0016 | -0.155 | base | Accuracy was 97.3% on 1.2M samples in 3 hours. | 97.3% accuracy was achieved on 1,200,000 samples in 3 hours. | Achieved 97.3% accuracy on 1,200,000 samples within a duration of 3 hours. |
| ins-0020 | -0.133 | base | 1) Prepare the dataset 2) Train the model 3) Evaluate on the dev set | 1. Prepare the dataset, 2. Train the model, 3. Evaluate the model on the development set. | 1. Prepare the dataset |

## Wasted compute (chosen=heavy, sh<=sb)

| id | Δ | chosen | gold | base | heavy |
|---|---:|---|---|---|---|
| ins-0003 | -0.667 | heavy | The meeting is scheduled for 3 p.m. | The scheduled meeting is set for 3:00 p.m. | [ERROR] HTTPSConnectionPool(host='openrouter.ai', port=443): Read timed out. (read timeout=120) |

## Missed opportunities (chosen=base, sh>sb)

| id | Δ | chosen | gold | base | heavy |
|---|---:|---|---|---|---|
| ins-0013 | +0.200 | base | • Goals\n• Timeline\n• Risks | - Goals | - Goals, - Timeline, - Risks |
| ins-0005 | +0.046 | base | This section is verbose and repetitive. | Simplify language for clarity and avoid repetition. | Simplify and eliminate repetition. |

