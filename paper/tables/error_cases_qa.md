# Error cases (qa)

- N=20; threshold delta=0.01
- Helped: 1  | Hurt: 0  | Wasted (FP): 0  | Missed (FN): 0

## Heavy helped (sh>sb)

| id | Δ | chosen | gold | base | heavy |
|---|---:|---|---|---|---|
| qa-0016 | +0.429 | heavy |  | George Orwell (Eric Arthur Blair | George Orwell |

## Heavy hurt (sh<sb)

| id | Δ | chosen | gold | base | heavy |
|---|---:|---|---|---|---|

## Wasted compute (chosen=heavy, sh<=sb)

| id | Δ | chosen | gold | base | heavy |
|---|---:|---|---|---|---|

## Missed opportunities (chosen=base, sh>sb)

| id | Δ | chosen | gold | base | heavy |
|---|---:|---|---|---|---|

