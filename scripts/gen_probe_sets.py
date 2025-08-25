# -*- coding: utf-8 -*-
import random, argparse, json, os

# -------------------------
# English lexicon (typed)
# -------------------------
EN_DOC   = ["report","paper","document","article","section","paragraph","file","page"]
EN_RES   = ["dataset","model","experiment"]
EN_PROC  = ["project","schedule","pipeline","system"]

# ---- Requests (verb -> allowed objects) ----
EN_REQ_ALLOWED = {
  "summarize": EN_DOC,
  "list": EN_DOC,
  "review": EN_DOC + EN_RES,
  "translate": EN_DOC,
  "open": ["report","document","file","page","paper","article"],
  "explain": EN_DOC + EN_RES + EN_PROC,
  "update": ["report","document","dataset","model","system","project"]
}

def en_req_phrase(verb, obj):
  if verb == "summarize":
    return f"Summarize the {obj} in bullet points."
  if verb == "list":
    return f"List the key points of the {obj}."
  if verb == "review":
    return f"Review the {obj}."
  if verb == "translate":
    return f"Translate the {obj}."
  if verb == "open":
    return f"Please open the {obj}."
  if verb == "explain":
    return f"Explain the {obj}."
  if verb == "update":
    return f"Please update the {obj}."
  return f"Please {verb} the {obj}."

# ---- Questions (template, allowed objects) ----
EN_Q_TPLS = {
  "why_important": ("Why is the {obj} important?", EN_DOC + EN_RES + EN_PROC),
  "how_work":      ("How does the {obj} work?", EN_RES + EN_PROC),
  "is_available":  ("Is the {obj} available now?", ["dataset","model","system"]),
  "when_start":    ("When does the {obj} start?", ["project","schedule","experiment","pipeline"]),
  "could_explain": ("Could you explain the {obj}?", EN_DOC + EN_RES + EN_PROC),
  # NEW: adds enough unique questions to meet quota
  "what_status":   ("What is the current status of the {obj}?", ["dataset","model","project","system","schedule","experiment","pipeline"]),
}

# ---- Statements (template, allowed pools) ----
EN_TOPICS = ["methods","results","discussion","introduction","evaluation"]
EN_S_TPLS = {
  "updated":   ("The {obj} was updated yesterday.", ["report","document","dataset","model","system","project"]),
  "available": ("The {obj} is available.", ["dataset","model","system"]),
  "includes":  ("The {container} includes a section on {topic}.", {
                 "container": ["report","paper","document","article"],
                 "topic": EN_TOPICS
               }),
  "important": ("This {obj} is important for the project.", EN_DOC + EN_RES)  # exclude 'project' itself
}

# ---- Promises / Expressives / Declarations ----
EN_PROM_OBJS = ["report","paper","document","dataset"]
EN_PROM = [
  "I promise to submit the {obj} tomorrow.",
  "I promise to deliver the {obj} tomorrow."
]

EN_EXPR = [
  "Thanks for your help!",
  "Thank you so much.",
  "I really appreciate your help.",
  "Sorry for the inconvenience.",
  "Apologies for the delay."
]

# Expand declaration objects + templates so we can fill 5 uniques
EN_DECL_OBJS = ["project","experiment","system","schedule","pipeline"]
EN_DECL_TPLS = [
  "We hereby declare the {obj} closed.",
  "We officially close the {obj}.",
  "We hereby announce the completion of the {obj}."
]

def pick(lst): return random.choice(lst)

# ---------- Build candidate pools (no duplicates), then sample ----------
def uniq(seq):
  seen=set(); out=[]
  for x in seq:
    if x not in seen:
      seen.add(x); out.append(x)
  return out

def build_en_requests():
  c=[]
  for v, pool in EN_REQ_ALLOWED.items():
    for obj in pool:
      c.append( (en_req_phrase(v, obj), "request") )
  return uniq(c)

def build_en_questions():
  c=[]
  for _, (tpl, pool) in EN_Q_TPLS.items():
    for obj in pool:
      c.append( (tpl.format(obj=obj), "question") )
  return uniq(c)

def build_en_statements():
  c=[]
  tpl, pool = EN_S_TPLS["updated"]
  for obj in pool:
    c.append( (tpl.format(obj=obj), "statement") )
  tpl, pool = EN_S_TPLS["available"]
  for obj in pool:
    c.append( (tpl.format(obj=obj), "statement") )
  # includes
  pools = EN_S_TPLS["includes"][1]
  for cont in pools["container"]:
    for topic in pools["topic"]:
      c.append( (EN_S_TPLS["includes"][0].format(container=cont, topic=topic), "statement") )
  tpl, pool = EN_S_TPLS["important"]
  for obj in pool:
    if obj == "project":  # just in case
      continue
    c.append( (tpl.format(obj=obj), "statement") )
  return uniq(c)

def build_en_promises():
  c=[]
  for obj in EN_PROM_OBJS:
    for tpl in EN_PROM:
      c.append( (tpl.format(obj=obj), "promise") )
  return uniq(c)

def build_en_expressives():
  return uniq([(x, "expressive") for x in EN_EXPR])

def build_en_declarations():
  c=[]
  for tpl in EN_DECL_TPLS:
    for obj in EN_DECL_OBJS:
      c.append( (tpl.format(obj=obj), "declaration") )
  return uniq(c)

def sample_from_pool(pool, n, rng):
  pool = list(pool)
  rng.shuffle(pool)
  assert len(pool) >= n, f"Pool size {len(pool)} < needed {n}"
  return pool[:n]

def expand_en(n_target=150, quotas=(48,48,39,5,5,5), seed=42):
  rng = random.Random(seed)
  NQ, NR, NS, NP, NX, ND = quotas
  Q = build_en_questions()
  R = build_en_requests()
  S = build_en_statements()
  P = build_en_promises()
  X = build_en_expressives()
  D = build_en_declarations()
  batch = []
  batch += sample_from_pool(Q, NQ, rng)
  batch += sample_from_pool(R, NR, rng)
  batch += sample_from_pool(S, NS, rng)
  batch += sample_from_pool(P, NP, rng)
  batch += sample_from_pool(X, NX, rng)
  batch += sample_from_pool(D, ND, rng)
  rng.shuffle(batch)
  assert len(batch) == n_target, f"EN size {len(batch)} != {n_target}"
  return batch

# -------------------------
# Chinese lexicon (typed)
# -------------------------
ZH_DOC   = ["报告","论文","文档","文章","段落","页面","文件"]
ZH_RES   = ["数据集","模型","实验"]
ZH_PROC  = ["项目","日程","流程","系统"]

# Requests (动词 -> 可搭配对象) — 已移除“计算”
ZH_REQ_ALLOWED = {
  "总结": ZH_DOC,
  "列出": ZH_DOC,
  "审阅": ZH_DOC,
  "翻译": ZH_DOC,
  "打开": ["报告","文档","文件","页面","文章","段落"],
  "解释": ZH_DOC + ZH_RES + ZH_PROC,
  "更新": ["报告","文档","数据集","模型","系统","项目"]
}

def zh_req_phrase(verb, obj):
  if verb == "总结":
    return f"请把{obj}用要点总结一下。"
  if verb == "列出":
    return f"请列出{obj}的关键点。"
  return f"请{verb}{obj}。"

# Questions
ZH_Q_TPLS = {
  "why_important": ("为何{obj}重要？", ZH_DOC + ZH_RES + ZH_PROC),
  "how_work":      ("{obj}是怎么运作的？", ZH_RES + ZH_PROC),
  "is_available":  ("{obj}现在可用吗？", ["数据集","模型","系统"]),
  "when_start":    ("{obj}什么时候开始？", ["项目","日程","实验","流程"]),
  "could_explain": ("你能解释一下{obj}吗？", ZH_DOC + ZH_RES + ZH_PROC),
  # NEW: 增加一个自然的询问模板，防止配额不足
  "what_status":   ("{obj}目前处于什么状态？", ["数据集","模型","项目","系统","日程","实验","流程"]),
}

# Statements
ZH_TOPICS = ["方法","结果","讨论","引言","评估"]
ZH_S_TPLS = {
  "updated":   ("{obj}昨天更新。", ["报告","文档","数据集","模型","系统","项目"]),
  "available": ("{obj}是可用的。", ["数据集","模型","系统"]),
  "includes":  ("{container}中包含{topic}部分。", {
                 "container": ["报告","论文","文档","文章"],
                 "topic": ZH_TOPICS
               }),
  "important": ("这个{obj}很重要。", ZH_DOC + ZH_RES + ["系统","流程"])
}

# Promises / Expressives / Declarations
ZH_PROM_OBJS = ["报告","文档","论文","数据集"]
ZH_PROM_TPLS = ["我保证明天提交{obj}。", "我承诺明天提交{obj}。"]   # +1 模板，确保≥5唯一

ZH_EXPR = [
  "谢谢你的帮助！",
  "非常感谢你的帮助！",
  "抱歉造成延误。",
  "十分抱歉给你带来不便。",
  "感谢你的耐心等待。"  # +1，确保≥5唯一
]

ZH_DECL_OBJS = ["项目","实验","流程","系统"]
ZH_DECL_TPLS = ["兹宣布{obj}结束。", "特此宣布{obj}完成。"]  # +1 模板，确保≥5唯一

# ---------- Build candidate pools and sample ----------
def uniq2(seq):
  seen=set(); out=[]
  for x in seq:
    if x not in seen:
      seen.add(x); out.append(x)
  return out

def build_zh_requests():
  c=[]
  for v, pool in ZH_REQ_ALLOWED.items():
    for obj in pool:
      c.append( (zh_req_phrase(v, obj), "request") )
  return uniq2(c)

def build_zh_questions():
  c=[]
  for _, (tpl, pool) in ZH_Q_TPLS.items():
    for obj in pool:
      c.append( (tpl.format(obj=obj), "question") )
  return uniq2(c)

def build_zh_statements():
  c=[]
  tpl, pool = ZH_S_TPLS["updated"]
  for obj in pool:
    c.append( (tpl.format(obj=obj), "statement") )
  tpl, pool = ZH_S_TPLS["available"]
  for obj in pool:
    c.append( (tpl.format(obj=obj), "statement") )
  pools = ZH_S_TPLS["includes"][1]
  for cont in pools["container"]:
    for topic in pools["topic"]:
      c.append( (ZH_S_TPLS["includes"][0].format(container=cont, topic=topic), "statement") )
  tpl, pool = ZH_S_TPLS["important"]
  for obj in pool:
    if obj == "项目":
      continue
    c.append( (tpl.format(obj=obj), "statement") )
  return uniq2(c)

def build_zh_promises():
  c=[]
  for obj in ZH_PROM_OBJS:
    for tpl in ZH_PROM_TPLS:
      c.append( (tpl.format(obj=obj), "promise") )
  return uniq2(c)

def build_zh_expressives():
  return uniq2([(x, "expressive") for x in ZH_EXPR])

def build_zh_declarations():
  c=[]
  for tpl in ZH_DECL_TPLS:
    for obj in ZH_DECL_OBJS:
      c.append( (tpl.format(obj=obj), "declaration") )
  return uniq2(c)

def sample_pool(pool, n, rng):
  pool = list(pool)
  rng.shuffle(pool)
  assert len(pool) >= n, f"Pool size {len(pool)} < needed {n}"
  return pool[:n]

def expand_zh(n_target=150, quotas=(48,48,39,5,5,5), seed=42):
  rng = random.Random(seed)
  NQ, NR, NS, NP, NX, ND = quotas
  Q = build_zh_questions()
  R = build_zh_requests()
  S = build_zh_statements()
  P = build_zh_promises()
  X = build_zh_expressives()
  D = build_zh_declarations()
  batch = []
  batch += sample_pool(Q, NQ, rng)
  batch += sample_pool(R, NR, rng)
  batch += sample_pool(S, NS, rng)
  batch += sample_pool(P, NP, rng)
  batch += sample_pool(X, NX, rng)
  batch += sample_pool(D, ND, rng)
  rng.shuffle(batch)
  assert len(batch) == n_target, f"ZH size {len(batch)} != {n_target}"
  return batch

if __name__ == "__main__":
  import argparse, json, os, random
  ap = argparse.ArgumentParser()
  ap.add_argument("--en", default="data/en/probe_150.jsonl")
  ap.add_argument("--zh", default="data/zh/probe_150.jsonl")
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()
  random.seed(args.seed)
  os.makedirs("data/en", exist_ok=True)
  os.makedirs("data/zh", exist_ok=True)

  def _to_dict(x):
    if isinstance(x, dict): 
      return x
    if isinstance(x, (list, tuple)) and len(x) == 2:
      return {"text": x[0], "gold": x[1]}
    raise ValueError(f"Unexpected sample format: {type(x)} -> {x}")

  with open(args.en, "w", encoding="utf-8") as f:
    for ex in expand_en():
      f.write(json.dumps(_to_dict(ex), ensure_ascii=False) + "\n")
  with open(args.zh, "w", encoding="utf-8") as f:
    for ex in expand_zh():
      f.write(json.dumps(_to_dict(ex), ensure_ascii=False) + "\n")
  print("Wrote:", args.en, "and", args.zh)
