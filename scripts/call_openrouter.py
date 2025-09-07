#!/usr/bin/env python3
import os, json, time, argparse, requests, sys, re, warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

MODEL = os.environ.get("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
API_KEY = os.environ.get("OPENROUTER_API_KEY")

def call(messages, max_tokens=256, temperature=0.0, extra=None):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "pragact-router"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # stop at first newline so we get exactly one line
        "stop": ["\n"]
    }
    if extra: payload.update(extra)
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if not r.ok:
        try: err = r.json()
        except Exception: err = {"text": r.text}
        raise RuntimeError(f"HTTP {r.status_code} from OpenRouter: {err}")
    data = r.json()
    msg = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    native = data.get("native_tokens")
    return msg, usage, native

def extract_final(text: str) -> str:
    tags = ("FINAL:", "Final:", "final:", "ANSWER:", "Answer:", "answer:")
    pos = max((text.rfind(t) for t in tags), default=-1)
    if pos != -1:
        out = text[pos:].split(":", 1)[1] if ":" in text[pos:] else text[pos:]
    else:
        out = text
    out = out.strip()
    out = re.sub(r"^[`'\"<>«»\[\(]+|[`'\"<>«»\]\)]+$", "", out).strip()
    return out

def run(task, mode, inp):
    if task == "qa":
        sys_msgs = [{"role":"system","content":"Return exactly one line: FINAL: <answer>. No other text."}]
        user = [{"role":"user","content": f"Question: {inp['question']}\nOutput one line exactly as: FINAL: <answer>"}]
        raw, usage, native = call(sys_msgs+user, max_tokens=(32 if mode=="base" else 96), temperature=0.0)
        return extract_final(raw), usage, native
    else:
        text = f"Instruction: {inp['instruction']}\nInput: {inp['input']}\nOutput one line exactly as: FINAL: <output>"
        sys_msgs = [{"role":"system","content":"Think briefly if needed, but output only one line starting with FINAL: and nothing else."}]
        user = [{"role":"user","content": text}]
        raw, usage, native = call(sys_msgs+user, max_tokens=(128 if mode=="base" else 256), temperature=0.0)
        return extract_final(raw), usage, native

def norm_cost(usage, native):
    if isinstance(usage, dict) and "total_tokens" in usage:
        return float(usage["total_tokens"])
    if native is not None:
        try: return float(native)
        except: pass
    return None

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--mode", choices=["base","heavy"], required=True)
    args = ap.parse_args()
    if not API_KEY:
        raise SystemExit("Set OPENROUTER_API_KEY in .env and `source scripts/use_env.sh`")
    out = open(args.out_jsonl,"w",encoding="utf-8")
    with open(args.in_jsonl,"r",encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            task = r.get("task","qa" if "question" in r else "instr")
            try:
                final_text, usage, native = run(task, args.mode, r)
                r[f"pred_{args.mode}"] = final_text
                c = norm_cost(usage, native)
                if c is not None: r[f"cost_{args.mode}"] = c
            except Exception as e:
                r[f"pred_{args.mode}"] = f"[ERROR] {e}"
            out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush(); time.sleep(0.3)
    out.close(); print("[ok] wrote", args.out_jsonl)
