#!/usr/bin/env python3
import os, requests, sys, json

MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/auto")
API_KEY = os.environ.get("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"

if not API_KEY:
    print("[debug] OPENROUTER_API_KEY is empty. Edit .env and run: source scripts/use_env.sh")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    # Optional but sometimes helpful for routing/identification:
    "X-Title": "pragact-router"
}
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Reply exactly: FINAL: ping"}],
    "max_tokens": 8,
    "temperature": 0.0
}

print("[debug] model:", MODEL)
r = requests.post(URL, headers=headers, json=payload, timeout=60)
print("[debug] status:", r.status_code)
print("[debug] body:", r.text[:1200])
if r.ok:
    data = r.json()
    print("[debug] choice:", data["choices"][0]["message"]["content"])
else:
    sys.exit(1)
