"""Send one long prompt to a running server and report time to first token.

Usage: served_prefill.py TARGET_TOKENS [BASE_URL]   (default http://127.0.0.1:8080)
The prompt is built from src/moespresso/runtime/qwen4/*.py at about 3.2
characters per token, so the served prompt length is approximate.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
target = int(sys.argv[1])
base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080"
body = ""
for path in sorted((REPO / "src/moespresso/runtime/qwen4").glob("*.py")):
    text = path.read_text()
    if len(body + text) > target * 3.2:
        break
    body += text
payload = {"messages": [{"role": "user",
                         "content": body + "\n\nSummarize the modules above in one sentence."}],
           "max_tokens": 1, "temperature": 0}
request = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
start = time.perf_counter()
response = json.loads(urllib.request.urlopen(request, timeout=3000).read())
elapsed = time.perf_counter() - start
usage = response["usage"]
print(f"SERVED prompt_tokens={usage['prompt_tokens']} ttft={elapsed:.1f}s "
      f"prefill={usage['prompt_tokens'] / elapsed:.1f} tok/s "
      f"cache={usage.get('prompt_cache', {}).get('event')}", flush=True)
