"""Cold time to first token for one prompt built from the Qwen4 runtime sources.

Usage: prefill_ttft.py TARGET_TOKENS
The prompt concatenates src/moespresso/runtime/qwen4/*.py up to TARGET_TOKENS
and asks for a one-sentence summary. STEP overrides the prefill chunk size.
"""
import os
import sys
import time

from _common import REPO, package_path, start_memory_guard
from moespresso.runtime.serve import generate_with_metadata, load_served_model

start_memory_guard("PREFILL")
step = int(os.environ["STEP"]) if os.environ.get("STEP") else None
parts = [p.read_text() for p in sorted((REPO / "src/moespresso/runtime/qwen4").glob("*.py"))]
model, tokenizer, _ = load_served_model(package_path())
target = int(sys.argv[1])
body = ""
for part in parts:
    if len(tokenizer.encode(body + part)) > target:
        break
    body += part
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": body + "\n\nSummarize the modules above in one sentence."}],
    add_generation_prompt=True, tokenize=False, enable_thinking=False)
n = len(tokenizer.encode(text))
marks = {}
start = time.perf_counter()
generate_with_metadata(model, tokenizer, text, max_tokens=1, temperature=0.0,
                       **({"prefill_step_size": step} if step else {}),
                       first_token_callback=lambda: marks.setdefault("t", time.perf_counter()))
ttft = marks["t"] - start
print(f"PREFILL step={step} prompt_tokens={n} ttft={ttft:.1f}s prefill={n / ttft:.1f} tok/s", flush=True)
