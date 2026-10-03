"""Cold time to first token and decode speed at long contexts, one load, one prompt per size.

Usage: ctx_sweep.py SIZES OUT_JSON   for example 4096,32768,65536,127000 sweep.json
Prompts concatenate the repository's src, docs and tests files up to each size
and ask for a step-by-step explanation. DECODE sets generated tokens (default 128).
"""
import json
import os
import sys
import time
from pathlib import Path

from _common import REPO, package_path, start_memory_guard
from moespresso.runtime.serve import generate_with_metadata, load_served_model

start_memory_guard("SWEEP")
sizes = [int(x) for x in sys.argv[1].split(",")]
out = Path(sys.argv[2])
decode = int(os.environ.get("DECODE", "128"))
model, tokenizer, _ = load_served_model(package_path())
files = (sorted((REPO / "src").rglob("*.py")) + sorted((REPO / "docs").rglob("*.md"))
         + sorted((REPO / "tests").rglob("*.py")))
parts, counts = [], []
for path in files:
    text = f"\n# file: {path.relative_to(REPO)}\n" + path.read_text()
    parts.append(text)
    counts.append(len(tokenizer.encode(text, add_special_tokens=False)))
print("SWEEP corpus tokens", sum(counts), flush=True)

results = []
for size in sizes:
    body, total = [], 0
    for text, n in zip(parts, counts):
        if total + n > size - 200:
            continue
        body.append(text)
        total += n
    question = "\n\nExplain in detail, step by step, how the code above loads and serves a model."
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "".join(body) + question}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)
    n_prompt = len(tokenizer.encode(prompt, add_special_tokens=False))
    print(f"SWEEP start target={size} prompt_tokens={n_prompt}", flush=True)
    stamps = []
    start = time.perf_counter()
    generate_with_metadata(model, tokenizer, prompt, max_tokens=decode, temperature=0.0,
                           response_callback=lambda _i, _r: stamps.append(time.perf_counter()))
    ttft = stamps[0] - start if stamps else float("nan")
    rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else float("nan")
    row = {"target": size, "prompt_tokens": n_prompt, "ttft_s": round(ttft, 1),
           "prefill_tps": round(n_prompt / ttft, 1), "decode_tokens": len(stamps),
           "decode_tps": round(rate, 2)}
    print("SWEEP", json.dumps(row), flush=True)
    results.append(row)
    out.write_text(json.dumps(results, indent=1))
