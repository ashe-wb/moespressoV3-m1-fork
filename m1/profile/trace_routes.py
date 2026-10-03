"""Record decode expert routes over three prompts with the built-in route tracer.

Usage: trace_routes.py PACKAGE TOKENS OUT_JSON
"""
import json
import sys
import time
from pathlib import Path

import moespresso.runtime.pooled_switchglu as psg
from moespresso.runtime.serve import generate_with_metadata, load_served_model

PROMPTS = [
    "Write a lock-free bounded queue in C++ and explain its memory ordering.",
    "Explain how the French Revolution changed European politics, in a few paragraphs.",
    "Solve step by step: a train leaves at 3pm at 80 km/h, another at 4pm at 100 km/h on the same track. "
    "When does the second catch up? Then generalize.",
]

package, tokens, out_path = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
model, tokenizer, _ = load_served_model(package)
out = []
for prompt in PROMPTS:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False,
        enable_thinking=False)
    psg.route_trace_start()
    t0 = time.perf_counter()
    result = generate_with_metadata(model, tokenizer, text, max_tokens=tokens, temperature=0.0)
    trace = psg.route_trace_stop()
    decode = [(int(e[1]), int(e[2]), [int(i) for i in e[3]]) for e in trace if e[0] == "decode"]
    out.append(decode)
    print("TRACE", len(result.generated_token_ids), "tokens", len(decode), "decode entries",
          round(time.perf_counter() - t0, 1), "s", flush=True)
out_path.write_text(json.dumps(out))
