"""Decode speed of one routing arm over four held-out prompts, 192 tokens each.

POLICY=off|<factor>,<protected> selects the routing (default 2,2, Cache-Prior
2/2). The reported rate excludes each prompt's first token.
"""
import json
import os
import statistics
import time

from _common import chat_ids, package_path, routing_kwargs, start_memory_guard
from moespresso.runtime.serve import generate_with_metadata, load_served_model

PROMPTS = [
    "Explain how a B-tree index works in a relational database, and when it beats a hash index.",
    "Write a Rust function that merges k sorted iterators, with an explanation of its complexity.",
    "Summarize the causes and consequences of the 2008 financial crisis in a few paragraphs.",
    "Describe how to set up a CI pipeline for a Python monorepo with caching and test sharding.",
]

start_memory_guard("DECODE")
policy = os.environ.get("POLICY", "2,2")
model, tokenizer, _ = load_served_model(package_path(), **routing_kwargs(policy))
tokens = seconds = 0.0
gaps = []
for prompt in PROMPTS:
    stamps = []
    generate_with_metadata(model, tokenizer, chat_ids(tokenizer, prompt), max_tokens=192,
                           temperature=0.0,
                           response_callback=lambda _i, _r: stamps.append(time.perf_counter()))
    tokens += len(stamps) - 1
    seconds += stamps[-1] - stamps[0]
    gaps += [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
print("DECODE", json.dumps({
    "policy": policy,
    "autonomous": os.environ.get("MOESPRESSO_QWEN4_AUTONOMOUS") == "1",
    "admit": os.environ.get("MOESPRESSO_QWEN4_AUTONOMOUS_ADMIT"),
    "tps": round(tokens / seconds, 2),
    "ms_p50": round(statistics.median(gaps), 1),
    "ms_p90": round(sorted(gaps)[int(len(gaps) * 0.9)], 1),
}), flush=True)
