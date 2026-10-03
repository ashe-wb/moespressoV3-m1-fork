"""Teacher-forced quality and decode speed of one routing arm.

POLICY=off|<factor>,<protected>. Run POLICY=off first: it writes greedy
exact-routing references to REF (default route_refs.json). Every arm then scores
each reference token under its own decode routing, reporting mean NLL and top-1
agreement, followed by a four-prompt greedy speed run.

References cover four chat prompts of TOK tokens (default 160) and 400 tokens of
docs/ssd_streaming.md after a 200-token prefix.
"""
import json
import os
import time
from pathlib import Path

import mlx.core as mx

from _common import REPO, chat_ids, package_path, routing_kwargs, start_memory_guard
from moespresso.runtime.serve import generate_with_metadata, load_served_model

PROMPTS = [
    "Design a REST API for a library management system, with endpoints, data models and error handling.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Explain the difference between TCP and UDP, with examples of when to use each.",
    "Write a Python function that parses a CSV file and computes per-column statistics, with tests.",
]
SPEED_PROMPTS = [
    "Explain how a B-tree index works in a relational database, and when it beats a hash index.",
    "Write a Rust function that merges k sorted iterators, with an explanation of its complexity.",
    "Summarize the causes and consequences of the 2008 financial crisis in a few paragraphs.",
    "Describe how to set up a CI pipeline for a Python monorepo with caching and test sharding.",
]

start_memory_guard("RQ")
policy = os.environ["POLICY"]
ref_path = Path(os.environ.get("REF", "route_refs.json"))
tok = int(os.environ.get("TOK", "160"))
model, tokenizer, _ = load_served_model(package_path(), **routing_kwargs(policy))

if not ref_path.exists():
    assert policy == "off", "references come from the exact-routing arm"
    refs = []
    for prompt in PROMPTS:
        ids = chat_ids(tokenizer, prompt)
        result = generate_with_metadata(model, tokenizer, ids, max_tokens=tok, temperature=0.0)
        refs.append({"prompt": ids, "ref": list(result.generated_token_ids)})
    text = (REPO / "docs/ssd_streaming.md").read_text()
    ids = tokenizer.encode(text, add_special_tokens=False)
    refs.append({"prompt": ids[:200], "ref": ids[200:600]})
    ref_path.write_text(json.dumps(refs))
    print("RQ refs written", [len(r["ref"]) for r in refs], flush=True)

refs = json.loads(ref_path.read_text())
per = []
for item in refs:
    prompt, ref = item["prompt"], item["ref"]
    nll, agree = [], []

    def factory(**_):
        def force(history, logits):
            want = ref[history.shape[0] - len(prompt)]
            wide = logits.astype(mx.float32)
            logprobs = wide - mx.logsumexp(wide, axis=-1, keepdims=True)
            mx.eval(logprobs)
            nll.append(-logprobs[0, want].item())
            agree.append(int(mx.argmax(logprobs[0]).item() == want))
            mask = mx.full(logits.shape, -1e9, dtype=logits.dtype)
            return mask.at[:, want].add(1e9)
        return [force]

    generate_with_metadata(model, tokenizer, prompt, max_tokens=len(ref), temperature=0.0,
                           presence_penalty=1e-9, logits_processors_factory=factory)
    per.append({"n": len(nll), "nll": sum(nll) / len(nll), "top1": sum(agree) / len(agree)})
    print("RQ item", json.dumps(per[-1]), flush=True)

total = sum(p["n"] for p in per)
summary = {"policy": policy,
           "autonomous": os.environ.get("MOESPRESSO_QWEN4_AUTONOMOUS") == "1",
           "nll": sum(p["nll"] * p["n"] for p in per) / total,
           "top1": sum(p["top1"] * p["n"] for p in per) / total}

tokens = seconds = 0.0
for prompt in SPEED_PROMPTS:
    mark = {}
    result = generate_with_metadata(
        model, tokenizer, chat_ids(tokenizer, prompt), max_tokens=192, temperature=0.0,
        first_token_callback=lambda: mark.setdefault("t", time.perf_counter()))
    tokens += len(result.generated_token_ids) - 1
    seconds += time.perf_counter() - mark["t"]
summary["tps"] = round(tokens / seconds, 2)
print("RQ", json.dumps(summary), flush=True)
