"""Simulate per-layer decaying-LFU expert pools on recorded routes.

Each layer's pool starts from the package's expert hotlist and is replayed at a
range of capacities. The output maps layer to capacity to miss count.

Usage: simulate_misses.py ROUTES_JSON PACKAGE OUT_JSON
"""
import json
import sys
from pathlib import Path

LAYERS = 48
DECAY = 128

routes = json.loads(Path(sys.argv[1]).read_text())
hot = json.loads((Path(sys.argv[2]) / "expert_hotlist.json").read_text())["layers"]
per_layer = {layer: [] for layer in range(LAYERS)}
for prompt in routes:
    for _seq, layer, ids in prompt:
        per_layer[layer].append(ids)


def simulate(layer, cap):
    ranked = sorted(hot[str(layer)].items(), key=lambda kv: -kv[1])
    resident = [int(e) for e, _ in ranked[:cap]]
    res = set(resident)
    freq, rec = {}, {}
    clock = touches = misses = 0
    for e in resident:
        clock += 1
        rec[e] = clock
    for ids in per_layer[layer]:
        active = set(ids)
        for e in sorted(active):
            clock += 1
            touches += 1
            freq[e] = freq.get(e, 0) + 1
            rec[e] = clock
            if e not in res:
                misses += 1
                if len(res) >= cap:
                    victim = min((x for x in res if x not in active),
                                 key=lambda x: (freq.get(x, 0), rec.get(x, 0)))
                    res.discard(victim)
                res.add(e)
            if DECAY and touches % DECAY == 0:
                for k in freq:
                    freq[k] //= 2
    return misses


caps = list(range(40, 512, 24)) + [512]
curves = {}
for layer in range(LAYERS):
    curves[layer] = {c: simulate(layer, c) for c in caps}
    print(layer, [curves[layer][c] for c in (112, 160, 208, 256, 352, 512)], flush=True)
Path(sys.argv[3]).write_text(json.dumps(curves))
