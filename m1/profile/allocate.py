"""Allocate a fixed total of expert slots across layers by dynamic programming.

The cost of a layer at a capacity is its simulated misses times the measured
cost of one bundle-row read, plus the per-token route-gate cost that a fully
resident layer avoids. ROW_MS and GATE_MS were measured on a 32 GB M1 Max, and
TOKENS is the decode-token count of the recorded traces.

Usage: allocate.py CURVES_JSON TOTAL_SLOTS PACKAGE_MANIFEST_ID OUT_PROFILE
"""
import json
import sys
from pathlib import Path

LAYERS = 48
TOKENS = 1139
ROW_MS = 1.64
GATE_MS = 0.30
UNIT = 8

curves = {int(layer): {int(c): m for c, m in v.items()}
          for layer, v in json.loads(Path(sys.argv[1]).read_text()).items()}
total = int(sys.argv[2])
choices = sorted(next(iter(curves.values())).keys())


def cost(layer, cap):
    return curves[layer][cap] * ROW_MS + (0 if cap >= 512 else GATE_MS * TOKENS)


budget = total // UNIT
INF = float("inf")
dp = [0.0] + [INF] * budget
picks = []
for layer in range(LAYERS):
    new = [INF] * (budget + 1)
    pick = [None] * (budget + 1)
    for b in range(budget + 1):
        if dp[b] == INF:
            continue
        for cap in choices:
            nb = b + (cap + UNIT - 1) // UNIT
            if nb > budget:
                continue
            value = dp[b] + cost(layer, cap)
            if value < new[nb]:
                new[nb] = value
                pick[nb] = (b, cap)
    dp = new
    picks.append(pick)

best = min(range(budget + 1), key=lambda b: dp[b])
alloc = [0] * LAYERS
b = best
for layer in range(LAYERS - 1, -1, -1):
    b, alloc[layer] = picks[layer][b]
uniform_cap = max(c for c in choices if c <= total // LAYERS)
uniform = sum(cost(layer, uniform_cap) for layer in range(LAYERS))
print("alloc", alloc, "slots", sum(alloc))
print(f"predicted ms/token: uniform{uniform_cap}={uniform / TOKENS:.2f} "
      f"optimized={dp[best] / TOKENS:.2f}")
Path(sys.argv[4]).write_text(json.dumps({
    "kind": "moespresso_qwen4_capacity_profile",
    "version": 1,
    "package_manifest_id": sys.argv[3],
    "layers": alloc,
    "source": "decode route traces; decaying-LFU simulation; DP allocation",
}, indent=1) + "\n")
print("wrote", sys.argv[4])
