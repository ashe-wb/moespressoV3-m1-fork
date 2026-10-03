#!/bin/zsh
# Regenerate a per-layer expert capacity profile from fresh decode route traces.
#
# Usage: m1/profile/make_profile.sh PACKAGE [GB] [SLOTS_PER_LAYER] [OUT]
#   GB               planner ceiling while tracing (default 24)
#   SLOTS_PER_LAYER  uniform capacity the planner resolves at GB (default 209,
#                    the 24 GB value on a 32 GB M1 Max; read it from the startup line)
#   OUT              profile path (default m1/qwen4-capacity-profile.json)
#
# Loads the model once for about two minutes of tracing, then simulates and
# allocates offline. Stop any running MoEspresso server first.
set -eu
HERE=${0:A:h}
M1=${HERE:h}
PACKAGE=${1:?usage: make_profile.sh PACKAGE [GB] [SLOTS_PER_LAYER] [OUT]}
GB=${2:-24}
SLOTS=${3:-209}
OUT=${4:-$M1/qwen4-capacity-profile.json}
WORK=$(mktemp -d)

# Trace with uniform capacity: an existing profile must not shape the routes.
MOESPRESSO_DISK_KV=off MOESPRESSO_SSD_MAX_MEMORY_GB=$GB MOESPRESSO_QWEN4_CAPACITY_PROFILE= \
  MOESPRESSO_QWEN4_AUTONOMOUS=0 \
  "$M1/moespresso-m1" run "$HERE/trace_routes.py" "$PACKAGE" 384 "$WORK/routes.json"
"$M1/moespresso-m1" run "$HERE/simulate_misses.py" "$WORK/routes.json" "$PACKAGE" "$WORK/curves.json"
MANIFEST_ID=$("$M1/moespresso-m1" run -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["artifact_id"])' "$PACKAGE/package_manifest.json")
"$M1/moespresso-m1" run "$HERE/allocate.py" "$WORK/curves.json" $((48 * SLOTS)) "$MANIFEST_ID" "$OUT"
rm -rf "$WORK"
