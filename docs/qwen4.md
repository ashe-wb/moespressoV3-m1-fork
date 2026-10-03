# Qwen4 architecture and serving

`runtime/qwen4` implements the Qwen3.8-Flash-Next text architecture identified
by `qwen4_exp` and `qwen4_exp_text` manifests, separately from the
`runtime/qwen` adapter used by Ornith. The package manifest selects the adapter
regardless of the directory name.

Loading requires a complete MoEspresso package containing its manifest,
tokenizer, model shards and PLE payload; neural weight shards alone are
insufficient. This package/runtime contract does not make arbitrary upstream
checkpoints loadable or declare a new public download.

## State and storage

The full512 graph has 48 layers and selects ten routed experts per layer. Its
hybrid state combines Gated DeltaNet recurrence, sparse QSA attention, and PLE
token/convolution history, all of which must describe the same committed token
frontier.

| Component | Runtime treatment |
| --- | --- |
| Dense projections, routers, norms and shared experts | Resident weights in the package's declared formats |
| Routed experts | Per-layer pools; full residency or demand loading from SSD |
| PLE/n-gram tables | Package-backed selected-row reads; no full GPU residency requirement |
| QSA cache | Mutable KVarN K4/V4 body with exact BF16 sink and recent suffix |
| GDN and PLE state | Included with QSA in composite prompt snapshots |

KVarN preserves an exact 128-token sink and at least the most recent 8,192
tokens, packing older body rows at tile boundaries. Short contexts do not
exercise the packed body. Ornith uses a separate Q8 attention-cache
implementation whose policy does not describe Qwen4.

The capacity planner reserves the configured context's KVarN buffers as well
as non-routed weights, workspace and safety headroom before allocating expert
slots. Increasing context can therefore reduce expert capacity. PLE/n-gram
files remain on SSD and are not charged as fully resident expert weights.
`--max-memory-gb` controls the startup planner without imposing a process-RSS
limit. Retained in-memory prompt snapshots have their own
`--prompt-cache-bytes` cap.

By default every routed layer receives the same slot count. Miss rates differ
by layer: layers 0 and 1 keep original routing and missed on 61-72% of decode
tokens at 209 slots, while several deeper layers missed on about 2%.
`--expert-capacity-profile PATH` (for `generate` and `serve`) redistributes the
planner's total slots in the proportions of a JSON profile:

```json
{"kind": "moespresso_qwen4_capacity_profile", "version": 1,
 "package_manifest_id": "pkg:...", "layers": [512, 512, 256, "... 48 values"]}
```

The profile is bound to one package manifest and fails closed on a mismatch.
Scaling never exceeds the planner total and keeps each layer between the
planner minimum and 512 slots, so pool memory is unchanged. A profile derived
from decode route traces raised four-prompt decode from 14.9 to 16.0 tokens/s
at 209 slots per layer on a 32 GB M1 Max. Cache-Prior routing depends on
residency, so outputs can differ from uniform capacity.

Generation raises MLX's wired limit only for full residency. Bounded pools
fill most of the wired-memory budget, and wiring them made the GPU driver
repeatedly wire and unwire pool buffers. On a 32 GB M1 Max running macOS 27,
greedy decode at 224-237 slots per layer measured 0.6-0.7 tokens/s with the
wired limit and 11-12 tokens/s without it, with identical tokens at equal
capacity.

## Ordinary generation

```bash
moespresso generate PACKAGE --prompt "Explain a binary search." \
  --thinking off --max-tokens 128
moespresso serve PACKAGE --thinking off
```

Serving targets 128K context unless an explicit supported limit is supplied.
Thinking defaults to on for this adapter; `--thinking off` selects the
non-thinking template. The packaged generation configuration supplies sampling
defaults unless the request overrides them.

Bounded full512 pools automatically use factor-two cache-conditioned routing
with two protected routes, while full residency remains unbiased and prefill
always uses original routing. See [Cache-Prior routing](cache_prior.md) for the
numerical tradeoff and the `--cache-routing off` override.

Ordinary decoding automatically prepares one row ahead when more than one
output token is requested, no history-dependent logits processor is active,
and the state supports the batch-one, unpadded single-token append lane.
The next row depends on the sampled token and uses no draft-model prediction.
Request ownership and slot publication protect in-flight readers; requests
outside these conditions use serial decoding. Stopping drains and discards
unpublished work.

Bounded pipelined decode runs the routed experts as two IQ_K dispatches that
read the host-published gate, up and down slots inside the kernels, after the
event wait. Routes are reduced in ascending source-expert order, so the output
matches the three-matrix path bit for bit. On a 32 GB M1 Max at 209 slots per
layer, greedy decode measured 13.93 tokens/s with three matrix dispatches and
14.16 tokens/s with two.

Full-resident prefill uses packed IQ_K matrix tiles when codec, geometry and
slot-map requirements hold. Smaller calls and unsupported layouts retain
their codec-specific paths. The runtime selects these paths automatically,
without requiring users to enable research flags.

Bounded prefill runs the same packed tiles over each over-capacity chunk's
routed pairs, addressed by pool slot, when the gate and up pools hold the same
slot map. The previous sorted path decoded every pool slot to FP16 per chunk,
so its cost followed pool size rather than routed pairs. Each bounded prefill
chunk also streams most routed experts from SSD again, so bounded pools use
4,096-token prefill chunks, disk checkpoints land on the same 4,096-token
frontiers, and large miss sets keep up to 24 row reads in flight ahead of the
ordered pool loads.

QSA prefill attends from a dense BF16 row bank: the prepared prior-plus-segment
rows, or the segment's own rows for a fresh prompt. A fused kernel reads each
query's selected rows from the bank by index instead of gathering them into a
new array, about 4 MB per query. It keeps the released arithmetic (BF16 logits,
exact scale, FP32 softmax, BF16 probabilities, FP32 value accumulation) and
differs only in accumulation order: 99.9% of outputs matched bit for bit on
random tests, the rest within one BF16 step. Greedy output can therefore flip
at near ties; measured flips were between candidates within 0.25 nats.
`MOESPRESSO_QWEN4_FUSED_QSA_PREFILL=0` selects the gathered path.

QSA decode uses a fused attention kernel with the same arithmetic contract.
About 99.995% of outputs matched the gathered path bit for bit.
`MOESPRESSO_QWEN4_FUSED_QSA_DECODE=0` selects the gathered path.

`MOESPRESSO_QWEN4_AUTONOMOUS=1` opts bounded decode into resident-only routing.
Each token routes only among experts already resident in every projection
pool, so the GPU never waits for the host within a token. The device records
the original top routes, and at token boundaries the host loads the strongest
missing originals in the background, up to `MOESPRESSO_QWEN4_AUTONOMOUS_ADMIT`
routes per token (default 4). On a 32 GB M1 Max at a 24 GB budget, decode rose
from 17.1 tokens/s with the default Cache-Prior 2/2 routing to 25.0 tokens/s,
with an mlx-kquant kernel change and a capacity profile that the repository
does not include.
Teacher-forced on exact-routing greedy references, NLL was 1.1618 for exact
routing, 1.1762 for Cache-Prior 2/2 and 1.2057 for autonomous decode, with top-1
agreement of 0.795, 0.769 and 0.761. Run it with `MLX_MAX_OPS_PER_BUFFER=100`.
The setting affects decode only.

On a 32 GB M1 Max at a 25 GB budget, these prefill changes took a 1,078-token
prompt from 34 s to 12 s to first token, a 5,783-token prompt from 125 s to
56 s, a 15,460-token prompt to 146 s, and a 3,286-token served prompt with
disk checkpoints from 108 s to 34 s.

## Prefix reuse and speculation

The HTTP adapter supports in-memory prompt reuse and default-on
[KVarN4 disk checkpoints](disk_kv.md#qwen4-kvarn4-checkpoints). Snapshots contain
the completed prefill state, with packed body data staying packed across save
and restore. On the next turn, generated text is processed as an unbiased
suffix.

Checkpoints omit expert residency and LFU counts, so a restored bounded request
can have different cache-biased decode choices even when its restored prompt
tensors are exact.

MTP is off in ordinary Qwen4 serving at every context length, including
short contexts. The retained [full-resident MTP command](qwen4_mtp.md) is an
explicit experimental path requiring a separate sidecar; its compiled verifier
is available through that command and does not activate MTP in `serve`.
