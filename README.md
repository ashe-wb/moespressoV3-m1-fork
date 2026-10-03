> **Disclaimer: this repository is an unofficial fork of
> [steadfastgaze/MoEspresso](https://github.com/steadfastgaze/MoEspresso).**
> MoEspresso is created and maintained by Riccardo Chiumiento
> ([steadfastgaze](https://github.com/steadfastgaze)). Its architecture,
> runtime, package format, quantization pipeline, model packages, benchmarks
> and documentation are his work. This fork, published by
> [ashe-wb](https://github.com/ashe-wb), adds only the changes described below
> and is not affiliated with or endorsed by the upstream project. Report
> issues with this fork's changes here, and use the upstream repository for
> everything else.

# moespressoV3-m1-fork

This fork tunes MoEspresso 3 for
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) on a 2021
M1 Max with a 32-core GPU and 32 GB of unified memory, running macOS 27. It
speeds up bounded decode and prefill, where most routed experts stay on SSD,
and adds an opt-in GPU-autonomous decode mode. It uses the upstream
[Qwen3.8-Flash-Next-MoEspressoV3](https://huggingface.co/steadfastgaze/Qwen3.8-Flash-Next-MoEspressoV3)
package unchanged.

For what MoEspresso is and how it runs a 145.75 GB package in 32 GB, read the
[upstream README](https://github.com/steadfastgaze/MoEspresso#readme). The
mechanisms and switches added here are documented in
[Qwen3.8 architecture and serving](docs/qwen4.md).

## Results

All measurements used a 24 GB planner ceiling, which resolves to 209 expert
slots per layer, greedy decoding and disk KV disabled unless noted.

| Measurement | Before | After |
|---|---:|---:|
| Decode, Cache-Prior 2/2 (default) | 14.0 tok/s | 17.1 tok/s |
| Decode, GPU-autonomous (opt-in) | | 25.0 tok/s |
| Time to first token, 1,078-token prompt | 34 s | 12 s |
| Time to first token, 5,783-token prompt | 125 s | 56 s |
| Time to first token, 3,286-token served prompt with disk checkpoints | 108 s | 34 s |

The 14.0 tokens/s baseline is upstream MoEspresso 3 with only the wired-limit
fix below, using the public mlx-kquant pin. The 17.1 and 25.0 tokens/s results
averaged four prompts with 192 generated tokens each, and the prefill runs
used a 25 GB ceiling.

These "after" runs and the sweep below used an additional mlx-kquant kernel
change and a per-layer capacity profile, neither of which this repository
includes, with the host's `iogpu.wired_lwm_mb` set to 20000. Without that
setting, about one run in four at 24 GB fell below 6 tokens/s when macOS
compressed idle pool pages. The published configuration has not been
re-measured, so expect somewhat lower decode rates.

A cold long-context sweep at 24 GB served the full 131,072-token limit without
a memory abort:

| Prompt tokens | Time to first token | Prefill tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 3,902 | 29 s | 134 | 12.4 |
| 32,558 | 284 s | 115 | 11.8 |
| 65,325 | 615 s | 106 | 11.3 |
| 126,736 | 1,218 s | 104 | 10.7 |

This sweep generated code explanations, a different workload from the
four-prompt decode benchmark, and ran without the MLX command-buffer settings
the server applies, so its decode rates are likely understated.

## Changes

- **Bounded pools no longer wire the expert working set.** On macOS 27 the
  mlx-lm wired limit made the GPU driver repeatedly wire and unwire pool
  buffers, and decode fell from 12.1 to 0.6 tokens/s. Skipping it for bounded
  pools restored decode with identical generated tokens. Full residency keeps
  the wired limit.
- **Faster bounded decode.** Concurrent row prefetch for multi-expert misses,
  fused ring and table routed kernels, fused QSA decode attention, a faster
  top-10 selection kernel, a stacked shared-expert projection and host-side
  caching of per-layer arguments. Except for QSA decode attention, each change
  is bit-identical to the path it replaces.
- **Faster bounded prefill.** Packed IQ_K tiles run over each chunk's routed
  pairs, rows stream ahead of pool loads, chunks and disk checkpoints use
  4,096-token frontiers, and a fused kernel computes QSA prefill attention.
- **Per-layer capacity profiles.** `--expert-capacity-profile PATH` or
  `MOESPRESSO_QWEN4_CAPACITY_PROFILE` redistributes expert slots across layers
  from a JSON weight profile. The repository reads profiles and includes no
  profile or profile generator.
- **Opt-in GPU-autonomous decode.** `MOESPRESSO_QWEN4_AUTONOMOUS=1` routes each
  token among resident experts only and loads missing originals in the
  background, so the GPU never waits for the host within a token.
- **Served-path fixes.** The roadtest server waits for the serve worker's
  process group to exit before a restart, and the startup planner re-reads
  available memory until it settles, so a restarted server does not plan
  against pools the old worker still holds.
- **Opt-in untyped tool-call fallback.** With
  `MOESPRESSO_TOOLCALL_UNTYPED_FALLBACK=1`, a structurally sound Qwen tool call
  whose values fail every type repair reaches the client with raw values. The
  client's schema check then returns a tool error to the model instead of the
  turn ending as plain text.

## Trade-offs

- **Fused QSA attention is approximate.** Fused prefill attention matched the
  gathered path bit for bit on 99.9% of outputs and fused decode attention on
  99.995%, with the remainder within one BF16 step. Greedy output can change
  at near ties. With Cache-Prior 2/2, teacher-forced NLL moved from 1.1762 to
  1.1788 and top-1 agreement from 0.769 to 0.771. Both kernels are on by
  default.
- **GPU-autonomous decode costs quality.** Teacher-forced on exact-routing
  greedy references, NLL was 1.1618 for exact routing, 1.1762 for Cache-Prior
  2/2 and 1.2057 for autonomous decode, with top-1 agreement of 0.795, 0.769
  and 0.761. It stays off by default.
- **Upstream's benchmark score was not repeated.** Upstream's 84.3%
  generated-answer result was measured on its default path and has not been
  rerun with this fork's fused attention kernels.

## Install and run

The fork installs from source and requires an arm64 Apple Silicon Mac running
macOS 26.2 or later, [uv](https://docs.astral.sh/uv/) and the Xcode command
line tools. The upstream Homebrew formula installs the upstream release, not
this fork.

```bash
git clone https://github.com/ashe-wb/moespressoV3-m1-fork
cd moespressoV3-m1-fork
uv sync --locked
```

Download the upstream package:

```bash
hf download steadfastgaze/Qwen3.8-Flash-Next-MoEspressoV3 \
  --local-dir ./models/qwen3.8-flash-next
```

Start the OpenAI-compatible server on `127.0.0.1:8080`:

```bash
uv run --locked moespresso serve ./models/qwen3.8-flash-next --max-memory-gb 24
```

Other commands, serving controls and client setup are unchanged from upstream
and described in the
[upstream README](https://github.com/steadfastgaze/MoEspresso#readme).

### Switches added or used by this fork

| Setting | Default | Effect |
|---|---|---|
| `MOESPRESSO_QWEN4_AUTONOMOUS=1` | off | GPU-autonomous decode. Run it with `MLX_MAX_OPS_PER_BUFFER=100`. |
| `MOESPRESSO_QWEN4_AUTONOMOUS_ADMIT` | 4 | Missing original routes loaded per token in autonomous mode. |
| `MOESPRESSO_QWEN4_FUSED_QSA_DECODE=0` | fused | Restores gathered QSA decode attention. |
| `MOESPRESSO_QWEN4_FUSED_QSA_PREFILL=0` | fused | Restores gathered QSA prefill attention. |
| `--expert-capacity-profile PATH` | none | Per-layer expert slot profile, also `MOESPRESSO_QWEN4_CAPACITY_PROFILE`. |
| `MOESPRESSO_TOOLCALL_UNTYPED_FALLBACK=1` | off | Returns unrepairable tool calls with raw values. |

GPU-autonomous decode example:

```bash
MOESPRESSO_QWEN4_AUTONOMOUS=1 MLX_MAX_OPS_PER_BUFFER=100 \
  uv run --locked moespresso serve ./models/qwen3.8-flash-next --max-memory-gb 24
```

## Verification status

- `make lint` and `make dist-check` pass.
- `make test` passes except three tolerance tests in
  `tests/test_qwen4_qsa_runtime.py`, which exceed their 2e-5 bound by about
  4e-5 on the measurement host and fail identically on the unmodified upstream
  tree.
- `make roadtest` passed at 24 GB with six server segments and five restarts
  before GPU-autonomous decode and the final kernel round were added. It has
  not been rerun since.

## Credit and license

MoEspresso is the work of
[Riccardo Chiumiento](https://github.com/steadfastgaze), who designed and wrote
the project and published its model packages. The third-party work it builds
on is acknowledged in the
[upstream README](https://github.com/steadfastgaze/MoEspresso#acknowledgements)
and recorded per file in `THIRD-PARTY-NOTICES`.

MoEspresso is dual-licensed under Apache 2.0 (`LICENSE-APACHE-2.0`) or MIT
(`LICENSE-MIT`), at your option. It is Copyright (c) 2026 Riccardo Chiumiento,
and his notice in `LICENSE-MIT` is retained unchanged. This fork modifies files
relative to upstream commit
[`6b96f27`](https://github.com/steadfastgaze/MoEspresso/commit/6b96f27) and
adds new files under `src/moespresso/runtime/qwen4/` and `tests/`. The commit
history records every changed file. The fork's changes are offered under the
same dual license.
