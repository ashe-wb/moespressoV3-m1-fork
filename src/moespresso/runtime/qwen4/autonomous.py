"""Deferred expert residency for GPU-autonomous bounded Qwen4 decode.

Autonomous decode routes each token only among experts resident in every
projection pool, so the GPU never waits for the host inside a token. The
original top routes are recorded on the device. At each token boundary the host
reads the completed tokens' originals, updates LFU counters and loads the
strongest nonresident originals into the pools in the background.

A slot referenced by any snapshot of an unfinished token is never written. A
reclaimed slot leaves the published slot map when it is reserved, and its
bytes are written only after every token built before that point has
completed. A loaded expert enters the slot map only after all three
projections have landed.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import time

import mlx.core as mx
import numpy as np

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qwen4-autonomous-load")


class AutonomousResidency:
    """Own snapshots, route records and background loads for one model."""

    def __init__(self, model, *, admit_routes: int = 4, layer_inflight: int = 8,
                 gated_layers: frozenset[int] = frozenset()):
        self.admit_routes = int(admit_routes)
        self.layer_inflight = int(layer_inflight)
        self.pools: dict[int, tuple] = {}
        self.layer_of: dict[int, int] = {}
        for index, layer in enumerate(model.layers):
            router = getattr(layer.mlp, "gate", None)
            provider = getattr(router, "_cache_routing_provider", None)
            if provider is None or index in gated_layers:
                continue
            self.pools[index] = provider.pools
            self.layer_of[id(router)] = index
            object.__setattr__(router, "_autonomous_residency", self)
        if not self.pools:
            raise ValueError("autonomous residency requires cache-routed layers")
        self.first_layer = min(self.pools)
        self.built = 0
        self.completed = 0
        self.records: deque = deque()
        self.reserved: list[tuple[int, int, tuple, int]] = []
        self.loading: list[tuple[int, int, tuple, object]] = []
        self.pending_experts: dict[int, set[int]] = {index: set() for index in self.pools}
        self.loads = 0
        self.admissions = 0
        self.failures = 0
        self.top1_missing = {index: 0 for index in self.pools}
        self.top2_missing = {index: 0 for index in self.pools}
        self.layer_tokens = 0
        self.boundary_seconds = 0.0
        self._decode_tables: dict[int, tuple] = {}
        self.snapshot_seconds = 0.0

    def snapshot(self, router):
        started = time.perf_counter()
        try:
            return self._snapshot(router)
        finally:
            self.snapshot_seconds += time.perf_counter() - started

    def _snapshot(self, router):
        layer = self.layer_of[id(router)]
        if layer == self.first_layer:
            self.built += 1
            self.records.append((self.built, []))
        pools = self.pools[layer]
        with ExitStack() as stack:
            for pool in pools:
                stack.enter_context(pool._bk_lock)
            if (all(p.capacity == p.num_experts for p in pools)
                    and all(len(p._slot_of) == p.num_experts for p in pools)):
                return None
            return tuple(pool._ensure_slot_table() for pool in pools)

    def decode_tables(self, router, maps):
        """Return slot tables bounded to each pool for the routed kernels.

        Selection reads the published maps, whose sentinel marks a missing
        expert. The routed kernels read a bounded copy, rebuilt only when the
        layer publishes new maps; the entry keeps the maps alive so their
        identity cannot be reused.
        """
        layer = self.layer_of[id(router)]
        cached = self._decode_tables.get(layer)
        if cached is not None and all(a is b for a, b in zip(cached[0], maps)):
            return cached[1]
        bounded = tuple(mx.minimum(table, pool.capacity - 1)
                        for table, pool in zip(maps, self.pools[layer]))
        self._decode_tables[layer] = (maps, bounded)
        return bounded

    def record(self, router, original):
        if self.records:
            self.records[-1][1].append((self.layer_of[id(router)], original))

    def boundary(self, *, queued: int) -> None:
        """Advance after the host has read a token; ``queued`` builds may still run."""
        started = time.perf_counter()
        try:
            self._boundary(queued)
        finally:
            self.boundary_seconds += time.perf_counter() - started

    def _boundary(self, queued: int) -> None:
        self.completed = max(self.completed, self.built - queued)
        self._publish_finished()
        self._start_retired()
        ready = []
        while self.records and self.records[0][0] <= self.completed:
            ready.append(self.records.popleft()[1])
        # Completed arrays are read without new device work, which would queue
        # behind the running token.
        for items in ready:
            for layer, original in items:
                self._admit(layer, np.asarray(original).reshape(-1).tolist())
        self._start_retired()

    def finish(self) -> None:
        """Complete every reservation after the request's GPU work has drained."""
        mx.synchronize()
        self.completed = self.built
        self.records.clear()
        self._start_retired()
        for *_rest, future in self.loading:
            try:
                future.result()
            except BaseException:
                pass
        self._publish_finished()

    def _admit(self, layer: int, originals: list[int]) -> None:
        pools = self.pools[layer]
        pending = self.pending_experts[layer]
        missing = [not all(e in pool._slot_of for pool in pools) for e in originals[:2]]
        self.top1_missing[layer] += missing[0]
        self.top2_missing[layer] += any(missing)
        self.layer_tokens += layer == self.first_layer
        for pool in pools:
            with pool._bk_lock:
                for expert in originals:
                    pool._touch(expert)
        for expert in originals[: self.admit_routes]:
            if expert in pending or len(pending) >= self.layer_inflight:
                continue
            if all(expert in pool._slot_of for pool in pools):
                continue
            protected = set(originals) | pending
            slots = []
            evicted_any = False
            try:
                for pool in pools:
                    with pool._bk_lock:
                        if expert in pool._slot_of:
                            slots.append(None)
                            continue
                        # A concurrent demand publication may still read its active set.
                        slot, evicted = pool._choose_slot(protected | pool._demand_protect)
                        pool._expert_at[slot] = expert
                        slots.append(slot)
                        evicted_any = evicted_any or evicted
            except Exception:
                self._release(pools, expert, slots)
                continue
            pending.add(expert)
            self.admissions += 1
            self.reserved.append((layer, expert, tuple(slots), self.built if evicted_any else 0))

    def _start_retired(self) -> None:
        keep = []
        for layer, expert, slots, retire in self.reserved:
            if retire > self.completed:
                keep.append((layer, expert, slots, retire))
                continue
            pools = self.pools[layer]
            future = _EXECUTOR.submit(_load, pools, expert, slots)
            self.loading.append((layer, expert, slots, future))
        self.reserved = keep

    def _publish_finished(self) -> None:
        keep = []
        for layer, expert, slots, future in self.loading:
            if not future.done():
                keep.append((layer, expert, slots, future))
                continue
            pools = self.pools[layer]
            self.pending_experts[layer].discard(expert)
            if future.exception() is not None:
                self.failures += 1
                self._release(pools, expert, slots)
                continue
            for pool, slot in zip(pools, slots):
                if slot is None:
                    continue
                with pool._bk_lock:
                    pool._slot_of[expert] = slot
                    pool._slot_table_dirty = True
                    pool.total_loads += 1
            self.loads += 1
        self.loading = keep

    @staticmethod
    def _release(pools, expert, slots) -> None:
        for pool, slot in zip(pools, slots):
            if slot is None:
                continue
            with pool._bk_lock:
                if pool._expert_at[slot] == expert:
                    pool._expert_at[slot] = None

    def stats(self) -> dict:
        return {"built": self.built, "completed": self.completed, "admissions": self.admissions,
                "loads": self.loads, "failures": self.failures,
                "reserved": len(self.reserved), "loading": len(self.loading),
                "boundary_seconds": round(self.boundary_seconds, 3),
                "snapshot_seconds": round(self.snapshot_seconds, 3),
                "tokens": self.layer_tokens,
                "top1_missing": self.top1_missing, "top2_missing": self.top2_missing}


def _load(pools, expert, slots) -> None:
    for pool, slot in zip(pools, slots):
        if slot is not None:
            pool._load_expert(expert=expert, slot=slot)


def install_autonomous_residency(model) -> AutonomousResidency:
    residency = getattr(model, "_autonomous_residency", None)
    if residency is None:
        import os

        gated = os.environ.get("MOESPRESSO_QWEN4_AUTONOMOUS_GATED", "")
        residency = AutonomousResidency(
            model, admit_routes=int(os.environ.get("MOESPRESSO_QWEN4_AUTONOMOUS_ADMIT", "4")),
            gated_layers=frozenset(int(x) for x in gated.split(",") if x.strip()))
        object.__setattr__(model, "_autonomous_residency", residency)
    return residency


def autonomous_boundary(model, *, queued: int) -> None:
    residency = getattr(model, "_autonomous_residency", None)
    if residency is not None:
        residency.boundary(queued=queued)


def autonomous_finish(model) -> None:
    residency = getattr(model, "_autonomous_residency", None)
    if residency is not None:
        residency.finish()
