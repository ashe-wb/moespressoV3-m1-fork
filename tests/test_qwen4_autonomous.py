"""Deferred residency never writes a slot that an unfinished token can read."""

import threading
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from moespresso.runtime.qwen4.autonomous import AutonomousResidency


class _Pool:
    def __init__(self, capacity, writes):
        self.capacity = capacity
        self.num_experts = 512
        self._bk_lock = threading.Lock()
        self._slot_of = {expert: expert for expert in range(capacity)}
        self._expert_at = list(range(capacity))
        self._freq = {}
        self._demand_protect = set()
        self._slot_table_dirty = True
        self.total_loads = 0
        self.writes = writes

    def _touch(self, expert):
        self._freq[expert] = self._freq.get(expert, 0) + 1

    def _choose_slot(self, protected):
        for slot, expert in enumerate(self._expert_at):
            if expert is None:
                return slot, False
        victim = min((e for e in self._slot_of if e not in protected),
                     key=lambda e: (self._freq.get(e, 0), e))
        slot = self._slot_of.pop(victim)
        self._expert_at[slot] = None
        self._slot_table_dirty = True
        return slot, True

    def _ensure_slot_table(self):
        table = np.full(512, 512, dtype=np.uint32)
        for expert, slot in self._slot_of.items():
            table[expert] = slot
        return mx.array(table)

    def _load_expert(self, *, expert, slot):
        self.writes.append((expert, slot))


def _model(writes):
    pools = tuple(_Pool(12, writes) for _ in range(3))
    router = SimpleNamespace(_cache_routing_provider=SimpleNamespace(pools=pools))
    layer = SimpleNamespace(mlp=SimpleNamespace(gate=router, experts=None))
    return SimpleNamespace(layers=[layer]), router, pools


def _mapped_slots(snapshot):
    return {int(s) for s in np.asarray(snapshot[0]) if s < 512}


def _drain(residency):
    for *_rest, future in residency.loading:
        future.result()


def test_reclaimed_slot_waits_for_every_earlier_build():
    writes = []
    model, router, pools = _model(writes)
    residency = AutonomousResidency(model, admit_routes=2)
    wanted = mx.array([40, 41, *range(8)], dtype=mx.uint32)

    first = residency.snapshot(router)
    residency.record(router, wanted)
    residency.boundary(queued=1)
    assert residency.admissions == 0

    second = residency.snapshot(router)
    residency.record(router, wanted)
    residency.boundary(queued=1)
    assert residency.admissions == 2
    assert writes == []
    victims = {slot for _layer, _expert, slots, _retire in residency.reserved for slot in slots}
    third = residency.snapshot(router)
    assert victims <= _mapped_slots(first) and victims <= _mapped_slots(second)
    assert not victims & _mapped_slots(third)

    residency.snapshot(router)
    residency.boundary(queued=1)
    _drain(residency)
    assert {slot for _expert, slot in writes} == victims
    residency.boundary(queued=1)
    assert all(40 in pool._slot_of and 41 in pool._slot_of for pool in pools)
    published = residency.snapshot(router)
    table = np.asarray(published[0])
    assert table[40] < 512 and table[41] < 512
