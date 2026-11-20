"""Deterministic 32-bit PRNG (Mulberry32) matching TypeScript src/engine/prng.ts."""

from __future__ import annotations

import math
from typing import List, TypeVar

T = TypeVar("T")


def _imul32(a: int, b: int) -> int:
    """Exact 32-bit signed integer multiplication wrapped to unsigned uint32."""
    a &= 0xFFFFFFFF
    b &= 0xFFFFFFFF
    if a & 0x80000000:
        a -= 0x100000000
    if b & 0x80000000:
        b -= 0x100000000
    return (a * b) & 0xFFFFFFFF


class Mulberry32:
    """Mulberry32 deterministic PRNG bit-compatible with TypeScript engine."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next_uint32(self) -> int:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = _imul32(self.state ^ (self.state >> 15), 1 | self.state)
        t = (t + _imul32(t ^ (t >> 7), 61 | t)) ^ t
        t &= 0xFFFFFFFF
        return (t ^ (t >> 14)) & 0xFFFFFFFF

    def random(self) -> float:
        return self.next_uint32() / 4294967296.0

    def randint(self, low: int, high_exclusive: int) -> int:
        return low + int(math.floor(self.random() * (high_exclusive - low)))

    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        u1 = max(1e-12, self.random())
        u2 = self.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z


def seeded_shuffle(items: List[T], rng: Mulberry32) -> List[T]:
    arr = list(items)
    for i in range(len(arr) - 1, 0, -1):
        j = rng.randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr
