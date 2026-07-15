"""Chaos / fault injection (master plan §9, borrowed from Strands Evals):
deterministic, seeded tool failures for experiment mode, paired with the
recovery_after_error evaluator. Wrap your agent's tools before an eval
run — NEVER in production paths:

    chaos = ChaosInjector(fail_rate=0.3, seed=7)
    lookup = chaos.wrap("lookup_customer", real_lookup)
"""

from __future__ import annotations

import random
from typing import Callable


class ChaosToolError(RuntimeError):
    pass


class ChaosInjector:
    def __init__(self, fail_rate: float = 0.2, seed: int = 42) -> None:
        if not 0.0 <= fail_rate <= 1.0:
            raise ValueError("fail_rate must be in [0, 1]")
        self.fail_rate = fail_rate
        self._rng = random.Random(seed)
        self.injected: list[str] = []  # audit of which calls were failed

    def wrap(self, name: str, fn: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            if self._rng.random() < self.fail_rate:
                self.injected.append(name)
                raise ChaosToolError(f"chaos: injected failure in tool '{name}'")
            return fn(*args, **kwargs)

        wrapped.__name__ = getattr(fn, "__name__", name)
        return wrapped
