"""Stopping criteria, separate from the engine.

When to stop is a different question from how to sample, and the same criterion
applies across engines. Keeping it separate means `Nested(dlogz=...)` stays the
sampler's own convergence knob while wall-clock and diagnostic-based limits are
uniform.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Stopping:
    """Any criterion that trips ends the run; None disables that one."""
    max_seconds: float | None = None       # wall-clock watchdog
    max_evals: int | None = None           # likelihood evaluations
    ess_min: float | None = None           # stop once ESS exceeds this
    rhat_max: float | None = None          # ...and Rhat is below this
    logz_tol: float | None = None          # evidence-based (nested/PT)
    check_every: int = 500

    def to_wire(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


WALL_CLOCK_1H = Stopping(max_seconds=3600)
CONVERGED = Stopping(ess_min=400, rhat_max=1.01)
