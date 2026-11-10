"""Stopping criteria, separate from the engine.

When to stop is a different question from how to sample, and the same criterion
applies across engines. Keeping it separate means `Nested(dlogz=...)` stays the
sampler's own convergence knob while wall-clock and diagnostic-based limits are
uniform.

A field maps to an engine option; an engine that cannot honour a field you set
makes the fit raise (naming the engines that can), it is never ignored:

=============  ==========================================  ==============================
field          engine option                               engines
=============  ==========================================  ==============================
max_seconds    ``max_seconds``                             PTEmcee, TransdimPTEmcee
max_evals      ``max_evals``                               PTEmcee, TransdimPTEmcee
rhat_max       ``rhat_threshold`` + ``convergence_stop``   PTEmcee, TransdimPTEmcee
ess_min        ``tail_ess_threshold`` + ``convergence_stop`` PTEmcee, TransdimPTEmcee
check_every    ``diag_every`` (steps between checks)       PTEmcee, TransdimPTEmcee
logz_tol       ``dlogz`` (``dlogz_batch``, NestedDynamic)  Nested, NestedINS, Daedalus,
                                                           NestedDynamic
=============  ==========================================  ==============================

A PT run that stops this way ends at the last recorded step and finishes
normally (chains, plots, tables); ``r.summary["stopped"]`` says why. A limit
reached during burn-in, when there are no draws to keep, is an error. Giving a
field *and* the engine option it maps to, with different values, is an error.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Stopping:
    """Any criterion that trips ends the run; None disables that one."""
    max_seconds: float | None = None       # wall-clock limit on the run
    max_evals: int | None = None           # likelihood evaluations
    ess_min: float | None = None           # stop once the worst tail-ESS exceeds this
    rhat_max: float | None = None          # ...and the worst R-hat is below this
    logz_tol: float | None = None          # remaining-evidence tolerance (nested)
    check_every: int | None = None         # steps between convergence checks

    def to_wire(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


WALL_CLOCK_1H = Stopping(max_seconds=3600)
CONVERGED = Stopping(ess_min=400, rhat_max=1.01)
