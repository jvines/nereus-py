"""What goes on the wire under `engine=`. No Julia, no daemon, no bundle.

The bug this exists for: `_run` used to send `engines.DEFAULT.to_wire()` when
the caller named no engine, so every payload carried the literal name
"pt_emcee". Julia's `fit_*` picks the sampler BY SHAPE when `engine === nothing`
-- pt_emcee fixed-dim, transdim_pt_emcee when a `transdim` block is present
(Nereus.jl src/api.jl:697) -- and naming one defeats that. Worse, api.jl then
attaches `td` to the named sampler's options and `run_engine` rejects options
the sampler does not declare, so every `fit_*(..., transdim=...)` from Python
died with "engine pt_emcee does not accept td" before sampling started.
"""
from __future__ import annotations

import pytest

from astronereus import _fit, engines


class _Channel:
    def to_wire(self):
        return {"kind": "RV", "data": {}}


class _Session:
    """Captures the payload instead of talking to a daemon."""

    def __init__(self):
        self.payload = None

    def raw(self, op, payload, **kw):
        self.payload = payload
        return {"status": "ok", "summary": {}}


def _payload(**kw):
    s = _Session()
    _fit._run(s, "fit_rv", [_Channel()], planets=1, engine=kw.pop("engine", None),
              stopping=None, output_dir=None, **kw)
    return s.payload


def test_no_engine_means_no_engine_on_the_wire():
    assert _payload()["engine"] is None


def test_transdim_still_sends_no_engine():
    """The regression that mattered: asking for trans-dim must be enough."""
    p = _payload(transdim=True)
    assert p["engine"] is None
    assert p["transdim"] is True


def test_an_explicit_engine_is_sent_verbatim():
    p = _payload(engine=engines.PT(n_rounds=4, n_chains=6))
    assert p["engine"] == {"engine": "pt",
                           "options": {"n_rounds": 4, "n_chains": 6}}


@pytest.mark.parametrize("cls", [engines.PTEmcee, engines.TransdimPTEmcee])
def test_prune_stranded_is_reachable(cls):
    """Nereus v0.6.0's burn-in pruning is on by default; `False` is the
    documented way back to the old behaviour, and a field the client does not
    declare cannot be passed at all."""
    assert cls(prune_stranded=False).options() == {"prune_stranded": False}


@pytest.mark.parametrize("cls", [engines.PTEmcee, engines.TransdimPTEmcee,
                                 engines.PTWhitening])
def test_node_flip_is_reachable(cls):
    """Nereus v0.6.1's astrometric (Ω, ω) -> (Ω+π, ω+π) move is on by default
    at 0.1; `0` is the way back, and it is reachable from here only because
    all three PT ensembles declare the field."""
    assert cls(node_flip=0).options() == {"node_flip": 0}
