"""Every engine field must be a keyword the Julia sampler actually declares.

`run_engine` (Nereus.jl src/api.jl:85-89) validates option names against
`Base.kwarg_decl` and throws on the first unknown one, so a field that Julia
does not declare is not a warning — it is a dead engine.

This is the test that would have caught the 0.2.0 breakage: `PTHMC(PT)`
inherited nine PT fields and `pt_hmc` shares exactly one keyword with `pt`.
It needs Julia but no daemon, no bundle and no fit, so it is cheap enough to
run on every change.

    NEREUS_PROJECT=/path/to/Nereus.jl python -m pytest tests/test_engine_fields.py -q
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess

import pytest

from astronereus import engines

ENGINE_CLASSES = [
    engines.PT, engines.PTHMC, engines.PTWhitening,
    engines.PTEmcee, engines.Nested, engines.NestedINS, engines.NestedDynamic,
    engines.MoMS, engines.Daedalus, engines.RJMCMC, engines.TransdimPTEmcee,
    engines.NUTS, engines.MAP, engines.SMC, engines.Ensemble, engines.ESS,
    engines.PA, engines.OFTI,
]

_INTROSPECT = r'''
using Nereus, JSON
d = Dict{String,Vector{String}}()
for (name, fn) in Nereus.ENGINES
    ks = String.(Base.kwarg_decl(first(methods(fn))))
    d[name] = filter(k -> !endswith(k, "..."), ks)
end
print(JSON.json(d))
'''


@pytest.fixture(scope="module")
def julia_kwargs() -> dict[str, list[str]]:
    julia = os.environ.get("NEREUS_JULIA", "julia")
    project = os.environ.get("NEREUS_PROJECT")
    if not project:
        pytest.skip("set NEREUS_PROJECT to a Nereus.jl checkout")
    out = subprocess.run(
        [julia, f"--project={project}", "--startup-file=no", "-e", _INTROSPECT],
        capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        pytest.fail(f"julia introspection failed:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout)


@pytest.mark.parametrize("cls", ENGINE_CLASSES, ids=lambda c: c.__name__)
def test_engine_name_exists(cls, julia_kwargs):
    assert cls.name in julia_kwargs, (
        f"{cls.__name__}.name = {cls.name!r} is not in Nereus.ENGINES. "
        f"Available: {sorted(julia_kwargs)}")


@pytest.mark.parametrize("cls", ENGINE_CLASSES, ids=lambda c: c.__name__)
def test_every_field_is_a_julia_keyword(cls, julia_kwargs):
    accepted = set(julia_kwargs.get(cls.name, ()))
    ours = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(ours - accepted)
    assert not unknown, (
        f"{cls.__name__} sends {unknown} which {cls.name} does not accept. "
        f"It accepts: {sorted(accepted)}")


@pytest.mark.parametrize("cls", ENGINE_CLASSES, ids=lambda c: c.__name__)
def test_unset_fields_are_not_sent(cls):
    """Defaults live in Julia, not here. A default-constructed engine must send
    an empty option set, so Julia's own default governs and cannot drift."""
    assert cls().options() == {}


def test_default_engine_is_usable():
    wire = engines.DEFAULT.to_wire()
    assert wire["engine"] == "pt"
    assert wire["options"] == {"n_rounds": 12, "n_chains": 8}
