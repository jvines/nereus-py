"""The stellar keywords the typed API accepts are the ones Nereus reads.

Needs a Nereus.jl checkout (NEREUS_PROJECT), no daemon, no fit.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from astronereus import _fit


def test_star_keywords_are_nereus_star_derived_keys():
    project = os.environ.get("NEREUS_PROJECT")
    if not project:
        pytest.skip("set NEREUS_PROJECT to a Nereus.jl checkout")
    julia = os.environ.get("NEREUS_JULIA", "julia")
    out = subprocess.run(
        [julia, f"--project={project}", "--startup-file=no", "-e",
         'using Nereus; print("[", join(string.(\'"\', Nereus._STAR_DERIVED_KEYS, \'"\'), ","), "]")'],
        capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-1500:]
    assert set(json.loads(out.stdout)) == set(_fit._STAR_KW)
