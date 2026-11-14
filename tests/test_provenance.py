"""`JobResult.provenance` reads the summary's provenance block. No Julia."""
from __future__ import annotations

from astronereus._result import JobResult


def test_provenance_block():
    p = {"engine": "nested", "nereus_version": "0.8.10", "nereus_commit": None,
         "priors": {"gamma_HARPS": {"family": "fixed", "fixed": True, "value": 5.0},
                    "P_k1": {"family": "loguniform", "fixed": False, "lo": 12.0, "hi": 12.6}}}
    r = JobResult({"status": "ok", "provenance": p})
    assert r.provenance["engine"] == "nested"
    assert r.provenance["priors"]["gamma_HARPS"]["value"] == 5.0


def test_provenance_absent_in_old_summaries():
    assert JobResult({"status": "ok"}).provenance == {}
