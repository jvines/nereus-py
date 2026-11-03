"""`JobResult` must read parameters from either summary shape. No Julia.

`fit_*` summaries carry the sampled parameters under "params"; `run_job`
returns them under "fitted", wrapped by the science contract as
{"conditioning": ..., "parameters": {name: {...}}}. Reading only "params" made
`session.run_job(cfg).params` an empty table, and `table()` with it.
"""
from __future__ import annotations

from astronereus._result import JobResult

K = {"value": 350.0, "err_lo": 40.0, "err_hi": 37.0, "unit": "m/s"}
P = {"value": 0.88205841, "err_lo": 1.7e-7, "err_hi": 1.5e-7, "unit": "d"}


def test_run_job_shape_is_read_from_fitted():
    r = JobResult({"status": "ok", "fitted": {
        "conditioning": {}, "parameters": {"K_k1": K, "P_k1": P}}})
    assert len(r.params) == 2
    assert float(r.params["K_k1"]) == 350.0
    assert "K_k1" in str(r.table())


def test_fit_shape_still_read_from_params():
    r = JobResult({"status": "ok", "params": {"K_k1": {
        "median": 350.0, "lo16": 310.0, "hi84": 387.0}}})
    assert len(r.params) == 1
    assert float(r.params["K_k1"]) == 350.0


def test_params_wins_when_both_are_present():
    r = JobResult({"status": "ok",
                   "params": {"K_k1": {"median": 1.0, "lo16": 0.9, "hi84": 1.1}},
                   "fitted": {"parameters": {"K_k1": K, "P_k1": P}}})
    assert len(r.params) == 1


def test_neither_gives_an_empty_table():
    assert len(JobResult({"status": "ok"}).params) == 0
