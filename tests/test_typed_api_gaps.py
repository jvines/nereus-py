"""The typed fit_* route reaches what a run_job config does. No Julia, no daemon:
this is what the client sends and refuses. The runtime half is
Nereus.jl/test/test_typed_api_gaps.jl and the end-to-end run in the changelog."""
from __future__ import annotations

import dataclasses
import inspect

import pytest

import astronereus
from astronereus import RV, TTV, Transit, _fit, engines, priors as pr
from astronereus import _features as F


class _Session:
    """Records the payload; answers each op with a canned summary."""

    def __init__(self, answers=None):
        self.calls = []
        self.answers = answers or {}

    @property
    def payload(self):
        return self.calls[-1][1]

    def raw(self, action, payload=None, **kw):
        self.calls.append((action, payload))
        ans = self.answers.get(action)
        if callable(ans):
            ans = ans(payload)
        return ans if ans is not None else {"status": "ok", "summary": {}}


_RV = {"A": {"t": [0.0, 1.0, 2.0], "rv": [1.0, 2.0, 1.5], "rv_err": [1.0, 1.0, 1.0]}}
_PH = {"T": {"t": [0.0, 1.0], "flux": [1.0, 1.0], "flux_err": [1e-3, 1e-3]}}
_TT = [{"planet": 1, "values": {"epoch": [0, 1], "tc": [1.0, 2.0], "tc_err": [1e-3, 1e-3]}}]


# --- 1. engine fields -----------------------------------------------------------

_ADDED = {
    "PTEmcee": {"checkpoint", "checkpoint_interval", "resume", "burnin_hold",
                "burnin_hold_frac", "progress_log", "swap_min", "evidence_postfit"},
    "TransdimPTEmcee": {"swap_min", "burnin_hold", "burnin_hold_frac",
                        "checkpoint", "checkpoint_interval", "resume"},
    "PTWhitening": {"burnin_hold", "burnin_hold_frac", "checkpoint",
                    "checkpoint_interval", "resume", "swap_min"},
    "PT": {"checkpoint", "checkpoint_interval", "resume", "swap_min", "n_warmup_rounds"},
    "PTHMC": {"checkpoint", "checkpoint_interval", "resume", "swap_min", "beta_min",
              "bridge_n", "evidence_tol", "laplace_switch_tol", "n_pilot", "n_scouts"},
    "Ensemble": {"checkpoint", "checkpoint_interval", "resume", "n_scouts",
                 "prune_stranded", "show_progress"},
    "ESS": {"checkpoint", "checkpoint_interval", "resume"},
    "MoMS": {"checkpoint", "checkpoint_interval", "resume", "n_scouts"},
    "RJMCMC": {"checkpoint", "checkpoint_interval", "resume", "n_scouts"},
    "NUTS": {"checkpoint", "checkpoint_interval", "resume", "n_scouts", "seed"},
    "Nested": {"seam_pilot", "show_progress"},
    "NestedINS": {"show_progress"},
    "NestedDynamic": {"show_progress"},
    "Daedalus": {"decorrelation"},
    "PA": {"decorrelation", "resample_ess"},
}


@pytest.mark.parametrize("cls,names", sorted(_ADDED.items()))
def test_engines_carry_the_julia_keywords(cls, names):
    fields = {f.name for f in dataclasses.fields(getattr(engines, cls))}
    assert names <= fields, sorted(names - fields)


def test_new_engine_fields_default_to_not_sent():
    assert engines.PTEmcee().options() == {}
    assert engines.TransdimPTEmcee().to_wire() == {"engine": "transdim_pt_emcee",
                                                   "options": {}}
    e = engines.PTEmcee(checkpoint="c.jls", resume=True, burnin_hold={"sigma_iad_G": 0.0},
                        burnin_hold_frac=0.4, progress_log="p.jsonl", swap_min=0.02,
                        evidence_postfit=True, checkpoint_interval=60.0)
    assert e.options() == {"checkpoint": "c.jls", "resume": True,
                           "burnin_hold": {"sigma_iad_G": 0.0},
                           "burnin_hold_frac": 0.4, "progress_log": "p.jsonl",
                           "swap_min": 0.02, "evidence_postfit": True,
                           "checkpoint_interval": 60.0}
    assert engines.PTWhitening(burnin_hold=True).options() == {"burnin_hold": True}


# --- 2. ttv_epochs, 6. noise / modes / indicators on the typed route ----------

def test_model_keywords_are_whitelisted():
    for k in ("ttv_epochs", "noise_models", "noise_menu", "planet_modes",
              "indicator_cols"):
        assert k in _fit._MODEL_KW


def test_fit_joint_passes_ttv_epochs_and_model_blocks():
    s = _Session()
    nm = [{"kind": "CeleriteRotation", "channel": "rv", "instruments": ["A"]}]
    _fit.fit_joint(RV(data=_RV), Transit(data=_PH), planets=1, session=s,
                   ttv_epochs={1: [0, 1]}, noise_models=nm, planet_modes=["RVPM_GD"],
                   noise_menu={"indicators": ["bis"]}, indicator_cols=["bis"])
    p = s.payload
    assert p["ttv_epochs"] == {1: [0, 1]}
    assert p["noise_models"] == nm and p["planet_modes"] == ["RVPM_GD"]
    assert p["noise_menu"] == {"indicators": ["bis"]} and p["indicator_cols"] == ["bis"]


def test_fit_rv_and_fit_transit_take_the_blocks():
    s = _Session()
    _fit.fit_rv(_RV, session=s, noise_models=[{"kind": "ErrorScale"}])
    assert s.payload["noise_models"] == [{"kind": "ErrorScale"}]
    _fit.fit_transit(_PH, session=s, planet_modes=["PM_GD"])
    assert s.payload["planet_modes"] == ["PM_GD"]
    _fit.fit_transit(_PH, gravity_darkening=True, session=s)
    assert s.payload["channels"][0]["gravity_darkening"] is True
    with pytest.raises(ValueError):
        Transit(data=_PH, gravity_darkening="yes")
    with pytest.raises(TypeError, match="unknown keyword"):
        _fit.fit_rv(_RV, session=s, bogus=1)


def test_rv_indicators_reach_the_wire():
    ind = {"A": {"bis": [0.1, 0.2, 0.3], "bis_err": [0.1, 0.1, 0.1]}}
    ch = RV(data=_RV, indicators=ind, indicator_cols=["bis"])
    w = ch.to_wire()
    assert w["indicators"] == ind and w["indicator_cols"] == ["bis"]
    assert RV(data=_RV, indicator_cols="bis").indicator_cols == ["bis"]
    with pytest.raises(ValueError, match="indicators"):
        RV(data=_RV, indicators={"bis": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not among"):
        RV(data=_RV, indicators={"Z": {"bis": [1.0]}})


# --- 3. fit_ttv phot / rv / nbody -----------------------------------------------

def test_fit_ttv_builds_the_three_channels():
    s = _Session()
    _fit.fit_ttv(_TT, phot=_PH, rv=RV(data=_RV), session=s, planets=1,
                 ttv_n_transits=3)
    srcs = [c["source"] for c in s.payload["channels"]]
    assert srcs == ["TTV", "PM", "RV"]
    assert s.payload["ttv_n_transits"] == 3 and "nbody" not in s.payload
    _fit.fit_ttv(None, phot=_PH, rv=_RV, nbody=True, session=s, planets=1)
    assert [c["source"] for c in s.payload["channels"]] == ["PM", "RV"]
    assert s.payload["nbody"] is True
    _fit.fit_ttv(_TT, session=s, planets=1)           # as before
    assert [c["source"] for c in s.payload["channels"]] == ["TTV"]


def test_fit_ttv_refusals():
    s = _Session()
    with pytest.raises(ValueError, match="transit_times"):
        _fit.fit_ttv(session=s)
    with pytest.raises(ValueError, match="needs rv"):
        _fit.fit_ttv(_TT, nbody=True, session=s)
    with pytest.raises(ValueError, match="fits none"):
        _fit.fit_ttv(_TT, rv=_RV, nbody=True, ttv_epochs={1: [0]}, session=s)
    with pytest.raises(TypeError):
        TTV(transit_times=_TT, nbody=True)


# --- 4. config_overrides on every fit_* ------------------------------------------

_CO = {"model": {"stability": "amd"}, "sampler": {"kwargs": {"n_steps": 5}}}


@pytest.mark.parametrize("call", [
    lambda s: _fit.fit_rv(_RV, session=s, config_overrides=_CO),
    lambda s: _fit.fit_transit(_PH, session=s, config_overrides=_CO),
    lambda s: _fit.fit_astrometry(iad={"catalogue": "hipparcos", "hip": 1},
                                  session=s, config_overrides=_CO),
    lambda s: _fit.fit_joint(RV(data=_RV), Transit(data=_PH), session=s,
                             config_overrides=_CO),
    lambda s: _fit.fit_rm(_RV, phot=_PH, session=s, config_overrides=_CO),
    lambda s: _fit.fit_ttv(_TT, session=s, config_overrides=_CO),
    lambda s: _fit.fit_binary(_RV, session=s, config_overrides=_CO),
])
def test_config_overrides_on_every_fit(call):
    s = _Session()
    call(s)
    assert s.payload["config_overrides"] == _CO


def test_config_overrides_is_a_documented_keyword_of_all_of_them():
    for name in ("fit_rv", "fit_transit", "fit_astrometry", "fit_rm", "fit_ttv",
                 "fit_binary", "fit_joint"):
        fn = getattr(_fit, name)
        assert "config_overrides" in inspect.signature(fn).parameters, name
        assert "config_overrides" in fn.__doc__, name
    with pytest.raises(TypeError, match="job config"):
        _fit.fit_rv(_RV, session=_Session(), config_overrides=[1])
    s = _Session()
    _fit.fit_rv(_RV, session=s)
    assert "config_overrides" not in s.payload


def test_config_route_fits_take_the_blocks():
    s = _Session()
    _fit.fit_rm(_RV, phot=_PH, session=s, planet_modes=["RVPM_RM_GD"],
                noise_menu={"indicators": ["bis"]}, indicator_cols=["bis"])
    assert s.payload["planet_modes"] == ["RVPM_RM_GD"]
    assert s.payload["indicator_cols"] == ["bis"]
    _fit.fit_binary(_RV, session=s, noise_models=[{"kind": "ErrorScale"}])
    assert s.payload["noise_models"] == [{"kind": "ErrorScale"}]


# --- 5. operations on a result ---------------------------------------------------

def _fit_result(session, **raw):
    res = _fit.fit_rv(_RV, session=session, output_dir="/o/fit", planets=1)
    res.raw.update({"output_dir": "/o/fit", **raw})
    return res


def test_result_blocks_are_returned_without_a_daemon_round_trip():
    s = _Session()
    res = _fit_result(s, loo={"elpd_loo": -3.0}, ppc={"chi2_N": 1.0},
                      fit_health={"overall": "ok"}, log_z=-10.0,
                      evidence={"reported": "bridge"},
                      detection_limits={"n_bins": 4})
    n = len(s.calls)
    sess = astronereus.Session.__new__(astronereus.Session)
    sess.raw = s.raw
    g = F._Group(lambda: sess, "diagnostics", F.DIAGNOSTICS)
    assert g.loo(result=res) == {"elpd_loo": -3.0}
    assert g.ppc(result=res) == {"chi2_N": 1.0}
    assert g.fit_health(result=res) == {"overall": "ok"}
    assert F.evidence(sess, result=res) == {"log_z": -10.0,
                                            "evidence": {"reported": "bridge"}}
    assert F.detection_limits(sess, result=res) == {"n_bins": 4}
    assert len(s.calls) == n


def test_recompute_replays_the_request():
    s = _Session({"diagnostics.loo": {"elpd_loo": -1.0}})
    res = _fit_result(s, loo={"elpd_loo": -3.0})
    sess = astronereus.Session.__new__(astronereus.Session)
    sess.raw = s.raw
    g = F._Group(lambda: sess, "diagnostics", F.DIAGNOSTICS)
    assert g.loo(result=res, recompute=True) == {"elpd_loo": -1.0}
    act, p = s.calls[-1]
    assert act == "diagnostics.loo"
    assert p["output_dir"] == "/o/fit" and p["channels"][0]["source"] == "RV"
    assert p["planets"] == 1 and "op" not in p and "plots" not in p
    # a skipped block is recomputed, not returned
    res.raw["loo"] = {"status": "skipped"}
    assert g.loo(result=res)["elpd_loo"] == -1.0
    # chains-only operations send the directory
    g.label_switching(result=res)
    assert s.calls[-1][0] == "diagnostics.label_switching"
    assert s.calls[-1][1]["output_dir"] == "/o/fit"


def test_result_without_a_request_or_directory_is_refused():
    sess = _Session()
    bare = astronereus.JobResult({"status": "ok", "output_dir": "/x"})
    with pytest.raises(ValueError, match="no record of the fit's channels"):
        F.run_op(sess, "diagnostics.loo", result=bare)
    with pytest.raises(ValueError, match="output_dir"):
        F.run_op(sess, "diagnostics.loo", result=astronereus.JobResult({"status": "ok"}))
    cfg = _fit.fit_ttv(_TT, session=sess)
    cfg.raw["output_dir"] = "/y"
    with pytest.raises(ValueError, match="job config"):
        F.run_op(sess, "diagnostics.ppc", result=cfg)


def test_not_implemented_stubs_are_gone():
    # the actions the client names all map to a real operation
    for n in ("loo", "ppc", "label_switching", "fit_health"):
        assert n in F.DIAGNOSTICS
    assert "lc_periodogram" in F.DETECT
    for n in ("evidence", "detection_limits", "select_planets", "select_noise",
              "pre_white"):
        assert callable(getattr(astronereus.Session, n))


def test_select_planets_runs_each_candidate_and_tabulates():
    def answer(p):
        n = p["planets"]
        return {"status": "ok", "log_z": {0: -50.0, 1: -20.0, 2: -22.0}[n],
                "evidence": {"reported": "bridge"}, "output_dir": p["output_dir"]}
    s = _Session({"fit_rv": answer})
    sess = astronereus.Session.__new__(astronereus.Session)
    sess.raw = s.raw
    out = F.select_planets(sess, RV(data=_RV), candidates=(0, 1, 2),
                           output_dir="/o/sel")
    assert out["best"] == 1
    t = {r["n_planets"]: r for r in out["table"]}
    assert t[1]["delta_log_z"] == 0.0 and t[2]["delta_log_z"] == -2.0
    assert t[2]["output_dir"] == "/o/sel/planets_2" and t[0]["reported"] == "bridge"
    assert set(out["results"]) == {0, 1, 2}
    with pytest.raises(TypeError):
        F.select_planets(sess)
    with pytest.raises(TypeError, match="not both"):
        F.select_planets(sess, RV(data=_RV), result=object())


def test_select_planets_on_a_transdim_result_reads_its_chains():
    s = _Session({"select_planets": {"best": 2, "table": []}})
    res = _fit_result(s)
    sess = astronereus.Session.__new__(astronereus.Session)
    sess.raw = s.raw
    assert F.select_planets(sess, result=res)["best"] == 2
    assert s.calls[-1][1]["output_dir"] == "/o/fit"


def test_select_noise_candidates():
    def answer(p):
        nm = p.get("noise_models")
        return {"status": "ok", "log_z": -5.0 if nm is None else -3.0,
                "output_dir": p["output_dir"]}
    s = _Session({"fit_rv": answer})
    sess = astronereus.Session.__new__(astronereus.Session)
    sess.raw = s.raw
    out = F.select_noise(sess, RV(data=_RV), output_dir="/o/n", candidates={
        "white": None, "rot": [{"kind": "CeleriteRotation"}],
        "menu": {"noise_menu": {"indicators": []}}})
    assert out["best"] == "rot"
    assert {r["model"] for r in out["table"]} == {"white", "rot", "menu"}
    assert s.calls[1][1]["noise_models"] == [{"kind": "CeleriteRotation"}]
    assert s.calls[2][1]["noise_menu"] == {"indicators": []}
    with pytest.raises(ValueError, match="unknown"):
        F.select_noise(sess, RV(data=_RV), candidates={"x": {"bogus": 1}})


def test_pre_white_payload():
    s = _Session({"pre_white": {"n_signals": 0}})
    sess = astronereus.Session.__new__(astronereus.Session)
    sess.raw = s.raw
    F.pre_white(sess, rv={"A": {"t": [2.0, 1.0], "rv": [3.0, 1.0], "rv_err": [1.0, 2.0]},
                          "B": {"t": [0.0], "rv": [5.0], "rv_err": [1.0]}},
                max_signals=3, period_max=40.0)
    p = s.calls[-1][1]
    assert p["t"] == [0.0, 1.0, 2.0]                       # time-sorted
    assert p["y"] == [0.0, -1.0, 1.0]                      # per-instrument mean removed
    assert p["y_err"] == [1.0, 2.0, 1.0] and p["max_signals"] == 3
    assert p["period_max"] == 40.0
    F.pre_white(sess, t=[0, 1], y=[1, 2], y_err=[1, 1])
    with pytest.raises(ValueError, match="needs t"):
        F.pre_white(sess, t=[0, 1])


# --- 7. priors ----------------------------------------------------------------------

def test_prior_constructors_make_the_wire_dicts():
    assert pr.modjeffreys(1.0, 100.0) == {"type": "modjeffreys", "knee": 1.0, "upper": 100.0}
    assert pr.beta(2, 5) == {"type": "beta", "a": 2.0, "b": 5.0, "lo": 0.0, "hi": 1.0}
    assert pr.beta(2, 5, lo=0.1, hi=0.9)["hi"] == 0.9
    assert pr.normal(0.0, 0.1) == {"type": "normal", "mu": 0.0, "sigma": 0.1}
    assert pr.normal(0.0, 0.1, lo=-0.5, hi=0.5) == {"type": "normal", "mu": 0.0,
                                                    "sigma": 0.1, "lo": -0.5, "hi": 0.5}
    assert pr.normal(1.0, 1.0, hi=2.0)["hi"] == 2.0 and "lo" not in pr.normal(1.0, 1.0, hi=2.0)
    assert pr.uniform(0, 1)["type"] == "uniform" and pr.sine() == {"type": "sine"}
    assert pr.loguniform(1, 10)["type"] == "loguniform"
    assert pr.fixed(3)["value"] == 3.0
    assert pr.wrapped_uniform(-3.14, 3.14)["type"] == "wrappeduniform"


@pytest.mark.parametrize("bad", [
    lambda: pr.modjeffreys(0.0, 1.0), lambda: pr.beta(-1, 2), lambda: pr.beta(1, 2, lo=1, hi=0),
    lambda: pr.normal(0, 0.0), lambda: pr.normal(0, 1, lo=2, hi=1),
    lambda: pr.uniform(1, 1), lambda: pr.loguniform(0, 1), lambda: pr.fixed(float("nan")),
    lambda: pr.uniform(True, 2)])
def test_prior_constructors_check_their_arguments(bad):
    with pytest.raises(ValueError):
        bad()


def test_priors_cross_the_wire_untouched():
    s = _Session()
    p = {"K_k1": pr.modjeffreys(1.0, 100.0), "q1": pr.beta(2, 5),
         "secosw_k1": pr.normal(0.0, 0.1, lo=-0.5, hi=0.5)}
    _fit.fit_rv(_RV, session=s, priors=p)
    assert s.payload["priors"] == p
