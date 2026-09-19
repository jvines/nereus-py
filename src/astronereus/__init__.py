"""Nereus — Nested-Evidence Recovery of Exoplanets by Unified Sampling.

Python frontend. Julia runs out-of-process in a warm daemon; you never need
Julia installed and nothing is compiled at install time.

    import astronereus
    astronereus.install(url=...)              # once: fetch the runtime bundle

    summary = nereus.run_job(cfg)        # one-shot

    with astronereus.session() as s:          # or reuse one warm daemon
        a = s.run_job(cfg_a)
        pg = s.detect.rv_periodogram(t=t, rv=rv, rv_err=err)
"""
from ._runtime import (JULIA_VERSION, CPU_TARGETS, BundleError, install,
                       runtime_parts,
                       find_julia, julia_home,
                       is_installed, platform_tag, runtime_dir, cache_root,
                       julia_env)
from ._daemon import JuliaDaemon, DaemonError
from ._api import Session, session, run_job, ping
from ._fit import (fit_rv, fit_transit, fit_astrometry, fit_rm,
                   fit_tomography, fit_ttv, fit_binary, fit_joint)
from .channels import RV, Transit, Astrometry, RM, Night, TTV, SB2
from .stopping import Stopping
from . import engines, channels
from ._result import JobResult, JobFailed, Figures

__all__ = ["JULIA_VERSION", "CPU_TARGETS", "BundleError", "DaemonError", "runtime_parts",
           "JuliaDaemon", "Session", "install", "is_installed", "platform_tag",
           "runtime_dir", "cache_root", "julia_env", "find_julia", "julia_home",
           "daemon", "session", "run_job", "ping",
           "JobResult", "JobFailed", "Figures",
           "fit_rv", "fit_transit", "fit_astrometry", "fit_rm",
           "fit_tomography", "fit_ttv", "fit_binary", "fit_joint",
           "RV", "Transit", "Astrometry", "RM", "Night", "TTV", "SB2",
           "Stopping", "engines", "channels"]
__version__ = "0.2.10"


def daemon(**kw) -> JuliaDaemon:
    """A warm Julia daemon. Use as a context manager."""
    return JuliaDaemon(**kw)
