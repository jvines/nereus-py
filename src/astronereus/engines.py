"""Sampling engines, one class per Julia sampler.

An engine carries the hyperparameters of ONE sampler. This exists because the
18 samplers take genuinely different arguments — `Nested` has `dlogz` and
`n_live`, `PTEmcee` has `n_temps`/`n_walkers`/`n_burnin`, `MAP` has `n_starts`
— and a single `fit(..., **kwargs)` would silently swallow the twelve that
don't apply to whichever sampler you picked.

EVERY FIELD DEFAULTS TO None, AND `options()` DROPS None. That is deliberate:
an unset field is not sent, so the Julia definition supplies the default. The
previous version restated Julia's defaults here and they drifted. Nothing is
duplicated now, so nothing can drift.

NO CLASS INHERITS ANOTHER'S FIELDS. `run_engine` (src/api.jl:85-89) validates
every option name against `Base.kwarg_decl` of the sampler and throws on the
first unknown one. `pt_hmc` shares no keyword but `seed` with `pt`, so a
`PTHMC(PT)` sent eight arguments `pt_hmc` rejects. The field lists below were
read out of Julia with `Base.kwarg_decl`; keep them that way.

Trans-dim: `td` is accepted by pt/moms/daedalus/rjmcmc/transdim_ptemcee
but is a Julia `TransDimConfig`, not JSON. It is deliberately not exposed —
trans-dim is selected by the `planets` argument to the fit_* entry points.

    fit_rv(rv, planets=2, engine=engines.PT(n_rounds=12, n_chains=8))
    fit_rv(rv, planets=2, engine=engines.Nested(n_live=500, dlogz=0.3))
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal, Sequence


class Engine:
    """Base. `name` is the Julia sampler; `options()` its keyword arguments."""

    name: str = ""

    def options(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_wire(self) -> dict[str, Any]:
        return {"engine": self.name, "options": self.options()}


# --- parallel tempering ------------------------------------------------------

@dataclass
class PT(Engine):
    """Parallel tempering. The default for fixed-dim evidence.

    `n_rounds` is a doubling schedule: total scans = 2^n_rounds, so 12 is 4096
    and 15 is 32768. Cost roughly doubles per round.
    """
    name = "pt"
    n_rounds: int | None = None
    n_chains: int | None = None
    seed: int | None = None
    within_model: Literal["slice", "rwm"] | None = None
    init_strategy: Literal["prior", "pathfinder"] | None = None
    n_pathfinder_runs: int | None = None
    n_pathfinder_draws: int | None = None
    early_stop_thresh: float | None = None
    early_stop_min_rounds: int | None = None
    show_report: bool | None = None


@dataclass
class PTHMC(Engine):
    """NUTS-within-PT. Fixed-dim, differentiable models only."""
    name = "pt_hmc"
    n_temps: int | None = None
    n_walkers_per_temp: int | None = None
    n_sweeps: int | None = None
    n_warmup: int | None = None
    swap_interval: int | None = None
    target_accept: float | None = None
    betas: Sequence[float] | None = None
    adapt_ladder: bool | None = None
    warm_start: bool | None = None
    seed: int | None = None
    progress: bool | None = None


@dataclass
class PTWhitening(Engine):
    """Normalising-flow-coupled PT; better swap acceptance on multimodal posteriors."""
    name = "pt_whitening"
    n_temps: int | None = None
    n_walkers: int | None = None
    n_steps: int | None = None
    n_burnin: int | None = None
    thin: int | None = None
    stretch_a: float | None = None
    proposal_scale: float | None = None
    warmup_swaps: int | None = None
    whiten_window: int | None = None
    whiten_refresh: int | None = None
    betas: Sequence[float] | None = None
    init_strategy: Literal["pathfinder", "prior"] | None = None
    seed: int | None = None
    show_progress: bool | None = None


@dataclass
class PTEmcee(Engine):
    """Affine-invariant parallel-tempered ensemble. Robust on multimodal RV.

    NOTE `init_strategy="prior"` matters for weak signals — prior-distributed
    walkers across chains and temperatures is what recovers low-K planets.
    """
    name = "ptemcee"
    n_temps: int | None = None
    n_walkers: int | None = None
    n_steps: int | None = None
    n_burnin: int | None = None
    min_steps: int | None = None
    stretch_a: float | None = None
    init_strategy: Literal["prior", "ball", "map"] | None = None
    thin: int | None = None
    seed: int | None = None
    convergence_stop: bool | None = None
    betas: Sequence[float] | None = None
    beta_min: float | None = None
    adapt_ladder: bool | None = None
    science_params: Sequence[str] | None = None
    rhat_threshold: float | None = None
    tail_ess_threshold: float | None = None
    n_converged_checks: int | None = None
    diag_every: int | None = None
    bridge_headline: bool | None = None
    bridge_n: int | None = None
    untemper_transit: bool | None = None
    show_progress: bool | None = None


# --- nested sampling ---------------------------------------------------------

@dataclass
class Nested(Engine):
    """Nested sampling. Gives log Z directly.

    `proposal="rslice"` under-mixes on thin curved ridges (the e-w-Mo
    degeneracy in eccentric RV); prefer PT + thermodynamic integration there.
    """
    name = "nested"
    n_live: int | None = None
    bounds: Literal["multi", "single", "mlfriends", "none"] | None = None
    proposal: Literal["rwalk", "rstagger", "slice", "rslice",
                      "unif", "hslice"] | None = None
    dlogz: float | None = None
    enlarge: float | None = None
    walk_scale: float | None = None
    n_walks: int | None = None
    slices: int | None = None
    batch_size: int | None = None
    parallel: bool | None = None
    seed: int | None = None


@dataclass
class NestedINS(Engine):
    """Importance nested sampling. Narrower proposal menu than `Nested`."""
    name = "nested_ins"
    n_live: int | None = None
    bounds: Literal["multi", "single", "mlfriends", "none"] | None = None
    proposal: Literal["rwalk", "slice", "unif"] | None = None
    dlogz: float | None = None
    enlarge: float | None = None
    walk_scale: float | None = None
    n_walks: int | None = None
    max_iter: int | None = None
    min_X_shrinkage: float | None = None
    bound_update_interval: int | None = None
    seed: int | None = None
    verbose: bool | None = None


@dataclass
class NestedDynamic(Engine):
    """Dynamic nested sampling — reallocates live points toward the posterior.

    Live points are split init/batch, so there is no single `n_live`, and the
    stopping tolerance is likewise split into `dlogz_init`/`dlogz_batch`.
    """
    name = "nested_dynamic"
    n_live_init: int | None = None
    n_live_batch: int | None = None
    dlogz_init: float | None = None
    dlogz_batch: float | None = None
    bounds: Literal["multi", "single"] | None = None
    proposal: Literal["rwalk", "slice", "unif"] | None = None
    enlarge: float | None = None
    walk_scale: float | None = None
    n_walks: int | None = None
    max_iter: int | None = None
    maxfrac: float | None = None
    pfrac: float | None = None
    pad: int | None = None
    bound_update_interval: int | None = None
    seed: int | None = None
    verbose: bool | None = None


# --- trans-dimensional -------------------------------------------------------

@dataclass
class MoMS(Engine):
    """Model-mixing variable selection. Trans-dim WITHOUT reversible jump.

    `informed_birth_fraction` around 0.7 is what makes it recover planets on
    real targets; the Julia default 0.0 is deliberately conservative.
    """
    name = "moms"
    n_samples: int | None = None
    n_warmup: int | None = None
    n_chains: int | None = None
    init_scale: float | None = None
    target_birth_accept: float | None = None
    inclusion_prior: float | None = None
    informed_birth_fraction: float | None = None
    within_model: Literal["slice", "rwm"] | None = None
    progress_every: int | None = None
    seed: int | None = None
    show_progress: bool | None = None


@dataclass
class Daedalus(Engine):
    """MoMS + nested sampling: trans-dim posterior AND evidence in one run.

    A nested run, not an MCMC one: it has `n_live`/`dlogz`, not
    `n_samples`/`n_warmup`.
    """
    name = "daedalus"
    n_live: int | None = None
    dlogz: float | None = None
    n_mcmc: int | None = None
    max_iter: int | None = None
    batch_size: int | None = None
    init_scale: float | None = None
    inclusion_prior: float | None = None
    informed_birth_fraction: float | None = None
    warm_start_points: Any = None
    progress_every: int | None = None
    seed: int | None = None
    show_progress: bool | None = None


@dataclass
class RJMCMC(Engine):
    """Reversible-jump MCMC over planet count."""
    name = "rjmcmc"
    n_samples: int | None = None
    n_warmup: int | None = None
    n_chains: int | None = None
    initial_scale: float | None = None
    target_accept: float | None = None
    within_model: Literal["slice", "rwm"] | None = None
    noise_swap_rate: float | None = None
    progress_every: int | None = None
    seed: int | None = None
    show_progress: bool | None = None


@dataclass
class TransdimPTEmcee(Engine):
    """Trans-dim parallel-tempered ensemble. The blind-search workhorse."""
    name = "transdim_ptemcee"
    n_temps: int | None = None
    n_walkers: int | None = None
    n_steps: int | None = None
    n_burnin: int | None = None
    thin: int | None = None
    stretch_a: float | None = None
    betas: Sequence[float] | None = None
    beta_min: float | None = None
    adapt_ladder: bool | None = None
    inclusion_prior: float | None = None
    informed_birth_fraction: float | None = None
    moms_init_scale: float | None = None
    target_birth_accept: float | None = None
    n_birth_tries: int | None = None
    n_birth_refine: int | None = None
    noise_swap: bool | None = None
    noise_swap_rate: float | None = None
    n_noise_bridge: int | None = None
    n_noise_relax: int | None = None
    untemper_transit: bool | None = None
    seed: int | None = None
    show_progress: bool | None = None


# --- gradient-based / point estimates ---------------------------------------

@dataclass
class NUTS(Engine):
    """No-U-Turn HMC. Fixed-dim, differentiable.

    `ad_backend="ForwardDiff"` is ~4x faster below ~15 parameters; switch to
    "Enzyme" for high-dimensional or GP-heavy models.
    """
    name = "nuts"
    n_samples: int | None = None
    n_warmup: int | None = None
    n_chains: int | None = None
    target_accept: float | None = None
    ad_backend: Literal["ForwardDiff", "Enzyme", "ReverseDiff"] | None = None
    compile_tape: bool | None = None
    warm_start: bool | None = None
    warm_temps: int | None = None
    warm_walkers: int | None = None
    warm_steps: int | None = None
    warm_burnin: int | None = None
    progress: bool | None = None


@dataclass
class MAP(Engine):
    """Multi-start MAP. A fail-loud point estimator, NOT a global search.

    Reports `railed` and `converged`; treat a railed result as a failure, not
    a fit.
    """
    name = "map"
    method: Literal["LBFGS", "NelderMead", "BFGS"] | None = None
    n_starts: int | None = None
    maxiter: int | None = None
    g_tol: float | None = None
    basin_rtol: float | None = None
    bound_rtol: float | None = None
    dom_margin: float | None = None
    seed: int | None = None


@dataclass
class SMC(Engine):
    """Sequential Monte Carlo over a temperature ladder.

    The Julia sampler takes exactly one keyword: the ladder itself.
    """
    name = "smc"
    betas: Sequence[float] | None = None


@dataclass
class Ensemble(Engine):
    """Plain affine-invariant ensemble (single temperature)."""
    name = "ensemble"
    n_walkers: int | None = None
    n_steps: int | None = None
    n_burnin: int | None = None
    n_chains: int | None = None
    thinning: int | None = None
    seed: int | None = None


@dataclass
class ESS(Engine):
    """Elliptical slice sampling — Gaussian-prior latent models."""
    name = "ess"
    n_samples: int | None = None
    n_burnin: int | None = None
    n_chains: int | None = None
    seed: int | None = None


@dataclass
class PA(Engine):
    """Particle annealing. Known to collapse above ~20 dimensions."""
    name = "pa"
    n_replicas: int | None = None
    n_mcmc: int | None = None
    max_steps: int | None = None
    max_mcmc_mult: int | None = None
    mcmc_target_moves: float | None = None
    ess_target: float | None = None
    step_scale: float | None = None
    stretch_a: float | None = None
    mutation_kernel: Literal["stretch", "rwm", "adaptive_cov"] | None = None
    seed: int | None = None
    show_progress: bool | None = None


@dataclass
class OFTI(Engine):
    """Orbits For The Impatient: rejection sampling for relative astrometry.

    Not an MCMC sampler -- it draws and rejects, so there are no chain
    diagnostics and no evidence. Suited to short-arc visual orbits where a
    chain would not mix.
    """
    name = "ofti"
    n_attempts: int | None = None
    n_calibrate: int | None = None
    planet_idx: int | None = None
    epoch_idx: int | None = None
    buffer: int | None = None
    seed: int | None = None
    show_progress: bool | None = None


DEFAULT = PT(n_rounds=12, n_chains=8)

__all__ = ["Engine", "PT", "PTHMC", "PTWhitening", "PTEmcee",
           "Nested", "NestedINS", "NestedDynamic", "MoMS", "Daedalus", "RJMCMC",
           "TransdimPTEmcee", "NUTS", "MAP", "SMC", "Ensemble", "ESS", "PA", "OFTI",
           "DEFAULT"]
