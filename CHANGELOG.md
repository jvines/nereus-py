# Changelog

## 0.4.50

- **Every engine carries every keyword its Julia sampler declares** (the field
  lists were read out of `Base.kwarg_decl`). Added, all defaulting to not sent:
  `PTEmcee` `checkpoint`, `checkpoint_interval`, `resume`, `burnin_hold`,
  `burnin_hold_frac`, `progress_log`, `swap_min`, `evidence_postfit`;
  `TransdimPTEmcee` `swap_min`, `burnin_hold`, `burnin_hold_frac`, `checkpoint`,
  `checkpoint_interval`, `resume`; `PTWhitening` `burnin_hold`, `burnin_hold_frac`,
  `checkpoint*`, `resume`, `swap_min`; and the checkpoint / `n_scouts` /
  `show_progress` / evidence keywords of `PT`, `PTHMC`, `Ensemble`, `ESS`, `MoMS`,
  `RJMCMC`, `NUTS` (also `seed`), `Nested*`, `Daedalus` and `PA`. Only `init`
  (a Julia vector), `td` and `rng` are not exposed.
- `ttv_epochs` is accepted by `fit_joint` and the other fits (it was missing from
  the keyword whitelist).
- **`fit_ttv(transit_times=None, *, phot=, rv=, nbody=)`**: transit times,
  photometry, or both; `rv` selects RVPM_TTV and `nbody=True` the N-body model.
- **`config_overrides: dict | None` on every `fit_*`** (not `fit_tomography`): a
  dict in a job config's shape applied last. Deep-merged into the config for
  `fit_rm`, `fit_ttv`, `fit_binary`; read block by block (`star`, `model`,
  `priors`, ...) for `fit_rv`, `fit_transit`, `fit_astrometry`, `fit_joint`.
- **`noise_models`, `noise_menu`, `planet_modes`, `indicator_cols` on every
  `fit_*`**, with `RV(indicators=..., indicator_cols=...)` for the activity
  columns. `Transit(gravity_darkening=True)` and `fit_transit(gravity_darkening=
  True)` fit the gravity-darkened transit instead of raising.
- **Operations on a result.** `s.evidence`, `s.detection_limits`,
  `s.diagnostics.loo / ppc / fit_health` take `result=`: they return the block
  the fit wrote, or with `recompute=True` evaluate again from its chains by
  replaying the fit's request. `s.diagnostics.label_switching(result=)`,
  `s.select_planets(result=)` (trans-dim chains) and `s.select_noise(result=)`
  read the chains. `s.select_planets(*channels, candidates=)` and
  `s.select_noise(*channels, candidates={name: noise})` run one fit per candidate
  and tabulate log Z. `s.pre_white(...)` prewhitens a series;
  `s.detect.lc_periodogram(...)` is a GLS periodogram of a light curve. These
  need a runtime with the matching Nereus.jl changes; `PY_API_VERSION` is unchanged
  (no payload changed shape).
- `planets=[1, 2]` (a range) or a trans-dim engine now sets up trans-dim mode on
  its own for `fit_rv`, `fit_transit`, `fit_astrometry` and `fit_joint`; it used
  to crash with `td not assigned` unless `transdim=True` was also given.
  `transdim=False` with either raises. Operations on a `fit_rm`, `fit_ttv` or
  `fit_binary` result (`recompute=True`) replay from the job config Julia wrote
  beside the chains. `select_planets(candidates=(0, 1, 2))` runs the
  zero-planet fit.
- `astronereus.priors`: constructors for every prior the typed route accepts
  (`modjeffreys`, `beta`, bounded `normal`, ...), with client-side checks.
  They return the same dicts as before.

## 0.4.49

- Every entry of a fit's `params` block now carries the unit Julia resolved for
  it, from the instrument's channel, so a photometric `gamma_`/`sigma_` is no
  longer read as m/s. The client uses that unit, `""` included, and falls back
  to its own name map only for an entry that carries none (a runtime that
  predates per-entry units). The raw chain unit is shown as sent: an angle in
  `rad` gets its degree reading from the unit, not from the name.
- Derived rows carry their unit from Julia too. The fallback map knows the new
  rows (`trend_P_min` in d, `trend_K_min` in m/s, `trend_Msini_min` and
  `trend_Mc_min_torres` in M_jup, `M_s_rho` in M_sun, `logg_s` in dex,
  `sep_mas` in mas, `omega_p_deg` and `pa_deg` in deg).

## 0.4.48

- Unit labels now mirror Nereus.jl's explicit unit rules: `rho_s` is in
  rho_sun, `T14`/`T23` in hr, `sigma_tt_k*` in d, `sigma_as_*`/`sigma_iad_*` in
  mas, and `dvdt`, `d2vdt2`, `i_star`, `v_sin_i_star`, the CCF and tomography
  slots, the instrument-independent noise hyperparameters and the derived
  columns (`mass_jup`, `radius_*`, `depth_ppm`, `Tsec`, `g_p`, ...) have units.
- A summary entry that carries a `unit` field is trusted even when it is `""`
  (dimensionless); name rules apply only to entries without one.

### Fixed

- **`fit_rv` channel options reach the model.** The daemon passed Julia's
  `fit_rv` the RV channel's `data` alone, so every option on it was dropped:
  `fit_rv(rv, trend_order=1)` fitted no trend (no `dvdt` in the result), and
  `jitter`, `gamma_per_instrument`, `share` and `marginalize_gamma` were
  ignored the same way. The whole channel is sent now and each option changes
  the model; `RV(indicators=...)`, which the typed fits cannot build, raises.
  Needs a runtime speaking API v4 (`PY_API_VERSION` 4).
- `RM(vsini=..., lambda_prior=..., beta=...)` were carried to the daemon and
  never read. They are priors on `v_sin_i_star` (m/s), `lambda_k1` (rad) and,
  for `flavour="arome"`, `beta_p` (m/s); `beta` with another kernel raises, as
  does the same parameter given in `priors` too.
- `fit_rv(RV(...), trend_order=...)` / `jitter=...` and
  `fit_transit(Transit(...), limb_darkening=...)` / `gravity_darkening=...`
  ignored the keyword next to the channel; they raise now.
- `fit_joint` with an `RM`, `TTV` or `SB2` channel fitted the other channels
  and dropped that one without a word; it raises now.
