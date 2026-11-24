# Changelog

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
