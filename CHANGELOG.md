# Changelog

## Unreleased

- Unit labels now mirror Nereus.jl's explicit unit rules: `rho_s` is in
  rho_sun, `T14`/`T23` in hr, `sigma_tt_k*` in d, `sigma_as_*`/`sigma_iad_*` in
  mas, and `dvdt`, `d2vdt2`, `i_star`, `v_sin_i_star`, the CCF and tomography
  slots, the instrument-independent noise hyperparameters and the derived
  columns (`mass_jup`, `radius_*`, `depth_ppm`, `Tsec`, `g_p`, ...) have units.
- A summary entry that carries a `unit` field is trusted even when it is `""`
  (dimensionless); name rules apply only to entries without one.
