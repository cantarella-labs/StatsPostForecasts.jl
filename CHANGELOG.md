# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [v0.1.0]

- Initial release
- Implements the Member-by-Member (MBM) ensemble post-processing method of
  [Van Schaeybroeck and Vannitsem (2015)][vsv2015], *Ensemble post-processing
  using member-by-member approaches: theoretical aspects*, Q. J. R. Meteorol.
  Soc. 141, 807–818, doi:10.1002/qj.2397 (BibTeX entry
  `van_schaeybroeck_ensemble_2015` in `docs/src/refs.bib`).
- Focus on ECMWF ensemble forecast, but should work for every ensemble forecast.
- Fit MBM coefficients by minimizing CRPS. Fit is done via Linear Programming using JUMP (fit_crps) and also using nonlinear optimization using LBFGS using Optim (fit_crps_naive)

- Apply fitted coefficients to a new, unseen run with `mbm_correction!`.
- Sub-daily interpolation of a corrected forecast (`interpolate_forecast`): the
  diurnal temperature cycle of Göttsche and Olesen (2001) is fitted per
  sunrise-to-sunrise window and evaluated on the fine grid, with the fit residual
  added back so the curve still passes through the forecast values.

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html
[vsv2015]: https://doi.org/10.1002/qj.2397

<!-- Versions -->

[unreleased]: https://github.com/cantarella-labs/StatsPostForecasts.jl/compare/v0.1.0...HEAD
