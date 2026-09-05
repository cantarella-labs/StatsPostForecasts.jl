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

- TODO: make the forecast machinery to 1. use unsampled forecast and adjust it based on the latest coefficients and 2. interpolate the results for a fine grained time profile. (for temp we will use a solar sunlight based model)

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html
[vsv2015]: https://doi.org/10.1002/qj.2397

<!-- Versions -->

[unreleased]: https://github.com/cantarella-labs/StatsPostForecasts.jl/compare/v0.1.0...HEAD
