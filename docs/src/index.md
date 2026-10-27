```@meta
CurrentModule = StatsPostForecasts
```

# StatsPostForecasts

Documentation for [StatsPostForecasts](https://github.com/cantarella-labs/StatsPostForecasts.jl).

Statistical post-processing of ensemble weather forecasts at a point: calibrate
a raw ensemble against station observations, interpolate it onto the time
resolution a product needs, and score the result out of sample.

```julia
ev = evaluate_forecast(MBM(), run, obs, params, ϕ, λ)
mean(ev.crps_raw), mean(ev.crps_corrected)
```

## Method

StatsPostForecasts implements the member-by-member (MBM) ensemble
post-processing of [van_schaeybroeck_ensemble_2015](@cite): every raw
member is mapped to a corrected member by an affine transformation of the
ensemble mean and of the member deviations, with parameters ``(α, β, γ₁, γ₂)``
fitted per lead time by minimising the ensemble CRPS (their "CRPS MIN"
variant, Sect. 3.5). The objective is piecewise linear, so the fit is solved
exactly as a linear program.

It also implements EMOS, the non-homogeneous Gaussian regression of
[gneiting_calibrated_2005](@cite): the forecast is a normal distribution whose
mean is affine in the ensemble mean and whose variance is affine in the
ensemble variance, fitted by minimising the closed-form normal CRPS. The
method is chosen by passing `MBM()` or `EMOS()` as the first argument. EMOS
also accepts forecasts published as quantiles rather than members.

Sub-daily interpolation uses the diurnal temperature cycle of
[gottsche_modelling_2001](@cite), fitted per sunrise-to-sunrise window, with the
fit residual added back so the curve still passes through every forecast value.

## User guide

- [Getting started](@ref) — installation, conventions, and the whole workflow on
  one page.
- [Data model](@ref data) — the containers, getting ECMWF open data in, and
  bringing your own.
- [Calibration](@ref calibration) — what the MBM and EMOS parameters mean and
  how to fit them.
- [Interpolation](@ref interpolation) — the diurnal-cycle model, and measured
  guidance on when it is worth using.
- [Evaluation](@ref evaluation) — scoring out of sample without fooling
  yourself.
- [Reference](@ref reference) — every exported function and type.

## Citing

If you use this package in published work, please cite the method paper
(BibTeX in `docs/src/refs.bib`, see the [Bibliography](@ref)) and the
software itself via `CITATION.cff`.
