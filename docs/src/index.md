```@meta
CurrentModule = StatsPostForecasts
```

# StatsPostForecasts

Documentation for [StatsPostForecasts](https://github.com/cantarella-labs/StatsPostForecasts.jl).

## Method

StatsPostForecasts implements the member-by-member (MBM) ensemble
post-processing of [van_schaeybroeck_ensemble_2015](@cite): every raw
member is mapped to a corrected member by an affine transformation of the
ensemble mean and of the member deviations, with parameters ``(α, β, γ₁, γ₂)``
fitted per lead time by minimising the ensemble CRPS (their "CRPS MIN"
variant, Sect. 3.5). The objective is piecewise linear, so the fit is solved
exactly as a linear program.

If you use this package in published work, please cite the method paper
(BibTeX in `docs/src/refs.bib`, see the [Bibliography](@ref)) and the
software itself via `CITATION.cff`.

