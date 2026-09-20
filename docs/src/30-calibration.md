```@meta
CurrentModule = StatsPostForecasts
DocTestSetup = quote
    using StatsPostForecasts
end
```

# [Calibration](@id calibration)

## The model

The member-by-member correction of [van_schaeybroeck_ensemble_2015](@cite) maps
every raw member to a corrected one with an affine transformation of the ensemble
mean and of the member deviations. For a raw ensemble ``x^{(1)} … x^{(M)}`` with
mean ``\bar{x}`` and absolute spread ``d``,

```math
\tilde{x}^{(m)} = α + β\bar{x} + τ\,(x^{(m)} - \bar{x}),
\qquad τ = γ_1 + γ_2/d
```

The four parameters do distinguishable jobs, which is what makes the fitted
values worth reading:

- **``α``** is an additive bias. At a station it absorbs representativeness — the
  model's grid point is not your site, and at 650 m elevation that alone can be
  2 K.
- **``β``** rescales the ensemble mean; ``β < 1`` damps the forecast anomaly
  towards climatology.
- **``γ_1``** scales the spread multiplicatively, ``γ_2`` adds a floor to it. The
  corrected spread is ``\tilde{d} = γ_1 d + γ_2``, so both must be non-negative
  for the result to be a spread at all — the objective does not enforce this, the
  solver's bounds do.

``d`` is the *absolute spread* (Gini mean difference), not the standard
deviation:

```math
d = \frac{1}{M^2}\sum_m\sum_{m'}\bigl|x^{(m)} - x^{(m')}\bigr|
```

[`mean_abs_diff`](@ref) computes it in ``O(M)`` after a sort, using the identity
``d = (2/M^2)\sum_k (2k-1-M)\,x_{(k)}``. For a Gaussian ensemble
``d \to 2σ/\sqrt{π} ≈ 1.13σ``, which is a convenient check.

```jldoctest
julia> mean_abs_diff([1.0, 2.0, 3.0])
0.8888888888888888
```

## Assembling the training data

[`TrainingObject`](@ref) is one lead time's worth of cases, with everything that
does not depend on the parameters precomputed:

```julia
t = TrainingObject(runs, obs, Hour(12))
ncases(t), nmembers(t)
```

It walks the runs in increasing initialisation time and keeps a case only when
the run is not already corrected, a forecast exists at that lead time, and an
observation exists at the valid time. Members are sorted, and ``\bar{x}``, ``d``
and the deviation matrix are stored. Spreads below `d_floor` (default `0.01`) are
raised, so that ``γ_2/d`` stays finite.

If no case survives, it throws rather than returning an empty object — an empty
training set is a data problem, not a valid fit.

## Fitting

[`fitting_crps`](@ref) minimises the mean ensemble CRPS over the training slice:

```math
J(θ) = \frac{1}{N}\sum_n\Bigl[\frac{1}{M}\sum_m\bigl|\tilde{x}_n^{(m)} - y_n\bigr|
       - \tfrac{1}{2}\bigl(γ_1 d_n + γ_2\bigr)\Bigr]
```

The corrected members are linear in the parameters and the objective is
piecewise linear, so this is solved *exactly* as a linear program with HiGHS —
not iterated to a tolerance. Each ``|r_{nm}|`` becomes an auxiliary variable with
two linear constraints.

```julia
p, model = fitting_crps(t)      # p = [α, β, γ₁, γ₂]
```

[`fitting_crps_naive`](@ref) solves the same problem with L-BFGS instead. It
exists to check the LP, and is slower and only locally optimal; prefer
`fitting_crps`.

[`crps_min`](@ref) evaluates the objective at any parameter vector, which makes
it the natural diagnostic as well as the thing being minimised. The raw ensemble
is the identity parameter set, so

```julia
crps_min([0, 1, 1, 0], t)       # what the raw ensemble scores
crps_min(p, t)                  # what the fit scores
```

is the in-sample skill of the correction. Both numbers are in the units of the
variable.

## One table per initialisation hour and lead time

[`MBMParameters`](@ref) holds a whole table for one initialisation hour: the
lead times, a `Dict` from lead time to ``(α, β, γ_1, γ_2)``, the training window,
and the in-sample CRPS per lead.

```julia
params = MBMParameters(Hour(0), leads, pbylead, (first_init, last_init), crps_train)
```

**00 UTC and 12 UTC runs are never pooled.** A +12 h forecast from a 00 UTC run
verifies at midday; the same lead from a 12 UTC run verifies at midnight. Those
have different bias and different spread, and averaging them gives a parameter
set correct for neither. Keep one `MBMParameters` per initialisation hour and
select on `Hour(hour(run.timestamp))`.

Lead time matters for the same reason, which is why the parameters are a `Dict`
keyed by it rather than a single vector.

## How much training data

The fit has four parameters per lead time. Thirty cases is workable and is what
the examples use; fewer starts to show. Two things to watch:

- ``α`` and ``β`` are collinear when the ensemble mean varies little over the
  training window, and you will see large compensating values (``α = 90``,
  ``β = 0.69``) that still predict sensibly but do not mean what they look like.
- In-sample CRPS always improves. It is not evidence. The only number worth
  quoting is out-of-sample, on runs the fit never saw — see
  [Evaluation](@ref evaluation).
