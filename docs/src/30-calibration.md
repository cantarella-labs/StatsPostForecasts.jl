```@meta
CurrentModule = StatsPostForecasts
DocTestSetup = quote
    using StatsPostForecasts
end
```

# [Calibration](@id calibration)

Two methods are implemented, selected by the first argument of
[`crps_min`](@ref), [`fitting_crps_naive`](@ref), [`crps`](@ref) and
[`evaluate_forecast`](@ref):

- **`MBM()`**, member-by-member: every member is corrected, so the result is
  still an ensemble.
- **`EMOS()`**, ensemble model output statistics: the result is a normal
  distribution built from the ensemble mean and variance.

Both have four parameters ``(α, β, γ_1, γ_2)`` per lead time, and both are fitted
by minimising the CRPS over a training slice.

## The MBM model

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

## Fitting MBM

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
p, model = fitting_crps(MBM(), t)      # p = [α, β, γ₁, γ₂]
```

[`fitting_crps_naive`](@ref)`(MBM(), t)` solves the same problem with L-BFGS
instead. It exists to check the LP, and is slower and only locally optimal;
prefer `fitting_crps`.

[`crps_min`](@ref) evaluates the objective at any parameter vector, which makes
it the natural diagnostic as well as the thing being minimised. The raw ensemble
is the identity parameter set, so

```julia
crps_min(MBM(), [0, 1, 1, 0], t)       # what the raw ensemble scores
crps_min(MBM(), p, t)                  # what the fit scores
```

is the in-sample skill of the correction. Both numbers are in the units of the
variable.

## The EMOS model

EMOS [gneiting_calibrated_2005](@cite) replaces the ensemble by a normal
distribution. With ``\bar{x}`` the ensemble mean and ``s^2`` its variance
(divided by ``M``, not ``M - 1``),

```math
Y \sim \mathcal{N}(μ, σ^2), \qquad μ = α + β\bar{x}, \qquad σ^2 = γ_1 + γ_2 s^2
```

[`emos_correction`](@ref) computes ``(μ, σ^2)`` for one ensemble. ``α`` and
``β`` mean what they do for MBM. ``γ_1`` is a variance floor and ``γ_2`` scales
the ensemble variance; both must be non-negative, and not both zero. The
identity, ``N(\bar{x}, s^2)``, is `[0, 1, 0, 1]`, not MBM's `[0, 1, 1, 0]`.

The CRPS of a normal distribution has a closed form, so the objective is [gneiting_calibrated_2005](@cite)

```math
J(θ) = \frac{1}{N}\sum_n σ_n\Bigl[z_n\bigl(2Φ(z_n) - 1\bigr) + 2φ(z_n)
       - \tfrac{1}{\sqrt{π}}\Bigr], \qquad z_n = \frac{y_n - μ_n}{σ_n}
```

## Fitting EMOS

The EMOS objective is smooth but not convex, so there is no LP.
[`fitting_crps_naive`](@ref) minimises it with box-constrained L-BFGS and
forward-mode AD:

```julia
p, result = fitting_crps_naive(EMOS(), t)                 # starts at the raw ensemble
p, result = fitting_crps_naive(EMOS(), t; p0 = p_start)   # or anywhere inside the box

crps_min(EMOS(), [0, 1, 0, 1], t)    # what N(x̄, s²) scores
crps_min(EMOS(), p, t)               # what the fit scores
```

Because the optimum is only local, check it from a second start. Ordinary least
squares of ``y`` on ``\bar{x}`` is a good one: take ``α`` and ``β`` from the
regression, ``γ_1`` from the residual variance, and ``γ_2 ≈ 0``. `p0` must lie
strictly inside the box, so use a small positive value such as `1e-10`, not
exactly zero.

`crps_min(EMOS(), [0, 1, 0, 1], t)` is the score of the *normal approximation*
to the raw ensemble, not of the ensemble itself, so it differs from
`crps_min(MBM(), [0, 1, 1, 0], t)`. When comparing the two methods, compare
out of sample (see [Evaluation](@ref evaluation)).

## [EMOS from quantile forecasts](@id quantile-emos)

Some products publish quantiles of the predictive distribution instead of
ensemble members. [`QuantileForecast`](@ref) holds one lead time as levels
``τ_k`` and values ``q_k``, and [`QuantileInitForecast`](@ref) holds a whole run.
Both mirror their ensemble counterparts, and the constructor rejects bad input:
levels outside ``(0, 1)`` or unsorted, and crossing quantiles.

EMOS only needs a location and a scale, and [`normal_stats`](@ref) gets them
from the quantiles by fitting a normal in least squares:

```math
q_k = m + s\,z_k, \qquad z_k = Φ^{-1}(τ_k)
```

This uses every level and does not need them evenly spaced. After that,
everything is the same as for an ensemble:

```julia
t = TrainingObject(quantile_runs, obs, Hour(12))   # xmean = m, s² = s²
p, _ = fitting_crps_naive(EMOS(), t)
apply_emos_correction(quantile_run, params)        # N(μ, σ²) per lead time
```

``m`` and ``s`` match an ensemble's mean and standard deviation only for normal
quantiles. That is fine, because ``β`` and ``γ_2`` absorb the difference as long
as training and prediction use the same summary, which they do. A slice built
from quantiles has no members (`nmembers(t) == 0`), so MBM throws on it.

## One table per initialisation hour and lead time

[`Parameters`](@ref) holds a whole table for one method and initialisation
hour: the method, the lead times, a `Dict` from lead time to
``(α, β, γ_1, γ_2)``, the training window, and the in-sample CRPS per lead.

```julia
params = Parameters(MBM(), Hour(0), leads, pbylead, (first_init, last_init), crps_train)
```

**00 UTC and 12 UTC runs are never pooled.** A +12 h forecast from a 00 UTC run
verifies at midday; the same lead from a 12 UTC run verifies at midnight. Those
have different bias and different spread, and averaging them gives a parameter
set correct for neither. Keep one `Parameters` per initialisation hour and
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
