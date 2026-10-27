```@meta
CurrentModule = StatsPostForecasts
DocTestSetup = quote
    using StatsPostForecasts
end
```

# [Evaluation](@id evaluation)

Training tells you what the parameters are. Evaluation tells you whether they
help, and it is the only number worth quoting.

## Applying the parameters

[`correct`](@ref) applies a fitted MBM table to every lead time of one run:

```julia
cor = correct(run, params)       # params::Parameters{…,MBM} for this init hour
```

Members keep the model's order: the map is elementwise, and only the spread
[`mean_abs_diff`](@ref) is computed on a sorted copy. Sort the result yourself
if you want an ensemble envelope, where member 1 is always the coldest.

The input run comes back untouched, so a raw run stays raw and correcting twice
gives the same answer. The result carries `corrected = true`, and correcting it
again throws. A lead time with no parameters throws rather than passing through
uncorrected.

For several runs at mixed initialisation hours, pass a `Dict` keyed by hour:

```julia
cor = correct(runs, Dict(Hour(0) => params_00, Hour(12) => params_12))
```

EMOS does not return an ensemble. [`apply_emos_correction`](@ref) gives the
predictive ``N(μ, σ^2)`` at every lead time of one run:

```julia
c = apply_emos_correction(run, params)   # params::Parameters{…,EMOS}
c.lead_times, c.μ, c.σ²
```

Passing EMOS parameters to `correct`, or MBM parameters to
`apply_emos_correction`, is a `MethodError`.

## The score

[`crps`](@ref) is the continuous ranked probability score of one forecast
against one observation, in the units of the variable, lower better. For an
ensemble (`MBM()`) it is

```math
\mathrm{CRPS} = \frac{1}{M}\sum_m\bigl|x^{(m)} - y\bigr|
              - \frac{1}{2M^2}\sum_m\sum_{m'}\bigl|x^{(m)} - x^{(m')}\bigr|
```

This is exact for the empirical CDF, so there is nothing to integrate
numerically — it is the same expression [`crps_min`](@ref) minimises, evaluated
for a single case.

```jldoctest
julia> crps(MBM(), [1.0, 2.0, 3.0], 2.0)
0.2222222222222222

julia> crps(MBM(), [2.0, 2.0, 2.0], 3.5)      # a deterministic forecast scores |x − y|
1.5
```

For a normal forecast (`EMOS()`) it is the closed form of
[gneiting_calibrated_2005](@cite), with ``z = (y - μ)/σ``:

```math
\mathrm{CRPS} = σ\Bigl[z\bigl(2Φ(z) - 1\bigr) + 2φ(z) - \tfrac{1}{\sqrt{π}}\Bigr]
```

```jldoctest
julia> crps(EMOS(), 0.0, 1.0, 0.0) ≈ (√2 - 1) / √π
true
```

It rewards sharpness and accuracy together: the first term punishes being wrong,
the second rewards being narrow, and a deterministic forecast collapses to
absolute error. Doing the same thing by quadrature over ``(F - H)^2`` is the hard
way and easy to get wrong — the integrand only vanishes outside
``[\min(x, y), \max(x, y)]``, so integrating over the ensemble range alone
silently drops everything whenever the observation falls outside the ensemble.

## Scoring a whole run

[`evaluate_forecast`](@ref) does correction, interpolation and scoring together:

```julia
ev = evaluate_forecast(MBM(), run, obs, params, ϕ, λ)

mean(ev.crps_raw)              # what the raw ensemble scored
mean(ev.crps_corrected)        # what the corrected one scored
```

It corrects the run, interpolates **every member** onto the observation times
with [`interpolate_forecast`](@ref), and scores both ensembles at each of those
times. The result is a named tuple:

| field | meaning |
|---|---|
| `times` | observation times inside the run's span |
| `observed` | the observed values there |
| `raw`, `corrected` | `length(times) × M` matrices, one column per member |
| `crps_raw`, `crps_corrected` | the score at each time |

With `EMOS()` the call and fields are the same, except that `corrected` is a
`length(times) × 2` matrix with columns `[μ σ²]`, and `crps_corrected` is the
normal CRPS. ``μ`` is interpolated with [`interpolate_forecast`](@ref), since it
follows the diurnal cycle; ``σ^2`` is interpolated linearly. `crps_raw` is still
the ensemble CRPS of the raw members, so the two methods can be compared on the
same baseline.

A `QuantileInitForecast` goes through the same call. `raw` then has one column
per quantile level, and `crps_raw` is the normal CRPS of the raw quantiles'
fitted normal ([`normal_stats`](@ref)). That is the EMOS identity `[0, 1, 0, 1]`,
the same baseline as in training. Quantiles have no members, so an ensemble
CRPS is not available. An average pinball loss over the levels would leave out
the tails beyond the outermost levels, so it would not be comparable with the
normal CRPS of the corrected forecast.

Only observations inside the run's own span are used. Outside it the
interpolation extrapolates, which is not evaluation — and an empty overlap
throws rather than returning nothing useful.

Because the matrices are returned, the same call feeds both the score and the
plot: the ensemble band is `extrema` across the columns of `ev.corrected`, and
the mean line is the row means, so the picture and the number come from the same
arithmetic.

## Setting up an honest test

Three things separate a real evaluation from a flattering one:

**Hold the test runs out of the fit.** `TRAIN_DATES` and `TEST_DATES` must be
disjoint, and it is worth making that visible in the code rather than deriving
one from the other.

**Use enough test runs.** A single run tells you about that day's weather. In
`examples/end_to_end.jl`, one held-out run gave 1.67 → 0.54 K for the ensemble
mean; the same fit over seven held-out runs gave 1.56 → 0.98 K. The single run
was a good day.

**Break the result down by lead time.** The pooled number hides where the skill
is. At the example station the MBM roughly halves the error at +0, +21 and +24 h,
where the raw ensemble is worst, and does nothing at +9 and +18 h — and at one
lead it was slightly *worse* than raw. That is what fitting four parameters on
thirty cases buys at leads where there was little bias to remove.

`examples/end_to_end.jl` does all three and plots the CRPS against lead time for
raw and corrected, which is the most informative single picture of what the
calibration is doing.

## Reading the two metrics

RMSE of the ensemble **mean** and CRPS of the **ensemble** answer different
questions, and the corrections move them differently. The correction adjusts level and
spread, so it improves the mean's RMSE *and* the ensemble's calibration; a
correction that only shifted the mean would improve RMSE and could leave CRPS
untouched or worse. Quote both.
