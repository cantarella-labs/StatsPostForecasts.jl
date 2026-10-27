import Dates: DateTime, Period, Hour, hour
import Statistics: mean
import DataInterpolations: LinearInterpolation

#=
Out-of-sample evaluation
========================

`training.jl` fits the parameters; this file spends them. Given a run the
fit has never seen, a parameter set and the station record, it corrects the
run, interpolates every member onto the observation times, and scores both the
raw and the corrected ensemble.

The score is the ensemble CRPS, in the units of the variable. For an ensemble
of M members it has a closed form (the same one `crps_min` minimises),

    CRPS = (1/M)·Σₘ|xₘ − y| − (1/2M²)·Σₘ Σₘ′|xₘ − xₘ′|

which is exact for the empirical CDF, so there is nothing to integrate
numerically. Quadrature over (F − H)² is the same quantity the hard way, and
it is easy to get wrong: the integrand is a step function, and the tails only
vanish outside [min(x, y), max(x, y)] — integrating over the ensemble range
alone silently drops the whole contribution whenever the observation falls
outside the ensemble, which is exactly the case worth scoring.
=#

"""
    crps(::MBM, ensemble, y) -> Real

Ensemble CRPS of one forecast against one observation, in the units of the
variable (lower is better). Uses the closed form above, so it agrees with
`crps_min(MBM(), …)` term for term and needs no quadrature.
"""
function crps(::MBM, x::AbstractVector, y)
    xs = sort(x)
    return mean(abs.(xs .- y)) - mean_abs_diff(xs) / 2
end

"""
    crps(::EMOS, μ, σ², y) -> Real

CRPS of the normal forecast N(μ, σ²) against one observation, in closed form
([gneiting_calibrated_2005](@cite)): σ·[z(2Φ(z) − 1) + 2φ(z) − 1/√π] with
z = (y − μ)/σ.
"""
function crps(::EMOS, μ, σ², y)
    σ = √σ²
    z = (y-μ)/σ
    crps = σ*(z * (2*cdf(std_normal, z)-1) + (2*pdf(std_normal, z)-1/√(π)))
    return crps
end

"""
    correct(run::InitForecast, params::Parameters{…,MBM}) -> InitForecast

Apply the fitted parameters to every lead time of one run, member by member.

Members keep the model's order (the map is elementwise; only the spread
`mean_abs_diff` is computed on a sorted copy). The input run is left
untouched, and correcting an already corrected run throws.

Throws if `params` has no entry for one of the run's lead times, rather than
silently leaving that lead uncorrected.
"""
function correct(run::InitForecast, params::Parameters{<:Any,<:Any,<:Any,MBM})
    run.corrected && throw(ArgumentError("run at $(run.timestamp) is already corrected"))
    fcs = map(run.forecasts) do fc
        haskey(params.p, fc.lead_time) ||
            throw(KeyError("no MBM parameters for lead time $(fc.lead_time)"))
        x = fc.ensemble
        xc = similar(x)
        mbm_correction!(xc, x, params.p[fc.lead_time], mean(x), mean_abs_diff(sort(x)))
        Forecast(fc.lead_time, xc)
    end
    return InitForecast(run.timestamp, fcs, true)
end

"""
    correct(runs, params_by_init_hour) -> Vector{InitForecast}

Correct several runs, picking the parameter set for each run's initialisation
hour from `params_by_init_hour` (keys are `Hour`s: `Hour(0)`, `Hour(12)`, …).
00 UTC and 12 UTC runs are never pooled, so they need separate entries.
"""
correct(runs::AbstractVector{<:InitForecast}, params::AbstractDict) =
    [correct(run, params[Hour(hour(run.timestamp))]) for run in runs]

"""
    evaluate_forecast(::MBM, run, obs, params, ϕ, λ) -> NamedTuple

Score one held-out run against the station record.

The run is corrected with `params`, each member is interpolated onto the
observation times with `interpolate_forecast` (diurnal-cycle fit per
sunrise-to-sunrise window), and the raw and corrected ensembles are scored
with `crps` at every one of those times.

Only observations inside the run's own span are used — outside it
`interpolate_forecast` extrapolates, which is not evaluation.

Returns `(; times, observed, raw, corrected, crps_raw, crps_corrected)`, where
`raw` and `corrected` are `length(times) × M` matrices, one column per member
in the model's member order.
"""
function evaluate_forecast(
    ::MBM,
    run::InitForecast,
    obs::Observations,
    params::Parameters,
    ϕ,
    λ,
)
    corrected = correct(run, params)
    fc_times = [run.timestamp + fc.lead_time for fc in run.forecasts]

    keep = findall(t -> first(fc_times) <= t <= last(fc_times), obs.times)
    isempty(keep) && throw(
        ArgumentError("no observations between $(first(fc_times)) and $(last(fc_times))"),
    )
    times, y = obs.times[keep], obs.values[keep]

    M = length(first(run.forecasts).ensemble)
    F = float(eltype(obs.values))
    raw = Matrix{F}(undef, length(times), M)
    cor = Matrix{F}(undef, length(times), M)
    for m in 1:M
        raw[:, m] .= interpolate_forecast(
            fc_times,
            [fc.ensemble[m] for fc in run.forecasts],
            times,
            ϕ,
            λ,
        )
        cor[:, m] .= interpolate_forecast(
            fc_times,
            [fc.ensemble[m] for fc in corrected.forecasts],
            times,
            ϕ,
            λ,
        )
    end

    return (
        times = times,
        observed = y,
        raw = raw,
        corrected = cor,
        crps_raw = [crps(MBM(), view(raw, i, :), y[i]) for i in eachindex(times)],
        crps_corrected = [crps(MBM(), view(cor, i, :), y[i]) for i in eachindex(times)],
    )
end

"""
    apply_emos_correction(run, params) -> (; timestamp, lead_times, μ, σ²)

Predictive distribution N(μ, σ²) at every lead time of one raw run, from
[`emos_correction`](@ref) applied to that lead's ensemble mean and
(uncorrected) variance. `μ` and `σ²` are vectors aligned with `lead_times`.

Throws if the run is already corrected, or if `params` has no entry for one of
the run's lead times.
"""
function apply_emos_correction(run::InitForecast, params::Parameters{<:Any,<:Any,<:Any,EMOS})
    run.corrected && throw(ArgumentError("run at $(run.timestamp) is already corrected"))
    stats = map(run.forecasts) do fc
        haskey(params.p, fc.lead_time) ||
            throw(KeyError("no EMOS parameters for lead time $(fc.lead_time)"))
        x = fc.ensemble
        x_mean = mean(x)
        emos_correction(params.p[fc.lead_time], x_mean, var(x; corrected = false, mean = x_mean))
    end
    return (
        timestamp = run.timestamp,
        lead_times = [fc.lead_time for fc in run.forecasts],
        μ = first.(stats),
        σ² = last.(stats),
    )
end

"""
    evaluate_forecast(::EMOS, run, obs, params, ϕ, λ) -> NamedTuple

EMOS counterpart of the MBM method: same observation window and raw-ensemble
scoring, but the corrected forecast is a normal distribution. Its mean μ is
interpolated with `interpolate_forecast` (it follows the diurnal cycle); its
variance σ² is interpolated linearly in time.

Returns `(; times, observed, raw, corrected, crps_raw, crps_corrected)`, where
`raw` is `length(times) × M` and `corrected` is `length(times) × 2` with
columns `[μ σ²]`. `crps_raw` is the ensemble CRPS, `crps_corrected` the
normal CRPS.
"""
function evaluate_forecast(
    ::EMOS,
    run::InitForecast,
    obs::Observations,
    params::Parameters,
    ϕ,
    λ,
)
    fc_times = [run.timestamp + fc.lead_time for fc in run.forecasts]
    correction = apply_emos_correction(run, params)

    keep = findall(t -> first(fc_times) <= t <= last(fc_times), obs.times)
    isempty(keep) && throw(
        ArgumentError("no observations between $(first(fc_times)) and $(last(fc_times))"),
    )
    times, y = obs.times[keep], obs.values[keep]

    F = float(eltype(obs.values))
    M = length(first(run.forecasts).ensemble)
    raw = Matrix{F}(undef, length(times), M) # one col per member
    cor = Matrix{F}(undef, length(times), 2) # [μ σ²]
    for m in 1:M
        raw[:, m] .= interpolate_forecast(
            fc_times,
            [fc.ensemble[m] for fc in run.forecasts],
            times,
            ϕ,
            λ,
        )
    end
    cor[:, 1] .= interpolate_forecast(fc_times, correction.μ, times, ϕ, λ)
    cor[:, 2] .= LinearInterpolation(correction.σ², Dates.value.(fc_times)).(Dates.value.(times))

    return (
        times = times,
        observed = y,
        raw = raw,
        corrected = cor,
        crps_raw = [crps(MBM(), view(raw, i, :), y[i]) for i in eachindex(times)],
        crps_corrected = [crps(EMOS(), cor[i, 1], cor[i, 2], y[i]) for i in eachindex(times)],
    )
end
