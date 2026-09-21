import Dates: DateTime, Period, Hour, hour
import Statistics: mean

#=
Out-of-sample evaluation
========================

`training.jl` fits the MBM parameters; this file spends them. Given a run the
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
    crps(ensemble, y) -> Real

Ensemble CRPS of one forecast against one observation, in the units of the
variable (lower is better). Uses the closed form above, so it agrees with
`crps_min` term for term and needs no quadrature.
"""
function crps(x::AbstractVector, y)
    xs = sort(x)
    return mean(abs.(xs .- y)) - mean_abs_diff(xs) / 2
end

"""
    correct(run::InitForecast, params::MBMParameters) -> InitForecast

Apply the fitted parameters to every lead time of one run, member by member.

Members are sorted ascending first, so member `k` of the result is the k-th
coldest at that lead time — the ordering `mean_abs_diff` needs and the one
that makes an ensemble envelope meaningful across lead times. The input run is
left untouched: a raw run stays raw, and correcting twice gives the same
answer.

Throws if `params` has no entry for one of the run's lead times, rather than
silently leaving that lead uncorrected.
"""
function correct(run::InitForecast, params::MBMParameters)
    run.corrected && throw(ArgumentError("run at $(run.timestamp) is already corrected"))
    fcs = map(run.forecasts) do fc
        haskey(params.p, fc.lead_time) ||
            throw(KeyError("no MBM parameters for lead time $(fc.lead_time)"))
        x = fc.ensemble
        xc = similar(x)
        mbm_correction!(xc, x, params.p[fc.lead_time], mean(x), mean_abs_diff(x))
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
    evaluate_forecast(run, obs, params, ϕ, λ) -> NamedTuple

Score one held-out run against the station record.

The run is corrected with `params`, each member is interpolated onto the
observation times with `interpolate_forecast` (diurnal-cycle fit per
sunrise-to-sunrise window), and the raw and corrected ensembles are scored
with `crps` at every one of those times.

Only observations inside the run's own span are used — outside it
`interpolate_forecast` extrapolates, which is not evaluation.

Returns `(; times, observed, raw, corrected, crps_raw, crps_corrected)`, where
`raw` and `corrected` are `length(times) × M` matrices with members sorted
ascending per lead time.
"""
function evaluate_forecast(
    run::InitForecast,
    obs::Observations,
    params::MBMParameters,
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
            [sort(fc.ensemble)[m] for fc in run.forecasts],
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
        crps_raw = [crps(view(raw, i, :), y[i]) for i in eachindex(times)],
        crps_corrected = [crps(view(cor, i, :), y[i]) for i in eachindex(times)],
    )
end
