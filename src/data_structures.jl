import Dates: DateTime, Period
import Statistics: mean

#=
Data structures for the member-by-member (MBM) forecast system
==============================================================

Three layers:

1. Raw material
   `Forecast`      — one ensemble at one lead time (a vector of M members).
   `InitForecast`  — all lead times of one model run (one initialisation).
   `Observations`  — the observed time series the forecasts are verified
                     against.

2. Training slice
   `TrainingObject` — everything the CRPS MIN objective needs for ONE lead
   time and ONE location/variable, with all parameter-independent
   quantities (ensemble means, spreads, deviations) precomputed. This is
   what `crps_min` consumes.

3. Fitted parameters
   `MBMParameters` — the four parameters (α, β, γ₁, γ₂) for one lead time,
   together with the training window they were fitted on.

Conventions
-----------
* Timestamps are UTC `DateTime`s. A forecast is identified by its
  initialisation time; its valid time is `timestamp + lead_time`.
* Lead times are `Dates.Period`s (e.g. `Hour(3)`), so that
  `init_time + lead_time` is a valid time without any unit bookkeeping.
* The "case" index n of the paper corresponds to one initialisation at a
  fixed lead time; M is the number of members.
* Members inside a `Forecast.ensemble` are kept in the order delivered by
  the model. Sorting happens once when a `TrainingObject` is built, because
  `mean_abs_diff` needs sorted input. Nothing else depends on member order.
* All fields are concrete (`Vector{F}`, `Matrix{F}`) so that code using
  these structs is type-stable.
=#


"""
    Forecast{P, F}

One ensemble forecast at one lead time.

# Fields
- `lead_time::P` (`P <: Dates.Period`): time since initialisation, e.g.
  `Hour(3)`.
- `ensemble::Vector{F}`: the `M` member values of one variable at one
  location. Member order is the model's order (not sorted).

A `Forecast` carries no timestamp of its own; its valid time is
`InitForecast.timestamp + lead_time`.
"""
struct Forecast{P<:Period,F<:Real}
    lead_time::P
    ensemble::Vector{F}
end


"""
    InitForecast{P, F}

All lead times of one model run (one initialisation) for one variable at
one location.

# Fields
- `timestamp::DateTime`: initialisation time (UTC).
- `forecasts::Vector{Forecast{P,F}}`: one `Forecast` per lead time, in
  increasing lead time.
- `corrected::Bool`: `false` for raw model output, `true` after the MBM map
  has been applied. Raw and corrected forecasts share the container; the
  flag prevents a corrected run from being used as training input or from
  being corrected twice.

The valid time of `forecasts[k]` is `timestamp + forecasts[k].lead_time`.
"""
struct InitForecast{P<:Period,F<:Real}
    timestamp::DateTime
    forecasts::Vector{Forecast{P,F}}
    corrected::Bool
end


"""
    Observations{F}

Observed time series of one variable at one location, used both to train
the MBM parameters and to verify forecasts.

# Fields
- `times::Vector{DateTime}`: observation times (UTC), strictly increasing.
- `values::Vector{F}`: observed values, same length as `times`.

Use [`observation_at`](@ref) to look up the value at a forecast's valid
time; a forecast whose valid time has no observation is simply not a
training case.
"""
struct Observations{F<:Real}
    times::Vector{DateTime}
    values::Vector{F}
    function Observations(times::Vector{DateTime}, values::Vector{F}) where {F<:Real}
        length(times) == length(values) ||
            throw(ArgumentError("times and values must have the same length"))
        issorted(times; lt = <) || throw(ArgumentError("times must be strictly increasing"))
        new{F}(times, values)
    end
end

"""
    observation_at(obs::Observations, t::DateTime) -> Union{F, Nothing}

Value observed exactly at time `t`, or `nothing` if `t` is not an
observation time. Uses binary search on the sorted `times` vector.
"""
function observation_at(obs::Observations, t::DateTime)
    i = searchsortedfirst(obs.times, t)
    (i <= length(obs.times) && obs.times[i] == t) ? obs.values[i] : nothing
end


"""
    TrainingObject{P, F}

Training slice for one lead time (and implicitly one variable and one
location): the input of `crps_min`. It holds `N` cases (initialisations)
and everything the objective needs that does not depend on the parameters.

For case n with raw members xₙ⁽¹⁾ … xₙ⁽ᴹ⁾:

    x̄ₙ     = (1/M) Σₘ xₙ⁽ᵐ⁾                      → `xmean[n]`
    dₙ     = (1/M²) Σₘ Σₘ′ |xₙ⁽ᵐ⁾ − xₙ⁽ᵐ′⁾|        → `d[n]`       (via `mean_abs_diff`)
    E[m,n] = xₙ⁽ᵐ⁾ − x̄ₙ                             (members sorted ascending)
    yₙ     = observation at init_times[n] + lead_time  → `y[n]`

# Fields
- `lead_time::P`: the lead time this slice belongs to.
- `init_times::Vector{DateTime}`: initialisation times of the `N` cases,
  increasing. Kept for traceability and for windowing.
- `y::Vector{F}`: observations, length `N`.
- `xmean::Vector{F}`: raw ensemble means, length `N`.
- `d::Vector{F}`: raw absolute spreads, length `N`, all `≥ d_floor > 0`.
- `E::Matrix{F}`: deviations, size `M × N`; column `n` is sorted ascending
  and sums to zero (up to rounding).

# Construction
Normally built from raw material with
`TrainingObject(inits, obs, lead_time; d_floor)`, see below. The inner
constructor only checks consistency of the sizes and positivity of `d`.
"""
struct TrainingObject{P<:Period,F<:Real}
    lead_time::P
    init_times::Vector{DateTime}
    y::Vector{F}
    xmean::Vector{F}
    d::Vector{F}
    E::Matrix{F}
    function TrainingObject(
        lead_time::P,
        init_times::Vector{DateTime},
        y::Vector{F},
        xmean::Vector{F},
        d::Vector{F},
        E::Matrix{F},
    ) where {P<:Period,F<:Real}
        N = length(y)
        (
            length(init_times) == N &&
            length(xmean) == N &&
            length(d) == N &&
            size(E, 2) == N
        ) || throw(ArgumentError("inconsistent number of cases N"))
        all(>(0), d) ||
            throw(ArgumentError("all spreads d must be strictly positive; apply a floor"))
        new{P,F}(lead_time, init_times, y, xmean, d, E)
    end
end


"""
    MBMParameters{P, F}

Fitted MBM parameters for one lead time.

# Fields
- `lead_time::P`.
- `p::Vector{F}`: `(α, β, γ₁, γ₂)`, in the order expected by
  `mbm_correction!` and `crps_min`.
- `window::Tuple{DateTime, DateTime}`: first and last initialisation time of
  the training cases, i.e. `extrema(training_object.init_times)`. Lets you
  see at a glance how old a parameter set is when a new forecast comes in.
- `crps_train::F`: in-sample mean CRPS at `p` (the minimised objective).
  Useful as a monitoring quantity together with the raw-ensemble CRPS
  `crps_min([0,1,1,0], t)`.

A full parameter table for one location/variable is a `Vector{MBMParameters}`
indexed by lead time; keep separate tables per initialisation hour
(00 UTC and 12 UTC runs are not pooled).
"""
struct MBMParameters{P<:Period,F<:Real}
    lead_time::P
    p::Vector{F}
    window::Tuple{DateTime,DateTime}
    crps_train::F
end
