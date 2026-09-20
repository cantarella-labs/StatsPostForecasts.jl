```@meta
CurrentModule = StatsPostForecasts
```

# Getting started

StatsPostForecasts turns a raw ensemble weather forecast (temperature only for now) into a calibrated one
at a specific site with observation data.
The module also interpolate cleverly the forecast to finer scale time resolution.

1. **Calibrate.** Fit the member-by-member (MBM) correction of
   [van_schaeybroeck_ensemble_2015](@cite) on past forecast–observation pairs,
   one parameter set per initialisation hour and lead time.
2. **Interpolate.** Turn a 3-hourly forecast into a finer one (e.g., 30-minute) by fitting a
   diurnal temperature cycle per sunrise-to-sunrise window. This is slightly better than linear interpolation specially for peaks and valleys of temperature.
3. **Evaluate.** Score a run against observations, with
   the ensemble CRPS.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/cantarella-labs/StatsPostForecasts.jl")
```

## Conventions

Please adopt these conventions when using this package:

- **Times are UTC `DateTime`s.** A forecast is identified by its initialisation
  time; its valid time is `timestamp + lead_time`. If your station reports local
  time, convert before building [`Observations`](@ref) — a one- or two-hour shift
  multiplies the CRPS several-fold and looks like a model problem.
- **Lead times are `Dates.Period`s**, e.g. `Hour(3)`, so `init + lead` is a valid
  time with no unit bookkeeping.
- **Units are whatever you put in**, but ECMWF fields are in kelvin, so the
  examples work in K throughout and convert only for plotting.
- **Stations are `(latitude, longitude)`** in degrees, longitude east-positive.
  Both the `±180` and `0–360` conventions are accepted when reading GRIB.

## The whole workflow

```julia
using Dates, Statistics
using StatsPostForecasts

station = (39.13, -3.10)            # (lat, lon), degrees
ϕ, λ = station

# 1. observations: your own reader, ending in an `Observations`
obs = Observations(times, values)   # times strictly increasing, UTC

# 2. forecasts: one InitForecast per run
path = download_ecmwf_ens(Date(2026, 7, 14), "00", ("2t",), 0:3:24, "run.grib2";
                          members = 1:10, base_url = ECMWF_GCS_MIRROR)
run = read_init_forecasts(path, "t2m", [station])[1]

# 3. fit one parameter set per lead time, on past runs only
leads = Hour.(0:3:24)
pbylead = Dict{Hour,AbstractVector{Float64}}()
crps_train = Float64[]
for lt in leads
    t = TrainingObject(past_runs, obs, lt)
    p, _ = fitting_crps(t)
    pbylead[lt] = p
    push!(crps_train, crps_min(p, t))
end
params = MBMParameters(Hour(0), leads, pbylead, window, crps_train)

# 4. correct a run the fit has never seen, and score it
ev = evaluate_forecast(run, obs, params, ϕ, λ)
mean(ev.crps_raw), mean(ev.crps_corrected)
```

Step 4 does the correction, the 30-minute interpolation of every member, and the
scoring in one call. If you only want the corrected ensemble, use
[`correct`](@ref); if you only want a finer time axis, use
[`interpolate_forecast`](@ref).

## The same thing, actually run

The block above is a sketch — it needs your data and a download. This one is
executed when these docs are built, so it is guaranteed to work against the
current API. It stands in synthetic forecasts for the real ones, and is
otherwise the same pipeline.

```@example workflow
using Dates, Statistics
using StatsPostForecasts

ϕ, λ = 39.13, -3.10
t0 = DateTime(2026, 1, 1)
leads = Hour.(0:3:24)
M = 10

# a smooth diurnal "truth", half-hourly, standing in for a station record
hours(t) = Dates.value(t - t0) / 3.6e6
truth(t) = 283.0 + 8 * sinpi((hours(t) - 9) / 12)
otimes = collect(t0:Minute(30):(t0 + Day(45)))
obs = Observations(otimes, truth.(otimes))

# 40 daily runs: the truth plus a warm bias that grows with lead time, and an
# ensemble that is too narrow — the two faults the MBM correction exists to fix
function fake_run(k)
    init = t0 + Day(k)
    fcs = map(leads) do lt
        b = 0.4 + 0.05 * Dates.value(lt)
        Forecast(lt, [truth(init + lt) + b + 0.35 * (m - (M + 1) / 2) +
                      0.3 * sinpi(0.37 * (k + m)) for m in 1:M])
    end
    InitForecast(init, fcs, false)
end
runs = fake_run.(0:39)
train, test = runs[1:30], runs[31:end]        # the fit never sees `test`
nothing # hide
```

Fit one parameter set per lead time, on the training runs only:

```@example workflow
pbylead = Dict{Hour,AbstractVector{Float64}}()
crps_train = Float64[]
for lt in leads
    t = TrainingObject(train, obs, lt)
    p, _ = fitting_crps(t)
    pbylead[lt] = p
    push!(crps_train, crps_min(p, t))
end
params = MBMParameters(Hour(0), leads, pbylead,
                       (first(train).timestamp, last(train).timestamp), crps_train)
pbylead[Hour(12)]        # (α, β, γ₁, γ₂) at +12 h
```

Then correct, interpolate and score a run the fit has never seen:

```@example workflow
ev = evaluate_forecast(first(test), obs, params, ϕ, λ)
(raw = mean(ev.crps_raw), corrected = mean(ev.crps_corrected))
```

## Runnable examples

`examples/` is a project of its own, so the plotting and DataFrames packages the
examples want stay out of the package:

```sh
julia --project=examples examples/quickstart.jl
```

- **`quickstart.jl`** — six runs, two lead times, in-sample CRPS. The cheap one,
  about 70 MB of download.
- **`end_to_end.jl`** — the full picture at one station: fit on a month of runs,
  correct the following week, interpolate, verify, and draw both the six-day
  forecast and the CRPS against lead time. Downloads about 11 GB the first time,
  then runs from cache.
- **`full_fit.jl`** — the whole station record, two lead times.

## Where to go next

- [Data model](@ref data) — what the containers hold, and how to get ECMWF data
  in (or bring your own).
- [Calibration](@ref calibration) — the MBM correction and how to fit it.
- [Interpolation](@ref interpolation) — the diurnal-cycle model, and when it is
  worth using at all.
- [Evaluation](@ref evaluation) — scoring out of sample without fooling yourself.
