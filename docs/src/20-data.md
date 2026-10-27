```@meta
CurrentModule = StatsPostForecasts
```

# [Data model](@id data)

Three containers carry everything, plus a quantile variant of the forecast
pair for products that do not publish members. They are deliberately small: the
package does not know where your data comes from, only what shape it arrives
in.

## The containers

[`Forecast`](@ref) is one ensemble at one lead time — a lead time and a vector of
`M` member values:

```julia
Forecast(Hour(12), [291.2, 292.0, 290.4])
```

Member order is whatever the model delivered. Nothing depends on it except
[`mean_abs_diff`](@ref), which requires sorted input, so sorting happens once
where it is needed rather than being imposed on the container.

[`InitForecast`](@ref) is all the lead times of one run, for one variable at one
location: an initialisation timestamp, a vector of `Forecast`s in increasing lead
time, and a `corrected::Bool` flag. The flag is what stops a corrected run being
fed back into training or corrected twice — [`correct`](@ref) throws on a run
that already carries it.

[`Observations`](@ref) is the verifying time series: strictly increasing UTC
times and values of equal length. The constructor enforces both. Look values up
with [`observation_at`](@ref), which returns `nothing` rather than throwing when
a time is not an observation time:

```julia
y = observation_at(obs, run.timestamp + Hour(12))
y === nothing && return          # not a training case, not an error
```

That `nothing` is deliberate. A forecast whose valid time has no observation is
simply not a case, and both [`TrainingObject`](@ref) and the evaluation path
skip it silently.

### Quantile forecasts

Some products publish quantiles of the predictive distribution instead of
members. [`QuantileForecast`](@ref) is one lead time as probability levels and
the values at those levels, and [`QuantileInitForecast`](@ref) is a whole run.
It has the same fields as `InitForecast`, with `QuantileForecast`s in place of
`Forecast`s:

```julia
QuantileForecast(Hour(12), [0.05, 0.25, 0.5, 0.75, 0.95],
                 [288.9, 290.3, 291.1, 291.8, 293.0])
```

Levels need not be evenly spaced. The constructor is the ingestion check: it
throws on mismatched lengths, fewer than two levels, levels outside ``(0, 1)``
or not strictly increasing, and crossing quantiles. Bad input fails there,
instead of later as a negative spread.

Quantile runs go through EMOS end to end: [`TrainingObject`](@ref),
[`fitting_crps_naive`](@ref), [`apply_emos_correction`](@ref) and
[`evaluate_forecast`](@ref). See [EMOS from quantile
forecasts](@ref quantile-emos) for how they are summarised. MBM corrects
members, so it refuses them.

## Getting ECMWF open data

[`download_ecmwf_ens`](@ref) fetches the perturbed members (`type = pf`) of
surface parameters for one run and concatenates the GRIB messages into one file:

```julia
path = download_ecmwf_ens(Date(2026, 7, 14), "00", ("2t",), 0:3:24, "run.grib2";
                          members = 1:10, base_url = ECMWF_GCS_MIRROR)
```

It drives byte-range requests off the run's `.index` files, so it downloads only
the messages asked for. **The cost model is worth internalising**: one message is
one global field, about 0.6 MB, and you need one per member per step. Ten members
over 3-hourly steps to +144 h is 490 messages, roughly 300 MB — for a single run.

Three roots are provided. [`ECMWF_OPEN_DATA`](@ref) is the live service, which
keeps only the last few days. [`ECMWF_GCS_MIRROR`](@ref) is the Google mirror and
has the archive back to February 2024 — use it for anything historical.
[`ECMWF_AWS_MIRROR`](@ref) is the same archive but answers `503 SlowDown` under
load. Transient failures (throttling, and stalled transfers that never return an
HTTP status at all) are retried with exponential backoff.

[`read_init_forecasts`](@ref) turns the file into one `InitForecast` per station,
taking the nearest grid point with no interpolation:

```julia
run = read_init_forecasts(path, "t2m", [(39.13, -3.10)])[1]
init_time(run)      # not needed: run.timestamp already has it
```

The file must hold exactly one run, which is what `download_ecmwf_ens` writes.
Each step's field block is read once and all stations are picked out of it, so
asking for many stations at once is much cheaper than looping.

For a long archive, download each run to a temporary directory, extract the
station values, and throw the GRIB away — `examples/full_fit.jl` does this and
keeps only a few hundred kilobytes for 873 runs.

## Bringing your own data

Nothing above is required. If you have forecasts from another model, build the
containers directly:

```julia
runs = [InitForecast(t0, [Forecast(Hour(h), members(t0, h)) for h in 0:3:48], false)
        for t0 in init_times]
obs = Observations(station_times, station_values)
```

For a quantile product, the same pattern with the quantile containers:

```julia
qruns = [QuantileInitForecast(t0, [QuantileForecast(Hour(h), levels, qs(t0, h))
                                   for h in 0:3:48], false)
         for t0 in init_times]
```

Everything downstream — training, correction, interpolation, evaluation — works
on these types alone and never touches GRIB. Readers for specific data sources
deliberately live outside the package; `examples/siar.jl` is one, for the Spanish
SIAR station CSVs used throughout the examples.
