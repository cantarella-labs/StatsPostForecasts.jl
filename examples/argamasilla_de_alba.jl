#=
Example: MBM calibration of 2 m temperature and dew point at
Argamasilla de Alba (Ciudad Real, Spain) against station observations
=======================================================================

Steps
-----
1. Read the station CSV (30-min means) into `Observations` for temperature
   and for dew point (derived from temperature and relative humidity).
2. Read a set of archived ECMWF ENS GRIB files (one per initialisation,
   same initialisation hour) with GRIBDatasets.jl.
3. Build one `TrainingObject` per lead time on a training subset of runs,
   fit the MBM parameters with the CRPS LP, and evaluate the CRPS of raw
   and corrected ensembles on the held-out runs.
4. Print a table CRPS_raw / CRPS_mbm / CRPSS per lead time and show one
   corrected run against the observations.

Assumptions you must check
--------------------------
* `GRIB_DIR` holds files named like `ecmwf-YYYYMMDDHH-ens-sfc.grib2`, each
  with the perturbed members of `2t` and `2d` for one run, all with the
  same initialisation hour (`INIT_HOUR`). Open-data only keeps the last
  few days online, so for 2024 runs these must come from an archive.
* The CSV timestamps are assumed to be UTC (`CSV_TZ_OFFSET = Hour(0)`). If
  the station reports local time set the offset accordingly — SIAR-style
  networks often use local time, and a 1–2 h shift is the single biggest
  error you can make in this comparison.
* Only lead times that are multiples of 30 min match an observation
  exactly; ENS steps are 3-hourly, so every step matches.

Requires the other files of the package to be loaded first.
=#

# Station metadata
#= 
[argamasilla_cr07]
name = "Argamasilla de Alba (SIAR CR07)"
lat = 39.13
lon = -3.09
elev_m = 650.0
siar_station = "13-7"
siar_ccaa = "CAM"
utc_offset_h = 0
 =#

using Dates, CSV, DataFrames, Statistics, Printf
for f in ("data_structures.jl", "MBM.jl", "training.jl", "parse_grib.jl")
    include(joinpath(@__DIR__, "..", "src", f))
end

# ---------------------------------------------------------------- settings
const STATION = (39.13, -3.10)          # Argamasilla de Alba, (lat, lon)
const CSV_PATH = "data/argamasilla_del_alba.csv"
const CSV_TZ_OFFSET = Hour(0)                 # CSV time → UTC: t_utc = t_csv - offset
const GRIB_DIR = "grib"
const INIT_HOUR = 0                       # use only 00 UTC runs
const N_TEST = 3                       # last runs are held out for evaluation

# --------------------------------------------------- 1. station observations
"""
    dewpoint(T_c, rh_pct) -> T_d in °C

Magnus (August–Roche–Magnus) inversion; use the same constants both ways.
"""
function dewpoint(T_c, rh_pct)
    a, b = 17.625, 243.04
    γ = log(rh_pct / 100) + a * T_c / (b + T_c)
    return b * γ / (a - γ)
end

function read_station(path)
    df = CSV.read(path, DataFrame; dateformat = "dd/mm/yyyy")
    # "Hora" is "H:MM" or "HH:MM"; build a DateTime per row, then shift to UTC
    times =
        [DateTime(Date(r.Fecha)) + Time(r.Hora, "H:M") - CSV_TZ_OFFSET for r in eachrow(df)]
    # station is in °C; ECMWF fields are in K → work in K throughout
    T_k = df.Temp_Media_C .+ 273.15
    Td_k = dewpoint.(df.Temp_Media_C, df.Hum_Media_pct) .+ 273.15
    # drop rows with missing values and enforce increasing time
    keep = .!ismissing.(T_k) .& .!ismissing.(Td_k)
    order = sortperm(times[keep])
    t = times[keep][order]
    return (
        temp = Observations(t, Float64.(T_k[keep][order])),
        dewp = Observations(t, Float64.(Td_k[keep][order])),
    )
end

obs = read_station(CSV_PATH)
@info "observations" first(obs.temp.times) last(obs.temp.times) length(obs.temp.times)

# ----------------------------------------------------------- 2. forecasts
grib_files = sort(filter(f -> endswith(f, ".grib2"), readdir(GRIB_DIR; join = true)))

runs_t = InitForecast{Hour,Float64}[]
runs_td = InitForecast{Hour,Float64}[]
for f in grib_files
    ds = GRIBDataset(f)
    hour(init_time(ds)) == INIT_HOUR || continue
    push!(runs_t, read_init_forecasts(ds, "t2m", [STATION])[1])
    push!(runs_td, read_init_forecasts(ds, "d2m", [STATION])[1])
end
@info "runs loaded" length(runs_t) first(runs_t).timestamp last(runs_t).timestamp

# ----------------------------------------------------- 3. fit and evaluate
"""
    crps_of_run(run, obs, p_by_lead) -> Vector{(lead_time, crps_raw, crps_mbm)}

For one held-out run, ensemble CRPS of the raw and of the MBM-corrected
ensemble at every lead time that has an observation. Uses the same CRPS
formula as `crps_min` (mean |member − y| minus half the absolute spread).
"""
function crps_of_run(run::InitForecast, obs::Observations, p_by_lead::Dict)
    out = Tuple{Hour,Float64,Float64}[]
    for fc in run.forecasts
        y = observation_at(obs, run.timestamp + fc.lead_time)
        (y === nothing || !haskey(p_by_lead, fc.lead_time)) && continue
        x = sort(fc.ensemble)
        m = mean(x)
        d = mean_abs_diff(x)
        crps_raw = mean(abs.(x .- y)) - d/2
        xc = similar(x)
        mbm_correction!(xc, x, p_by_lead[fc.lead_time], m, d)
        crps_mbm = mean(abs.(xc .- y)) - mean_abs_diff(sort(xc))/2
        push!(out, (fc.lead_time, crps_raw, crps_mbm))
    end
    return out
end

function fit_all_leads(train_runs, obs)
    p_by_lead = Dict{Hour,Vector{Float64}}()
    for lt in lead_times_of(train_runs)
        t = try
            TrainingObject(train_runs, obs, lt)
        catch e
            e isa ArgumentError && continue        # no cases at this lead time
            rethrow()
        end
        p, _ = fitting_crps(t)
        p_by_lead[lt] = p
    end
    return p_by_lead
end

lead_times_of(runs) = sort(unique(fc.lead_time for r in runs for fc in r.forecasts))

function evaluate(runs, obs, label)
    train = runs[1:(end-N_TEST)]
    test = runs[(end-N_TEST+1):end]
    p_by_lead = fit_all_leads(train, obs)

    # pool CRPS over the test runs, per lead time
    acc = Dict{Hour,Vector{Tuple{Float64,Float64}}}()
    for run in test, (lt, cr, cm) in crps_of_run(run, obs, p_by_lead)
        push!(get!(acc, lt, Tuple{Float64,Float64}[]), (cr, cm))
    end

    println("\n$label — trained on $(length(train)) runs, evaluated on $(length(test))")
    println(rpad("lead", 8), rpad("CRPS raw", 12), rpad("CRPS mbm", 12), "CRPSS")
    for lt in sort(collect(keys(acc)))
        cr = mean(first.(acc[lt]))
        cm = mean(last.(acc[lt]))
        @printf("%-8s%-12.3f%-12.3f%+.3f\n", string(lt), cr, cm, 1 - cm/cr)
    end
    return p_by_lead
end

p_t = evaluate(runs_t, obs.temp, "2 m temperature (K)")
p_td = evaluate(runs_td, obs.dewp, "2 m dew point (K)")

# -------------------------------------------- 4. one corrected run, by eye
last_run = runs_t[end]
println(
    "\nLast run $(last_run.timestamp): observation, raw mean ± sd, corrected mean ± sd (°C)",
)
for fc in last_run.forecasts
    y = observation_at(obs.temp, last_run.timestamp + fc.lead_time)
    (y === nothing || !haskey(p_t, fc.lead_time)) && continue
    x = sort(fc.ensemble)
    m = mean(x)
    d = mean_abs_diff(x)
    xc = similar(x)
    mbm_correction!(xc, x, p_t[fc.lead_time], m, d)
    @printf(
        "%-8s obs %6.2f   raw %6.2f ± %4.2f   mbm %6.2f ± %4.2f\n",
        string(fc.lead_time),
        y - 273.15,
        m - 273.15,
        std(x),
        mean(xc) - 273.15,
        std(xc)
    )
end
