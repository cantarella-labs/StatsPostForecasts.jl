#=
MBM fit of 2 m temperature at Argamasilla de Alba on the whole station
record: every 00 UTC ENS run on the Google mirror of ECMWF open data (the
0.25° ENS starts there in early 2024) up to the last observation, 2026-07-21.

A GRIB message is a global field (~0.65 MB), so a run costs
members × steps × 0.65 MB and one HTTP request per message. Each run is
downloaded to a temporary directory, the station values are extracted and
the directory is deleted; only the extracted ensembles are kept, in CACHE,
so re-running this script skips straight to the fit. Defaults: 10 members,
+12 h and +24 h, about 19k requests, ~30 min at 8 concurrent downloads.

Run:  julia --project examples/full_fit.jl
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
using Dates, Printf, Serialization, Statistics
using StatsPostForecasts

const STATION = (39.13, -3.10)                 # Argamasilla de Alba, (lat, lon)
const CSV_PATH = joinpath(
    @__DIR__,
    "data",
    "SIAR_Argamasilla_de_Alba_CR07_horario_2024-01-01_a_2026-07-21.csv",
)
const CACHE = joinpath(@__DIR__, "cache", "argamasilla_2t_00z.jls")
const MEMBERS = 1:10
const STEPS = (12, 24)
const RUN_DATES = Date(2024, 2, 29):Day(1):Date(2026, 7, 20)   # first 0.25° ENS run on the mirror

# ---- 1. observations (same reader as the quickstart)
function read_siar_temperature(path)
    times, temp = DateTime[], Float64[]
    for line in Iterators.drop(eachline(path), 1)
        f = split(line, ',')
        t = DateTime(f[1] * " " * f[2], dateformat"dd/mm/yyyy H:M")
        push!(times, f[2] == "24:00" ? t - Day(1) : t)    # "24:00" is 00:00 of that date
        push!(temp, parse(Float64, f[3]) + 273.15)
    end
    return Observations(times, temp)
end
obs = read_siar_temperature(CSV_PATH)

# ---- 2. forecasts: fetch what the cache lacks, 8 runs at a time, saving every 40
runs = isfile(CACHE) ? deserialize(CACHE) : Dict{Date,InitForecast{Hour,Float64}}()
todo = [d for d in RUN_DATES if !haskey(runs, d)]
skipped = Date[]
@info "forecasts" cached = length(runs) to_fetch = length(todo)
mkpath(dirname(CACHE))
for chunk in Iterators.partition(todo, 40)
    asyncmap(chunk; ntasks = 8) do date
        try
            mktempdir() do dir                    # the GRIB lives only until extracted
                path = download_ecmwf_ens(
                    date,
                    "00",
                    ("2t",),
                    STEPS,
                    joinpath(dir, "run.grib2");
                    members = MEMBERS,
                    base_url = ECMWF_GCS_MIRROR,
                )
                runs[date] = read_init_forecasts(path, "t2m", [STATION])[1]
            end
        catch e
            push!(skipped, date)
            @warn "skipping $date: $(first(sprint(showerror, e), 120))"
        end
    end
    serialize(CACHE, runs)
    @info "fetched $(length(runs)) / $(length(RUN_DATES)) runs"
end
isempty(skipped) || @warn "$(length(skipped)) runs not on the mirror" skipped

# ---- 3. fit per lead time on all runs; the parameter table is what a live system would load
inits = sort(collect(values(runs)); by = r -> r.timestamp)
println(
    "\n",
    rpad("lead", 10),
    rpad("N", 6),
    rpad("CRPS raw", 12),
    rpad("CRPS mbm", 12),
    "(α, β, γ₁, γ₂)",
)
table = map(Hour.(STEPS)) do lt
    t = TrainingObject(inits, obs, lt)
    p, _ = fitting_crps(t)
    @printf(
        "%-10s%-6d%-12.3f%-12.3f(%.2f, %.3f, %.3f, %.3f)\n",
        string(lt),
        ncases(t),
        crps_min([0, 1, 1, 0], t),
        crps_min(p, t),
        p[1],
        p[2],
        p[3],
        p[4]
    )
    MBMParameters(lt, p, extrema(t.init_times), crps_min(p, t))
end
serialize(joinpath(dirname(CACHE), "argamasilla_2t_00z_mbm.jls"), table)
