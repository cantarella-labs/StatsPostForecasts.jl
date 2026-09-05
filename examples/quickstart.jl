#=
Quick check: a few ECMWF ENS runs at one station, fitted against the
station's own observations.

data.ecmwf.int keeps only the last ~3 days, so the runs come from the Google
Cloud mirror, which has the archive back to 2024. Six 00 UTC runs of July 2026,
10 members, lead times 12 h and 24 h, verified against the SIAR half-hourly
temperatures in examples/data.

The CSV times are UTC (checked: the 12 UTC ensemble mean matches the 12:00
row within 0.1 K; a 2 h shift multiplies the CRPS by seven). Midnight is
written "24:00" with the date of the day it starts ("02/01/2024,24:00" sits
between 01/01 23:30 and 02/01 0:30); Dates parses 24:00 as the next day,
hence the one-day shift below.

Run from the package directory:  julia --project examples/quickstart.jl
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

using Dates
using StatsPostForecasts

const STATION = (39.13, -3.10)                  # Argamasilla de Alba, (lat, lon)
const CSV_PATH = joinpath(
    @__DIR__,
    "data",
    "SIAR_Argamasilla_de_Alba_CR07_horario_2024-01-01_a_2026-07-21.csv",
)
const GRIB_DIR = joinpath(@__DIR__, "grib")
const DATES = Date(2026, 7, 14):Day(1):Date(2026, 7, 19)
mkpath(GRIB_DIR)

# ---- 1. observations: 2 m temperature in K. CSV columns: Fecha,Hora,Temp_Media_C,...
obs = let times = DateTime[], temp = Float64[]
    for line in Iterators.drop(eachline(CSV_PATH), 1)
        f = split(line, ',')
        t = DateTime(f[1] * " " * f[2], dateformat"dd/mm/yyyy H:M")
        push!(times, f[2] == "24:00" ? t - Day(1) : t)
        push!(temp, parse(Float64, f[3]) + 273.15)
    end
    Observations(times, temp)
end
@info "observations" first(obs.times) last(obs.times) length(obs.times)

# ---- 2. forecasts: one file per run, 10 members at +12 h and +24 h (~13 MB each)
runs = map(DATES) do date
    path = joinpath(GRIB_DIR, "ens-2t-$(Dates.format(date, "yyyymmdd"))00.grib2")
    isfile(path) || download_ecmwf_ens(
        date,
        "00",
        ("2t",),
        (12, 24),
        path;
        members = 1:10,
        base_url = ECMWF_GCS_MIRROR,
    )
    read_init_forecasts(path, "t2m", [STATION])[1]
end
@info "runs loaded" [r.timestamp for r in runs]

# ---- 3. fit per lead time, compare in-sample CRPS with the raw ensemble
for lt in (Hour(12), Hour(24))
    t = TrainingObject(runs, obs, lt)
    p, _ = fitting_crps(t)
    println("lead $lt: N = $(ncases(t)) cases, M = $(nmembers(t)) members")
    println("  raw    CRPS = $(round(crps_min([0, 1, 1, 0], t); digits = 3)) K")
    println(
        "  fitted CRPS = $(round(crps_min(p, t); digits = 3)) K   (α, β, γ₁, γ₂) = $(round.(p; digits = 3))",
    )
end
