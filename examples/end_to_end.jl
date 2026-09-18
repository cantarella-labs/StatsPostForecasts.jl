#=
What the package does, end to end, at one station
=================================================

Argamasilla de Alba (Ciudad Real). ECMWF ENS 2 m temperature, 10 members,
3-hourly out to +144 h, verified against the SIAR station record (30-minute
means).

    1. fit    the MBM parameters on past runs — one set per lead time, for
              one initialisation hour
    2. correct a week of later runs, none of which the fit has seen
    3. interpolate each corrected forecast onto the 30-minute grid the
       product needs, through the diurnal cycle rather than by chords
    4. compare all of it with what the station actually measured, and draw
       the last run: six days of corrected ensemble against the station

Parameters are per (initialisation hour, lead time). 00 UTC and 12 UTC runs
have different error structure — a +12 h forecast from 00 UTC verifies at
midday, one from 12 UTC verifies at midnight — so they are never pooled.
`INIT_HOUR` picks one, and the cache and the printed table are labelled with
it; to build both tables, run the script once per hour.

+144 h is where the open-data ENS stops being 3-hourly (6-hourly beyond).
The diurnal-cycle fit needs five points in a sunrise-to-sunrise window, and
6-hourly spacing gives four, so it would fall back to linear past that.

Run:  julia --project=examples examples/end_to_end.jl

The runs come from the Google mirror of ECMWF open data (the archive back to
Feb 2024) and the station values are cached in examples/cache, so only the
first run of the script pays for them: ~0.6 MB per member per step, here
about 11 GB in total, some tens of minutes at 8 concurrent downloads.
=#

using Dates, Printf, Serialization, Statistics
using DataFrames
using GLMakie
using StatsPostForecasts
include("siar.jl")

const STATION = (39.13, -3.10)                 # Argamasilla de Alba, (lat, lon)
const CSV_PATH = joinpath(
    @__DIR__,
    "data",
    "SIAR_Argamasilla_de_Alba_CR07_horario_2024-01-01_a_2026-07-21.csv",
)
const INIT_HOUR = "00"                         # "00", "06", "12" or "18" — never pooled
const MEMBERS = 1:10
const STEPS = 0:3:144                          # six days, 3-hourly
const CACHE =
    joinpath(@__DIR__, "cache", "argamasilla_2t_$(INIT_HOUR)z_6d.jls")   # one cache per hour
# a week of held-out runs, the last of which the station record still covers
# to its full +144 h, and the month of runs before them to train on
const TEST_DATES = Date(2026, 7, 9):Day(1):Date(2026, 7, 15)
const TRAIN_DATES = (first(TEST_DATES) - Day(30)):Day(1):(first(TEST_DATES) - Day(1))
const REPORT_LEADS = Hour.(0:12:144)           # printing all 49 leads is a wall of text

# ---- 1. observations: the station's own 30-minute record
obs = read_siar_temperature(CSV_PATH)
@info "observations" first(obs.times) last(obs.times) length(obs.times)

# ---- 2. forecasts: one run per day, station values cached so a re-run is instant
runs = isfile(CACHE) ? deserialize(CACHE) : Dict{Date,InitForecast{Hour,Float64}}()
todo = [d for d in [TRAIN_DATES; TEST_DATES] if !haskey(runs, d)]
if !isempty(todo)
    @info "downloading $(length(todo)) runs (the GRIB is thrown away once read)"
    mkpath(dirname(CACHE))
    # eight at a time, cached after every batch: this is ~11 GB and hours of
    # requests, so a failure must cost one batch, not the whole download. Re-run
    # the script to pick up where it stopped.
    for batch in Iterators.partition(todo, 8)
        asyncmap(batch; ntasks = 8) do date
            mktempdir() do dir
                path = download_ecmwf_ens(
                    date, INIT_HOUR, ("2t",), STEPS, joinpath(dir, "run.grib2");
                    members = MEMBERS, base_url = ECMWF_GCS_MIRROR,
                )
                runs[date] = read_init_forecasts(path, "t2m", [STATION])[1]
            end
        end
        serialize(CACHE, runs)
        @info "cached $(length(runs)) of $(length(TRAIN_DATES) + length(TEST_DATES)) runs"
    end
end

# ---- 3. fit: one parameter set per lead time, trained on past runs only
train = [runs[d] for d in TRAIN_DATES]
println("\nMBM fit, $(INIT_HOUR) UTC runs, $(length(train)) of them, " *
        "$(first(TRAIN_DATES)) … $(last(TRAIN_DATES))   (every 12 h shown)")
println(rpad("lead", 10), rpad("N", 5), rpad("CRPS raw", 11), rpad("CRPS mbm", 11),
        "(α, β, γ₁, γ₂)")
params = map(Hour.(STEPS)) do lt
    t = TrainingObject(train, obs, lt)
    p, _ = fitting_crps(t)
    lt in REPORT_LEADS && @printf("%-10s%-5d%-11.3f%-11.3f(%.2f, %.3f, %.3f, %.3f)\n",
                                  string(lt), ncases(t), crps_min([0, 1, 1, 0], t),
                                  crps_min(p, t), p...)
    MBMParameters(lt, p, extrema(t.init_times), crps_min(p, t))
end

"Apply the fitted parameters to every lead time of one run, member by member."
function correct(run, params)
    fcs = map(run.forecasts) do fc
        p = params[findfirst(q -> q.lead_time == fc.lead_time, params)]
        x = sort(fc.ensemble)                  # sorted, so member k is the k-th coldest
        xc = similar(x)
        mbm_correction!(xc, x, p.p, mean(x), mean_abs_diff(x))
        Forecast(fc.lead_time, xc)
    end
    return InitForecast(run.timestamp, fcs, true)
end

"The run's valid times, and the 30-minute grid the product wants them on."
function grids(run)
    t = [run.timestamp + fc.lead_time for fc in run.forecasts]
    return (fc_times = t, out_times = first(t):Minute(30):last(t))
end

# ---- 4. every held-out run: correct it, interpolate it, verify it
ϕ, λ = STATION
points = DataFrame(init = DateTime[], lead = Hour[],             # at the 3-hourly steps
                   observed = Float64[], raw = Float64[], mbm = Float64[])
fine = DataFrame(init = DateTime[], time = DateTime[],           # on the 30-minute grid
                 observed = Float64[], linear = Float64[], dtc = Float64[])

for date in TEST_DATES
    run = runs[date]
    corrected = correct(run, params)

    for (fc, cfc) in zip(run.forecasts, corrected.forecasts)
        y = observation_at(obs, run.timestamp + fc.lead_time)
        y === nothing && continue
        push!(points, (run.timestamp, fc.lead_time, y, mean(fc.ensemble), mean(cfc.ensemble)))
    end

    fc_times, out_times = grids(corrected)
    fc_values = [mean(fc.ensemble) for fc in corrected.forecasts]
    dtc = interpolate_forecast(fc_times, fc_values, out_times, ϕ, λ)
    chords =
        interpolate_forecast(fc_times, fc_values, out_times, ϕ, λ; min_points = typemax(Int))
    y = [observation_at(obs, t) for t in out_times]
    keep = .!isnothing.(y)                     # the station misses a reading now and then
    append!(fine, DataFrame(init = run.timestamp, time = collect(out_times)[keep],
                            observed = Float64.(y[keep]), linear = chords[keep],
                            dtc = dtc[keep]))
end

# ---- 5. what it bought, pooled over the week
rmse(a, b) = sqrt(mean(abs2, a .- b))
println("\n$(length(TEST_DATES)) held-out runs, $(first(TEST_DATES)) … $(last(TEST_DATES)), " *
        "each to +144 h, verified against the station (K)")
@printf("  3-hourly, raw ensemble mean        RMSE = %.2f   (N = %d)\n",
        rmse(points.raw, points.observed), nrow(points))
@printf("  3-hourly, MBM-corrected mean       RMSE = %.2f\n", rmse(points.mbm, points.observed))
@printf("  30-minute, linear between the 3 h  RMSE = %.2f   (N = %d)\n",
        rmse(fine.linear, fine.observed), nrow(fine))
@printf("  30-minute, diurnal-cycle fit       RMSE = %.2f\n", rmse(fine.dtc, fine.observed))

bylead = combine(groupby(points, :lead),
                 [:raw, :observed] => rmse => :raw,
                 [:mbm, :observed] => rmse => :mbm,
                 nrow => :N)
println("\nby lead time (K, every 12 h):")
println(bylead[in(REPORT_LEADS).(bylead.lead), :])

# ---- 6. the last run, all six days of it, ensemble and all
run = runs[last(TEST_DATES)]
corrected = correct(run, params)
fc_times, out_times = grids(corrected)
M = length(first(corrected.forecasts).ensemble)
# members are sorted per lead time, so member 1 and member M are the envelope
member(m) = interpolate_forecast(fc_times, [fc.ensemble[m] for fc in corrected.forecasts],
                                 out_times, ϕ, λ)
lo, hi = member(1), member(M)
mid = interpolate_forecast(fc_times, [mean(fc.ensemble) for fc in corrected.forecasts],
                           out_times, ϕ, λ)
f = fine[fine.init .== run.timestamp, :]

# x is lead time in days: it is what the plot is about, and Makie's `band!`
# does not take DateTime (`lines!` does, which is a trap worth avoiding here)
lead_days(t) = Dates.value(t - run.timestamp) / 86_400_000
fig = Figure(size = (1300, 520))
ax = Axis(fig[1, 1];
          xlabel = "lead time (days from $(run.timestamp) UTC)",
          ylabel = "2 m temperature (°C)",
          xticks = 0:(last(STEPS) ÷ 24),
          title = "Argamasilla de Alba — ENS run $(run.timestamp), six days ahead")
band!(ax, lead_days.(out_times), lo .- 273.15, hi .- 273.15;
      color = (:crimson, 0.18), label = "corrected ensemble, coldest to warmest member")
lines!(ax, lead_days.(out_times), mid .- 273.15; color = :crimson, linewidth = 2,
       label = "corrected ensemble mean, diurnal-cycle fit (30 min)")
lines!(ax, lead_days.(fc_times), [mean(fc.ensemble) for fc in run.forecasts] .- 273.15;
       color = :steelblue, linewidth = 1, linestyle = :dash,
       label = "raw ensemble mean (3 h)")
lines!(ax, lead_days.(f.time), f.observed .- 273.15; color = :black, linewidth = 1.5,
       label = "station (30 min)")
axislegend(ax; position = :lt, framevisible = false)
png = joinpath(@__DIR__, "cache", "end_to_end.png")
save(png, fig)
println("\nfigure: $png")
