#=
What the package does, end to end, at one station
=================================================

Argamasilla de Alba (Ciudad Real). ECMWF ENS 2 m temperature, 10 members,
3-hourly out to +144 h, verified against the SIAR station record (30-minute
means).

    1. fit    the MBM and the EMOS parameters on past runs — one set per lead
              time, for one initialisation hour
    2. correct a week of later runs, none of which the fit has seen. MBM
       returns a corrected ensemble, EMOS a normal distribution N(μ, σ²)
    3. interpolate each corrected forecast onto the 30-minute grid the
       product needs, through the diurnal cycle rather than by chords
    4. compare all of it with what the station actually measured, and draw
       the last run: six days of both corrections against the station

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
using CairoMakie
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
const CACHE = joinpath(@__DIR__, "cache", "argamasilla_2t_$(INIT_HOUR)z_6d.jls")   # one cache per hour
# a week of held-out runs, the last of which the station record still covers
# to its full +144 h, and the month of runs before them to train on
const TEST_DATES = Date(2026, 7, 9):Day(1):Date(2026, 7, 15)
const TRAIN_DATES = (first(TEST_DATES)-Day(30)):Day(1):(first(TEST_DATES)-Day(1))
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
                    date,
                    INIT_HOUR,
                    ("2t",),
                    STEPS,
                    joinpath(dir, "run.grib2");
                    members = MEMBERS,
                    base_url = ECMWF_GCS_MIRROR,
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
println(
    "\nMBM and EMOS fits, $(INIT_HOUR) UTC runs, $(length(train)) of them, " *
    "$(first(TRAIN_DATES)) … $(last(TRAIN_DATES))   (every 12 h shown)",
)
println(
    rpad("lead", 10),
    rpad("N", 5),
    rpad("CRPS raw", 11),
    rpad("CRPS mbm", 11),
    rpad("CRPS emos", 11),
    rpad("MBM (α, β, γ₁, γ₂)", 32),
    "EMOS (α, β, γ₁, γ₂)",
)
leads = Hour.(STEPS)
pbylead = Dict(m => Dict{Hour,AbstractVector{Float64}}() for m in (MBM(), EMOS()))
crps_train = Dict(m => Float64[] for m in (MBM(), EMOS()))
window = (DateTime(first(TRAIN_DATES)), DateTime(last(TRAIN_DATES)))
for lt in leads
    t = TrainingObject(train, obs, lt)
    pm, _ = fitting_crps(MBM(), t)            # exact LP
    pe, _ = fitting_crps_naive(EMOS(), t)     # L-BFGS, from N(x̄, s²)
    for (m, p) in ((MBM(), pm), (EMOS(), pe))
        pbylead[m][lt] = p
        push!(crps_train[m], crps_min(m, p, t))
    end
    lt in REPORT_LEADS && @printf(
        "%-10s%-5d%-11.3f%-11.3f%-11.3f%-32s%s\n",
        string(lt),
        ncases(t),
        crps_min(MBM(), [0, 1, 1, 0], t),
        last(crps_train[MBM()]),
        last(crps_train[EMOS()]),
        @sprintf("(%.2f, %.3f, %.3f, %.3f)", pm...),
        @sprintf("(%.2f, %.3f, %.3f, %.3f)", pe...)
    )
end
# one table per method and initialisation hour — 00 and 12 UTC are never pooled
init_hour = Hour(parse(Int, INIT_HOUR))
params_mbm = Parameters(MBM(), init_hour, leads, pbylead[MBM()], window, crps_train[MBM()])
params_emos = Parameters(EMOS(), init_hour, leads, pbylead[EMOS()], window, crps_train[EMOS()])

# ---- 4. every held-out run: correct it, interpolate, score it
#         `evaluate_forecast` does all three and returns the raw and corrected
#         CRPS at every observation time inside the run's span
ϕ, λ = STATION
evals_mbm = [evaluate_forecast(MBM(), runs[d], obs, params_mbm, ϕ, λ) for d in TEST_DATES]
evals_emos = [evaluate_forecast(EMOS(), runs[d], obs, params_emos, ϕ, λ) for d in TEST_DATES]

points = DataFrame(
    init = DateTime[],
    lead = Hour[],             # at the 3-hourly steps
    observed = Float64[],
    raw = Float64[],
    mbm = Float64[],           # corrected ensemble mean
    emos = Float64[],          # μ of the predictive normal
)
for d in TEST_DATES
    run = runs[d]
    cor = correct(run, params_mbm)
    nrm = apply_emos_correction(run, params_emos)
    for (fc, cfc, μ) in zip(run.forecasts, cor.forecasts, nrm.μ)
        y = observation_at(obs, run.timestamp + fc.lead_time)
        y === nothing && continue
        push!(
            points,
            (run.timestamp, fc.lead_time, y, mean(fc.ensemble), mean(cfc.ensemble), μ),
        )
    end
end

# ---- 5. what it bought, pooled over the week
rmse(a, b) = sqrt(mean(abs2, a .- b))
allraw = reduce(vcat, [ev.crps_raw for ev in evals_mbm])
allmbm = reduce(vcat, [ev.crps_corrected for ev in evals_mbm])
allemos = reduce(vcat, [ev.crps_corrected for ev in evals_emos])
println(
    "\n$(length(TEST_DATES)) held-out runs, $(first(TEST_DATES)) … $(last(TEST_DATES)), " *
    "each to +$(last(STEPS)) h, verified against the station (K)",
)
@printf(
    "  3-hourly, raw ensemble mean        RMSE = %.2f   (N = %d)\n",
    rmse(points.raw, points.observed),
    nrow(points)
)
@printf(
    "  3-hourly, MBM-corrected mean       RMSE = %.2f\n",
    rmse(points.mbm, points.observed)
)
@printf(
    "  3-hourly, EMOS μ                   RMSE = %.2f\n",
    rmse(points.emos, points.observed)
)
@printf(
    "  30-minute, CRPS raw ensemble       %.3f   (N = %d)\n",
    mean(allraw),
    length(allraw)
)
@printf("  30-minute, CRPS MBM ensemble       %.3f\n", mean(allmbm))
@printf("  30-minute, CRPS EMOS normal        %.3f\n", mean(allemos))

bylead = combine(
    groupby(points, :lead),
    [:raw, :observed] => rmse => :raw,
    [:mbm, :observed] => rmse => :mbm,
    [:emos, :observed] => rmse => :emos,
    nrow => :N,
)
println("\nRMSE of the forecast mean by lead time (K, every 12 h):")
println(bylead[in(REPORT_LEADS).(bylead.lead), :])

# ---- 6. the figure: the forecast itself, and what the CRPS does with lead time
ev = last(evals_mbm)                              # the last held-out run
ee = last(evals_emos)
shown = runs[last(TEST_DATES)]
lead_h(t) = Dates.value(t - shown.timestamp) / 3.6e6
lo = [minimum(view(ev.corrected, i, :)) for i in eachindex(ev.times)]
hi = [maximum(view(ev.corrected, i, :)) for i in eachindex(ev.times)]
mid = [mean(view(ev.corrected, i, :)) for i in eachindex(ev.times)]
μ, σ = ee.corrected[:, 1], sqrt.(ee.corrected[:, 2])

fig = Figure(size = (1400, 860))
ax1 = Axis(
    fig[1, 1];
    ylabel = "2 m temperature (°C)",
    xticks = 0:24:last(STEPS),
    title = "Argamasilla de Alba — ENS run $(shown.timestamp), $(last(STEPS) ÷ 24) days ahead",
)
band!(
    ax1,
    lead_h.(ev.times),
    lo .- 273.15,
    hi .- 273.15;
    color = (:crimson, 0.18),
    label = "MBM ensemble, coldest to warmest member",
)
lines!(
    ax1,
    lead_h.(ev.times),
    mid .- 273.15;
    color = :crimson,
    linewidth = 2,
    label = "MBM ensemble mean, diurnal-cycle fit (30 min)",
)
band!(
    ax1,
    lead_h.(ee.times),
    μ .- 1.645 .* σ .- 273.15,
    μ .+ 1.645 .* σ .- 273.15;
    color = (:darkorange, 0.15),
    label = "EMOS 5–95 % interval",
)
lines!(
    ax1,
    lead_h.(ee.times),
    μ .- 273.15;
    color = :darkorange,
    linewidth = 2,
    label = "EMOS μ",
)
lines!(
    ax1,
    lead_h.(ev.times),
    [mean(view(ev.raw, i, :)) for i in eachindex(ev.times)] .- 273.15;
    color = :steelblue,
    linewidth = 1,
    linestyle = :dash,
    label = "raw ensemble mean",
)
lines!(
    ax1,
    lead_h.(ev.times),
    ev.observed .- 273.15;
    color = :black,
    linewidth = 1.5,
    label = "station (30 min)",
)
axislegend(ax1; position = :lt, framevisible = false, labelsize = 11)

# CRPS against lead time, averaged over every held-out run
ax2 = Axis(
    fig[2, 1];
    xlabel = "lead time (h)",
    ylabel = "CRPS (K)",
    xticks = 0:24:last(STEPS),
    title = "CRPS of the interpolated forecast, mean over $(length(TEST_DATES)) held-out runs",
)
# keyed by lead time, not by position: a run with a missing observation must
# not shift every later point into the wrong lead
crpsdf = DataFrame(lead = Float64[], raw = Float64[], mbm = Float64[], emos = Float64[])
for (d, e, f) in zip(TEST_DATES, evals_mbm, evals_emos), i in eachindex(e.times)
    push!(
        crpsdf,
        (
            Dates.value(e.times[i] - runs[d].timestamp) / 3.6e6,
            e.crps_raw[i],
            e.crps_corrected[i],
            f.crps_corrected[i],    # same times: both use the run's own span
        ),
    )
end
bylead_crps = sort(
    combine(
        groupby(crpsdf, :lead),
        :raw => mean => :raw,
        :mbm => mean => :mbm,
        :emos => mean => :emos,
    ),
    :lead,
)
lines!(
    ax2,
    bylead_crps.lead,
    bylead_crps.raw;
    color = :steelblue,
    linewidth = 1.5,
    label = "raw ensemble",
)
lines!(
    ax2,
    bylead_crps.lead,
    bylead_crps.mbm;
    color = :crimson,
    linewidth = 2,
    label = "MBM-corrected ensemble",
)
lines!(
    ax2,
    bylead_crps.lead,
    bylead_crps.emos;
    color = :darkorange,
    linewidth = 2,
    label = "EMOS normal",
)
vlines!(ax2, 0:24:last(STEPS); color = (:black, 0.12))
axislegend(ax2; position = :lt, framevisible = false, labelsize = 11)
rowsize!(fig.layout, 2, Relative(0.36))
png = joinpath(@__DIR__, "cache", "end_to_end.png")
save(png, fig)
println("\nfigure: $png")
