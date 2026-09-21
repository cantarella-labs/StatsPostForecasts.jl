@testitem "CRPS and the MBM correction" tags=[:unit] begin
    using Dates
    using Statistics: mean
    import StatsPostForecasts as SPF

    # ---- crps against the definition it is a closed form of: ∫(F(t) − 1{t ≥ y})² dt.
    # The integrand vanishes only outside [min(x, y), max(x, y)] — integrating over
    # the ensemble range alone silently drops the whole contribution when the
    # observation falls outside the ensemble, which is the case worth scoring.
    function crps_integrated(x, y; n = 200_001)
        xs = sort(x)
        lo, hi = min(first(xs), y) - 2.0, max(last(xs), y) + 2.0
        h = (hi - lo) / (n - 1)
        s = 0.0
        for i in 1:n
            t = lo + (i - 1) * h
            s += abs2(count(<=(t), xs) / length(xs) - (t >= y ? 1.0 : 0.0))
        end
        return s * h
    end

    ens = [1.0, 2.5, 3.0, 4.75, 6.0]
    for y in (-4.0, 0.5, 3.0, 5.5, 12.0)        # inside the ensemble, and far outside it
        @test SPF.crps(ens, y) ≈ crps_integrated(ens, y) rtol = 2e-3
    end

    # a deterministic forecast scores its absolute error; a perfect one scores zero
    @test SPF.crps([4.0, 4.0, 4.0], 6.5) ≈ 2.5
    @test SPF.crps([4.0, 4.0, 4.0], 4.0) ≈ 0 atol = 1e-12
    @test SPF.crps(ens, 3.0) ≥ 0

    # invariances: shifting both, scaling both, and reordering the members
    @test SPF.crps(ens .+ 10, 13.0) ≈ SPF.crps(ens, 3.0)
    @test SPF.crps(2 .* ens, 6.0) ≈ 2 * SPF.crps(ens, 3.0)
    @test SPF.crps(reverse(ens), 3.0) ≈ SPF.crps(ens, 3.0)

    # ---- correct
    t0 = DateTime(2026, 1, 1)
    raw = [3.0, 1.0, 2.0, 5.0]                  # deliberately unsorted
    run = SPF.InitForecast(t0, [SPF.Forecast(Hour(6), copy(raw))], false)
    pars(lt, p) = SPF.MBMParameters(
        Hour(0),
        [lt],
        Dict{Hour,AbstractVector{Float64}}(lt => p),
        (t0, t0),
        [0.0],
    )

    # (α, β, γ₁, γ₂) = (0, 1, 1, 0) is the identity map: μ = x̄ and τ = 1
    id = SPF.correct(run, pars(Hour(6), [0.0, 1.0, 1.0, 0.0]))
    @test id.corrected

    # a known affine map: x̃ = α + β·x̄ + (γ₁ + γ₂/d)·(x − x̄)
    p = [2.0, 0.5, 1.5, 0.75]
    x = raw
    x̄, d = mean(x), SPF.mean_abs_diff(sort(x))
    @test SPF.correct(run, pars(Hour(6), p)).forecasts[1].ensemble ≈
          p[1] .+ p[2] * x̄ .+ (p[3] + p[4] / d) .* (x .- x̄)

    # the raw run must come back untouched: not reordered, not overwritten
    @test run.forecasts[1].ensemble == raw
    @test !run.corrected

    # guards, rather than silently doing the wrong thing
    @test_throws ArgumentError SPF.correct(id, pars(Hour(6), [0.0, 1.0, 1.0, 0.0]))
    @test_throws KeyError SPF.correct(run, pars(Hour(3), [0.0, 1.0, 1.0, 0.0]))
end

@testitem "evaluate_forecast on an ENS slice" tags=[:integration, :network] begin
    using Dates
    using Statistics: mean
    import StatsPostForecasts as SPF

    # a real run, 3 members, 3-hourly over one day — enough points in the
    # sunrise-to-sunrise window for the diurnal-cycle fit rather than the fallback
    date = Dates.today() - Day(1)
    steps = 0:3:24
    path =
        joinpath(@__DIR__, "data", "ens-2t-eval-$(Dates.format(date, "yyyymmdd"))00.grib2")
    mkpath(dirname(path))
    isfile(path) || SPF.download_ecmwf_ens(date, "00", ("2t",), steps, path; members = 1:3)

    station = (39.13, -3.10)                     # Argamasilla de Alba
    ϕ, λ = station
    run = SPF.read_init_forecasts(path, "t2m", [station])[1]

    # observations every 30 min across the run. Only their timing matters to what
    # is asserted here; the forecast being scored is real ECMWF data.
    grid = collect(run.timestamp:Minute(30):(run.timestamp+Hour(last(steps))))
    hrs = [Dates.value(t - run.timestamp) / 3.6e6 for t in grid]
    obs = SPF.Observations(grid, 290.0 .+ 5.0 .* sin.(2π .* hrs ./ 24))

    leads = [fc.lead_time for fc in run.forecasts]
    identity = SPF.MBMParameters(
        Hour(0),
        leads,
        Dict{Hour,AbstractVector{Float64}}(lt => [0.0, 1.0, 1.0, 0.0] for lt in leads),
        (run.timestamp, run.timestamp),
        zeros(length(leads)),
    )

    ev = SPF.evaluate_forecast(run, obs, identity, ϕ, λ)

    # the evaluation stays inside the run's span — outside it the interpolation
    # extrapolates, which is not evaluation
    @test first(ev.times) ≥ run.timestamp
    @test last(ev.times) ≤ run.timestamp + Hour(last(steps))
    M = length(first(run.forecasts).ensemble)
    @test size(ev.raw) == (length(ev.times), M)
    @test size(ev.corrected) == size(ev.raw)
    @test length(ev.crps_raw) == length(ev.times)

    # the identity parameters leave the ensemble alone, so both scores must agree
    @test all(isapprox.(ev.corrected, ev.raw; atol = 1e-6))
    @test all(isapprox.(ev.crps_corrected, ev.crps_raw; atol = 1e-6))

    # the residual add-back makes the interpolation exact at the forecast times
    k = findfirst(==(run.timestamp + Hour(12)), ev.times)
    @test k !== nothing
    fc12 = run.forecasts[findfirst(f -> f.lead_time == Hour(12), run.forecasts)]
    @test ev.raw[k, :] ≈ (fc12.ensemble)

    # and the reported score is what `crps` gives for that row
    @test ev.crps_raw[k] ≈ SPF.crps(ev.raw[k, :], ev.observed[k])
    @test all(≥(0), ev.crps_raw)

    # no overlap with the observations is an error, not an empty result
    far = SPF.Observations([run.timestamp - Day(10)], [290.0])
    @test_throws ArgumentError SPF.evaluate_forecast(run, far, identity, ϕ, λ)
end
