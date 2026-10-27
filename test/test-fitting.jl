@testsnippet Data begin
    using Dates, Statistics
    import StatsPostForecasts as SPF
    # SIAR half-hourly CSV (Fecha,Hora,Temp_Media_C,…), times in UTC (checked: the
    # 12 UTC ensemble mean matches the 12:00 row within 0.1 K). Midnight is written
    # "24:00" with the date of the day it starts; Dates rolls that into the next
    # day, hence the one-day shift.
    function read_siar_temperature(path)
        times, temp = DateTime[], Float64[]
        for line in Iterators.drop(eachline(path), 1)
            f = split(line, ',')
            t = DateTime(f[1] * " " * f[2], dateformat"dd/mm/yyyy H:M")
            push!(times, f[2] == "24:00" ? t - Day(1) : t)
            push!(temp, parse(Float64, f[3]) + 273.15)
        end
        return SPF.Observations(times, temp)
    end
    obs = read_siar_temperature(
        joinpath(
            @__DIR__,
            "..",
            "examples",
            "data",
            "SIAR_Argamasilla_de_Alba_CR07_horario_2024-01-01_a_2026-07-21.csv",
        ),
    )

    # six 00 UTC runs, 10 members at +12 h and +24 h, ~12 MB each, fetched once
    station = (39.13, -3.10)                                   # Argamasilla de Alba
    dir = joinpath(@__DIR__, "data")
    mkpath(dir)
    runs = map(Date(2026, 7, 14):Day(1):Date(2026, 7, 19)) do date
        path = joinpath(dir, "ens-2t-$(Dates.format(date, "yyyymmdd"))00.grib2")
        isfile(path) || SPF.download_ecmwf_ens(
            date,
            "00",
            ("2t",),
            (12, 24),
            path;
            members = 1:10,
            base_url = SPF.ECMWF_GCS_MIRROR,
        )
        SPF.read_init_forecasts(path, "t2m", [station])[1]
    end
end

@testitem "CRPS - MBM fit on six July 2026 ENS runs" setup = [Data] tags=[:integration, :network, :slow] begin
    using Dates, Statistics
    import StatsPostForecasts as SPF


    raw = [0.0, 1.0, 1.0, 0.0]
    for lt in (Hour(12), Hour(24))
        t = SPF.TrainingObject(runs, obs, lt)
        @test SPF.ncases(t) == 6
        @test SPF.nmembers(t) == 10

        # identity parameters: the correction returns the raw members and the objective
        # equals the ensemble CRPS computed from scratch, E|X − y| − ½ E|X − X′|
        direct = mean(runs) do r
            x = sort(r.forecasts[findfirst(f -> f.lead_time == lt, r.forecasts)].ensemble)
            y = SPF.observation_at(obs, r.timestamp + lt)
            mean(abs.(x .- y)) - sum(abs(a - b) for a in x, b in x) / (2 * length(x)^2)
        end
        @test SPF.crps_min(SPF.MBM(), raw, t) ≈ direct
        x = sort(runs[1].forecasts[1].ensemble)
        xc = similar(x)
        SPF.mbm_correction!(xc, x, raw, mean(x), SPF.mean_abs_diff(x))
        @test xc ≈ x

        # LP fit: beats the raw ensemble, objective consistent, spread parameters ≥ 0
        p, model = SPF.fitting_crps(SPF.MBM(), t)
        J = SPF.crps_min(SPF.MBM(), p, t)
        @test J < SPF.crps_min(SPF.MBM(), raw, t)
        @test J ≈ SPF.objective_value(model) atol = 1e-6
        @test p[3] >= 0 && p[4] >= 0

        # naive fit (box-constrained L-BFGS on the same objective) reaches the LP optimum
        pn, _ = SPF.fitting_crps_naive(SPF.MBM(), t)
        @test SPF.crps_min(SPF.MBM(), pn, t) ≈ J rtol = 1e-4
        @test pn[3] >= 0 && pn[4] >= 0
    end
end


@testitem "CRPS - EMOS fit on six July 2026 ENS runs" setup = [Data] tags=[:integration, :network, :slow] begin
    using Dates, Statistics
    import StatsPostForecasts as SPF

    identity = [0.0, 1.0, 0.0, 1.0]
    for lt in (Hour(12), Hour(24))
        t = SPF.TrainingObject(runs, obs, lt)
        @test SPF.ncases(t) == 6
        @test SPF.nmembers(t) == 10

        # identity parameters give N(x̄, s²); the objective is the mean normal CRPS of
        # those, computed here from the raw runs (not the ensemble CRPS of the members)
        direct = mean(runs) do r
            x = r.forecasts[findfirst(f -> f.lead_time == lt, r.forecasts)].ensemble
            y = SPF.observation_at(obs, r.timestamp + lt)
            SPF.crps(SPF.EMOS(), mean(x), var(x; corrected = false), y)
        end
        J_id = SPF.crps_min(SPF.EMOS(), identity, t)
        @test J_id ≈ direct

        # fit from the raw ensemble: converges, beats it, variance parameters ≥ 0
        p, res = SPF.fitting_crps_naive(SPF.EMOS(), t)
        J = SPF.crps_min(SPF.EMOS(), p, t)
        @test SPF.Optim.converged(res)
        @test J < J_id
        @test p[3] >= 0 && p[4] >= 0

        # the objective is not convex: an OLS start (α, β from y ~ x̄, γ₁ the residual
        # variance) must reach the same optimum
        X = [ones(SPF.ncases(t)) t.xmean]
        ab = X \ t.y
        p0 = [ab; var(t.y - X * ab; corrected = false); 1e-10]
        p_ols, _ = SPF.fitting_crps_naive(SPF.EMOS(), t; p0 = p0)
        @test SPF.crps_min(SPF.EMOS(), p_ols, t) ≈ J rtol = 1e-3
    end
end
