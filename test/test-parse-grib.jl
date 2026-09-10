@testitem "ECMWF open-data ENS slice" tags=[:integration, :network] begin
    using Dates
    import StatsPostForecasts as SPF

    # Yesterday's 00 UTC run is fully published; only the last few days stay
    # online, so the slice is fetched once per day into test/data (3 members,
    # steps 0 and 6 of 2t, ~4 MB).
    date = Dates.today() - Day(1)
    path = joinpath(@__DIR__, "data", "ens-2t-$(Dates.format(date, "yyyymmdd"))00.grib2")
    mkpath(dirname(path))
    isfile(path) || SPF.download_ecmwf_ens(date, "00", ("2t",), (0, 6), path; members = 1:3)

    ds = SPF.GRIBDataset(path)
    @test SPF.GRIBDatasets.dimnames(ds["t2m"]) ==
          ["lon", "lat", "heightAboveGround", "number", "valid_time"]
    @test ds["number"][:] == [1, 2, 3]
    @test SPF.init_time(ds) == DateTime(date)
    @test SPF.lead_times(ds) == [Hour(0), Hour(6)]

    station = (39.13, -3.10)                          # Argamasilla de Alba
    r = SPF.read_init_forecasts(ds, "t2m", [station])[1]
    @test r.timestamp == DateTime(date)
    @test !r.corrected
    @test [f.lead_time for f in r.forecasts] == [Hour(0), Hour(6)]
    @test all(f -> length(f.ensemble) == 3, r.forecasts)
    @test all(f -> all(250 .< f.ensemble .< 320), r.forecasts)   # K, plausible for Spain

    # members come out in file order, straight from the raw field
    lat, lon = ds["lat"][:], ds["lon"][:]
    i, j = argmin(abs.(lon .- station[2])), argmin(abs.(lat .- station[1]))
    @test r.forecasts[2].ensemble == ds["t2m"][i, j, 1, :, 2]

    # the same station on the 0–360 convention hits the same grid point
    r360 = SPF.read_init_forecasts(ds, "t2m", [(station[1], station[2] + 360)])[1]
    @test r360.forecasts[1].ensemble == r.forecasts[1].ensemble

    # path convenience method
    @test SPF.read_init_forecasts(path, "t2m", [station])[1].forecasts[1].ensemble ==
          r.forecasts[1].ensemble
end
