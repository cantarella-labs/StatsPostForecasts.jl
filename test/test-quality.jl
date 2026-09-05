@testitem "Aqua" tags=[:quality] begin
    using Aqua
    Aqua.test_all(StatsPostForecasts)
end

@testitem "JET" tags=[:quality] begin
    using JET
    JET.test_package(StatsPostForecasts; target_modules = (StatsPostForecasts,))
end
