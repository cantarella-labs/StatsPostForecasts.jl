@testitem "DTC interpolation of a 3-hourly series" tags=[:unit] begin
    using Dates
    using Statistics: mean
    import StatsPostForecasts as SPF

    ϕ, λ = 39.12861, -3.09032                       # Argamasilla de Alba
    i0 = DateTime(2025, 5, 1)
    fine = [i0 + Minute(30i) for i in 0:144]        # 3 days at 30 min: what the product needs

    # truth: one DTC per sunrise-to-sunrise window with a fixed shape, T₀ chained
    # across the sunrises so the series is continuous
    Tₐ, tₘ, tₛ = 22.9, 12.1, 16.1
    truth = zeros(length(fine))
    let T₀ = 3.7                                    # `let`: a testitem body is top-level scope
        for w in SPF.sunrise_windows(fine, ϕ, λ)
            θ = π / w.ω * (tₛ - tₘ)
            k = w.ω / π * cot(θ)
            truth[w.mask] .= SPF.dtc.(SPF.window_solar_hours(fine[w.mask], w, λ),
                                      T₀, Tₐ, tₘ, θ, k, w.ω)
            # value at the closing sunrise, re-expressed in the next window's frame
            T_end = SPF.dtc(SPF.solar_hours(w.t_end, w.ref_date, λ, w.eot_min),
                            T₀, Tₐ, tₘ, θ, k, w.ω)
            d = w.ref_date + Day(1)
            eot = SPF.equation_of_time(year(d), dayofyear(d))
            ω = SPF.daylight_hours(ϕ, SPF.declination_cooper(dayofyear(d)))
            T₀ = T_end - Tₐ * cos(π / ω * (SPF.solar_hours(w.t_end, d, λ, eot) - tₘ))
        end
    end

    fc_times, fc_values = fine[1:6:end], truth[1:6:end]   # what the forecast delivers, every 3 h

    fit = SPF.interpolate_forecast(fc_times, fc_values, fine, ϕ, λ)
    # min_points above the window size sends every window to the linear fallback
    lin = SPF.interpolate_forecast(fc_times, fc_values, fine, ϕ, λ; min_points = typemax(Int))

    rmse(a, b) = sqrt(mean(abs2, a .- b))
    @test rmse(lin, truth) > 0.5                 # 3-hourly chords do lose the curvature
    @test rmse(fit, truth) < 0.5 * rmse(lin, truth)

    # the leading window holds 2 forecast points, too few to fit, so both methods
    # interpolate that first night linearly. Past the first sunrise the DTC is
    # actually fitted, and there the curve comes back to machine precision.
    windows = SPF.sunrise_windows(fc_times, ϕ, λ)
    @test count(w -> count(w.mask) >= 5, windows) == 3
    day = fine .>= windows[1].t_end
    @test rmse(lin[day], truth[day]) > 0.5
    @test rmse(fit[day], truth[day]) < 0.01

    # the windows tile the output grid: no time is left unassigned (truth is 4–29 °C
    # here, so a zero can only mean an uncovered output time)
    @test !any(iszero, fit)
    # the residual add-back reproduces the forecast values exactly
    @test SPF.interpolate_forecast(fc_times, fc_values, fc_times, ϕ, λ) ≈ fc_values
end
