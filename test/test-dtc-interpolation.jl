#= 
The solar interpolation algorithm uses the diurnal cycle to interpolate temperature across the domain.
Now we have forecasts every 3 hours in the future that we want to interpolate.
For that we are going to do a few things:
=#
using Dates
using DataFrames
using DataFramesMeta
# run at 00 UTC
i0 = DateTime("2025-05-01")
dts = [i0 + i*Hour(3) for i in 1:24]
eot = equation_of_time.(year.(dts), dayofyear.(dts))
# longitude:
λ = -3.09032
# latitude
ϕ = 39.12861
sh = solar_hours.(dts, Date.(year.(dts), month.(dts), day.(dts)), λ, eot)
# solar declination (degrees)
δ = declination_cooper.(dayofyear.(dts))
ω = daylight_hours.(ϕ, δ)

#= 
Calculating DTS:
- divide the data in sampling windows (from sunrise to sunset)
- correct the solar hours to a continuous scale in the sampling window (reference date of sunrise, not current date)
- calculate daylight hours
- add parameters and fit the model
=#

# the groups are withing the sunrises between dates:
ref_dates = unique(Date.(dts))
sunrises = [sunrise_sunset(ref_date, ϕ, λ,
    declination_cooper.(dayofyear.(ref_date)), equation_of_time.(year.(ref_date), day.(ref_date))
)[1] for ref_date in ref_dates]

sampling_dataset = DataFrame(:timestamp => dts)
sampling_dataset[:, :dtc] .= 0.
sampling_dataset[:, :dtc_e] .= 0.
analysis_dataset = DataFrame(:timestamp =>
    [i0 + i*Minute(30) for i in 1:(24*6)])
analysis_dataset[:, :dtc] .= 0.

T₀ = 3.7 #°C # inital T₀, the rest is from the sunrise temp of the previous model.
# I actually have on more group than sunrises:
for i in 1:(length(sunrises)+1)
    if i == 1 #first index:
        filter = dts .< sunrises[i]
        ds = dts[filter]
        filter_a = analysis_dataset[:, :timestamp] .< sunrises[i]

    elseif i == length(sunrises)+1 # last index
        filter = dts .> sunrises[i-1]
        filter_a = analysis_dataset[:, :timestamp] .> sunrises[i-1]
        ds = dts[filter]
    else    #general case
        filter = (dts .> sunrises[i-1]) .&& (dts .< sunrises[i])
        ds = dts[filter]
        filter_a = (analysis_dataset[:, :timestamp] .> sunrises[i-1]) .&&
            (analysis_dataset[:, :timestamp] .< sunrises[i])
    end
    
    length(ds) == 0 && continue

    # calculate dts at each location:
    ref_date = i== 1 ? Date(ds[1])-Day(1) : Date(ds[1])

    eot = equation_of_time(year(ref_date), dayofyear(ref_date))
    solar_hrs = solar_hours.(ds, ref_date, λ, eot)
    solar_hrs_a = solar_hours.(analysis_dataset[filter_a, :timestamp],
        ref_date, λ, eot)
    δ = declination_cooper(dayofyear(ref_date))
    # T₀ = 3.7 #°C
    Tₐ = 22.9 #°C
    tₘ = 12.1 # solar h
    ω = daylight_hours(ϕ, δ)
    tₛ = tₘ + 4 # solar h
    θ = π/ω * (tₛ - tₘ)
    k = ω/π*cot(θ)
    t_c = dtc.(solar_hrs, T₀, Tₐ, tₘ, θ, k, ω)
    t_a = dtc.(solar_hrs_a, T₀, Tₐ, tₘ, θ, k, ω)
    sampling_dataset[filter, :dtc] .= t_c
    sampling_dataset[filter, :dtc_e] .= t_c .+ randn()*2
    analysis_dataset[filter_a, :dtc] .= t_a
    # Calculating the next series: 
    # We need to adjust the next T₀, to have a smooth curve:
    next_sunrise = solar_hours.(sunrises[i], ref_date, λ, eot)
    T_end = dtc(next_sunrise, T₀, Tₐ, tₘ, θ, k, ω)
    ref_next = ref_date + Day(1)
    eot_next = equation_of_time(year(ref_next), dayofyear(ref_next))
    δ_next = declination_cooper(dayofyear(ref_next))
    ω_next = daylight_hours(ϕ, δ)
    t_sunrise_new = solar_hours(sunrises[i], ref_next, λ, eot_next)
    T₀ = T_end - Tₐ*cos(π/ω_next*(t_sunrise_new-tₘ))
end

# Strategy 1: just use linear interpolation
using DataInterpolations
A = LinearInterpolation(sampling_dataset[:, :dtc_e],
    Dates.value.(sampling_dataset[:, :timestamp]);
    extrapolation = ExtrapolationType.Linear)
analysis_dataset[:, :li] .= A(Dates.value.(analysis_dataset[:, :timestamp]))

# Strategy 2: find the parameters of the DTC equation and fit a model to the data at each sunrise interval:
analysis_dataset[:, :dtc_fit] .= 0.

using TrustRegionLeastSquares
for i in 1:(length(sunrises)+1)
    if i == 1 #first index:
        filter = dts .< sunrises[i]
        ds = dts[filter]
        filter_a = analysis_dataset[:, :timestamp] .< sunrises[i]

    elseif i == length(sunrises)+1 # last index
        filter = dts .> sunrises[i-1]
        filter_a = analysis_dataset[:, :timestamp] .> sunrises[i-1]
        ds = dts[filter]
    else    #general case
        filter = (dts .> sunrises[i-1]) .&& (dts .< sunrises[i])
        ds = dts[filter]
        filter_a = (analysis_dataset[:, :timestamp] .> sunrises[i-1]) .&&
            (analysis_dataset[:, :timestamp] .< sunrises[i])
    end
    
    length(ds) == 0 && continue

    if length(ds) < 5
        analysis_dataset[filter_a, :dtc_fit] .= A(Dates.value.(analysis_dataset[filter_a, :timestamp]))
        continue
    end
    t_f = sampling_dataset[filter, :dtc_e]
    # calculate dts at each location:
    ref_date = i== 1 ? Date(ds[1])-Day(1) : Date(ds[1])

    eot = equation_of_time(year(ref_date), dayofyear(ref_date))
    solar_hrs = solar_hours.(ds, ref_date, λ, eot)
    solar_hrs_a = solar_hours.(analysis_dataset[filter_a, :timestamp],
        ref_date, λ, eot)
    δ = declination_cooper(dayofyear(ref_date))
    # T₀ = 3.7 #°C
    T₀ = minimum(t_f)
    Tₐ = max(maximum(t_f) - T₀, 0.5) #°C
    tₘ = 12.5 # solar h
    tₛ = 17.0
    ω = daylight_hours(ϕ, δ)
    θ = clamp(π / ω * (tₛ - tₘ), 0, π)
    k = ω/π*cot(θ)
    p0 = [T₀, Tₐ, tₘ, θ, k]
    lb = [-Inf, 0, 0, 0, 0]
    ub = [Inf, Inf, Inf, π, Inf]
    function residual!(f,p)
        T₀, Tₐ, tₘ, θ, k = p
        f .= t_f .- dtc.(solar_hrs, T₀, Tₐ, tₘ, θ, k, ω)
    end
    f = zeros(length(ds))
    residual!(f, p0)

    jacobian!(J, p) = ForwardDiff.jacobian!(J, residual!, zeros(length(ds)), p)
    J = zeros((length(ds), length(p0)))
    jacobian!(J, p0)
   
    sol = lm_trust_region!(residual!, jacobian!, p0, length(ds); lb = lb,  ub = ub)
    T₀, Tₐ, tₘ, θ, k = sol[1]
    t_a = dtc.(solar_hrs_a, T₀, Tₐ, tₘ, θ, k, ω)
    # Now the residuals:
    t_c = dtc.(solar_hrs, T₀, Tₐ, tₘ, θ, k, ω)
    Ar = LinearInterpolation(t_f .- t_c, Dates.value.(ds);
        extrapolation = ExtrapolationType.Linear)
    res = Ar(Dates.value.(analysis_dataset[filter_a, :timestamp]))
    
    analysis_dataset[filter_a, :dtc_fit] .= t_a .+ res
    
end

using GLMakie
fig = Figure()
ax = Axis(fig[1,1])
scatter!(ax, sampling_dataset[:, :timestamp], sampling_dataset[:, :dtc_e])
lines!(ax, analysis_dataset[:, :timestamp], analysis_dataset[:, :dtc])
lines!(ax, analysis_dataset[:, :timestamp], analysis_dataset[:, :li])
lines!(ax, analysis_dataset[:, :timestamp], analysis_dataset[:, :dtc_fit])