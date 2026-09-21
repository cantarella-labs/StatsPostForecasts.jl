#=
MWE: the DTC window fit throws PosDefException.

Data: raw ECMWF ENS 2 m temperature (K), member 6 of the 2024-03-05 00 UTC run at
Argamasilla de Alba (39.13 N, 3.10 W), the eight 3-hourly steps of the
sunrise-to-sunrise window starting 2024-03-08, as solar hours in that window's
frame (what `interpolate_forecast` passes to `fit_dtc_window`).

Part A is the call as StatsPostForecasts makes it. Part B is the same problem with
only TrustRegionLeastSquares + ForwardDiff.
=#

import StatsPostForecasts as SPF
using TrustRegionLeastSquares

t = [8.606572244235696, 11.606572244235696, 14.606572244235696, 17.606572244235696,
     20.606572244235696, 23.606572244235696, 26.606572244235696, 29.606572244235696]
T = [276.3280334472656, 279.72369384765625, 282.4947204589844, 280.06300354003906,
     277.80686950683594, 278.65545654296875, 279.20155334472656, 280.1933135986328]
ω = 11.433423174345474

# ---- A: StatsPostForecasts
try
    SPF.fit_dtc_window(t, T, ω)
catch e
    println("A: ", sprint(showerror, e))
end

# ---- B: TrustRegionLeastSquares only (dtc, x0 and bounds as in StatsPostForecasts)
function dtc(t, T₀, Tₐ, tₘ, θ, k, ω)
    tₛ = tₘ + ω / π * θ
    δT = Tₐ * (cos(θ) - k * π / ω * sin(θ))
    a = Tₐ * k * π / ω * sin(θ)
    return t < tₛ ? T₀ + Tₐ * cos(π / ω * (t - tₘ)) : (T₀ + δT) + a * exp(-(t - tₛ) / k)
end
θ0 = π / ω * (17.0 - 12.5)
x0 = [minimum(T), maximum(T) - minimum(T), 12.5, θ0, ω / π * cot(θ0)]
lb = [-Inf, 1e-3, 6.0, 1e-3, 1e-3]
ub = [Inf, Inf, 18.0, π - 1e-3, Inf]
res!(f, x) = (f .= T .- dtc.(t, x[1], x[2], x[3], x[4], x[5], ω))
jac!(J, x) = ForwardDiff.jacobian!(J, res!, zeros(eltype(x), length(t)), x)
try
    lm_trust_region!(res!, jac!, copy(x0), length(t), QRStrategy(); lb = lb, ub = ub, verbose = true)
catch e
    println("B: ", sprint(showerror, e))
end


x1 = [0.9499393039279513, 0.02882295057508423, 0.9150346922009791, 1.852263326104978, 2.7567528618706407, 3.61846670450544, 4.428100162985661, 5.066363643368769, 5.63675464569449, 6.133055974188368  …  10.677949911135796, 9.57599436283228, 8.455815070288162, 7.319588132278252, 6.1894158178528125, 5.0517804888483075, 3.9127770749516095, 2.7794688746227574, 1.6596980597952389, 0.5618879530164931]
x2 = [0.9499393039279513, 0.02882295057508423, 0.9150346922009791, 1.852263326104978, 2.7567528618706407, 3.61846670450544, 4.428100162985661, 5.060771014607199, 5.625569388171337, 6.116278087903646  …  10.677348738303667, 9.575336536095241, 8.455368788128977, 7.319588132278252, 6.189049803461962, 5.051225473197014, 3.9121853318391056, 2.778970395993964, 1.6594028132358796, 0.5618879530164931]