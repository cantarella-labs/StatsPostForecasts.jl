import Dates: DateTime, Date, Day, Dates, dayofyear
import Statistics: median
#=
Diurnal temperature cycle (DTC) model
=====================================

Semi-empirical model of the cloud-free diurnal cycle of surface
temperature from

    Göttsche, F.-M. and Olesen, F. S. (2001). Modelling of diurnal cycles
    of brightness temperature extracted from METEOSAT data.
    Remote Sens. Environ. 76, 337–348.

used here as the deterministic part of the 30-minute interpolation of the
calibrated 3-hourly ensemble forecast: the DTC is fitted to (or predicted
for) a day, and the residual is modelled separately.

Model (paper Eq. 1)
-------------------
With t in hours of local solar time, a cosine hump during the day and an
exponential decay after the "thermal sunset" tₛ:

    T₁(t) = T₀ + Tₐ·cos(π/ω·(t − tₘ))                              t <  tₛ
    T₂(t) = (T₀ + δT) + [Tₐ·cos(π/ω·(tₛ − tₘ)) − δT]·exp(−(t − tₛ)/k)   t ≥ tₛ

    T₀  residual (early-morning) temperature           [K or °C]
    Tₐ  amplitude, maximum is T₀ + Tₐ at t = tₘ         [K]
    ω   full width of the cosine hump, = daylight hours [h]   (Eq. 2, not fitted)
    tₘ  time of the maximum                             [h solar time]
    tₛ  start of the exponential attenuation            [h solar time]
    k   attenuation time constant                       [h]   (Eq. 5, not fitted)
    δT  T₀ − T(t→∞): offset of the night asymptote      [K]

Units and conventions
---------------------
* All times are decimal hours of local solar time, counted from solar
  midnight of a reference day and allowed to exceed 24 for the following
  night. Eq. (1) is written in radians (π/ω carries the conversion).
* Eqs. (2)–(3) of the paper are in degrees: N = 2/15·arccos(−tanφ·tanδ)
  with arccos in degrees, δ = 23.45·sin(360·(284+n)/365) in degrees.
* Eq. (5), k = ω/π·[tan⁻¹θ − (δT/Tₐ)·sin⁻¹θ] with θ = π/ω·(tₛ − tₘ), uses
  tan⁻¹ and sin⁻¹ for the RECIPROCALS (cot, csc), not the inverse
  functions. It follows from equal slopes of T₁ and T₂ at tₛ.

Bounded parametrisation used here
---------------------------------
Positivity of k requires cosθ > δT/Tₐ, which couples tₛ and δT. Instead of
(tₛ, δT) the model is written in

    θ = π/ω·(tₛ − tₘ)   ∈ (0, π/2)          tₛ = tₘ + ω/π·θ
    u = δT/(Tₐ·cosθ)    < 1                 δT = u·Tₐ·cosθ

so that

    k = ω/π·(1 − u)·cotθ  > 0

for every point in a box. Fitted vector: x = (T₀, Tₐ, tₘ, θ, u); ω is a
known input per day and latitude. Table-1 quantities are recovered with
`to_table1`. Value continuity at tₛ holds by construction, slope
continuity by Eq. (5), and the model is smooth in x on the whole box —
which is what Levenberg–Marquardt needs.

Fitting
-------
Nonlinear least squares on one
sunrise-to-next-sunrise window, with optional iteratively reweighted
(Huber) residuals as a substitute for the paper's median-based robust
estimator. Initial values follow the paper: T₀, Tₐ from the window's
min/max, tₘ = 12.5 h, tₛ = 17 h (→ θ), δT = 0.5 K (→ u).

The solar-geometry inputs (declination, equation of time) are taken as
arguments so this file plugs into the product's own solar code;
`declination_cooper` (paper Eq. 3) is provided only as a fallback.
=#


# ------------------------------------------------------------- solar helpers
"""
    declination_cooper(n) -> δ in degrees

Solar declination from the day of year `n` by the Cooper formula, paper
Eq. (3): δ = 23.45·sin(360·(284 + n)/365) in degrees. Accuracy ~1°.
Prefer the product's own declination routine; this is a fallback.
"""
function declination_cooper(n)
    return 23.45 * sind(360 * (284 + n) / 365)
end

"""
    daylight_hours(φ, δ) -> N in hours

Hours of daylight for latitude `φ` and solar declination `δ`, both in
degrees, paper Eq. (2): N = 2/15·arccos(−tanφ·tanδ) with arccos in
degrees. This is the cosine width ω of the DTC model. The argument is
clamped to [−1, 1] so polar day/night give 24 or 0 h instead of a
domain error.
"""
function daylight_hours(φ, δ)
    x = clamp(-tand(φ) * tand(δ), -1.0, 1.0)
    return 2 / 15 * acosd(x)
end

"""
    solar_hours(t_utc::DateTime, ref_day::Date, λ, eot_min) -> t in hours

Local solar time of `t_utc` as decimal hours since solar midnight of
`ref_day`, for longitude `λ` (degrees, east positive) and equation of time
`eot_min` (minutes, positive when the sun is ahead of mean time):

    t = (t_utc − ref_day 00:00 UTC) [h] + λ/15 + eot_min/60

Values run past 24 for times on the following day, which is what the
sunrise-to-sunrise fitting window needs. Called once on the whole
observation vector; the model itself never touches `DateTime`s.
"""
function solar_hours(t_utc::DateTime, ref_day::Date, λ, eot_min)
    h = Dates.value(t_utc - DateTime(ref_day)) / 3.6e6
    return h + λ / 15 + eot_min / 60
end


# "Start of attenuation tₛ = tₘ + ω/π·θ [h]."
# ts(p::DTCParams) = p.tₘ + p.ω / π * p.θ
# "Night-asymptote offset δT = u·Tₐ·cosθ [K]."
# deltaT(p::DTCParams) = p.u * p.Tₐ * cos(p.θ)
# "Attenuation time constant k = ω/π·(1 − u)·cotθ [h], paper Eq. (5)."
# kdecay(p::DTCParams) = p.ω / π * (1 - p.u) * cot(p.θ)


"""
    dtc(t, T₀, Tₐ, tₘ, θ, u, ω) -> T

Diurnal-cycle temperature at solar hour `t` (paper Eq. 1) in the bounded
parametrisation. Using Tₐ·cosθ − δT = Tₐ·cosθ·(1 − u):

    t <  tₛ:  T₀ + Tₐ·cos(π/ω·(t − tₘ))
    t ≥ tₛ:  (T₀ + δT) + Tₐ·cosθ·(1 − u)·exp(−(t − tₛ)/k)

with tₛ, δT, k as in `DTCParams`. Scalar, allocation-free; safe to call
from an optimiser's inner loop.
"""
function dtc(t, T₀, Tₐ, tₘ, θ, u, ω)
    tₛ = tₘ + ω / π * θ
    if t < tₛ
        return T₀ + Tₐ * cos(π / ω * (t - tₘ))
    else
        c = cos(θ)
        δT = u * Tₐ * c
        k = ω / π * (1 - u) * cot(θ)
        return (T₀ + δT) + Tₐ * c * (1 - u) * exp(-(t - tₛ) / k)
    end
end

# ------------------------------------------------------------------ fitting

"""
    dtc_bounds(ω; θmin=0.2, θmax=π/2 - 0.05, umin=-1.0, umax=0.95)
        -> (lower, upper)

Box constraints for the fitted vector (T₀, Tₐ, tₘ, θ, u). The θ and u
bounds are what make k > 0 and keep the optimiser away from the two
degenerate corners: θ → 0 (k → ∞, flat evening) and u → 1 (k → 0, step
cooling). T₀ and tₘ are only loosely bounded (tₘ within the daylight
hump), Tₐ must be positive.
"""
function dtc_bounds(ω; θmin = 0.2, θmax = π / 2 - 0.05, umin = -1.0, umax = 0.95)
    lower = [-Inf, 0.1, 12 - ω / 2, θmin, umin]
    upper = [Inf, Inf, 12 + ω / 2, θmax, umax]
    return lower, upper
end

"""
    dtc_initial(t, T, ω; tₘ=12.5, tₛ=17.0, δT=0.5) -> Vector

Initial parameter vector following the paper: T₀ and Tₐ from the window's
minimum and maximum, tₘ = 12.5 h, tₛ = 17 h and δT = 0.5 K translated to
θ = π/ω·(tₛ − tₘ) and u = δT/(Tₐ·cosθ), each clamped into `dtc_bounds`.
"""
function dtc_initial(t, T, ω; tₘ = 12.5, tₛ = 17.0, δT = 0.5)
    lower, upper = dtc_bounds(ω)
    T₀ = minimum(T)
    Tₐ = max(maximum(T) - T₀, 0.5)
    θ = clamp(π / ω * (tₛ - tₘ), lower[4], upper[4])
    u = clamp(δT / (Tₐ * cos(θ)), lower[5], upper[5])
    return [T₀, Tₐ, clamp(tₘ, lower[3], upper[3]), θ, u]
end
