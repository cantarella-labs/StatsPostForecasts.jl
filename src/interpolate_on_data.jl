import Dates: DateTime, Date, Day, Hour, Minute, Dates, dayofyear, year
using ForwardDiff
using DataInterpolations
using TrustRegionLeastSquares

#=
Sub-daily interpolation of 3-hourly forecasts with the DTC model
================================================================

The calibrated ensemble forecast arrives every 3 h; the product needs
30-minute values. A linear interpolation across a 3-hour gap loses the
curvature of the diurnal cycle — most visibly around the afternoon
maximum and the evening bend. Instead the series is split into
sunrise-to-sunrise windows, the DTC of `diurnal_cycle.jl` is fitted to
the forecast points of each window, and the fitted curve is evaluated on
the fine grid.

Because the DTC is a smooth 5-parameter shape it does not pass exactly
through the 3-hourly points. The difference at those points (the
residual) is interpolated linearly onto the fine grid and added back, so
the result reproduces the forecast values exactly at the forecast times
and follows the diurnal shape between them:

    T̂(t) = DTC(t; p̂) + Linear{ Tᵢ − DTC(tᵢ; p̂) }(t)

Windows
-------
Each window runs from one sunrise to the next, because that is the
interval Eq. (1) describes: cosine hump, then exponential decay, then
the next sunrise starts a new hump. Within a window the solar hours are
counted from solar midnight of the window's *first* day and run past 24
into the following night (`solar_hours` with the window's reference
date). Sunrise is taken from `sunrise_sunset`, i.e. from the same ω that
the model uses, so the window edges and the cosine zeros are consistent.

A 3-hourly series usually has a partial first window (the run starts
mid-night) and a partial last one. Windows with fewer points than free
parameters cannot be fitted and fall back to linear interpolation.

Fitting
-------
Five parameters x = (T₀, Tₐ, tₘ, θ, k) per window, ω known from the
window's day and latitude. Levenberg–Marquardt with a trust region
(`TrustRegionLeastSquares`) and box constraints; the Jacobian comes from
ForwardDiff, which works because `dtc` is generic and its branch point
tₛ is smooth in the parameters. Initial values follow the paper: T₀ and
Tₐ from the window's min/max, tₘ = 12.5 h, tₛ = 17 h → θ, and k from
δT = 0 (k = ω/π·cotθ) rather than guessed.

Caveats
-------
* 7–8 points per window against 5 parameters leaves 2–3 degrees of
  freedom, and tₘ and θ are the weakest identified: the 3-hourly spacing
  cannot locate the maximum to better than about an hour. Expect them to
  jump between days. If that matters, fit the shape (tₘ, θ, k) once on
  station data and re-fit only (T₀, Tₐ) per window — see
  `fit_dtc_window` with `fixed_shape`.
* The residual add-back makes the curve interpolating, not smoothing; if
  the forecast points are noisy the residual carries that noise. A
  state-space (Kalman/GP) residual would smooth instead, at the cost of
  no longer honouring the forecast values exactly.
* Windows are independent. The curve is continuous inside a window but
  the residual add-back also forces agreement at the window edges only
  if a forecast point sits there; otherwise a small step at sunrise
  remains. It is bounded by the residual magnitude, which is the
  quantity to watch.
=#


"""
    sunrise_windows(times, ϕ, λ; declination=declination_cooper,
                    eot=equation_of_time) -> Vector{NamedTuple}

Split a sorted vector of UTC `times` into sunrise-to-sunrise windows.

Sunrises are computed with `sunrise_sunset` for every calendar day the
series touches, using the same ω as the model. The first window holds
everything before the first sunrise (it belongs to the previous day, so
its reference date is one day earlier) and the last holds everything
after the last sunrise.

Returns one NamedTuple per non-empty window with
- `mask::BitVector`  — which elements of `times` belong to the window;
- `ref_date::Date`   — reference day for `solar_hours` (the day the
                       window starts on);
- `ω::Float64`       — daylight hours of `ref_date`, the model's width;
- `eot_min::Float64` — equation of time of `ref_date`, in minutes;
- `t_end::DateTime`  — the sunrise that closes the window, or the last
                       time of the series for the final window.

`declination` and `eot` are passed as functions so the product's own
solar routines can replace the paper's approximations; they are called
as `declination(dayofyear)` and `eot(year, dayofyear)`.
"""
function sunrise_windows(
    times::AbstractVector{DateTime},
    ϕ,
    λ;
    declination = declination_cooper,
    eot = equation_of_time,
)
    days = unique(Date.(times))
    sunrises = [
        sunrise_sunset(d, ϕ, λ, declination(dayofyear(d)), eot(year(d), dayofyear(d)))[1]
        for d in days
    ]

    windows = NamedTuple[]
    for i in 1:(length(sunrises)+1)
        mask = if i == 1
            times .< sunrises[1]
        elseif i == length(sunrises) + 1
            times .>= sunrises[end]
        else
            (times .>= sunrises[i-1]) .& (times .< sunrises[i])
        end
        any(mask) || continue

        # the window starts on the day of its first element, except the
        # leading partial window, which belongs to the previous day
        first_time = times[findfirst(mask)]
        ref_date = i == 1 ? Date(first_time) - Day(1) : Date(first_time)
        t_end = i == length(sunrises) + 1 ? times[findlast(mask)] : sunrises[i]

        push!(
            windows,
            (
                mask = mask,
                ref_date = ref_date,
                ω = daylight_hours(ϕ, declination(dayofyear(ref_date))),
                eot_min = eot(year(ref_date), dayofyear(ref_date)),
                t_end = t_end,
            ),
        )
    end
    return windows
end

"""
    window_solar_hours(times, w, λ) -> Vector

Solar hours of `times` in the frame of window `w` (from `sunrise_windows`):
counted from solar midnight of `w.ref_date`, so values run past 24 for
the night that follows. All times handed to the model must be converted
this way, with the window's own `ref_date` and `eot_min`.
"""
window_solar_hours(times::AbstractVector{DateTime}, w, λ) =
    solar_hours.(times, w.ref_date, λ, w.eot_min)


"""
    dtc_initial(t, T, ω; tₘ=12.5, tₛ=17.0) -> Vector

Initial parameter vector `[T₀, Tₐ, tₘ, θ, k]` for one window, following
the paper's initialisation: T₀ and Tₐ from the window's minimum and
maximum, tₘ = 12.5 h and tₛ = 17 h (→ θ = π/ω·(tₛ − tₘ) ≈ 1.0).

`k` is not guessed but taken from the δT = 0 condition,
k = ω/π·cotθ ≈ 3.3 h, which is both inside the paper's fitted range
(02:38–05:15) and the value for which the night asymptote coincides
with T₀ — a neutral starting point that biases neither a warming nor a
cooling day.
"""
function dtc_initial(t, T, ω; tₘ = 12.5, tₛ = 17.0)
    T₀ = minimum(T)
    Tₐ = max(maximum(T) - T₀, 0.5)
    θ = clamp(π / ω * (tₛ - tₘ), 0.05, π - 0.05)
    k = ω / π * cot(θ)
    return [T₀, Tₐ, tₘ, θ, k]
end

"""
    dtc_bounds() -> (lb, ub)

Box constraints for `[T₀, Tₐ, tₘ, θ, k]`.

* `Tₐ > 0` breaks the sign degeneracy: (T₀, −Tₐ, tₘ) and (T₀, Tₐ, tₘ ± ω)
  give the identical curve, so without a convention the problem has two
  disconnected optima.
* `θ ∈ (0, π)` keeps sinθ > 0, i.e. tₛ after tₘ; θ ≤ 0 makes the decay
  constant negative and the exponential diverge.
* `k > 0` is required for a decay rather than a blow-up. With θ and k as
  the free pair, δT = Tₐ(cosθ − (πk/ω)·sinθ) comes out with either sign
  on its own, so no further constraint is needed.
* `tₘ` is bounded loosely around solar noon; the fitted values in the
  paper sit between 12:13 and 13:24.
"""
dtc_bounds() = ([-Inf, 1e-3, 6.0, 1e-3, 1e-3], [Inf, Inf, 18.0, π - 1e-3, Inf])

"""
    fit_dtc_window(t, T, ω; x0=dtc_initial(t, T, ω), fixed_shape=nothing)
        -> (x, sol)

Fit the DTC to the temperatures `T` at solar hours `t` of one window,
with fixed width `ω`.

Trust-region Levenberg–Marquardt (`lm_trust_region!`) on the residual
r(x) = T − DTC(t; x), with the box of `dtc_bounds` and a ForwardDiff
Jacobian. Returns the parameter vector `[T₀, Tₐ, tₘ, θ, k]` and the
solver result.

With `fixed_shape = (tₘ, θ, k)` only `T₀` and `Tₐ` are fitted and the
shape is held at the given values. That is the recommended mode for
3-hourly data: the shape is a site property best estimated once on
30-minute station observations, while T₀ and Tₐ are synoptic and change
daily. In that mode the model is linear in the two free parameters, so
the fit is a 2×2 least-squares solve and the LM call is only there for
uniformity.

Throws if there are fewer points than free parameters; callers should
fall back to linear interpolation in that case.
"""
function fit_dtc_window(
    t::AbstractVector,
    T::AbstractVector,
    ω;
    x0 = dtc_initial(t, T, ω),
    fixed_shape = nothing,
)
    lb, ub = dtc_bounds()
    n = length(t)

    if fixed_shape === nothing
        n >= 5 || throw(ArgumentError("need ≥5 points to fit 5 parameters, got $n"))
        residual!(f, x) = (f .= T .- dtc.(t, x[1], x[2], x[3], x[4], x[5], ω))
        jacobian!(J, x) = ForwardDiff.jacobian!(J, residual!, zeros(eltype(x), n), x)
        sol = lm_trust_region!(residual!, jacobian!, copy(x0), n; lb = lb, ub = ub)
        return sol[1], sol
    else
        n >= 2 || throw(ArgumentError("need ≥2 points to fit T₀ and Tₐ, got $n"))
        tₘ, θ, k = fixed_shape
        res2!(f, x) = (f .= T .- dtc.(t, x[1], x[2], tₘ, θ, k, ω))
        jac2!(J, x) = ForwardDiff.jacobian!(J, res2!, zeros(eltype(x), n), x)
        sol = lm_trust_region!(res2!, jac2!, [x0[1], x0[2]], n; lb = lb[1:2], ub = ub[1:2])
        return [sol[1][1], sol[1][2], tₘ, θ, k], sol
    end
end


"""
    interpolate_forecast(fc_times, fc_values, out_times, ϕ, λ;
                         fixed_shape=nothing, min_points=5) -> Vector

Interpolate a 3-hourly forecast series onto the finer `out_times` grid
using a per-window DTC fit plus a linearly interpolated residual.

For each sunrise-to-sunrise window (`sunrise_windows`):
1. convert the window's forecast times and output times to solar hours
   in the window's own frame;
2. fit the DTC to the window's forecast values;
3. evaluate the fitted curve on the output times;
4. interpolate the fit residuals (forecast − fitted, at the forecast
   times) linearly onto the output times and add them back, so the
   result passes exactly through the forecast values.

Windows with fewer than `min_points` forecast points fall back to a
plain linear interpolation of the whole series over that window — this
is the usual fate of the leading and trailing partial windows.

`fixed_shape = (tₘ, θ, k)` holds the diurnal shape fixed and fits only
level and amplitude per window; see `fit_dtc_window`.

`fc_times` must be sorted and `out_times` should lie within their span
(points outside are handled by the linear extrapolation of the fallback).
"""
function interpolate_forecast(
    fc_times::AbstractVector{DateTime},
    fc_values::AbstractVector,
    out_times::AbstractVector{DateTime},
    ϕ,
    λ;
    fixed_shape = nothing,
    min_points = 5,
)
    out = zeros(float(eltype(fc_values)), length(out_times))

    # whole-series linear interpolation, used as the fallback for short windows
    linear = LinearInterpolation(
        fc_values,
        Dates.value.(fc_times);
        extrapolation = ExtrapolationType.Linear,
    )

    windows = sunrise_windows(fc_times, ϕ, λ)
    t_start = typemin(DateTime)        # the first window also owns what precedes it
    for (i, w) in enumerate(windows)
        # a window owns the output times of its own sunrise-to-sunrise span, and
        # the last one everything after it, so the tiling leaves no output time
        # unassigned — a forecast point need not sit on the window edge
        upper = i == length(windows) ? typemax(DateTime) : w.t_end
        mask_out = (out_times .>= t_start) .& (out_times .< upper)
        t_start = w.t_end
        any(mask_out) || continue

        t_fc = window_solar_hours(fc_times[w.mask], w, λ)
        T_fc = fc_values[w.mask]
        t_out = window_solar_hours(out_times[mask_out], w, λ)

        if length(t_fc) < min_points
            out[mask_out] .= linear(Dates.value.(out_times[mask_out]))
            continue
        end

        x, _ = fit_dtc_window(t_fc, T_fc, w.ω; fixed_shape = fixed_shape)

        fitted_out = dtc.(t_out, x[1], x[2], x[3], x[4], x[5], w.ω)
        fitted_fc = dtc.(t_fc, x[1], x[2], x[3], x[4], x[5], w.ω)

        # residual add-back: linear in time, so the curve hits every forecast point
        res = LinearInterpolation(
            T_fc .- fitted_fc,
            Dates.value.(fc_times[w.mask]);
            extrapolation = ExtrapolationType.Linear,
        )
        out[mask_out] .= fitted_out .+ res(Dates.value.(out_times[mask_out]))
    end
    return out
end
