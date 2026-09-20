```@meta
CurrentModule = StatsPostForecasts
DocTestSetup = quote
    using StatsPostForecasts
end
```

# [Interpolation](@id interpolation)

The calibrated ensemble arrives every 3 hours; a product often needs 30-minute
values. A straight line across a 3-hour gap loses the curvature of the diurnal
cycle, most visibly around the afternoon maximum and the evening bend.
[`interpolate_forecast`](@ref) instead fits a diurnal temperature cycle to each
sunrise-to-sunrise window and evaluates it on the fine grid.

```julia
fine = interpolate_forecast(fc_times, fc_values, out_times, ϕ, λ)
```

**Read the [caveats](@ref interpolation-caveats) before adopting this.** On real
station data it beats linear interpolation in winter and loses to it for most of
the rest of the year, and there is a crash you can hit.

## The model

The diurnal cycle of [gottsche_modelling_2001](@cite) is a cosine hump during
the day and an exponential decay after a "thermal sunset" ``t_s``:

```math
T(t) = \begin{cases}
T_0 + T_a\cos\!\bigl(\tfrac{π}{ω}(t - t_m)\bigr) & t < t_s \\
(T_0 + δT) + \tfrac{πk}{ω}\sin θ \; e^{-(t - t_s)/k} & t \ge t_s
\end{cases}
```

with ``t`` in hours of local solar time. ``T_0`` is the residual early-morning
temperature, ``T_a`` the amplitude, ``t_m`` the time of the maximum, and ``ω``
the daylight hours — which is *not* fitted but computed from latitude and solar
declination ([`daylight_hours`](@ref)).

The package uses ``(T_0, T_a, t_m, θ, k)`` with ``θ = \tfrac{π}{ω}(t_s - t_m)``
in place of ``(t_s, δT)``, so that ``δT`` follows from slope continuity and the
model is smooth in its parameters across the whole box — which is what
Levenberg–Marquardt needs. [`dtc`](@ref) evaluates it.

Solar geometry helpers are exposed so you can substitute your own:
[`solar_hours`](@ref), [`sunrise_sunset`](@ref), [`daylight_hours`](@ref),
[`declination_cooper`](@ref), [`equation_of_time`](@ref).

```jldoctest
julia> StatsPostForecasts.daylight_hours(0.0, 0.0)   # equator, equinox
12.0
```

## Windows, fitting and the residual

[`sunrise_windows`](@ref) splits the series from one sunrise to the next, because
that is the interval the model describes: hump, decay, then the next sunrise
starts a new hump. Within a window, solar hours are counted from solar midnight
of the window's first day and run past 24 into the following night.

Each window is fitted independently by trust-region Levenberg–Marquardt with box
constraints and a ForwardDiff Jacobian ([`fit_dtc_window`](@ref)). Initial values
follow the paper.

Because the fitted curve does not pass exactly through the 3-hourly points, the
residual at those points is interpolated linearly onto the fine grid and added
back:

```math
\hat{T}(t) = \mathrm{DTC}(t; \hat{p}) + \mathrm{Linear}\bigl\{T_i - \mathrm{DTC}(t_i; \hat{p})\bigr\}(t)
```

So the result **reproduces the forecast values exactly at the forecast times**
and follows the diurnal shape between them. That property is worth relying on:
it means interpolation can never move a forecast value, only fill between them.

Windows with fewer than `min_points` (default 5) forecast points fall back to a
plain linear interpolation. This is the usual fate of the leading partial window
of a run, which starts mid-night. Setting `min_points` above the window size
forces linear everywhere, which is the honest way to get a baseline to compare
against:

```julia
chords = interpolate_forecast(fc_times, fc_values, out_times, ϕ, λ;
                              min_points = typemax(Int))
```

`fixed_shape = (tₘ, θ, k)` holds the diurnal shape and fits only level and
amplitude per window. For 3-hourly data that is the better-posed problem: the
shape is a site property best estimated once on high-resolution station data,
while ``T_0`` and ``T_a`` are synoptic and change daily.

## [Caveats](@id interpolation-caveats)

These are measured at one station (39.13 °N, 650 m, continental Spain) over
29 months of 30-minute records, by sampling the station's own series every
3 hours and asking the interpolator to rebuild it — which isolates interpolation
error from forecast error.

**It only beats linear interpolation in winter.** The diurnal-cycle fit won in
8 months of 29, and every one of them was November–February. From March to
October a straight line was better. Pooled over the whole record, linear scored
0.751 K and the diurnal fit 0.797 K. This is the regime the model was built for —
clear-sky radiative cooling, long stable nights; in the convective half of the
year the record has sub-3-hourly structure that a smooth five-parameter curve
cannot represent, and fitting one adds error rather than removing it.

For low-temperature applications this cuts a useful way: the months it wins are
the months frost matters.

**The fit can throw.** `lm_trust_region!` raised `PosDefException` on 2 months of
29. There is no guard around it, so one bad window aborts the whole call. Wrap
`interpolate_forecast` if you are running it unattended.

**3-hourly input is required for the diurnal fit.** A window needs five points;
6-hourly spacing gives four, so everything silently falls back to linear. For
ECMWF open data that caps the useful range at +144 h, where the ENS stops being
3-hourly.

**Windows are independent.** The curve is continuous inside a window but there is
a step at each sunrise where one window hands over to the next. Measured over 42
boundaries it averaged −0.38 K with a worst case of −2.98 K. It is not a defect
worth patching: at those boundaries the fit lands +1.24 K from the station where
linear lands +2.52 K, and the large downward steps move the curve *towards* the
observation. Several fixes were tried — restricting the fit to the window's own
span, cross-fading, chaining ``T_0``, switching where the two window curves
cross — and all were worse or within noise.
