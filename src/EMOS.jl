# EMOS / non-homogeneous Gaussian regression (Gneiting et al., 2005), fitted by
# minimising the closed-form CRPS of the normal predictive distribution.
using Distributions

const std_normal = Normal(0, 1)

"""
    normal_stats(fc::Forecast) -> (m, s²)
    normal_stats(fc::QuantileForecast) -> (m, s²)

The location and squared scale EMOS regresses on.

For an ensemble, its mean and uncorrected variance (divided by `M`).

For quantiles, the normal that fits them best in least squares: with
zₖ = Φ⁻¹(τₖ), regress qₖ = m + s·zₖ, so s is the slope and m the intercept.
This uses every level, does not need them evenly spaced, and leaves the
tails no more weight than the centre. For a non-normal distribution m and s
are not its mean and standard deviation, but EMOS only needs them to be
computed the same way in training and at prediction time.
"""
function normal_stats(fc::Forecast)
    m = mean(fc.ensemble)
    return m, var(fc.ensemble; corrected = false, mean = m)
end

function normal_stats(fc::QuantileForecast)
    z = quantile.(std_normal, fc.levels)
    z̄, q̄ = mean(z), mean(fc.quantiles)
    s = sum((z .- z̄) .* (fc.quantiles .- q̄)) / sum(abs2, z .- z̄)
    return q̄ - s * z̄, s^2
end

"""
    emos_correction(p, x_mean, s²) -> (μ, σ²)

Predictive normal distribution N(μ, σ²) of one raw ensemble with mean
`x_mean` and (uncorrected) variance `s²`, for parameters `p = (α, β, γ₁, γ₂)`:

    μ = α + β·x̄,      σ² = γ₁ + γ₂·s²
"""
function emos_correction(p, x_mean, s²)
    α, β, γ₁, γ₂ = p
    return α + β * x_mean, γ₁ + γ₂ * s²
end

"""
    crps_min(::EMOS, p, training_object) -> J

Mean CRPS of the EMOS predictive distribution over all training cases, as a
function of the parameter vector `p = (α, β, γ₁, γ₂)`. This is the CRPS MIN
objective of [gneiting_calibrated_2005](@cite):

    J(θ) = (1/N) Σₙ σₙ·[ zₙ(2Φ(zₙ) − 1) + 2φ(zₙ) − 1/√π ]

with N(μₙ, σₙ²) from [`emos_correction`](@ref) and zₙ = (yₙ − μₙ)/σₙ.

# Arguments
- `p`: parameter vector `(α, β, γ₁, γ₂)`. Use a floating-point element
  type; the accumulator takes `eltype(p)`, so dual numbers are supported.
- `training_object`: a `TrainingObject`; only `y`, `xmean` and `s²` are used.

# Properties
- σₙ² must be positive: γ₁, γ₂ ≥ 0 and not both zero.
- Smooth in `p` but, unlike the MBM objective, not convex: the fit depends
  on the starting point.
- `crps_min(EMOS(), [0, 1, 0, 1], t)` is the mean CRPS of N(x̄ₙ, sₙ²), not
  the ensemble CRPS of the raw members.
"""
function crps_min(::EMOS, p, training_object::TrainingObject)
    sum_J = zero(eltype(p))
    y = training_object.y
    xmean = training_object.xmean
    s² = training_object.s²
    # N: training cases (initializations) at one lead time
    N = length(y)
    for i in 1:N
        μ, σ² = emos_correction(p, xmean[i], s²[i])
        sum_J += crps(EMOS(), μ, σ², y[i])
    end
    sum_J /= N
    return sum_J
end
