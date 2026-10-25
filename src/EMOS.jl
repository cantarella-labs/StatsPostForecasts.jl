# implements EMOS correction to data using CRPS.
using Distributions

"""
    mbm_correction!(x_out, x_sorted, p, x_mean, dₙ)

Apply the member-by-member correction to one raw ensemble, in place.

For each raw member x⁽ᵐ⁾ the corrected member is

    x̃⁽ᵐ⁾ = α + β·x̄ + τ·(x⁽ᵐ⁾ − x̄),      τ = γ₁ + γ₂ / d

where x̄ is the raw ensemble mean and d its absolute spread.

# Arguments
- `x_out`: preallocated output vector of length `M`, overwritten with the
  corrected members (same order as the input).
- `x_sorted`: vector of the `M` **raw** members. Sorting is *not* required
  by this function (the map is applied elementwise); the name reflects that
  the same sorted vector is normally passed to `mean_abs_diff`.
- `p`: parameter vector `(α, β, γ₁, γ₂)`.
- `x_mean`: raw ensemble mean x̄.
- `dₙ`: raw absolute spread d of this ensemble.

This is the prediction-time function: it is applied to a new ensemble
using its own mean and spread together with previously fitted parameters.
Do **not** pass deviations (`x - x_mean`) as `x_sorted`; the mean is
subtracted inside.
"""
function emos_correction!(x_out, p::AbstractArray, x_mean, s²)
    #unpacking the parameters
    α, β, γ₁, γ₂ = p
    τ = γ₁ + γ₂/dₙ

    # applying the correction on the ensemble vector.
    x_out .= α .+ β*x_mean .+ τ .* (x_sorted .- x_mean)
end

const std_normal = Normal(0, 1)

"""
    crps_min(p, training_object) -> J

Mean CRPS of the EMOS-corrected ensemble over all training cases, as a
function of the parameter vector `p = (α, β, γ₁, γ₂)`. This is the
objective of the CRPS MIN method ([gneiting_calibrated_2005](@cite)):

    J(θ) = (1/N) Σₙ [ (1/M) Σₘ |x̃ₙ⁽ᵐ⁾ − yₙ|  −  ½·(γ₁·dₙ + γ₂) ]

with corrected members

    x̃ₙ⁽ᵐ⁾ = μₙ,      μₙ = α + β·x̄ₙ,      τₙ = γ₁ + γ₂*σ,

The term ½·(γ₁·dₙ + γ₂) is half the absolute spread d̃ₙ = τₙ·dₙ of the
corrected ensemble; the closed form is exact only for τₙ ≥ 0, i.e. for
γ₁, γ₂ ≥ 0.

# Arguments
- `p`: parameter vector `(α, β, γ₁, γ₂)`. Use a floating-point element
  type; the accumulators take `eltype(p)`, so dual numbers are supported.
- `training_object`: any object with fields
  - `y`     — observations, length `N`;
  - `xmean` — raw ensemble means x̄ₙ, length `N`;
  - `d`     — raw absolute spreads dₙ (from `mean_abs_diff`), length `N`,
              strictly positive;
  - `E`     — deviation matrix, size `M × N`, `E[m, n] = xₙ⁽ᵐ⁾ − x̄ₙ`.

# Properties
- J is convex and piecewise linear in `p`, with kinks wherever a corrected
  member coincides with an observation; it is not differentiable
  everywhere. A subgradient is

      ∂J/∂θ = (1/NM) Σₙ Σₘ sign(x̃ₙ⁽ᵐ⁾ − yₙ)·∂x̃ₙ⁽ᵐ⁾/∂θ  −  (1/2N) Σₙ ∂d̃ₙ/∂θ

  with ∂x̃/∂(α, β, γ₁, γ₂) = (1, x̄ₙ, E[m,n], E[m,n]/dₙ) and
  ∂d̃ₙ/∂(γ₁, γ₂) = (dₙ, 1).
- `crps_min([0, 1, 1, 0], t)` equals the mean CRPS of the raw ensemble.
- The function allocates nothing; all parameter-independent quantities
  live in `training_object`.

# Notes
`E[m, i]` is a deviation, so the mean is *not* subtracted here (contrast
`mbm_correction!`, which takes raw members).
"""
function crps_min(::EMOS, p, training_object::TrainingObject)
    sum_J = zero(eltype(p))
    α, β, γ₁, γ₂ = p
    y = training_object.y
    xmean = training_object.xmean
    s² = training_object.s²
    # N: training cases (initializations) at one lead time
    N = length(y)
    for i in 1:N
        x_mean = xmean[i]
        s²_i = s²[i]
        z_i = (y[i] - (α + β*x_mean))/√(γ₁ + γ₂*s²_i)
        sum_J += √(γ₁ + γ₂*s²_i)* (z_i*(2*cdf(std_normal, z_i)-1) + 2*pdf(std_normal, z_i)-1√π)
    end
    sum_J /= N
    return sum_J
end
