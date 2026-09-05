import Statistics: mean

#=
Member-by-Member (MBM) ensemble post-processing
================================================

Implementation of the CRPS-minimisation variant ("CRPS MIN") of the
member-by-member calibration method of

    Van Schaeybroeck, B. and Vannitsem, S. (2015). Ensemble post-processing
    using member-by-member approaches: theoretical aspects.
    Q. J. R. Meteorol. Soc. 141, 807–818. doi:10.1002/qj.2397

Idea
----
Every raw ensemble member is mapped to a corrected member by an affine
transformation that acts separately on the ensemble mean and on the
deviations of the members from that mean. For one forecast case n
(one initialisation time at a fixed lead time and location), with raw
members x_n^(m), m = 1..M, raw ensemble mean x̄_n and deviations
e_n^(m) = x_n^(m) − x̄_n, the corrected member is

    x̃_n^(m) = α + β x̄_n + τ_n e_n^(m),           τ_n = γ₁ + γ₂ / d_n,

where d_n is the raw absolute spread of case n (see `mean_abs_diff`).
The parameters (α, β) shift and scale the ensemble mean; (γ₁, γ₂) scale
the ensemble spread and add a spread floor ("nudging"). Because the map
is affine with τ_n ≥ 0, member ranks, normalised higher moments and the
dependence structure across variables, locations and lead times of the
raw ensemble are preserved — the property that distinguishes MBM from
distribution-based methods such as NGR/EMOS.

Fitting
-------
The four parameters θ = (α, β, γ₁, γ₂) are obtained for each lead time
and location separately by minimising the mean continuous ranked
probability score (CRPS) of the corrected ensemble over N training cases:

    J(θ) = (1/N) Σ_n [ (1/M) Σ_m |x̃_n^(m) − y_n|  −  ½ d̃_n ],
    d̃_n = τ_n d_n = γ₁ d_n + γ₂.

Since x̃_n^(m) is linear in θ and the second term is linear in θ, J is a
convex piecewise-linear function of θ (an L1-regression problem with a
linear term). Any local minimum is therefore global, but J is non-smooth,
which should be taken into account when choosing the solver. The raw
ensemble corresponds to θ = (0, 1, 1, 0), so the in-sample CRPS of the fit
is never worse than that of the raw ensemble.

Conventions
-----------
* M : number of ensemble members
* N : number of training cases (initialisations) at one lead time
* A training object bundles, for one (lead time, location) slice, the
  observations `y` (length N), the raw ensemble means `xmean` (length N),
  the raw absolute spreads `d` (length N) and the deviation matrix
  `E` (M × N, `E[m, n] = e_n^(m)`). All of these are parameter
  independent and are computed once at assembly time.
* `d` must be strictly positive; apply a floor when building the training
  object, not inside the objective.
* γ₁ ≥ 0 and γ₂ ≥ 0 are required for d̃_n = γ₁ d_n + γ₂ to be the spread of
  the corrected ensemble; the objective does not enforce this.
=#


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
function mbm_correction!(x_out, x_sorted::AbstractArray, p::AbstractArray, x_mean, dₙ)
    #unpacking the parameters
    α, β, γ₁, γ₂ = p
    τ = γ₁ + γ₂/dₙ

    # applying the correction on the ensemble vector.
    x_out .= α .+ β*x_mean .+ τ .* (x_sorted .- x_mean)
end


"""
    mean_abs_diff(x_sorted) -> d

Absolute spread of an ensemble: the mean absolute difference between two
members drawn at random (Gini mean difference),

    d = (1/M²) Σₘ Σₘ′ |x⁽ᵐ⁾ − x⁽ᵐ′⁾|

This is δₙ in Eq. (2b) of Van Schaeybroeck & Vannitsem (2015) and appears
as the second term of the ensemble CRPS.

**Requires `x_sorted` to be sorted in ascending order.** For sorted values
x₍₁₎ ≤ … ≤ x₍ₘ₎ the double sum reduces to

    d = (2/M²) Σₖ (2k − 1 − M)·x₍ₖ₎,      k = 1..M

which is evaluated here in O(M) after the O(M log M) sort, instead of
O(M²) for the double sum. The identity follows from the fact that x₍ₖ₎
exceeds k−1 members and is exceeded by M−k members.

For a Gaussian ensemble with standard deviation σ, d → 2σ/√π ≈ 1.13σ as
M → ∞, which is a convenient check of the implementation.
"""
function mean_abs_diff(x_sorted::AbstractArray)
    M = length(x_sorted)
    sum_J = zero(eltype(x_sorted))
    for k in 1:M
        sum_J += (2*k - 1 - M)*x_sorted[k]
    end
    sum_J *= 2/M^2
    return sum_J
end


"""
    crps_min(p, training_object) -> J

Mean CRPS of the MBM-corrected ensemble over all training cases, as a
function of the parameter vector `p = (α, β, γ₁, γ₂)`. This is the
objective of the CRPS MIN method (Sect. 3.5, Eq. (20) of
[van_schaeybroeck_ensemble_2015](@cite)):

    J(θ) = (1/N) Σₙ [ (1/M) Σₘ |x̃ₙ⁽ᵐ⁾ − yₙ|  −  ½·(γ₁·dₙ + γ₂) ]

with corrected members

    x̃ₙ⁽ᵐ⁾ = μₙ + τₙ·E[m,n],      μₙ = α + β·x̄ₙ,      τₙ = γ₁ + γ₂/dₙ,
    E[m,n] = xₙ⁽ᵐ⁾ − x̄ₙ

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
function crps_min(p, training_object::TrainingObject)
    sum_J = zero(eltype(p))
    α, β, γ₁, γ₂ = p
    y = training_object.y
    xmean = training_object.xmean
    d = training_object.d
    E = training_object.E
    # N: training cases (initializations) at one lead time
    N = length(y)
    # M: number of ensemble members
    M = size(E)[1]
    for i in 1:N
        x_mean = xmean[i]
        dₙ = d[i]
        τ = γ₁ + γ₂/dₙ            # spread factor of case i
        μ = α + β*x_mean          # corrected ensemble mean of case i
        s = zero(eltype(p))
        for m in 1:M
            s += abs(μ + τ*E[m, i] - y[i])
        end
        sum_J += s/M - 0.5*(γ₁*d[i] + γ₂)   # ensemble CRPS of case i
    end
    sum_J /= N
    return sum_J
end

"""
Instead of the loss function this function return the coefficients of the Linear Programming version of the problem
"""
function crps_min_lp(p, training_object::TrainingObject)
    sum_J = zero(eltype(p))
    α, β, γ₁, γ₂ = p
    y = training_object.y
    xmean = training_object.xmean
    d = training_object.d
    E = training_object.E
    # N: training cases (initializations) at one lead time
    N = length(y)
    # M: number of ensemble members
    M = size(E)[1]
    a = 0
    c = zeros(eltype(y), length(p))
    for i in 1:N
        x_mean = xmean[i]
        dₙ = d[i]
        τ = γ₁ + γ₂/dₙ            # spread factor of case i
        μ = α + β*x_mean          # corrected ensemble mean of case i
        s = zero(eltype(p))
        for m in 1:M
            s += abs(μ + τ*E[m, i] - y[i])
        end
        sum_J += s/M - 0.5*(γ₁*d[i] + γ₂)   # ensemble CRPS of case i
    end
    sum_J /= N
    return sum_J
end
