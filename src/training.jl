using JuMP, HiGHS
using Optim, ADTypes, ForwardDiff

#=
Fitting the MBM parameters by CRPS minimisation
================================================

The CRPS MIN objective `crps_min` is convex and piecewise linear in the
parameters θ = (α, β, γ₁, γ₂). It is therefore solved exactly as a linear
program rather than with a generic nonlinear optimiser. The absolute values
are removed with the epigraph reformulation: every residual gets an
auxiliary variable t that is bounded below by ±residual, and the sum of the
t's is minimised in their place. At the optimum every t equals the absolute
residual, so the LP objective equals `crps_min` at the same θ.

The paper (Van Schaeybroeck & Vannitsem 2015, Sect. 3.5) gives no solver;
it only notes the N·M cost of the objective. It also imposes no reliability
constraints on CRPS MIN (Table 1 marks CR/WER/SER as "±", i.e. satisfied
approximately in the experiments, not enforced). The only constraint used
here, γ₁ ≥ 0 and γ₂ ≥ 0, is not stated in the paper but is required for
Eq. (4b), d̃ₙ = γ₁·dₙ + γ₂, to be the spread of the corrected ensemble.
=#


"""
    TrainingObject(inits, obs, lead_time; d_floor) -> TrainingObject

Assemble the training slice for `lead_time` from a collection of raw
`InitForecast`s and an `Observations` series.

For every initialisation the forecast at `lead_time` is looked up, its
valid time `timestamp + lead_time` is matched against `obs`, and the case
is kept only if (i) the run is not `corrected`, (ii) a forecast at that
lead time exists, and (iii) an observation exists at the valid time.
Members are sorted, and `xmean`, `d` and the deviation column are
computed. Spreads below `d_floor` are raised to `d_floor` so that
`γ₂ / dₙ` stays finite; choose `d_floor` well below the observation
precision (e.g. 0.01 for temperature in K).

Cases are returned in increasing initialisation time. The number of
members `M` is taken from the first kept case and must be the same for
all cases.
"""
function TrainingObject(
    inits::AbstractVector{InitForecast{P,F}},
    obs::Observations{F},
    lead_time::P;
    d_floor::F = F(1e-2),
) where {P<:Period,F<:Real}
    init_times = DateTime[]
    y = F[]
    xmean = F[]
    d = F[]
    cols = Vector{Vector{F}}()

    for run in sort(inits; by = r -> r.timestamp)
        run.corrected && continue
        k = findfirst(f -> f.lead_time == lead_time, run.forecasts)
        k === nothing && continue
        obs_value = observation_at(obs, run.timestamp + lead_time)
        obs_value === nothing && continue

        x = sort(run.forecasts[k].ensemble)
        m = mean(x)
        push!(init_times, run.timestamp)
        push!(y, obs_value)
        push!(xmean, m)
        push!(d, max(mean_abs_diff(x), d_floor))
        push!(cols, x .- m)
    end

    isempty(y) && throw(ArgumentError("no training cases found for lead time $lead_time"))
    M = length(cols[1])
    all(c -> length(c) == M, cols) ||
        throw(ArgumentError("all cases must have the same number of members"))
    E = reduce(hcat, cols)::Matrix{F}

    return TrainingObject(lead_time, init_times, y, xmean, d, E)
end


"Number of training cases `N` (initialisations) in `t`."
ncases(t::TrainingObject) = length(t.y)
"Number of ensemble members `M` in `t`."
nmembers(t::TrainingObject) = size(t.E, 1)

"""
    fitting_crps(training_object::TrainingObject) -> (p, model)
 
Fit the MBM parameters `p = [α, β, γ₁, γ₂]` for one lead time by minimising
the mean ensemble CRPS `crps_min` over the training slice, solved exactly
as a linear program with HiGHS.
 
# Problem
With x̄ₙ, dₙ, E[m,n] and yₙ taken from `training_object` (N cases, M
members), the corrected members are linear in the parameters,
 
    x̃ₙ⁽ᵐ⁾ = α + β·x̄ₙ + γ₁·E[m,n] + γ₂·E[m,n]/dₙ,      rₙₘ = x̃ₙ⁽ᵐ⁾ − yₙ
 
and the objective to minimise is
 
    J(θ) = (1/N) Σₙ [ (1/M) Σₘ |rₙₘ|  −  ½·(γ₁·dₙ + γ₂) ].
 
Each |rₙₘ| is replaced by an auxiliary variable tₙₘ with the two linear
constraints
 
    tₙₘ ≥ rₙₘ,      tₙₘ ≥ −rₙₘ,
 
which together say tₙₘ ≥ |rₙₘ|; minimising Σ tₙₘ then forces tₙₘ = |rₙₘ|.
The resulting LP is
 
    min  (1/NM) Σₙ Σₘ tₙₘ  −  (1/2N) Σₙ (γ₁·dₙ + γ₂)
    s.t. tₙₘ ≥ ±rₙₘ   for all n, m
         γ₁ ≥ 0,  γ₂ ≥ 0
 
with 4 + N·M variables and 2·N·M inequality constraints. The parameters
are decision variables of the LP; the optimum's parameter part is the
minimiser of `crps_min`, and the tₙₘ are discarded.
 
# Arguments
- `training_object`: a `TrainingObject` for one lead time (and one
  initialisation hour, location and variable).
 
# Returns
- `p::Vector{Float64}`: `[α, β, γ₁, γ₂]`, in the order expected by
  `mbm_correction!` and `crps_min`.
- `model::JuMP.Model`: the solved model, kept for inspection
  (`objective_value(model)` is the in-sample CRPS at `p`;
  `termination_status(model)` the solver status).
 
# Properties and checks
- The objective is bounded below by 0 (an ensemble CRPS is an integral of
  a squared CDF difference), so the LP is always bounded.
- The raw ensemble θ = (0, 1, 1, 0) is feasible, hence
  `crps_min(p, training_object) ≤ crps_min([0,1,1,0], training_object)`.
- `objective_value(model) == crps_min(p, training_object)` up to solver
  tolerance. If not, the bug is in the data path (`E`, `d`, `y`), not in
  the solver.
- The minimiser need not be unique: `J` is piecewise linear and can be
  flat along a face at its minimum, in which case the solver returns one
  vertex of that face. Parameter curves across lead time may therefore look
  jumpier than the objective values; smoothing across lead time or a small
  regulariser toward a previous solution can be added if needed.
 
# Notes
- No reliability constraint (CR, WER, SER; Eqs. 7–9 of the paper) is
  imposed. Compute χ²/N (Eq. 9) and the variance ratio (Eq. 7) on the
  fitted slice as diagnostics if required.
- The 1/(N·M) and 1/(2N) scalings give small objective coefficients; they
  are harmless for HiGHS, and multiplying the objective by N·M would not
  change the minimiser.
- Throws if the solver does not return an optimal, feasible solution.
"""
function fitting_crps(training_object::TrainingObject; set_silence::Bool = true)
    y = training_object.y
    xmean = training_object.xmean
    d = training_object.d
    E = training_object.E
    # N: training cases (initializations) at one lead time
    N = length(y)
    # M: number of ensemble members
    M = size(E)[1]

    model = Model(HiGHS.Optimizer)

    if set_silence
        set_silent(model) # keep
    end

    @variable(model, α)
    @variable(model, β)
    @variable(model, γ₁ >= 0)
    @variable(model, γ₂ >= 0)
    @variable(model, t[1:M, 1:N] >= 0)

    @expression(
        model,
        r[k=1:M, i=1:N],
        α + β*xmean[i] + γ₁*E[k, i] + γ₂*E[k, i]/d[i] - y[i]
    )

    @constraint(model, [k=1:M, i=1:N], t[k, i] >= r[k, i])
    @constraint(model, [k=1:M, i=1:N], t[k, i] >= -r[k, i])

    @objective(model, Min, sum(t)/(N*M) - sum(γ₁*d[i] + γ₂ for i in 1:N)/(2N))

    optimize!(model)
    is_solved_and_feasible(model) ||
        error("CRPS LP did not solve: $(termination_status(model))")
    p = [value(α), value(β), value(γ₁), value(γ₂)]
    return p, model
end


"""
    fitting_crps_naive(training_object) -> (p, result)

Same objective as `fitting_crps`, minimised with box-constrained L-BFGS
(`Optim.Fminbox`) and forward-mode AD instead of the exact LP. Kept as a
cross-check of the LP; `result` is the `Optim` result object.
"""
function fitting_crps_naive(training_object::TrainingObject)
    p0 = [1e-10, 1.0, 1.0, 1e-10]        # the raw ensemble, nudged inside the box
    lower = [-Inf, -Inf, 0.0, 0.0]
    upper = fill(Inf, 4)
    f(p) = crps_min(p, training_object)
    # ponytail: L-BFGS on a piecewise-linear objective; CG and GD stall at kinks, this did not
    result =
        Optim.optimize(f, lower, upper, p0, Fminbox(LBFGS()); autodiff = AutoForwardDiff())
    return Optim.minimizer(result), result
end
