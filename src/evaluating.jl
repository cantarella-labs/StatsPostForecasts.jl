## Module to Evaluate fits.
# Instead of the trining modules, here we want to evaluate data out of sample (testing phase.)
using QuadGK
using Statistics

"Apply the fitted parameters to every lead time of one run, member by member, inplace."
function correct(runs::AbstractVector{InitForecast}, params::Dict{Period, MBMParameters})
    corr_runs = InitForecast[]
    for run in runs
        launch_time = Hour(hour(run.timestamp))
        corrs_params = params[launch_time] # grabbing the parameters fixed for this launch time.
        fors = Forecast[]
        for forecast in run.forecasts
            p = corrs_params.p[forecast.lead_time]
            sort!(forecast.ensemble)
            x_mean = mean(forecast.ensemble)
            dₙ = [abs(x-x_mean) for x in forecast.ensemble]
            x = similar(forecast.ensemble)
            mbm_correction!(x, forecast.ensemble, p, dₙ)
            cor_for = Forecast(forecast.lead_time, x)
            push!(fors, cor_for)
        end
        cor_run = InitForecast(run.timestamp, fros, true)
        push!(corr_runs, cor_run)
    end
    return corr_runs
end



"Apply the fitted parameters to just one run. needs to make sure params is at the correct launch_time from the run"
function correct(run::InitForecast, params::MBMParameters)
    fors = Forecast[]
    for forecast in run.forecasts
        p = params.p[forecast.lead_time]
        sort!(forecast.ensemble)
        x_mean = mean(forecast.ensemble)
        dₙ = [abs(x-x_mean) for x in forecast.ensemble]
        x = similar(forecast.ensemble)
        mbm_correction!(x, forecast.ensemble, p, dₙ)
        cor_for = Forecast(forecast.lead_time, x)
        push!(fors, cor_for)
    end
    cor_run = InitForecast(run.timestamp, fros, true)
    return cor_run
end

function crps(x::AbstractArray{R}, y::R) where R
    x_s = sort(x)
    n = length(x_s)
    F = x-> searchsortedlast(x_s, x)/n
    H = x -> x < y ? 0.0 : 1.0
    Inte(x) = x-> (F(x)- H(x))^2
    sol = quadgk(Inte(x), minimum(x_s), maximum(x_s))
    return sol[1]
end

function evaluate_forecast(init_for::InitForecast, obs::Observations, params::MBMParameters, ϕ, λ)

    corrected = correct(init_for, params)
    # get the initial forecast as reference to the number of ensembles:
    ensemble_length = length(corrected.forecasts[1].ensemble)

    fc_times = [corrected.timestamp + fc.lead_time for fc in corrected.forecasts]
    model = Matrix(eltype(obs.values), length(obs.times), ensemble_length)

    for i in 1:ensemble_length
        fc_values = [fc.ensemble[i] for fc in corrected.forecasts]
        model[:, i] .= interpolate_forecast(fc_times, fc_values, obs.times, ϕ, λ)
    end

    crps_t = zeros(eltype(model), length(obs.times))
    for i in 1:length(obs.times)
        crps_t[i] = crps(model[i,:], obs.values[i])
    end    

    return crps_t, model
end




