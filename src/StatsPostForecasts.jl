module StatsPostForecasts

include("data_structures.jl")
include("MBM.jl")
include("EMOS.jl")
include("training.jl")
include("parse_grib.jl")
include("solar_interpolation.jl")
include("interpolate_on_data.jl")
include("evaluating.jl")

export Forecast,
    MBM,
    EMOS,
    InitForecast,
    Observations,
    observation_at,
    TrainingObject,
    Parameters,
    ncases,
    nmembers,
    fitting_crps,
    fitting_crps_naive,
    crps_min,
    emos_correction,
    apply_emos_correction,
    mbm_correction!,
    mean_abs_diff,
    read_init_forecasts,
    init_time,
    lead_times,
    download_ecmwf_ens,
    ECMWF_OPEN_DATA,
    ECMWF_GCS_MIRROR,
    ECMWF_AWS_MIRROR,
    interpolate_forecast,
    correct,
    crps,
    evaluate_forecast


end
