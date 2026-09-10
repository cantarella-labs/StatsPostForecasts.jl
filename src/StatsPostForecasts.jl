module StatsPostForecasts

include("data_structures.jl")
include("MBM.jl")
include("training.jl")
include("parse_grib.jl")

export Forecast,
    InitForecast,
    Observations,
    observation_at,
    TrainingObject,
    MBMParameters,
    ncases,
    nmembers,
    fitting_crps,
    crps_min,
    mbm_correction!,
    mean_abs_diff,
    read_init_forecasts,
    init_time,
    lead_times,
    download_ecmwf_ens,
    ECMWF_OPEN_DATA,
    ECMWF_GCS_MIRROR,
    ECMWF_AWS_MIRROR


end
