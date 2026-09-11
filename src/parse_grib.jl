import Dates: DateTime, Hour, Dates, unix2datetime
import Downloads, JSON
using GRIBDatasets

#=
GRIBDatasets.jl → Forecast / InitForecast, one InitForecast per station.

GRIBDatasets lays a variable out as (lon, lat, level, number, valid_time)
and has no time × step product, so a file must hold exactly one run —
which is what `download_ecmwf_ens` writes. Steps are valid_time minus the
reference time. One (lon, lat, number) block is read per step; the whole
variable is never materialised (~260 MB per step per parameter for a
global 0.25° ENS file). A file mixing the control run with the perturbed
members opens with `GRIBDataset(path; filter_by_values = Dict("dataType" => "pf"))`.
=#

"Initialisation time (UTC) of the single run in `ds`; errors if `ds` holds several."
init_time(ds::GRIBDataset) = unix2datetime(GRIBDatasets.getone(ds.index, "time"))
"Forecast steps of `ds` as `Hour`s, from its `valid_time` coordinate."
lead_times(ds::GRIBDataset) = Hour.(ds["valid_time"][:] .- init_time(ds))

"""
    read_init_forecasts(ds_or_path, varname, stations; F=Float64)

Ensemble forecasts of `varname` at each `(lat, lon)` station (nearest grid
point, no interpolation), one `InitForecast` per station. The file must
hold a single run. Each step's field block is read once and all stations
are picked from it. A `missing` at a station is an error. Member order is
the file's.
"""
function read_init_forecasts(ds::GRIBDataset, varname, stations; F = Float64)
    var = ds[varname]
    d = GRIBDatasets.dimnames(var)
    # ponytail: ECMWF ENS files always come out of GRIBDatasets in this order;
    # anything else (e.g. one member, which it squeezes away) is an error.
    (length(d) == 5 && d[[1, 2, 4, 5]] == ["lon", "lat", "number", "valid_time"]) ||
        throw(ArgumentError("unexpected dimensions $d"))
    lat, lon = ds["lat"][:], ds["lon"][:]
    ilat = [argmin(abs.(lat .- s[1])) for s in stations]
    ilon = [argmin(abs.(mod.(lon .- s[2] .+ 180, 360) .- 180)) for s in stations]  # 0–360 vs ±180 safe
    steps = lead_times(ds)
    t0 = init_time(ds)
    fcs = [Vector{Forecast{Hour,F}}(undef, length(steps)) for _ in stations]
    for (k, h) in enumerate(steps)
        block = var[:, :, 1, :, k]                 # (lon, lat, number)
        for s in eachindex(stations)
            m = block[ilon[s], ilat[s], :]
            any(ismissing, m) && error("missing $varname at $(stations[s]), step $h")
            fcs[s][k] = Forecast(h, F.(m))
        end
    end
    return [InitForecast(t0, f, false) for f in fcs]
end

read_init_forecasts(path::AbstractString, args...; kw...) =
    read_init_forecasts(GRIBDataset(path), args...; kw...)


"ECMWF open-data root; keeps only the last few days of runs."
const ECMWF_OPEN_DATA = "https://data.ecmwf.int/forecasts"
"Google Cloud mirror of ECMWF open data; archive since February 2024."
const ECMWF_GCS_MIRROR = "https://storage.googleapis.com/ecmwf-open-data"
"AWS mirror of ECMWF open data; same archive, but often answers `503 SlowDown`."
const ECMWF_AWS_MIRROR = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"

# ponytail: S3 answers "503 SlowDown" under load; back off and retry, no smarter client.
# Fetches into memory so a failed attempt never leaves partial bytes in the output file.
function _fetch(url; kw...)
    for attempt in 1:6
        try
            return take!(Downloads.download(url, IOBuffer(); kw...))
        catch e
            (
                attempt < 6 &&
                e isa Downloads.RequestError &&
                e.response.status in (429, 503)
            ) || rethrow()
            sleep(2.0^attempt)
        end
    end
end

"""
    download_ecmwf_ens(date, hour, params, steps, out_path;
                       resolution="0p25", members=nothing, base_url=ECMWF_OPEN_DATA)

Fetch only the perturbed ensemble members (`type = pf`) of the surface
parameters `params` (ECMWF short names, e.g. `("2t",)`) for the given
`steps` (hours) of the open-data ENS run initialised at `date` (`Date`)
and `hour` (`"00"`, `"06"`, `"12"`, `"18"`), concatenating the GRIB
messages into `out_path` (byte-range requests driven by the run's `.index` files).
`members` restricts the download to those member numbers (default: all 50).

Files live under `base_url` at

    {yyyymmdd}/{HH}z/ifs/{res}/enfo/{yyyymmdd}{HH}0000-{step}h-enfo-ef.grib2

data.ecmwf.int keeps only the last few days; pass `ECMWF_GCS_MIRROR` (or
`ECMWF_AWS_MIRROR`) for older runs.
"""
function download_ecmwf_ens(
    date::Dates.Date,
    hour::AbstractString,
    params,
    steps,
    out_path::AbstractString;
    resolution::AbstractString = "0p25",
    members = nothing,
    base_url::AbstractString = ECMWF_OPEN_DATA,
)
    yyyymmdd = Dates.format(date, "yyyymmdd")
    base = "$base_url/$yyyymmdd/$(hour)z/ifs/$resolution/enfo"
    prefix = "$(yyyymmdd)$(hour)0000"
    part = out_path * ".part"          # an interrupted download must never look complete
    open(part, "w") do out
        for step in steps
            name = "$prefix-$(step)h-enfo-ef"
            for line in eachline(IOBuffer(_fetch("$base/$name.index")))
                msg = JSON.parse(line)
                (
                    msg["levtype"] == "sfc" &&
                    msg["param"] in params &&
                    get(msg, "type", "") == "pf" &&
                    (members === nothing || parse(Int, msg["number"]) in members)
                ) || continue
                first_byte = msg["_offset"]
                last_byte = first_byte + msg["_length"] - 1
                write(
                    out,
                    _fetch(
                        "$base/$name.grib2";
                        headers = ["Range" => "bytes=$first_byte-$last_byte"],
                    ),
                )
            end
        end
    end
    mv(part, out_path; force = true)
    return out_path
end
