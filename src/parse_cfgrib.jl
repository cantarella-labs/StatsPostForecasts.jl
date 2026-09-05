import Dates: DateTime, Hour, Dates, unix2datetime
import Downloads, JSON
using CfGRIB

#=
CfGRIB.jl → Forecast / InitForecast, one InitForecast per (run, station).

Bare `DataSet` on purpose: the AxisArrays / DimensionalData backends read
the whole variable into memory on `convert` (~260 MB per step per parameter
for a global 0.25° ENS file). Here one (number, lon, lat) block is read per
step instead.

Known limitation: CfGRIB.jl has no `filter_by_keys` yet, so a file mixing
the control run (`type = cf`) with the perturbed members may not open
cleanly. `download_ecmwf_ens` fetches the perturbed members only.
=#

# `[x;]`: CfGRIB squeezes single-run files to a scalar; this makes it a vector
"Initialisation time(s) of the runs in `ds` (UTC), from its `time` coordinate."
init_times(ds::CfGRIB.DataSet) = unix2datetime.([ds.variables["time"].data;])
"Forecast steps of `ds` as `Hour`s, from its `step` coordinate."
lead_times(ds::CfGRIB.DataSet) = Hour.([ds.variables["step"].data;])

# ponytail: ECMWF ENS files always come out of CfGRIB in this dimension
# order; anything else is an error, not a permutedims.
function _block(var::CfGRIB.Variable, itime, k)
    d = var.dimensions
    A = var.data::Union{CfGRIB.OnDiskArray,Array}   # a field is never the squeezed scalar
    d == ("number", "step", "longitude", "latitude") && return A[:, k, :, :]
    d == ("number", "time", "step", "longitude", "latitude") && return A[:, itime, k, :, :]
    throw(ArgumentError("unexpected dimensions $d"))
end

"""
    read_init_forecasts(ds_or_path, varname, stations; itime=1, F=Float64)

Ensemble forecasts of `varname` at each `(lat, lon)` station (nearest grid
point, no interpolation) for run `itime`, one `InitForecast` per station.
Each step's field block is read once and all stations are picked from it.
A `missing` at a station is an error. Member order is the file's.
"""
function read_init_forecasts(ds::CfGRIB.DataSet, varname, stations; itime = 1, F = Float64)
    var = ds.variables[varname]
    lat, lon = ds.variables["latitude"].data, ds.variables["longitude"].data
    ilat = [argmin(abs.(lat .- s[1])) for s in stations]
    ilon = [argmin(abs.(mod.(lon .- s[2] .+ 180, 360) .- 180)) for s in stations]  # 0–360 vs ±180 safe
    steps = lead_times(ds)
    fcs = [Vector{Forecast{Hour,F}}(undef, length(steps)) for _ in stations]
    for (k, h) in enumerate(steps)
        block = _block(var, itime, k)
        for s in eachindex(stations)
            m = block[:, ilon[s], ilat[s]]
            any(ismissing, m) && error("missing $varname at $(stations[s]), step $h")
            fcs[s][k] = Forecast(h, F.(m))
        end
    end
    t0 = init_times(ds)[itime]
    return [InitForecast(t0, f, false) for f in fcs]
end

read_init_forecasts(path::AbstractString, args...; kw...) =
    read_init_forecasts(DataSet(path), args...; kw...)


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
messages into `out_path`. Byte-range download as in the CfGRIB.jl manual.
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
