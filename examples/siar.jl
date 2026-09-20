#=
Reader for the SIAR station CSV in examples/data — the observations every
example verifies against. It lives here, not in src: the package takes
`Observations`, it does not know where they come from.

Columns: Fecha,Hora,Temp_Media_C,Hum_Media_pct,... — 30-minute means, °C,
times in UTC. Midnight is written "24:00" carrying the date of the day it
*starts* ("02/01/2024,24:00" sits between 01/01 23:30 and 02/01 00:30), and
Dates parses 24:00 as the next day, hence the one-day shift.
=#

using Dates
using StatsPostForecasts: Observations

"Station 2 m temperature from a SIAR half-hourly CSV, in K."
function read_siar_temperature(path)
    times, temp = DateTime[], Float64[]
    for line in Iterators.drop(eachline(path), 1)
        f = split(line, ',')
        t = DateTime(f[1] * " " * f[2], dateformat"dd/mm/yyyy H:M")
        push!(times, f[2] == "24:00" ? t - Day(1) : t)
        push!(temp, parse(Float64, f[3]) + 273.15)
    end
    return Observations(times, temp)
end
