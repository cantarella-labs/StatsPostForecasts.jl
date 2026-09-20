# StatsPostForecasts

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://cantarella-labs.github.io/StatsPostForecasts.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://cantarella-labs.github.io/StatsPostForecasts.jl/dev)
[![Test workflow status](https://github.com/cantarella-labs/StatsPostForecasts.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/cantarella-labs/StatsPostForecasts.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/cantarella-labs/StatsPostForecasts.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cantarella-labs/StatsPostForecasts.jl)
[![Docs workflow Status](https://github.com/cantarella-labs/StatsPostForecasts.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/cantarella-labs/StatsPostForecasts.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

## Examples

`examples/` is its own project, so the plotting and DataFrames packages the
examples want stay out of the package itself:

```sh
julia --project=examples examples/end_to_end.jl
```

- `end_to_end.jl` — the whole thing at one station: fit the MBM on a month of past
  ENS runs (one parameter set per initialisation hour and lead time), correct the
  following week of runs with it, interpolate each onto the 30-minute grid through
  the diurnal cycle, and compare all of it with what the station measured.
- `quickstart.jl` — six runs, two lead times, in-sample CRPS. The cheap one.
- `full_fit.jl` — the whole station record (a long download, cached).

## How to Cite

If you use StatsPostForecasts.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/cantarella-labs/StatsPostForecasts.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://cantarella-labs.github.io/StatsPostForecasts.jl/dev/90-contributing/)