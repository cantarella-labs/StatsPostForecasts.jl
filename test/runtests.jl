using StatsPostForecasts
using TestItemRunner

# TEST_LEVEL=fast skips tests tagged :slow (multi-run downloads, minutes).
# CI: pull requests run fast, pushes to main run full.
const LEVEL = get(ENV, "TEST_LEVEL", "full")
@run_package_tests filter = ti -> (LEVEL == "full" || !(:slow in ti.tags)) verbose = true
