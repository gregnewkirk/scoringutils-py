# Example data: source and licence

`example_quantile.csv.gz`, `example_sample_continuous.csv.gz`,
`example_sample_discrete.csv.gz`, `example_point.csv.gz` and
`example_binary.csv.gz` are exported unchanged (all rows, all
columns, full precision) from the datasets of the same name in the R package
scoringutils 2.3.0 by `r/generate_fixtures.R`.

In scoringutils they are distributed under the MIT licence:

    Copyright (c) 2020 EpiForecasts

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.

The forecasts and observations themselves come from the European COVID-19
Forecast Hub (https://covid19forecasthub.eu/), as described in the
scoringutils documentation (`?example_quantile`). The script upstream used
to build them is `inst/create-example-data.R` in the scoringutils source.
