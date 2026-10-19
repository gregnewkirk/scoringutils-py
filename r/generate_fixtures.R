# Generate parity fixtures for scoringutils-py from R scoringutils.
#
# Usage (from the repository root):
#   Rscript r/generate_fixtures.R
#
# Requires the exact versions recorded in tests/fixtures/versions.json
# (scoringutils 2.3.0, scoringRules 1.1.3). The script refuses to run with
# other versions so that fixtures never drift silently.
#
# Outputs:
#   src/scoringutils_py/data/*.csv.gz   upstream example datasets
#   tests/fixtures/*.csv                data-frame level expected outputs
#   tests/fixtures/metric_cases.json    vector/matrix level expected outputs
#   tests/fixtures/versions.json        R and package versions used

suppressPackageStartupMessages({
  library(scoringutils)
  library(scoringRules)
  library(data.table)
  library(jsonlite)
})

stopifnot(
  as.character(packageVersion("scoringutils")) == "2.3.0",
  as.character(packageVersion("scoringRules")) == "1.1.3"
)

fixture_dir <- file.path("tests", "fixtures")
data_dir <- file.path("src", "scoringutils_py", "data")
dir.create(fixture_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Write a data.frame to CSV with doubles at 17 significant digits, so the
# values round-trip exactly into Python floats.
write_fixture_csv <- function(x, path) {
  dt <- copy(as.data.table(x))
  for (col in names(dt)) {
    v <- dt[[col]]
    if (inherits(v, "Date")) {
      set(dt, j = col, value = format(v))
    } else if (is.double(v)) {
      out <- ifelse(is.na(v), NA_character_, sprintf("%.17g", v))
      set(dt, j = col, value = out)
    } else if (is.factor(v)) {
      set(dt, j = col, value = as.character(v))
    }
  }
  if (grepl("\\.gz$", path)) {
    # data.table may be built without zlib; gzip -n keeps output reproducible
    plain <- sub("\\.gz$", "", path)
    fwrite(dt, plain, na = "")
    unlink(path)
    system2("gzip", c("-n", "-9", shQuote(plain)))
  } else {
    fwrite(dt, path, na = "")
  }
  invisible(path)
}

fx <- function(name) file.path(fixture_dir, name)

# ---------------------------------------------------------------------------
# 1. Example data shipped with the Python package
# ---------------------------------------------------------------------------

write_fixture_csv(example_quantile, file.path(data_dir, "example_quantile.csv.gz"))
write_fixture_csv(
  example_sample_continuous,
  file.path(data_dir, "example_sample_continuous.csv.gz")
)
write_fixture_csv(
  example_sample_discrete,
  file.path(data_dir, "example_sample_discrete.csv.gz")
)

# ---------------------------------------------------------------------------
# 2. score()
# ---------------------------------------------------------------------------

scores_quantile <- score(example_quantile)
scores_sample_continuous <- score(example_sample_continuous)
scores_sample_discrete <- suppressWarnings(score(example_sample_discrete))

write_fixture_csv(scores_quantile, fx("scores_quantile.csv"))
write_fixture_csv(scores_sample_continuous, fx("scores_sample_continuous.csv"))
write_fixture_csv(scores_sample_discrete, fx("scores_sample_discrete.csv"))

# a model without the median: ae_median fails for that group only
test <- copy(as.data.table(example_quantile))
no_median <- test[model == "epiforecasts-EpiNow2" & quantile_level != 0.5]
test <- rbind(test[model != "epiforecasts-EpiNow2"], no_median)
test <- suppressWarnings(as_forecast_quantile(test))
scores_no_median <- suppressWarnings(suppressMessages(score(test)))
write_fixture_csv(scores_no_median, fx("scores_quantile_no_median.csv"))

# asymmetric quantile levels scored with wis(na.rm = TRUE)
asymm <- example_quantile[quantile_level <= 0.6]
metrics <- get_metrics(
  example_quantile,
  exclude = c("overprediction", "underprediction", "dispersion")
)
metrics$wis <- purrr::partial(wis, na.rm = TRUE)
scores_asymm <- suppressWarnings(suppressMessages(score(asymm, metrics = metrics)))
write_fixture_csv(scores_asymm, fx("scores_quantile_asymmetric.csv"))

# sample -> quantile conversion, then scoring (issue 557)
probs_23 <- c(0.01, 0.025, seq(0.05, 0.95, 0.05), 0.975, 0.99)
discrete_q <- as_forecast_quantile(na.omit(example_sample_discrete), probs = probs_23)
write_fixture_csv(
  suppressMessages(score(discrete_q)),
  fx("scores_sample_discrete_to_quantile.csv")
)

# all nine R quantile types on a subset of the continuous samples
cont_sub <- na.omit(example_sample_continuous)[model == "EuroCOVIDhub-ensemble"]
conv <- rbindlist(lapply(1:9, function(type) {
  q <- as.data.table(
    as_forecast_quantile(cont_sub, probs = c(0.05, 0.25, 0.5, 0.75, 0.95), type = type)
  )
  q[, quantile_type := type]
  q
}))
write_fixture_csv(conv, fx("sample_to_quantile_types.csv"))

# ---------------------------------------------------------------------------
# 3. summarise_scores()
# ---------------------------------------------------------------------------

write_fixture_csv(
  summarise_scores(scores_quantile, by = "model"),
  fx("summarise_quantile_model.csv")
)
write_fixture_csv(
  summarise_scores(scores_quantile, by = c("model", "target_type")),
  fx("summarise_quantile_model_target.csv")
)
write_fixture_csv(
  # plain `median` fails in R on the logical coverage columns (odd group
  # sizes return a logical), so coerce to numeric first
  summarise_scores(
    scores_quantile, by = "model", fun = function(x) median(as.numeric(x))
  ),
  fx("summarise_quantile_model_median.csv")
)
write_fixture_csv(
  summarise_scores(scores_sample_continuous, by = c("model", "target_type")),
  fx("summarise_sample_continuous_model_target.csv")
)
write_fixture_csv(
  summarise_scores(scores_sample_discrete, by = "model", fun = sd),
  fx("summarise_sample_discrete_model_sd.csv")
)

# ---------------------------------------------------------------------------
# 4. pairwise comparisons and relative skill
# ---------------------------------------------------------------------------

write_fixture_csv(
  get_pairwise_comparisons(scores_quantile),
  fx("pairwise_quantile.csv")
)
write_fixture_csv(
  get_pairwise_comparisons(
    scores_quantile, by = "target_type", baseline = "EuroCOVIDhub-baseline"
  ),
  fx("pairwise_quantile_by_target_baseline.csv")
)
write_fixture_csv(
  get_pairwise_comparisons(scores_quantile, metric = "ae_median", test_type = NULL),
  fx("pairwise_quantile_ae_median_notest.csv")
)
write_fixture_csv(
  get_pairwise_comparisons(scores_sample_continuous, by = "target_type"),
  fx("pairwise_sample_continuous_by_target.csv")
)

# forecasts that do not overlap perfectly, one pair with no overlap at all
partial <- copy(scores_quantile)
partial <- partial[!(model == "EuroCOVIDhub-ensemble" & location == "DE")]
partial <- partial[!(model == "EuroCOVIDhub-ensemble" & target_type == "Cases")]
partial <- partial[!(model == "epiforecasts-EpiNow2" & target_type == "Deaths")]
write_fixture_csv(partial, fx("scores_quantile_partial_overlap.csv"))
write_fixture_csv(
  get_pairwise_comparisons(partial, metric = "wis"),
  fx("pairwise_quantile_partial_overlap.csv")
)

write_fixture_csv(
  add_relative_skill(scores_quantile),
  fx("relskill_quantile.csv")
)
rel <- add_relative_skill(
  scores_quantile, by = "target_type", baseline = "EuroCOVIDhub-baseline"
)
write_fixture_csv(rel, fx("relskill_quantile_by_target_baseline.csv"))
write_fixture_csv(
  summarise_scores(rel, by = c("model", "target_type")),
  fx("relskill_quantile_by_target_baseline_summary.csv")
)
summ_disc <- summarise_scores(scores_sample_discrete, by = c("model", "target_type"))
write_fixture_csv(
  add_relative_skill(summ_disc, metric = "crps", baseline = "EuroCOVIDhub-baseline"),
  fx("relskill_sample_discrete_summarised.csv")
)

# ---------------------------------------------------------------------------
# 5. get_coverage()
# ---------------------------------------------------------------------------

write_fixture_csv(get_coverage(example_quantile, by = "model"), fx("coverage_model.csv"))
write_fixture_csv(
  get_coverage(example_quantile, by = c("model", "target_type")),
  fx("coverage_model_target.csv")
)
write_fixture_csv(
  get_coverage(example_quantile, by = get_forecast_unit(example_quantile)),
  fx("coverage_forecast_unit.csv")
)
asym_cov <- copy(example_quantile)[!quantile_level %in% c(0.2, 0.3, 0.5)]
write_fixture_csv(get_coverage(asym_cov), fx("coverage_asymmetric.csv"))

# ---------------------------------------------------------------------------
# 6. duplicates
# ---------------------------------------------------------------------------

bad <- rbind(as.data.table(example_quantile), as.data.table(example_quantile)[1000:1010])
write_fixture_csv(
  get_duplicate_forecasts(bad, type = "quantile"),
  fx("duplicates_quantile.csv")
)
write_fixture_csv(
  get_duplicate_forecasts(bad, type = "quantile", counts = TRUE),
  fx("duplicates_quantile_counts.csv")
)

# ---------------------------------------------------------------------------
# 7. vector / matrix level metric cases
# ---------------------------------------------------------------------------

cases <- list()
add_case <- function(id, fn, args, kwargs = list()) {
  f <- if (exists(fn, envir = asNamespace("scoringutils"), inherits = FALSE)) {
    get(fn, envir = asNamespace("scoringutils"))
  } else {
    get(fn)
  }
  result <- suppressWarnings(suppressMessages(do.call(f, c(args, kwargs))))
  cases[[length(cases) + 1]] <<- list(
    id = id, fn = fn, args = args, kwargs = kwargs, result = result
  )
}

obs3 <- c(1, -15, 22)
pred3 <- rbind(c(-1, 0, 1, 2, 3), c(-2, 1, 2, 2, 4), c(-2, 0, 3, 3, 4))
ql5 <- c(0.1, 0.25, 0.5, 0.75, 0.9)
qargs <- list(observed = obs3, predicted = pred3, quantile_level = ql5)

for (sep in c(FALSE, TRUE)) {
  for (w in c(TRUE, FALSE)) {
    for (cm in c(FALSE, TRUE)) {
      add_case(
        sprintf("wis_basic_sep%s_weigh%s_cm%s", sep, w, cm), "wis", qargs,
        list(separate_results = sep, weigh = w, count_median_twice = cm)
      )
    }
  }
}
for (fn in c("dispersion_quantile", "overprediction_quantile",
             "underprediction_quantile", "bias_quantile",
             "ae_median_quantile")) {
  add_case(paste0(fn, "_basic"), fn, qargs)
}
add_case("quantile_score_basic", "quantile_score", qargs)
add_case("quantile_score_unweighted", "quantile_score", qargs, list(weigh = FALSE))
for (r in c(50, 80)) {
  add_case(paste0("interval_coverage_", r), "interval_coverage", qargs,
           list(interval_range = r))
}

# quantile levels 0 and 1
p01 <- matrix(c(1, 3, 5, 7, 9), nrow = 1)
q01 <- c(0, 0.25, 0.5, 0.75, 1)
for (o in c(5, 10, -3)) {
  add_case(sprintf("wis_q01_obs%s", o), "wis",
           list(observed = o, predicted = p01, quantile_level = q01),
           list(separate_results = TRUE))
  add_case(sprintf("wis_q01_obs%s_unweighted", o), "wis",
           list(observed = o, predicted = p01, quantile_level = q01),
           list(separate_results = TRUE, weigh = FALSE))
  add_case(sprintf("quantile_score_q01_obs%s", o), "quantile_score",
           list(observed = o, predicted = p01, quantile_level = q01))
  add_case(sprintf("quantile_score_q01_obs%s_unweighted", o), "quantile_score",
           list(observed = o, predicted = p01, quantile_level = q01),
           list(weigh = FALSE))
}

# random quantile forecasts (values generated here, stored in the fixture)
set.seed(2026)
n <- 40
ql23 <- probs_23
obs_r <- rnorm(n, mean = 10, sd = 3)
pred_r <- t(sapply(seq_len(n), function(i) {
  qnorm(ql23, mean = rnorm(1, 10, 2), sd = runif(1, 0.5, 4))
}))
rargs <- list(observed = obs_r, predicted = pred_r, quantile_level = ql23)
add_case("wis_random_sep", "wis", rargs, list(separate_results = TRUE))
add_case("wis_random_unweighted", "wis", rargs, list(weigh = FALSE))
add_case("wis_random_cm", "wis", rargs, list(count_median_twice = TRUE))
add_case("bias_quantile_random", "bias_quantile", rargs)
add_case("ae_median_quantile_random", "ae_median_quantile", rargs)
add_case("quantile_score_random", "quantile_score", rargs)
add_case("quantile_score_random_unweighted", "quantile_score", rargs, list(weigh = FALSE))
for (r in c(10, 30, 50, 90, 98)) {
  add_case(paste0("interval_coverage_random_", r), "interval_coverage", rargs,
           list(interval_range = r))
}

# median missing: bias interpolates
no_med <- which(ql23 != 0.5)
add_case("bias_quantile_random_no_median", "bias_quantile",
         list(observed = obs_r, predicted = pred_r[, no_med],
              quantile_level = ql23[no_med]))
# incomplete intervals with na.rm = TRUE
keep <- which(ql23 <= 0.6)
add_case("wis_random_asymmetric_narm", "wis",
         list(observed = obs_r, predicted = pred_r[, keep], quantile_level = ql23[keep]),
         list(na.rm = TRUE, separate_results = TRUE))

# bias_quantile edge cases from the upstream tests
add_case("bias_q_at_median", "bias_quantile",
         list(observed = 2, predicted = c(1, 2, 3), quantile_level = c(0.1, 0.5, 0.9)))
add_case("bias_q_minus08", "bias_quantile",
         list(observed = 2, predicted = c(0, 1, 2), quantile_level = c(0.1, 0.5, 0.9)))
add_case("bias_q_matrix", "bias_quantile",
         list(observed = seq(10, 0, length.out = 4), predicted = matrix(1:12, ncol = 3),
              quantile_level = c(0.1, 0.5, 0.9)))
add_case("bias_q_23", "bias_quantile",
         list(observed = 8062,
              predicted = c(705.500, 1127.000, 4006.250, 4341.500, 4709.000, 4821.996,
                            5340.500, 5451.000, 5703.500, 6087.014, 6329.500, 6341.000,
                            6352.500, 6594.986, 6978.500, 7231.000, 7341.500, 7860.004,
                            7973.000, 8340.500, 8675.750, 11555.000, 11976.500),
              quantile_level = c(0.01, 0.025, seq(0.05, 0.95, 0.05), 0.975, 0.99)))
add_case("bias_q_below_median_no_median", "bias_quantile",
         list(observed = 1, predicted = c(1, 2, 4, 5), quantile_level = c(0.1, 0.3, 0.7, 0.9)))
add_case("bias_q_above_median_no_median", "bias_quantile",
         list(observed = 5, predicted = c(1, 2, 4, 5), quantile_level = c(0.1, 0.3, 0.7, 0.9)))
add_case("bias_q_unordered", "bias_quantile",
         list(observed = 1.5, predicted = c(3, 1, 2), quantile_level = c(0.75, 0.25, 0.5)))

# interval helpers
add_case("get_range_from_quantile", "get_range_from_quantile",
         list(quantile_level = c(0, 0.01, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.99, 1)))
add_case("interpolate_median_sym", "interpolate_median",
         list(predicted = c(1, 10), quantile_level = c(0.4, 0.6)))
add_case("interpolate_median_asym", "interpolate_median",
         list(predicted = c(1, 10), quantile_level = c(0.3, 0.6)))
set.seed(11)
o_is <- rnorm(30, mean = 1:30)
lo <- qnorm(0.05, rnorm(30, mean = 1:30))
up <- qnorm(0.95, rnorm(30, mean = 11:40))
for (w in c(TRUE, FALSE)) {
  add_case(paste0("interval_score_weigh", w), "interval_score",
           list(observed = o_is, lower = lo, upper = up, interval_range = 90),
           list(weigh = w, separate_results = TRUE))
}

# sample metrics: continuous and integer
set.seed(7)
obs_c <- rnorm(30, mean = 1:30)
pred_c <- replicate(50, rnorm(30, mean = 1:30, sd = 2))
obs_i <- rpois(30, lambda = 1:30)
pred_i <- replicate(50, rpois(30, lambda = 1:30))
sample_fns <- c("crps_sample", "dispersion_sample", "overprediction_sample",
                "underprediction_sample", "bias_sample", "dss_sample",
                "logs_sample", "ae_median_sample", "se_mean_sample")
for (fn in sample_fns) {
  add_case(paste0(fn, "_continuous"), fn, list(observed = obs_c, predicted = pred_c))
  add_case(paste0(fn, "_integer"), fn, list(observed = obs_i, predicted = pred_i))
  add_case(paste0(fn, "_integer_as_double"), fn,
           list(observed = as.numeric(obs_i), predicted = pred_i * 1.0))
}
add_case("crps_sample_continuous_sep", "crps_sample",
         list(observed = obs_c, predicted = pred_c), list(separate_results = TRUE))
add_case("mad_sample_continuous", "mad_sample", list(predicted = pred_c))
add_case("mad_sample_integer", "mad_sample", list(predicted = pred_i))
add_case("mad_sample_vector", "mad_sample", list(predicted = c(0, 2, 4)))
# ties with the observation
add_case("bias_sample_ties_continuous", "bias_sample",
         list(observed = 1.5, predicted = c(0.5, 1.5, 1.5, 2.5, 4.5)))
add_case("bias_sample_vector_integer", "bias_sample",
         list(observed = 1, predicted = c(0, 2, 4)))
add_case("crps_sample_single", "crps_sample",
         list(observed = 2.5, predicted = seq(1.5, 10.5)), list(separate_results = TRUE))
add_case("ae_median_sample_vector", "ae_median_sample",
         list(observed = 1, predicted = c(0, 2, 4)))
add_case("se_mean_sample_vector", "se_mean_sample",
         list(observed = 1, predicted = c(0, 2, 4)))
add_case("bw_nrd", "bw.nrd", list(x = pred_c[1, ]))

# R helpers
add_case("get_type_int", "get_type", list(x = c(1, 2, 3)))
add_case("get_type_cont", "get_type", list(x = c(1, 2.5, 3)))
add_case("get_type_near_int", "get_type", list(x = c(1, 2 + 1e-10, 3)))
add_case("get_type_big", "get_type", list(x = c(1, 3e9, 3)))
add_case("p_adjust_holm", "p.adjust",
         list(p = c(0.01, 0.04, 0.03, 0.2, NA, 0.5)))

# Wilcoxon signed rank (paired) as used by compare_scores()
wilcox_cases <- list(
  exact = list(x = c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30),
               y = c(0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29)),
  ties = list(x = c(1, 2, 3, 4, 5, 6), y = c(0, 1, 1, 2, 7, 3)),
  zeros = list(x = c(1, 2, 3, 4, 5, 6, 7), y = c(1, 1, 2, 4, 3, 9, 1)),
  large = list(x = abs(rnorm(80, 3)), y = abs(rnorm(80, 3.2))),
  all_positive = list(x = c(2, 3, 4, 5, 6), y = c(1, 1, 1, 1, 1))
)
for (nm in names(wilcox_cases)) {
  add_case(paste0("compare_scores_", nm), "compare_scores",
           list(values_x = wilcox_cases[[nm]]$x, values_y = wilcox_cases[[nm]]$y))
}

write_json(
  cases, fx("metric_cases.json"),
  digits = I(17), auto_unbox = TRUE, na = "string", pretty = FALSE,
  matrix = "rowmajor"
)

# ---------------------------------------------------------------------------
# 8. versions
# ---------------------------------------------------------------------------

write_json(
  list(
    R = R.version.string,
    scoringutils = as.character(packageVersion("scoringutils")),
    scoringRules = as.character(packageVersion("scoringRules")),
    data.table = as.character(packageVersion("data.table")),
    generated = format(Sys.Date())
  ),
  fx("versions.json"),
  auto_unbox = TRUE, pretty = TRUE
)

cat("Fixtures written to", fixture_dir, "\n")
