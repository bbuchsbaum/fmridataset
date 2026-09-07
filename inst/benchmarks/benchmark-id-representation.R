# Benchmark the 1.0 durable feature-ID representation.
# Run from an installed package or the package root after devtools::load_all().

sizes <- c(50000L, 100000L, 1000000L)
result <- do.call(rbind, lapply(sizes, function(n) {
  ids <- sprintf("feature-atlas-%07d", seq_len(n))
  data.frame(
    n_feature = n,
    character_bytes = as.numeric(object.size(ids)),
    integer_bytes = as.numeric(object.size(seq_len(n))),
    generation_seconds = unname(system.time(
      sprintf("feature-atlas-%07d", seq_len(n))
    )[["elapsed"]])
  )
}))
result$bytes_per_feature <- result$character_bytes / result$n_feature
print(result, row.names = FALSE)
