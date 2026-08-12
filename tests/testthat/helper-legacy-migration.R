legacy_matrix_dataset <- function(values, TR, run_length, event_table = data.frame()) {
  structure(
    list(
      datamat = values,
      TR = TR,
      nruns = length(run_length),
      event_table = event_table,
      sampling_frame = structure(
        list(blocklens = run_length, TR = rep(TR, length(run_length))),
        class = "sampling_frame"
      ),
      mask = rep(TRUE, ncol(values))
    ),
    class = c("matrix_dataset", "fmri_dataset", "list")
  )
}
