make_dummy_subjects <- function(n = 3L) {
  data.frame(
    sub = sprintf("sub-%02d", seq_len(n)),
    age = seq_len(n) + 30L,
    dataset = I(lapply(seq_len(n), function(i) structure(list(id = i), class = "fmri_dataset"))),
    stringsAsFactors = FALSE
  )
}

make_dummy_group <- function(n = 3L, ...) {
  fmri_group(make_dummy_subjects(n), id = "sub", dataset_col = "dataset", ...)
}
