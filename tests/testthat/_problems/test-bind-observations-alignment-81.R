# Extracted from test-bind-observations-alignment.R:81

# prequel ----------------------------------------------------------------------
bind_space <- function() {
  volume_space(dim = c(2, 2, 2), affine = diag(4), support = 1:4, template = "t")
}
bind_frame <- function(ids, components, parcel = c("a", "a", "b", "b"),
                       block_values = NULL, tables = list(), metadata = list()) {
  sp <- bind_space()
  n <- length(ids)
  block <- axis_block(
    if (is.null(block_values)) matrix(seq_len(n * 2), n, 2) else block_values,
    components = tibble::tibble(.component_id = components)
  )
  fmri_frame(
    assays = list(beta = memory_source(matrix(seq_len(n * 4), n, 4))),
    observations = axis_frame(
      tibble::tibble(.obs_id = ids, grp = rep("g", n)),
      blocks = list(motion = block)
    ),
    features = feature_axis(
      tibble::tibble(.feature_id = feature_ids(sp), parcel = parcel),
      space = sp
    ),
    tables = tables,
    metadata = metadata
  )
}
block_matrix <- function(frame) {
  as.matrix(source_read(as_array_source(axis_block_data(obs_blocks(frame)$motion))))
}

# test -------------------------------------------------------------------------
a <- bind_frame(c("o1", "o2"), c("translation", "rotation"))
b <- bind_frame(c("o3", "o4"), c("translation", "scaling"))
expect_error(
    bind_observations(a, b),
    class = "fmridataset_error_alignment"
  )
