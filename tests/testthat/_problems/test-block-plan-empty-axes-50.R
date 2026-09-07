# Extracted from test-block-plan-empty-axes.R:50

# prequel ----------------------------------------------------------------------
empty_axis_frame <- function() {
  fmri_frame(
    list(a = matrix(rnorm(200), 20, 10)),
    observations = data.frame(.obs_id = sprintf("o%02d", 1:20))
  )
}

# test -------------------------------------------------------------------------
frame <- empty_axis_frame()[, integer(0)]
plan <- plan_blocks(frame)
