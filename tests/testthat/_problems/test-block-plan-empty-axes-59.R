# Extracted from test-block-plan-empty-axes.R:59

# prequel ----------------------------------------------------------------------
empty_axis_frame <- function() {
  fmri_frame(
    list(a = matrix(rnorm(200), 20, 10)),
    observations = data.frame(.obs_id = sprintf("o%02d", 1:20))
  )
}

# test -------------------------------------------------------------------------
frame <- empty_axis_frame()
expect_equal(plan_blocks(filter_obs(frame, rep(FALSE, 20)))$n_blocks, 0L)
