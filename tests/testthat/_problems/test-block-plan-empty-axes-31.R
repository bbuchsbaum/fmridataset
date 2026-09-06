# Extracted from test-block-plan-empty-axes.R:31

# prequel ----------------------------------------------------------------------
empty_axis_frame <- function() {
  fmri_frame(
    list(a = matrix(rnorm(200), 20, 10)),
    observations = data.frame(.obs_id = sprintf("o%02d", 1:20))
  )
}

# test -------------------------------------------------------------------------
full <- empty_axis_frame()
cases <- list(
    "no features" = full[, integer(0)],
    "no observations" = full[integer(0), ],
    "neither" = full[integer(0), integer(0)],
    "single row, no features" = full[1, integer(0)]
  )
for (label in names(cases)) {
    frame <- cases[[label]]
    for (layout in c("balanced", "imagewise", "featurewise")) {
      plan <- plan_blocks(frame, layout = layout)
      expect_s3_class(plan, "frame_block_plan")
      expect_equal(plan$n_blocks, 0L, info = paste(label, layout))
      expect_equal(plan$total_bytes, 0, info = paste(label, layout))
    }
  }
