.print_lines <- function(x) capture.output(print(x))

test_that("feature spaces print a compact summary", {
  voxels <- volume_space(c(4, 4, 2), support = 1:10, template = "toy")
  lines <- .print_lines(voxels)
  expect_identical(lines[[1L]], "<volume_space> 10 voxels in a 4 x 4 x 2 grid")
  expect_match(lines[[3L]], substr(space_digest(voxels), 1, 12), fixed = TRUE)
  capture.output(printed <- withVisible(print(voxels)))
  expect_false(printed$visible)
  expect_identical(printed$value, voxels)

  surface <- surface_space(sprintf("v%d", 1:6), rep(c("left", "right"), each = 3))
  expect_identical(.print_lines(surface)[[1L]], "<surface_space> 6 vertices (left 3, right 3)")

  parcels <- parcel_space(
    voxels, c("a", "b"),
    Matrix::sparseMatrix(i = 1:10, j = rep(1:2, 5), x = 1, dims = c(10, 2)),
    atlas = "at"
  )
  expect_identical(.print_lines(parcels)[1:2], c(
    "<parcel_space> 2 parcels (atlas \"at\")",
    "  parent:      volume_space, 10 features"
  ))

  synthesis <- basis_space_from_decoder(voxels, c("c1", "c2"), diag(10)[, 1:2], encoder = "none")
  expect_true(any(grepl("none (synthesis only)", .print_lines(synthesis), fixed = TRUE)))

  index <- index_space(6, ids = letters[1:6])
  expect_identical(.print_lines(index)[[2L]], "  ids:    a, b, c, d, ... (6 in all)")

  mixed <- composite_space(list(lh = surface, vol = voxels))
  expect_identical(.print_lines(mixed)[1:3], c(
    "<composite_space> 16 features in 2 parts",
    "  lh:   surface_space, 6 features",
    "  vol:  volume_space, 10 features"
  ))
})

test_that("explain() keeps its list fields and prints a summary", {
  voxels <- volume_space(c(2, 2, 1), template = "toy")
  frame <- fmri_frame(
    list(bold = matrix(0, 3, 4), var = matrix(1, 3, 4)),
    data.frame(.obs_id = c("a", "b", "c")),
    space = voxels
  )
  report <- explain(frame[3:2, 1:2])
  expect_s3_class(report, "frame_explanation")
  expect_type(report, "list")
  expect_identical(report$active_assay, "bold")
  lines <- .print_lines(report)
  expect_identical(lines[[1L]], "<frame explanation> 2 x 2 view of a 3 x 4 frame")
  expect_true(any(grepl("bold .*\\(active\\)$", lines)))
  expect_false(any(grepl(" $", lines)))
})

test_that("provenance graphs list their records", {
  space <- index_space(2, ids = c("x", "y"))
  make <- function(id) fmri_frame(list(b = matrix(0, 1, 2)), data.frame(.obs_id = id), space = space)
  bound <- bind_observations(make("a"), make("b"))
  lines <- .print_lines(bound$provenance)
  expect_identical(lines[[1L]], "<provenance_graph> 1 record")
  expect_match(lines[[2L]], "bind_observations.*\\(tip\\)$")
})

test_that("unaligned records print without their class attributes", {
  record <- unaligned_record(list(site = "A", scanner = list(vendor = "X")))
  lines <- .print_lines(record)
  expect_identical(lines[[1L]], "<unaligned_record> 2 fields")
  expect_false(any(grepl("attr", lines)))
  expect_identical(.print_lines(unaligned_record()), "<unaligned_record> (empty)")
})
