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
  left_only <- surface_space(sprintf("v%d", 1:3), hemisphere = "left")
  expect_identical(.print_lines(left_only)[[1L]], "<surface_space> 3 vertices (left hemisphere)")

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
    "  lh:  surface_space, 6 features",
    "  vol: volume_space, 10 features"
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

test_that("surface and composite maps print their parts", {
  left <- surface_space(sprintf("L%d", 1:5), hemisphere = "left", medial_wall = c(rep(FALSE, 4), TRUE))
  expect_identical(
    .print_lines(left)[[2L]],
    "  excluded: 1 medial-wall vertex"
  )
  mixed <- composite_space(list(lh = left, vol = volume_space(c(2, 1, 1))))
  frame <- fmri_frame(list(b = matrix(as.double(1:6), 1)), data.frame(.obs_id = "a"), space = mixed)
  map <- spatial_map(frame, "a")
  expect_identical(.print_lines(map), c("<composite_map> 2 parts", "  lh:  surface_map", "  vol: DenseNeuroVol"))
  expect_identical(.print_lines(map$parts$lh)[[1L]], "<surface_map> 5 vertices (4 with values)")
})

test_that("long run labels print by the part that tells them apart", {
  labels <- sprintf(
    "derivatives/fmriprep/sub-01/func/sub-01_task-memory_run-%02d_space-MNI_desc-preproc_bold", 1:2
  )
  expect_identical(
    fmridataset:::.abbreviate_labels(labels),
    c("sub-01_task-memory_run-01", "sub-01_task-memory_run-02")
  )
  same_file <- c("a/ses-1/func/run-1_bold", "a/ses-2/func/run-1_bold")
  # Repeated file names keep the directory that tells them apart.
  expect_identical(fmridataset:::.abbreviate_labels(same_file, 10L), c("a/ses-1", "a/ses-2"))
  expect_identical(fmridataset:::.abbreviate_labels(c("run-1", "run-2")), c("run-1", "run-2"))
  expect_identical(fmridataset:::.abbreviate_labels(strrep("x", 50), 10L), "...xxxxxxx")
})

test_that("identity descriptors print their domain and short digest", {
  frame <- fmri_frame(
    list(b = matrix(0, 2, 4)), data.frame(.obs_id = c("a", "b")),
    space = volume_space(c(2, 2, 1))
  )
  identity <- identity_descriptor(frame)
  lines <- .print_lines(identity)
  expect_identical(lines[[1L]], paste("<fmri_identity> semantic", substr(identity$digest, 1, 12)))
  expect_match(lines[[2L]], "^  canonicalization: ")
})

test_that("links, studies, collections, and entities print their contents", {
  subjects <- entity_frame(
    data.frame(subject_id = c("s1", "s2"), group = c("a", "b")),
    key = "subject_id", entity_type = "subject"
  )
  expect_identical(.print_lines(subjects), c(
    "<entity_frame> 2 subject records", "  key:     subject_id", "  columns: group"
  ))
  space <- index_space(2, ids = c("x", "y"))
  make <- function(prefix) {
    fmri_frame(
      list(b = matrix(0, 2, 2)),
      data.frame(.obs_id = paste0(prefix, 1:2), subject_id = c("s1", "s2")),
      space = space, entities = list(subject = subjects),
      relations = list(os = key_relation("subject_id", target = "subject"))
    )
  }
  link <- frame_link(
    "first", "second", type = "correspondence",
    map = data.frame(.source_id = c("a1", "a2"), .target_id = c("b1", "b2"))
  )
  expect_identical(.print_lines(link), c(
    "<frame_link> first -> second (correspondence)",
    "  map: 2 pairs (observation IDs -> observation IDs)"
  ))
  study <- fmri_study(
    list(first = make("a"), second = make("b")),
    entities = list(subject = subjects), links = list(onset = link)
  )
  expect_identical(.print_lines(study), c(
    "<fmri_study> 2 representations",
    "  first:  fmri_frame 2 x 2",
    "  second: fmri_frame 2 x 2",
    "  entities: subject",
    "  links:    onset (first -> second)"
  ))
  collection <- fmri_collection(list(one = make("a"), two = make("b")))
  expect_identical(.print_lines(collection)[1:2], c("<fmri_collection> 2 frames", "  frames:         one, two"))
})

test_that("numbers get the article they are read with", {
  article <- fmridataset:::.article
  expect_identical(
    vapply(c(1, 8, 11, 18, 80, 110, 800, 1800, 11000, 18500, 110000), article, ""),
    c("a", "an", "an", "an", "an", "a", "an", "a", "an", "an", "a")
  )
})
