# Regression tests for defects found in the pre-CRAN correctness audit. Each
# block reproduces one defect with the smallest input that showed it.

.audit_entities <- function() {
  subject <- entity_frame(
    data.frame(subject_id = c("s1", "s2", "s3"), age = c(20, 60, 40)),
    key = "subject_id"
  )
  run <- entity_frame(
    data.frame(
      run_id = c("r1", "r2", "r3", "r4"),
      subject_id = c("s1", "s1", "s2", "s3")
    ),
    key = "run_id"
  )
  list(
    entities = list(subject = subject, run = run),
    relations = list(
      obs_run = key_relation("run_id", target = "run"),
      run_subject = key_relation("subject_id", target = "subject", source = "run")
    ),
    space = index_space(4, ids = paste0("f", 1:4)),
    observations = data.frame(.obs_id = paste0("o", 1:4), run_id = paste0("r", 1:4))
  )
}

# Feature spaces ---------------------------------------------------------------

test_that("symmetric and diagonal memberships keep every entry", {
  parent <- index_space(2, ids = c("a", "b"))
  full <- parcel_space(parent, c("p1", "p2"), matrix(1, 2, 2), atlas = "at")
  expect_equal(as.matrix(parcel_membership(full)), matrix(1, 2, 2), ignore_attr = TRUE)
  expect_equal(vectorize_space(full, c(10, 20)), c(15, 15), ignore_attr = TRUE)

  decoded <- parcel_space(
    parent, c("p1", "p2"), matrix(1, 2, 2),
    atlas = "at", decoder = matrix(0.5, 2, 2)
  )
  expect_equal(reconstruct_space(decoded, c(2, 4)), c(3, 3), ignore_attr = TRUE)

  expect_s3_class(parcel_space(parent, c("p1", "p2"), diag(2), atlas = "at"), "parcel_space")
  expect_s3_class(
    parcel_space(parent, c("p1", "p2"), Matrix::Diagonal(2), atlas = "at"),
    "parcel_space"
  )
})

test_that("diagonal operators are accepted by basis spaces and feature maps", {
  parent <- volume_space(c(5, 1, 1))
  ids <- paste0("c", 1:5)
  expect_s3_class(
    basis_space(parent, ids, Matrix::Diagonal(5), Matrix::Diagonal(5)),
    "basis_space"
  )
  expect_s3_class(feature_map(parent, parent, Matrix::Diagonal(5, x = 2)), "feature_map")
})

test_that("stored zeros and negative zero do not change spatial identity", {
  parent <- index_space(3, ids = c("a", "b", "c"))
  plain <- Matrix::sparseMatrix(i = 1:3, j = c(1, 1, 2), x = 1, dims = c(3, 2))
  padded <- Matrix::sparseMatrix(
    i = c(1, 2, 3, 1), j = c(1, 1, 2, 2), x = c(1, 1, 1, 0), dims = c(3, 2)
  )
  expect_true(same_space(
    parcel_space(parent, c("x", "y"), plain, atlas = "at"),
    parcel_space(parent, c("x", "y"), padded, atlas = "at")
  )$same)

  flipped <- diag(c(2, 2, 2, 1))
  flipped[1, ] <- -flipped[1, ]
  expect_true(same_space(
    volume_space(c(2, 2, 2), flipped),
    volume_space(c(2, 2, 2), diag(c(-2, 2, 2, 1)))
  )$same)
})

test_that("complex assays map through sparse operators", {
  voxels <- volume_space(c(4, 1, 1))
  parcels <- parcel_space(
    voxels, c("a", "b"),
    Matrix::sparseMatrix(i = 1:4, j = c(1, 1, 2, 2), x = 1),
    atlas = "at"
  )
  values <- matrix(complex(real = 1:8, imaginary = 1), 2)
  frame <- fmri_frame(
    assays = list(s = memory_source(values, dtype = "complex128")),
    observations = data.frame(.obs_id = c("o1", "o2")),
    space = voxels
  )
  expect_equal(
    collect_assay(map_features(frame, target = parcels)),
    unname(values %*% t(as.matrix(parcel_aggregation(parcels))))
  )
})

test_that("space constructors refuse fractional sizes and mislabelled data", {
  expect_error(
    volume_space(c(2, 2, 1), support = c(1.9, 2.2)),
    class = "fmridataset_error_space_mismatch"
  )
  expect_error(volume_space(c(2.9, 2, 1)), class = "fmridataset_error_space_mismatch")
  expect_error(
    index_space(2.7, ids = c("a", "b")),
    class = "fmridataset_error_space_mismatch"
  )
  expect_error(
    index_space(2, ids = c("a", "b"), data = data.frame(.feature_id = c("b", "a"), w = 1:2)),
    class = "fmridataset_error_space_mismatch"
  )
})

test_that("an fmrilatent basis refuses a parent that is not its own mask", {
  skip_if_not_installed("fmrilatent")
  mask <- neuroim2::LogicalNeuroVol(
    array(c(TRUE, TRUE, FALSE, TRUE, TRUE, FALSE), c(3, 2, 1)),
    neuroim2::NeuroSpace(c(3, 2, 1))
  )
  latent <- fmrilatent::LatentNeuroVec(
    basis = Matrix::Matrix(matrix(as.double(1:6), 3)),
    loadings = Matrix::Matrix(matrix(as.double(1:8), 4), sparse = TRUE),
    space = neuroim2::NeuroSpace(c(3, 2, 1, 3)), mask = mask, offset = rep(0, 4)
  )
  expect_s3_class(
    basis_space_from_fmrilatent(latent, volume_space(c(3, 2, 1), support = c(1, 2, 4, 5))),
    "basis_space"
  )
  # Same feature count, other voxels.
  expect_error(
    basis_space_from_fmrilatent(latent, volume_space(c(3, 2, 1), support = c(1, 2, 3, 6))),
    class = "fmridataset_error_space_mismatch"
  )
})

# Array sources ----------------------------------------------------------------

test_that("whole-image plans are limited by the budget, not the preference", {
  frame <- fmri_frame(
    assays = list(signal = memory_source(matrix(0, 2, 600000))),
    observations = data.frame(.obs_id = c("a", "b"))
  )
  plan <- plan_blocks(frame, layout = "imagewise")
  expect_identical(nrow(block_manifest(plan)), 2L)
  expect_error(
    plan_blocks(frame, layout = "imagewise", memory_budget = 1e6),
    class = "fmridataset_error_budget"
  )
})

test_that("a NIfTI mask file on another grid is refused", {
  dims <- c(4L, 5L, 3L)
  dir <- withr::local_tempdir()
  bold <- file.path(dir, "bold.nii")
  other <- file.path(dir, "other.nii")
  same <- file.path(dir, "same.nii")
  neuroim2::write_vec(
    neuroim2::NeuroVec(
      array(as.double(seq_len(180)), c(dims, 3L)),
      neuroim2::NeuroSpace(c(dims, 3L), spacing = c(2, 2, 2))
    ),
    bold
  )
  neuroim2::write_vol(
    neuroim2::NeuroVol(
      array(1, dims),
      neuroim2::NeuroSpace(dims, spacing = c(3, 3, 3), origin = c(50, 50, 50))
    ),
    other
  )
  neuroim2::write_vol(
    neuroim2::NeuroVol(array(1, dims), neuroim2::NeuroSpace(dims, spacing = c(2, 2, 2))),
    same
  )
  expect_error(nifti_source(bold, other), class = "fmridataset_error_space_mismatch")
  expect_error(read_nifti_bold(bold, mask = other), class = "fmridataset_error_space_mismatch")
  expect_identical(source_shape(nifti_source(bold, same)), c(3L, 60L))

  # Dimnames on an otherwise equal affine are not a different grid.
  affine <- unname(neuroim2::trans(neuroim2::read_header(bold)))
  dimnames(affine) <- list(letters[1:4], LETTERS[1:4])
  expect_s3_class(
    nifti_source(bold, volume_space(dim = dims, affine = affine, support = 1:60)),
    "nifti_array_source"
  )
})

test_that("source metadata is validated before it is coerced", {
  expect_error(
    memory_source(matrix(1:4, 2), chunks = c(NA, 2)),
    class = "fmridataset_error_alignment"
  )
  expect_error(
    memory_source(matrix(1:4, 2), chunks = c(1.9, 2)),
    class = "fmridataset_error_alignment"
  )
  expect_error(
    zarr_source("x.zarr", shape = c(5.9, 6), dtype = "float64", chunks = c(2.7, 3)),
    class = "fmridataset_error_source_contract"
  )
  # A chunk larger than the array is legal and is clamped to the array.
  clamped <- zarr_source("x.zarr", shape = c(3, 4), dtype = "float64", chunks = c(10, 10))
  expect_identical(source_chunks(clamped), c(3L, 4L))
})

# Frames -----------------------------------------------------------------------

test_that("a run relation names the run column whatever its key is called", {
  run <- entity_frame(
    data.frame(acq = c("ses1-r1", "ses2-r1")),
    key = "acq", entity_type = "run"
  )
  frame <- fmri_frame(
    list(b = matrix(0, 4, 3)),
    data.frame(
      .obs_id = sprintf("o%d", 1:4),
      acq = rep(c("ses1-r1", "ses2-r1"), each = 2L),
      run_id = "run-1", TR = 2
    ),
    space = index_space(3, ids = c("f1", "f2", "f3")),
    entities = list(run = run),
    relations = list(observation_run = key_relation("acq", target = "run"))
  )
  schema <- temporal_schema(frame)
  expect_identical(schema$columns$run, "acq")
  expect_equal(unname(schema$run_lengths), c(2L, 2L))
})

test_that("deterministic IDs depend on key values, not on factor levels", {
  ids <- function(data) {
    axis_ids(axis_frame(
      data,
      id_policy = "deterministic", id_keys = c("sub", "vol"), id_namespace = "ns"
    ))
  }
  two <- ids(data.frame(sub = factor(c("a", "b")), vol = 1:2))
  three <- ids(data.frame(sub = factor(c("a", "b", "c")), vol = 1:3))
  plain <- ids(data.frame(sub = c("a", "b"), vol = 1:2))
  expect_identical(two, three[1:2])
  expect_identical(two, plain)
})

test_that("binding compares entity values exactly", {
  make <- function(prefix, t0) {
    fmri_frame(
      list(b = matrix(as.double(1:6), 2, 3)),
      data.frame(.obs_id = paste0(prefix, 1:2), sid = c("s1", "s2")),
      space = index_space(3, ids = c("f1", "f2", "f3")),
      entities = list(subj = entity_frame(
        data.frame(sid = c("s1", "s2"), t0 = t0),
        key = "sid", entity_type = "subject"
      )),
      relations = list(os = key_relation("sid", target = "subj"))
    )
  }
  # 10 seconds apart, but equal under all.equal()'s relative tolerance.
  expect_error(
    bind_observations(
      make("a", c(1700000000, 1700000100)),
      make("b", c(1700000010, 1700000100))
    ),
    class = "fmridataset_error_entity"
  )
  expect_identical(
    dim(bind_observations(make("a", c(1, 2)), make("b", c(1, 2)))),
    c(4L, 3L)
  )
})

test_that("block_apply accepts the largest integer as one block", {
  frame <- fmri_frame(
    list(b = matrix(1:6, 2, 3)),
    data.frame(.obs_id = c("x", "y"))
  )
  blocks <- block_apply(frame, function(values, ids) length(ids),
    block_size = .Machine$integer.max
  )
  expect_identical(blocks, list(3L))
})

# Studies ----------------------------------------------------------------------

test_that("collection and study digests cover blocks, tables, and metadata", {
  fx <- .audit_entities()
  source <- memory_source(matrix(as.double(1:16), 4))
  make <- function(observations = fx$observations, ...) {
    fmri_frame(
      assays = list(bold = source), observations = observations,
      space = fx$space, entities = fx$entities, relations = fx$relations, ...
    )
  }
  digest <- function(frame) collection_digest(fmri_collection(list(a = frame)))
  with_block <- function(values) {
    make(axis_frame(
      tibble::as_tibble(fx$observations),
      blocks = list(motion = axis_block(matrix(values, 4, 2), role = "confound"))
    ))
  }
  events <- function(onset) {
    list(ev = event_table(data.frame(event_id = c("e1", "e2"), onset = onset)))
  }

  expect_identical(digest(with_block(1:8 / 10)), digest(with_block(1:8 / 10)))
  expect_false(identical(digest(with_block(1:8 / 10)), digest(with_block(8:1 / 10))))
  expect_false(identical(
    digest(make(tables = events(c(0, 5)))),
    digest(make(tables = events(c(0, 9))))
  ))
  expect_false(identical(
    study_digest(fmri_study(list(a = make(metadata = list(note = "a"))), entities = fx$entities)),
    study_digest(fmri_study(list(a = make(metadata = list(note = "b"))), entities = fx$entities))
  ))
})

test_that("a study refuses local entity block values that contradict the shared entity", {
  block <- function(values) {
    axis_block(
      values,
      components = tibble::tibble(.component_id = c("c1", "c2")),
      role = "continuous"
    )
  }
  shared <- entity_frame(
    data.frame(stimulus_id = c("a", "b", "c")),
    key = "stimulus_id",
    blocks = list(emb = block(matrix(c(1, 2, 3, 10, 20, 30), 3)))
  )
  member <- function(values) {
    fmri_frame(
      assays = list(bold = matrix(1, 3, 4)),
      observations = data.frame(.obs_id = paste0("o", 1:3), stimulus_id = c("a", "c", "a")),
      space = index_space(4, ids = paste0("f", 1:4)),
      entities = list(stimulus = entity_frame(
        data.frame(stimulus_id = c("c", "a")),
        key = "stimulus_id", blocks = list(emb = block(values))
      )),
      relations = list(os = key_relation("stimulus_id", target = "stimulus"))
    )
  }
  expect_s3_class(
    fmri_study(list(m = member(matrix(c(3, 1, 30, 10), 2))), entities = list(stimulus = shared)),
    "fmri_study"
  )
  expect_error(
    fmri_study(list(m = member(matrix(c(-7, -8, -70, -80), 2))), entities = list(stimulus = shared)),
    class = "fmridataset_error_study"
  )
})

test_that("numeric link map IDs follow the key rule", {
  space <- index_space(2, ids = c("f1", "f2"))
  frame <- fmri_frame(
    assays = list(b = matrix(1, 2, 2)),
    observations = data.frame(.obs_id = c(100000, 200000)),
    space = space
  )
  link <- frame_link(
    "a", "b",
    type = "correspondence",
    map = data.frame(.source_id = c(100000, 200000), .target_id = c(100000, 200000))
  )
  expect_identical(link$map$.source_id, c("100000", "200000"))
  expect_s3_class(fmri_study(list(a = frame, b = frame), links = list(l = link)), "fmri_study")
  expect_error(
    frame_link(
      "a", "b",
      type = "correspondence",
      map = data.frame(.source_id = c(1.5, 2), .target_id = c("x", "y"))
    ),
    class = "fmridataset_error_study"
  )
})

test_that("filter_entities restricts frame tables and refuses orphaned features", {
  fx <- .audit_entities()
  events <- event_table(data.frame(
    event_id = paste0("e", 1:4), run_id = paste0("r", 1:4), onset = 0:3
  ))
  frame <- fmri_frame(
    assays = list(bold = matrix(1, 4, 4)), observations = fx$observations,
    space = fx$space, entities = fx$entities, relations = fx$relations,
    tables = list(ev = events)
  )
  older <- filter_entities(fmri_study(list(m = frame), entities = fx$entities), subject, age > 30)
  kept <- study_frame(older, "m")
  expect_identical(event_data(kept$tables$ev)$run_id, c("r3", "r4"))

  parcel <- entity_frame(
    data.frame(parcel_id = c("p1", "p2"), lobe = c("x", "y")),
    key = "parcel_id"
  )
  by_parcel <- fmri_frame(
    assays = list(bold = matrix(1, 3, 4)),
    observations = data.frame(.obs_id = paste0("o", 1:3)),
    features = feature_axis(
      data.frame(parcel_id = c("p1", "p1", "p2", "p2")),
      space = fx$space
    ),
    entities = list(parcel = parcel),
    relations = list(fp = key_relation("parcel_id", target = "parcel", source = "feature"))
  )
  study <- fmri_study(list(m = by_parcel), entities = list(parcel = parcel))
  expect_error(
    filter_entities(study, parcel, lobe == "x"),
    "features reference",
    class = "fmridataset_error_study"
  )
  expect_s3_class(filter_entities(study, parcel, lobe %in% c("x", "y")), "fmri_study")
})

test_that("an entity may point at its own records", {
  region <- entity_frame(
    data.frame(region_id = c("a", "b", "c"), parent_id = c(NA, "a", "a")),
    key = "region_id"
  )
  frame <- fmri_frame(
    assays = list(bold = matrix(1, 3, 4)),
    observations = data.frame(.obs_id = paste0("o", 1:3), region_id = c("b", "c", "a")),
    space = index_space(4, ids = paste0("f", 1:4)),
    entities = list(region = region),
    relations = list(
      or = key_relation("region_id", target = "region"),
      rp = key_relation("parent_id", target = "region", source = "region", allow_missing = TRUE)
    )
  )
  expect_identical(observations(frame, resolve = TRUE)$region.parent_id, c("a", "a", NA))
})

# Identity and the new constructors ---------------------------------------------

test_that("named dimnames keep their positions in the canonical encoding", {
  rows_ab <- matrix(1:4, 2, dimnames = list(z = c("a", "b"), y = c("c", "d")))
  rows_cd <- matrix(1:4, 2, dimnames = list(y = c("c", "d"), z = c("a", "b")))
  expect_false(identical(canonical_sha256(rows_ab), canonical_sha256(rows_cd)))
  expect_false(identical(
    canonical_sha256(Matrix::Matrix(rows_ab, sparse = TRUE)),
    canonical_sha256(Matrix::Matrix(rows_cd, sparse = TRUE))
  ))
  # A named, non-square table used to fail outright.
  expect_type(
    canonical_sha256(table(z = c("a", "b", "b"), h = c("x", "y", "w"))),
    "character"
  )
})

test_that("a one-voxel mask yields one feature column", {
  dims <- c(3L, 4L, 2L)
  values <- array(as.double(seq_len(prod(dims) * 5L)), c(dims, 5L))
  vec <- neuroim2::NeuroVec(values, neuroim2::NeuroSpace(c(dims, 5L)))
  mask <- array(FALSE, dims)
  mask[2, 3, 1] <- TRUE
  frame <- as_fmri_frame(vec, mask = neuroim2::LogicalNeuroVol(mask, neuroim2::NeuroSpace(dims)))
  expect_identical(dim(frame), c(5L, 1L))
  expect_equal(as.vector(collect_assay(frame)), values[2, 3, 1, ])
})

test_that("a sparse NeuroVec must cover the mask", {
  dims <- c(3L, 4L, 2L)
  own <- array(FALSE, dims)
  own[1:3] <- TRUE
  wider <- array(FALSE, dims)
  wider[1:6] <- TRUE
  sparse <- neuroim2::SparseNeuroVec(
    matrix(as.double(1:15), 5), neuroim2::NeuroSpace(c(dims, 5L)),
    mask = own
  )
  expect_error(
    as_fmri_frame(sparse, mask = neuroim2::LogicalNeuroVol(wider, neuroim2::NeuroSpace(dims))),
    class = "fmridataset_error_space_mismatch"
  )
  covered <- as_fmri_frame(
    sparse,
    mask = neuroim2::LogicalNeuroVol(own, neuroim2::NeuroSpace(dims))
  )
  expect_equal(collect_assay(covered), matrix(as.double(1:15), 5))
})

test_that("read_nifti_bold keeps the caller's space and a location-free manifest", {
  path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(path), "neuroim2 NIfTI fixture is unavailable")
  mask <- suppressWarnings(neuroim2::read_vol(path))
  declared <- volume_space(
    dim = dim(mask),
    affine = neuroim2::trans(neuroim2::space(mask)),
    support = which(as.vector(mask) != 0)
  )
  from_space <- read_nifti_bold(path, mask = declared)
  expect_true(same_space(space(from_space), declared)$same)
  in_memory <- as_fmri_frame(suppressWarnings(neuroim2::read_vec(path)), mask = mask)
  expect_true(same_space(space(read_nifti_bold(path, mask = mask)), space(in_memory))$same)
  expect_identical(execution_path(from_space, "spatial"), "native")

  first <- file.path(withr::local_tempdir(), "bold.nii")
  second <- file.path(withr::local_tempdir(), "bold.nii")
  file.copy(path, first)
  file.copy(path, second)
  digest <- function(file) {
    fds_frame_manifest_digest(fds_frame_manifest(read_nifti_bold(file, mask = file, TR = 2)))
  }
  expect_identical(digest(first), digest(second))
})

test_that("frame I/O leaves the caller's own deprecation warnings alone", {
  seen <- 0L
  result <- withCallingHandlers(
    fmridataset:::.muffle_store_deprecations(identity("store result")),
    deprecatedWarning = function(w) seen <<- seen + 1L
  )
  expect_identical(result, "store result")
  frame <- fmri_frame(
    list(beta = matrix(as.double(1:12), 4)),
    data.frame(.obs_id = paste0("o", 1:4), run_id = rep(c("a", "b"), each = 2L)),
    space = index_space(3L, id_policy = "deterministic", namespace = "audit")
  )
  testthat::local_mocked_bindings(
    .require_frame_store = function(...) invisible(TRUE),
    .package = "fmridataset"
  )
  skip_if_not_installed("fmristore")
  path <- withr::local_tempfile(fileext = ".h5")
  expect_warning(
    write_frame(filter_obs(frame, run_id == "a"), path),
    class = "deprecatedWarning"
  )
})
