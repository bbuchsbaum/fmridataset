.feature_map_fixture <- function(operator_backend = c("matrix", "sparse", "source"),
                                 instrument = FALSE) {
  operator_backend <- match.arg(operator_backend)
  source_space <- index_space(4L,
    ids = paste0("voxel-", 1:4),
    namespace = "map-source"
  )
  target_space <- index_space(3L,
    ids = paste0("parcel-", 1:3),
    namespace = "map-target"
  )
  operator <- matrix(
    c(
      0.5, 0.5, 0, 0,
      0, 0, 1, 0,
      0.25, 0, 0.25, 0.5
    ),
    nrow = 3L, byrow = TRUE
  )
  stored_operator <- switch(operator_backend,
    matrix = operator,
    sparse = Matrix::Matrix(operator, sparse = TRUE),
    source = memory_source(operator, chunks = c(2L, 2L))
  )
  map <- feature_map(
    from = source_space,
    to = target_space,
    operator = stored_operator,
    map_type = "toy_aggregation",
    traits = list(linear = TRUE, preserves_constant = TRUE),
    provenance = list(method = "analytic", version = "1")
  )
  values <- matrix(seq_len(20L), nrow = 5L, byrow = TRUE)
  source <- memory_source(values, chunks = c(2L, 2L))
  if (instrument) source <- counting_source(source)
  list(
    map = map, source_space = source_space, target_space = target_space,
    operator = operator, values = values, source = source
  )
}

test_that("feature_map has stable typed spatial identity", {
  fx <- .feature_map_fixture()
  x <- fx$map

  expect_s3_class(x, "feature_map")
  expect_invisible(validate_feature_map(x))
  expect_identical(
    space_digest(feature_map_source_space(x)),
    space_digest(fx$source_space)
  )
  expect_identical(
    space_digest(feature_map_target_space(x)),
    space_digest(fx$target_space)
  )
  expect_identical(dim(feature_map_operator(x)), c(3L, 4L))
  expect_match(feature_map_digest(x), "^[0-9a-f]{64}$")
  expect_identical(
    feature_map_digest(unserialize(serialize(x, NULL))),
    feature_map_digest(x)
  )
  expect_output(print(x), "3 target features")

  changed <- x
  changed$operator[1L, 1L] <- 0.25
  expect_false(identical(feature_map_digest(changed), feature_map_digest(x)))
})

test_that("feature_map rejects ambiguous or unserializable mappings", {
  fx <- .feature_map_fixture()

  expect_error(
    feature_map(fx$source_space, fx$target_space, matrix(1, 4L, 3L)),
    class = "fmridataset_error_feature_map"
  )
  bad <- fx$operator
  bad[1L, 1L] <- Inf
  expect_error(
    feature_map(fx$source_space, fx$target_space, bad),
    "finite"
  )
  expect_error(
    feature_map(fx$source_space, fx$target_space, fx$operator,
      metadata = list(loader = function() NULL)
    ),
    "serializable"
  )
  expect_error(
    feature_map(fx$source_space, fx$target_space, fx$operator,
      traits = list(TRUE)
    ),
    "named"
  )
})

test_that("feature_mapped_source is lazy and backend invariant", {
  dense <- .feature_map_fixture("matrix")
  sparse <- .feature_map_fixture("sparse")
  lazy_operator <- .feature_map_fixture("source")
  expected <- dense$values %*% t(dense$operator)

  for (fx in list(dense, sparse, lazy_operator)) {
    source <- feature_mapped_source(fx$source, fx$map)
    expect_array_source_conformance(source, expected)
    expect_equal(
      source_read(source, observations = c(5L, 2L), features = c(3L, 1L)),
      expected[c(5L, 2L), c(3L, 1L), drop = FALSE]
    )
    expect_equal(delarr::collect(as_delarr(source)), expected)
  }
})

test_that("mapped target blocks read only contributing source features", {
  fx <- .feature_map_fixture(instrument = TRUE)
  source <- feature_mapped_source(fx$source, fx$map)

  expect_equal(source_counts(fx$source)$bytes, 0)
  result <- source_read(source, observations = c(1L, 4L), features = 2L)
  expect_equal(result, fx$values[c(1L, 4L), 3L, drop = FALSE])
  counts <- source_counts(fx$source)
  expect_identical(counts$reads, 1)
  expect_identical(counts$values, 2)
})

test_that("independent variance mapping uses squared linear weights", {
  fx <- .feature_map_fixture()
  source <- feature_mapped_source(
    memory_source(fx$values / 100), fx$map,
    rule = "independent_variance"
  )
  expect_equal(
    source_read(source),
    (fx$values / 100) %*% t(fx$operator^2)
  )
})

test_that("provenance records form a deterministic acyclic graph", {
  first <- provenance_record(
    "import",
    inputs = list(source_fingerprint = "abc"),
    outputs = list(frame = "native")
  )
  graph <- provenance_graph(first)
  second <- provenance_record(
    "map_features",
    parents = provenance_tips(graph),
    inputs = list(feature_map = "def"),
    parameters = list(variance = "independent"),
    outputs = list(frame = "parcel")
  )
  graph <- append_provenance(graph, second)

  expect_s3_class(graph, "provenance_graph")
  expect_identical(provenance_tips(graph), second$id)
  expect_identical(names(provenance_records(graph)), c(first$id, second$id))
  expect_identical(
    provenance_digest(unserialize(serialize(graph, NULL))),
    provenance_digest(graph)
  )
  expect_output(print(graph), "2 records")
  cycle <- first
  cycle$parents <- first$id
  expect_error(provenance_graph(cycle), "acyclic")
})

test_that("map_features changes only the feature domain and declared assay rules", {
  fx <- .feature_map_fixture(instrument = TRUE)
  observations <- data.frame(
    .obs_id = paste0("obs-", 1:5),
    subject_id = rep(c("sub-1", "sub-2"), c(2L, 3L))
  )
  subject <- entity_frame(
    data.frame(subject_id = c("sub-1", "sub-2"), age = c(60, 70)),
    key = "subject_id"
  )
  old_feature_relation <- sparse_relation(
    data.frame(
      .from_id = feature_ids(fx$source_space),
      .to_id = rep("sub-1", 4L)
    ),
    from = "feature", to = "subject"
  )
  frame <- fmri_frame(
    assays = list(
      beta = fx$source,
      variance = memory_source(fx$values / 100)
    ),
    observations = observations,
    space = fx$source_space,
    entities = list(subject = subject),
    relations = list(
      observation_subject = key_relation("subject_id"),
      feature_subject = old_feature_relation
    ),
    provenance = provenance_graph(provenance_record("import"))
  )

  wrong_space <- index_space(
    4L,
    ids = feature_ids(fx$source_space), namespace = "wrong-space"
  )
  wrong_map <- feature_map(wrong_space, fx$target_space, fx$operator)
  expect_error(
    map_features(frame, map = wrong_map),
    class = "fmridataset_error_space_mismatch"
  )

  mapped <- map_features(
    frame,
    map = fx$map,
    assay_rules = c(beta = "linear", variance = "independent_variance")
  )
  expect_s3_class(mapped, "fmri_frame")
  expect_identical(observation_ids(mapped), observation_ids(frame))
  expect_identical(feature_ids(mapped), feature_ids(fx$target_space))
  expect_identical(space_digest(space(mapped)), space_digest(fx$target_space))
  expect_identical(names(relations(mapped)), "observation_subject")
  expect_equal(collect_assay(mapped, "beta"), fx$values %*% t(fx$operator))
  expect_equal(
    collect_assay(mapped, "variance"),
    (fx$values / 100) %*% t(fx$operator^2)
  )
  expect_equal(source_counts(fx$source)$reads, 1)
  records <- provenance_records(mapped$provenance)
  expect_identical(records[[length(records)]]$operation, "map_features")
  expect_identical(
    records[[length(records)]]$inputs$feature_map,
    feature_map_digest(fx$map)
  )
})

test_that("map_features derives canonical maps for parcel and basis spaces", {
  parent <- volume_space(c(2, 2, 1), support = 1:4, template = "toy")
  frame <- fmri_frame(
    assays = list(signal = matrix(1:12, nrow = 3L)),
    observations = data.frame(.obs_id = paste0("o", 1:3)),
    space = parent
  )
  parcels <- parcel_space(
    parent,
    parcel_ids = c("left", "right"),
    membership = Matrix::sparseMatrix(
      i = 1:4, j = c(1L, 1L, 2L, 2L), x = 1, dims = c(4L, 2L)
    ),
    atlas = "toy-atlas"
  )
  parcel_frame <- map_features(frame, target = parcels)
  expect_equal(
    collect_assay(parcel_frame),
    collect_assay(frame) %*% t(as.matrix(parcel_aggregation(parcels)))
  )

  basis <- basis_space_from_decoder(
    parent,
    component_ids = c("mean", "contrast"),
    decoder = cbind(c(1, 1, 1, 1), c(1, -1, 1, -1))
  )
  basis_frame <- map_features(frame, target = basis)
  expect_equal(
    collect_assay(basis_frame),
    collect_assay(frame) %*% t(basis_analysis(basis))
  )
})

test_that("mapped frames retain feature-map provenance through FDS", {
  fx <- .feature_map_fixture()
  frame <- fmri_frame(
    assays = list(signal = fx$values),
    observations = data.frame(.obs_id = paste0("obs-", 1:5)),
    space = fx$source_space
  )
  mapped <- map_features(frame, map = fx$map)
  manifest <- fds_frame_manifest(mapped)
  restored <- frame_from_fds_manifest(
    manifest,
    bindings = list("assays/signal" = feature_mapped_source(
      memory_source(fx$values), fx$map
    ))
  )

  expect_identical(
    provenance_digest(restored$provenance),
    provenance_digest(mapped$provenance)
  )
  expect_identical(space_digest(space(restored)), space_digest(fx$target_space))
  expect_equal(collect_assay(restored), fx$values %*% t(fx$operator))
})

test_that("study mapped_from links validate their typed feature map", {
  fx <- .feature_map_fixture()
  source_frame <- fmri_frame(
    assays = list(signal = fx$values),
    observations = data.frame(.obs_id = paste0("obs-", 1:5)),
    space = fx$source_space
  )
  target_frame <- map_features(source_frame, map = fx$map)
  link <- frame_link(
    from = "parcel", to = "native", type = "mapped_from",
    from_axis = "feature", to_axis = "feature",
    feature_map = fx$map
  )

  expect_s3_class(
    study <- fmri_study(
      frames = list(native = source_frame, parcel = target_frame),
      links = list(parcel_from_native = link)
    ),
    "fmri_study"
  )
  expect_invisible(validate_fds_study_manifest(fds_study_manifest(study)))
  expect_identical(
    feature_map_digest(link$metadata$feature_map),
    feature_map_digest(fx$map)
  )
  positional_metadata <- frame_link(
    "parcel", "native", "corresponds_to", NULL,
    "feature", "feature", list(legacy = TRUE)
  )
  expect_true(positional_metadata$metadata$legacy)
  expect_error(
    frame_link("parcel", "native", "derived_from", feature_map = fx$map),
    "feature-to-feature mapped_from"
  )

  bad_link <- link
  bad_link$metadata$feature_map <- feature_map(
    fx$target_space, fx$source_space, t(fx$operator)
  )
  expect_error(
    fmri_study(
      frames = list(native = source_frame, parcel = target_frame),
      links = list(parcel_from_native = bad_link)
    ),
    class = "fmridataset_error_space_mismatch"
  )
})
