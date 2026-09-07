# Observation axes, feature axes, entities, event tables, auxiliary tables,
# and relation edge tables share one set of keyed-domain validators. These
# tests pin the shared rules through each public constructor and check that
# every domain keeps its own error class and structured fields.

realized_block <- function(block) {
  unname(as.matrix(source_read(as_array_source(axis_block_data(block)))))
}

keyed_space <- function() {
  volume_space(dim = c(2, 2, 2), affine = diag(4), support = 1:4, template = "kd")
}

test_that("the stable-key rule is shared but each domain keeps its class", {
  expect_error(
    axis_frame(data.frame(.obs_id = c("a", "a"))),
    class = "fmridataset_error_alignment"
  )
  expect_error(
    axis_frame(data.frame(.obs_id = c("a", ""))),
    class = "fmridataset_error_alignment"
  )
  expect_error(
    entity_frame(tibble::tibble(subject_id = c("s1", "s1")), key = "subject_id"),
    class = "fmridataset_error_alignment"
  )

  duplicated_event <- expect_error(
    event_table(tibble::tibble(event_id = c("e1", "e1"))),
    class = "fmridataset_error_event"
  )
  expect_identical(duplicated_event$field, "event_id")
  expect_error(
    event_table(tibble::tibble(event_id = c("e1", NA))),
    class = "fmridataset_error_event"
  )

  duplicated_aux <- expect_error(
    auxiliary_table(tibble::tibble(id = c("x", "x")), key = "id"),
    class = "fmridataset_error_table"
  )
  expect_identical(duplicated_aux$field, "id")
  expect_error(
    auxiliary_table(tibble::tibble(id = c("x", "")), key = "id"),
    class = "fmridataset_error_table"
  )

  fx <- make_frame_fixture()
  manifest <- fds_frame_manifest(fx$frame)
  manifest$axes$observation$ids[[2L]] <- manifest$axes$observation$ids[[1L]]
  manifest$axes$observation$data$.obs_id <- manifest$axes$observation$ids
  duplicated_manifest <- expect_error(
    validate_fds_manifest(manifest),
    class = "fmridataset_error_schema"
  )
  expect_identical(duplicated_manifest$field, "axes.observation.ids")
})

test_that("the scalar-column rule is shared and names the offending columns", {
  list_column <- tibble::tibble(id = c("a", "b"), opaque = list(1, 2))

  entity_error <- expect_error(
    entity_frame(list_column, key = "id"),
    class = "fmridataset_error_entity"
  )
  expect_identical(entity_error$columns, "opaque")

  matrix_column <- tibble::tibble(id = c("a", "b"), wide = matrix(1:4, 2))
  expect_error(
    entity_frame(matrix_column, key = "id"),
    class = "fmridataset_error_entity"
  )

  event_error <- expect_error(
    event_table(list_column, key = "id"),
    class = "fmridataset_error_event"
  )
  expect_identical(event_error$field, "data")

  table_error <- expect_error(
    auxiliary_table(list_column, key = "id"),
    class = "fmridataset_error_table"
  )
  expect_identical(table_error$columns, "opaque")

  relation_error <- expect_error(
    sparse_relation(
      tibble::tibble(.from_id = "a", .to_id = "b", w = list(1)),
      from = "observation", to = "entity:stimulus"
    ),
    class = "fmridataset_error_relation"
  )
  expect_identical(relation_error$columns, "w")

  link_error <- expect_error(
    frame_link(
      "a", "b",
      type = "derivation",
      map = tibble::tibble(.source_id = "x", .target_id = "y", extra = list(1))
    ),
    class = "fmridataset_error_study"
  )
  expect_identical(link_error$field, "map")
})

test_that("one-string fields are validated identically across domains", {
  expect_error(
    entity_frame(tibble::tibble(subject_id = "s1"), key = c("a", "b")),
    class = "fmridataset_error_entity"
  )
  expect_error(
    entity_frame(tibble::tibble(subject_id = "s1"), key = "subject_id", entity_type = ""),
    class = "fmridataset_error_entity"
  )
  expect_error(
    auxiliary_table(tibble::tibble(id = "x"), role = NA_character_),
    class = "fmridataset_error_table"
  )
  expect_error(key_relation(key = ""), class = "fmridataset_error_relation")
  expect_error(
    frame_link(source = character(), target = "b"),
    class = "fmridataset_error_study"
  )
})

test_that("axis blocks must be two-dimensional", {
  tensor <- array(seq_len(12), dim = c(2L, 2L, 3L))

  error <- expect_error(axis_block(tensor), class = "fmridataset_error_alignment")
  expect_identical(error$shape, c(2L, 2L, 3L))
  expect_identical(error$dims, 3L)
  expect_match(conditionMessage(error), "2 x 2 x 3")
  expect_match(conditionMessage(error), "two-dimensional")

  expect_error(axis_block(1:3), class = "fmridataset_error_alignment")

  expect_s3_class(axis_block(matrix(1:6, 3, 2)), "axis_block")
  expect_s3_class(
    axis_block(memory_source(matrix(as.double(1:6), 3, 2))),
    "axis_block"
  )
})

test_that("blocks built around the constructor are rejected by name", {
  tensor <- array(seq_len(12), dim = c(2L, 2L, 3L))
  forged <- structure(
    list(
      data = tensor,
      components = data.frame(.component_id = c("a", "b")),
      role = "continuous", units = NULL, metadata = list()
    ),
    class = "axis_block"
  )

  axis_error <- expect_error(
    axis_frame(data.frame(.obs_id = c("o1", "o2")), blocks = list(tensor = forged)),
    class = "fmridataset_error_alignment"
  )
  expect_identical(axis_error$block, "tensor")
  expect_identical(axis_error$shape, c(2L, 2L, 3L))

  registry <- entity_registry(
    subject = entity_frame(tibble::tibble(subject_id = c("s1", "s2")), key = "subject_id")
  )
  registry$subject$blocks <- list(tensor = forged)
  entity_error <- expect_error(
    validate_entity_registry(registry),
    class = "fmridataset_error_entity"
  )
  expect_identical(entity_error$entity, "subject")
  expect_identical(entity_error$block, "tensor")
})

test_that("block alignment errors name the block and both lengths", {
  block <- axis_block(matrix(1:6, 3, 2))

  error <- expect_error(
    axis_frame(data.frame(.obs_id = c("a", "b")), blocks = list(embed = block)),
    class = "fmridataset_error_alignment"
  )
  expect_identical(error$block, "embed")
  expect_identical(error$expected, 2L)
  expect_identical(error$actual, 3L)

  expect_error(
    axis_frame(data.frame(.obs_id = "a"), blocks = list(axis_block(matrix(1, 1, 1)))),
    class = "fmridataset_error_alignment"
  )
  expect_error(
    axis_frame(data.frame(.obs_id = "a"), blocks = list(x = matrix(1, 1, 1))),
    class = "fmridataset_error_alignment"
  )

  entity_error <- expect_error(
    entity_frame(
      tibble::tibble(subject_id = c("s1", "s2")),
      key = "subject_id",
      blocks = list(scores = block)
    ),
    class = "fmridataset_error_alignment"
  )
  expect_identical(entity_error$block, "scores")
})

test_that("empty axes, entities, and tables validate and subset", {
  empty_axis <- axis_frame(
    tibble::tibble(.obs_id = character()),
    blocks = list(embed = axis_block(
      matrix(numeric(), 0L, 2L),
      components = data.frame(.component_id = c("x", "y"))
    ))
  )
  expect_length(empty_axis, 0L)
  expect_identical(block_component_ids(axis_blocks(empty_axis)$embed), c("x", "y"))
  expect_length(empty_axis[integer()], 0L)
  expect_identical(
    dim(axis_block_data(axis_blocks(empty_axis[integer()])$embed)),
    c(0L, 2L)
  )

  empty_entity <- entity_frame(
    tibble::tibble(subject_id = character(), age = numeric()),
    key = "subject_id"
  )
  expect_length(empty_entity, 0L)
  expect_length(entity_registry(subject = empty_entity), 1L)
  expect_length(entity_registry(), 0L)

  expect_s3_class(
    event_table(tibble::tibble(event_id = character(), onset = numeric())),
    "fmri_event_table"
  )
  expect_s3_class(
    auxiliary_table(tibble::tibble(id = character()), key = "id"),
    "fmri_auxiliary_table"
  )
})

test_that("subsetting keeps scalar rows and every block synchronized", {
  eager <- matrix(as.double(1:10), 5, 2)
  lazy <- matrix(as.double(11:20), 5, 2)
  x <- axis_frame(
    tibble::tibble(.obs_id = paste0("o", 1:5), v = 1:5),
    blocks = list(
      embed = axis_block(eager, components = data.frame(.component_id = c("x", "y"))),
      lazy = axis_block(memory_source(lazy))
    )
  )

  index <- c(5L, 2L, 3L)
  selected <- x[index]
  expect_identical(axis_ids(selected), paste0("o", index))
  expect_identical(axis_data(selected)$v, index)
  expect_identical(
    axis_block_data(axis_blocks(selected)$embed),
    eager[index, , drop = FALSE]
  )
  expect_equal(realized_block(axis_blocks(selected)$lazy), lazy[index, , drop = FALSE])
  expect_identical(block_component_ids(axis_blocks(selected)$embed), c("x", "y"))

  # Re-subsetting restores the original order and values.
  restored <- selected[order(index)]
  expect_identical(axis_ids(restored), paste0("o", sort(index)))
  expect_identical(
    axis_block_data(axis_blocks(restored)$embed),
    eager[sort(index), , drop = FALSE]
  )

  entity <- entity_frame(
    tibble::tibble(subject_id = paste0("s", 1:3)),
    key = "subject_id",
    blocks = list(scores = axis_block(matrix(as.double(1:6), 3, 2)))
  )
  expect_identical(entity_ids(entity[3:2]), c("s3", "s2"))
  expect_identical(
    axis_block_data(entity_blocks(entity[3:2])$scores),
    matrix(as.double(1:6), 3, 2)[3:2, , drop = FALSE]
  )
})

test_that("frames with observation and feature blocks round-trip through FDS", {
  sp <- keyed_space()
  motion <- axis_block(
    matrix(as.double(1:6), 3, 2),
    components = data.frame(.component_id = c("tx", "ty")),
    role = "confound"
  )
  pcs <- axis_block(
    matrix(as.double(1:12), 4, 3),
    components = data.frame(.component_id = c("pc1", "pc2", "pc3")),
    role = "embedding"
  )
  frame <- fmri_frame(
    assays = list(beta = memory_source(matrix(as.double(1:12), 3, 4))),
    observations = axis_frame(
      tibble::tibble(.obs_id = c("o1", "o2", "o3")),
      blocks = list(motion = motion)
    ),
    features = feature_axis(
      tibble::tibble(.feature_id = feature_ids(sp)),
      space = sp,
      blocks = list(pcs = pcs)
    )
  )

  manifest <- fds_frame_manifest(frame)
  expect_identical(manifest$arrays[["axis/observation/blocks/motion"]]$shape, c(3L, 2L))
  expect_identical(
    manifest$arrays[["axis/feature/blocks/pcs"]]$axes,
    c("feature", "component:axis/feature/blocks/pcs")
  )

  rebuilt <- frame_from_fds_manifest(manifest, fds_frame_bindings(frame))
  expect_identical(block_component_ids(obs_blocks(rebuilt)$motion), c("tx", "ty"))
  expect_identical(block_component_ids(feature_blocks(rebuilt)$pcs), c("pc1", "pc2", "pc3"))
  expect_equal(realized_block(obs_blocks(rebuilt)$motion), matrix(as.double(1:6), 3, 2))
  expect_equal(realized_block(feature_blocks(rebuilt)$pcs), matrix(as.double(1:12), 4, 3))
  expect_identical(fds_frame_manifest(rebuilt), manifest)
})

test_that("binding frames with blocks preserves component IDs and values", {
  sp <- keyed_space()
  pcs <- axis_block(
    matrix(as.double(1:8), 4, 2),
    components = data.frame(.component_id = c("pc1", "pc2"))
  )
  build <- function(ids, motion_values, feature_block = pcs, lazy = FALSE) {
    n <- length(ids)
    data <- if (lazy) memory_source(motion_values) else motion_values
    fmri_frame(
      assays = list(beta = memory_source(matrix(as.double(seq_len(n * 4)), n, 4))),
      observations = axis_frame(
        tibble::tibble(.obs_id = ids),
        blocks = list(motion = axis_block(
          data, components = data.frame(.component_id = c("tx", "ty"))
        ))
      ),
      features = feature_axis(
        tibble::tibble(.feature_id = feature_ids(sp)),
        space = sp,
        blocks = list(pcs = feature_block)
      )
    )
  }
  a_values <- matrix(c(1, 2, 10, 20), 2, 2)
  b_values <- matrix(c(3, 4, 5, 30, 40, 50), 3, 2)
  a <- build(c("o1", "o2"), a_values)
  b <- build(c("o3", "o4", "o5"), b_values, lazy = TRUE)

  bound <- bind_observations(a, b)
  expect_identical(observation_ids(bound), paste0("o", 1:5))
  expect_identical(block_component_ids(obs_blocks(bound)$motion), c("tx", "ty"))
  expect_equal(realized_block(obs_blocks(bound)$motion), rbind(a_values, b_values))
  expect_identical(dim(realized_block(obs_blocks(bound)$motion)), c(5L, 2L))
  expect_identical(block_component_ids(feature_blocks(bound)$pcs), c("pc1", "pc2"))
  expect_equal(realized_block(feature_blocks(bound)$pcs), matrix(as.double(1:8), 4, 2))

  # Feature blocks are kept from the first frame, so the others must agree.
  other_values <- axis_block(
    matrix(as.double(8:1), 4, 2),
    components = data.frame(.component_id = c("pc1", "pc2"))
  )
  error <- expect_error(
    bind_observations(a, build(c("o6", "o7"), a_values, feature_block = other_values)),
    class = "fmridataset_error_alignment"
  )
  expect_match(conditionMessage(error), "feature block")
})
