.validate_budget_scalar <- function(value, name) {
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
    !is.finite(value) || value <= 0) {
    .frame_abort(
      sprintf("%s must be one positive finite number of bytes.", name),
      "fmridataset_error_budget",
      field = name,
      actual = value
    )
  }
  as.double(value)
}

.frame_plan_fingerprint <- function(x, assay) {
  selection <- .frame_selection(x)
  descriptor <- assay(selection$base, assay)
  .canonical_digest(list(
    assay = assay,
    source = source_fingerprint(descriptor$source),
    observations = selection$observations,
    features = selection$features
  ))
}

.plan_block_shape <- function(shape, chunks, layout, capacity) {
  if (any(shape == 0L)) {
    return(c(0L, 0L))
  }
  n_observation <- shape[[1L]]
  n_feature <- shape[[2L]]
  if (layout == "imagewise") {
    if (n_feature > capacity) {
      .frame_abort(
        "One complete image exceeds the block memory budget; use layout = 'balanced'.",
        "fmridataset_error_budget",
        required_values = n_feature,
        available_values = capacity,
        layout = layout
      )
    }
    return(c(min(n_observation, max(1L, floor(capacity / n_feature))), n_feature))
  }
  if (layout == "featurewise") {
    if (n_observation > capacity) {
      .frame_abort(
        "One complete feature column exceeds the block memory budget; use layout = 'balanced'.",
        "fmridataset_error_budget",
        required_values = n_observation,
        available_values = capacity,
        layout = layout
      )
    }
    return(c(n_observation, min(n_feature, max(1L, floor(capacity / n_observation)))))
  }

  chunks <- pmin(as.integer(chunks), pmax(1L, shape))
  observation_block <- floor(sqrt(capacity * chunks[[1L]] / chunks[[2L]]))
  observation_block <- min(n_observation, max(1L, observation_block))
  feature_block <- min(n_feature, max(1L, floor(capacity / observation_block)))
  observation_block <- min(
    n_observation,
    max(1L, floor(capacity / feature_block))
  )
  as.integer(c(observation_block, feature_block))
}

.axis_block_ranges <- function(n, block_size, prefix) {
  if (n == 0L) {
    out <- data.frame(start = integer(), end = integer(), size = integer())
  } else {
    start <- seq.int(1L, n, by = block_size)
    end <- pmin.int(n, start + block_size - 1L)
    out <- data.frame(start = start, end = end, size = end - start + 1L)
  }
  names(out) <- paste0(".", prefix, c("_start", "_end", "_size"))
  out
}

.block_grid <- function(shape, block_shape, dtype_bytes, layout) {
  observation <- .axis_block_ranges(shape[[1L]], block_shape[[1L]], "observation")
  feature <- .axis_block_ranges(shape[[2L]], block_shape[[2L]], "feature")
  if (!nrow(observation) || !nrow(feature)) {
    return(data.frame(
      .block_id = integer(),
      .observation_start = integer(),
      .observation_end = integer(),
      .n_observation = integer(),
      .feature_start = integer(),
      .feature_end = integer(),
      .n_feature = integer(),
      .bytes = numeric()
    ))
  }
  index <- if (layout == "imagewise") {
    expand.grid(observation = seq_len(nrow(observation)), feature = seq_len(nrow(feature)))
  } else {
    expand.grid(feature = seq_len(nrow(feature)), observation = seq_len(nrow(observation)))
  }
  out <- data.frame(
    .block_id = seq_len(nrow(index)),
    .observation_start = observation[[".observation_start"]][index$observation],
    .observation_end = observation[[".observation_end"]][index$observation],
    .n_observation = observation[[".observation_size"]][index$observation],
    .feature_start = feature[[".feature_start"]][index$feature],
    .feature_end = feature[[".feature_end"]][index$feature],
    .n_feature = feature[[".feature_size"]][index$feature]
  )
  out$.bytes <- as.double(out$.n_observation) * out$.n_feature * dtype_bytes
  out
}

#' Plan bounded observation-by-feature blocks
#'
#' `plan_blocks()` uses source chunk hints and an explicit byte budget to build
#' a serializable, metadata-only execution plan. `"imagewise"` blocks retain
#' complete feature rows, `"featurewise"` blocks retain complete observation
#' columns, and `"balanced"` blocks scale both axes in the source chunk ratio.
#'
#' @param x An `fmri_frame` or lazy view.
#' @param assay Assay name.
#' @param layout One of `"balanced"`, `"imagewise"`, or `"featurewise"`.
#' @param memory_budget Hard maximum bytes for one input block.
#' @param target_block_bytes Preferred block size, capped by `memory_budget`.
#' @return A serializable `frame_block_plan`.
#' @export
plan_blocks <- function(
  x,
  assay = active_assay(x),
  layout = c("balanced", "imagewise", "featurewise"),
  memory_budget = getOption("fmridataset.block_budget", 512 * 1024^2),
  target_block_bytes = getOption("fmridataset.target_block_bytes", 4 * 1024^2)
) {
  if (!inherits(x, "fmri_frame")) {
    .frame_abort("x must be an fmri_frame or fmri_view.", "fmridataset_error_alignment")
  }
  layout <- match.arg(layout)
  memory_budget <- .validate_budget_scalar(memory_budget, "memory_budget")
  target_block_bytes <- .validate_budget_scalar(target_block_bytes, "target_block_bytes")
  selection <- .frame_selection(x)
  descriptor <- assay(selection$base, assay)
  # A block must fit in memory once realized, not once read from storage.
  dtype_bytes <- .realized_dtype_bytes(descriptor$dtype)
  capacity <- floor(min(memory_budget, target_block_bytes) / dtype_bytes)
  if (capacity < 1) {
    .frame_abort(
      "The block memory budget cannot hold one assay value.",
      "fmridataset_error_budget",
      dtype = descriptor$dtype,
      dtype_bytes = dtype_bytes,
      memory_budget = memory_budget
    )
  }
  shape <- as.integer(dim(x))
  chunks <- pmin(source_chunks(descriptor$source), pmax(1L, shape))
  block_shape <- .plan_block_shape(shape, chunks, layout, capacity)
  blocks <- .block_grid(shape, block_shape, dtype_bytes, layout)
  max_block_bytes <- if (nrow(blocks)) max(blocks$.bytes) else 0
  if (max_block_bytes > memory_budget) {
    .frame_abort(
      "The planned block exceeds memory_budget.",
      "fmridataset_error_budget",
      planned_bytes = max_block_bytes,
      memory_budget = memory_budget
    )
  }
  structure(
    list(
      schema_version = 1L,
      assay = assay,
      layout = layout,
      shape = shape,
      dtype = descriptor$dtype,
      dtype_bytes = dtype_bytes,
      source_chunks = as.integer(chunks),
      source_capabilities = source_capabilities(descriptor$source),
      selection_fingerprint = .frame_plan_fingerprint(x, assay),
      block_shape = block_shape,
      blocks = blocks,
      n_blocks = nrow(blocks),
      max_block_bytes = max_block_bytes,
      total_values = prod(as.double(shape)),
      total_bytes = prod(as.double(shape)) * dtype_bytes,
      memory_budget = memory_budget,
      target_block_bytes = target_block_bytes
    ),
    class = "frame_block_plan"
  )
}

#' Inspect a frame block plan
#'
#' @param plan A `frame_block_plan`.
#' @return A data frame containing logical block bounds and byte estimates.
#' @export
block_manifest <- function(plan) {
  if (!inherits(plan, "frame_block_plan")) {
    .frame_abort("plan must be a frame_block_plan.", "fmridataset_error_source_contract")
  }
  plan$blocks
}

#' Execute a bounded frame block plan
#'
#' @param x The same frame or view used to construct `plan`.
#' @param plan A `frame_block_plan`.
#' @param FUN Function receiving `values`, `observation_ids`, `feature_ids`, and
#'   the one-row block manifest entry.
#' @param ... Additional arguments passed to `FUN`.
#' @param assay Assay name; defaults to the planned assay.
#' @return A list containing one result per planned block.
#' @export
execute_block_plan <- function(x, plan, FUN, ..., assay = plan$assay) {
  if (!inherits(plan, "frame_block_plan")) {
    .frame_abort("plan must be a frame_block_plan.", "fmridataset_error_source_contract")
  }
  if (!is.function(FUN)) {
    .frame_abort("FUN must be a function.", "fmridataset_error_source_contract")
  }
  actual <- .frame_plan_fingerprint(x, assay)
  if (!identical(actual, plan$selection_fingerprint) || !identical(assay, plan$assay)) {
    .frame_abort(
      "The block plan does not match this frame selection and assay.",
      "fmridataset_error_source_contract",
      expected = plan$selection_fingerprint,
      actual = actual
    )
  }
  observation_id <- observation_ids(x)
  feature_id <- feature_ids(x)
  lapply(seq_len(nrow(plan$blocks)), function(index) {
    block <- plan$blocks[index, , drop = FALSE]
    observations <- block$.observation_start:block$.observation_end
    features <- block$.feature_start:block$.feature_end
    values <- collect_assay(
      x[observations, features],
      assay = assay,
      memory_budget = plan$memory_budget
    )
    FUN(
      values = values,
      observation_ids = observation_id[observations],
      feature_ids = feature_id[features],
      block = block,
      ...
    )
  })
}

#' @export
print.frame_block_plan <- function(x, ...) {
  cat("<frame_block_plan>", x$layout, "\n")
  cat("  shape:", paste(x$shape, collapse = " x "), "\n")
  cat("  block shape:", paste(x$block_shape, collapse = " x "), "\n")
  cat("  blocks:", x$n_blocks, "\n")
  cat("  maximum block bytes:", format(x$max_block_bytes, scientific = FALSE), "\n")
  invisible(x)
}
