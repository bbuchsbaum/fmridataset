.api_developer_exports <- c(
  "counting_source", "source_counts", "reset_source_counts", "fault_source"
)

.api_extension_exports <- c(
  "aligned_assay_set", "as_array_source", "as_delarr", "assert_compatible_space",
  "axis_block_data", "axis_blocks", "axis_data", "axis_ids", "block_component_ids",
  "block_components", "block_manifest", "collection_digest", "compatible_space",
  "entity_registry_digest", "execute_block_plan", "execution_path",
  "fds_frame_bindings", "fds_frame_manifest", "fds_manifest_digest", "fds_schema",
  "fds_schema_version", "fds_study_bindings", "fds_study_manifest",
  "fds_study_manifest_digest", "fds_study_representations", "feature_map_digest",
  "frame_schema", "frame_schema_digest", "compare_frame_schema",
  "feature_map_operator", "feature_map_source_space", "feature_map_target_space",
  "feature_mapped_source", "frame_from_fds_manifest", "hierarchy_complete",
  "hierarchy_digest", "hierarchy_groups", "hierarchy_ids", "hierarchy_levels",
  "hierarchy_relations", "locate_source_rows", "mask_bank_digest", "plan_blocks",
  "provenance_digest", "relation_registry_digest", "row_bound_source",
  "row_sharded_source", "shard_manifest", "source_capabilities", "source_chunks",
  "source_close", "source_descriptor", "source_dtype", "source_fingerprint",
  "source_open", "source_read", "source_read_native", "source_shape", "source_view",
  "study_digest", "study_from_fds_manifest", "validate_array_source",
  "validate_entity_feature_validity", "validate_entity_registry", "validate_event_table",
  "validate_fds_manifest", "validate_fds_study_manifest", "validate_feature_map",
  "validate_frame_schema", "validate_against_schema",
  "validate_fmri_collection", "validate_fmri_study", "validate_mask_bank",
  "validate_provenance_graph", "validate_relation_registry", "validity_masked_source"
)

.api_user_exports <- c(
  "active_assay", "adjacency", "append_provenance", "append_source_shards",
  "apply_feature_validity", "as_fmri_frame", "assay", "assays", "axis_block",
  "axis_frame", "basis_analysis", "basis_projection_info", "basis_space",
  "basis_space_from_decoder", "basis_space_from_fmrilatent", "basis_synthesis",
  "bind_observations", "block_apply", "collect_assay", "collect_spatial_maps",
  "collection_common_space", "collection_frame", "collection_frames",
  "collection_ids", "collection_space_data", "composite_part",
  "composite_part_names", "composite_parts", "composite_space", "entities",
  "entity", "entity_blocks", "entity_data", "entity_feature_validity",
  "entity_frame", "entity_ids", "entity_key", "entity_names", "entity_registry",
  "event_data", "event_key", "event_table", "events", "execute_spatial", "explain",
  "feature_axis", "feature_blocks", "feature_data", "feature_ids", "feature_map",
  "feature_map_from_target", "features", "filter_entities", "filter_obs",
  "fmri_collection", "fmri_frame", "fmri_study", "frame_link", "hierarchy_index",
  "index_space", "key_relation", "map_features", "mask_bank", "mask_values",
  "memory_source", "n_features", "n_masks", "native_shape", "nifti_array_source",
  "nifti_source_space", "obs_blocks", "observation_axis", "observation_ids",
  "observation_validity", "observations", "open_frame", "parcel_aggregation",
  "parcel_membership", "parcel_space", "parcel_space_from_atlas", "parent_space",
  "provenance_graph", "provenance_record", "provenance_records", "provenance_tips",
  "reconstruct_space", "relation", "relation_names", "relation_registry", "relations",
  "restrict_space", "select_features", "space", "space_digest", "sparse_relation",
  "spatial_map", "study_frame", "study_frames", "study_ids", "study_link",
  "study_links", "study_table", "study_tables", "surface_space",
  "surface_space_from_neurosurf", "upgrade_dataset", "validity_coverage",
  "validity_entity", "validity_entity_ids", "validity_mask_bank", "validity_matrix",
  "validity_space", "vectorize_space", "volume_space", "write_frame",
  "zarr_array_source"
)

test_that("every ordinary export has one documented audience", {
  exports <- getNamespaceExports("fmridataset")
  method_exports <- grep(
    "^(nrow|ncol)\\.(fmri_frame|fmri_view)$",
    exports,
    value = TRUE
  )
  audiences <- c(.api_user_exports, .api_extension_exports, .api_developer_exports)

  expect_length(intersect(.api_developer_exports, .api_extension_exports), 0L)
  expect_length(intersect(.api_user_exports, .api_extension_exports), 0L)
  expect_length(intersect(.api_user_exports, .api_developer_exports), 0L)
  expect_identical(anyDuplicated(audiences), 0L)
  expect_setequal(exports, c(audiences, method_exports))
  expect_true(file.exists(system.file(
    "architecture", "API-AUDIENCES.md", package = "fmridataset"
  )))
})

test_that("implementation conveniences are absent from the public namespace", {
  internal <- "%||%"
  removed <- c(
    "analyze_run", "generate_benchmark_data",
    "generate_example_events", "generate_example_fmri_data",
    "generate_example_mask", "generate_example_paths", "print_dataset_info"
  )
  exports <- getNamespaceExports("fmridataset")

  expect_length(intersect(c(internal, removed), exports), 0L)
  expect_true(exists(internal, envir = asNamespace("fmridataset"), inherits = FALSE))
  expect_false(any(vapply(
    removed, exists, logical(1), envir = asNamespace("fmridataset"), inherits = FALSE
  )))
})

test_that("developer-only exports retain explicit audience documentation", {
  counted <- help("counting_source", package = "fmridataset")
  faulted <- help("fault_source", package = "fmridataset")

  expect_true(length(counted) > 0L)
  expect_true(length(faulted) > 0L)
})
