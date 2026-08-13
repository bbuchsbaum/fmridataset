# Package index

## All functions

- [`aligned_assay_set()`](https://bbuchsbaum.github.io/fmridataset/reference/aligned_assay_set.md)
  : Construct a strictly aligned assay set

- [`all_selector()`](https://bbuchsbaum.github.io/fmridataset/reference/all_selector.md)
  : All Voxels Series Selector

- [`append_source_shards()`](https://bbuchsbaum.github.io/fmridataset/reference/append_source_shards.md)
  : Append immutable source shards

- [`apply_feature_validity()`](https://bbuchsbaum.github.io/fmridataset/reference/apply_feature_validity.md)
  : Apply one validity relation lazily to frame assays

- [`as_array_source()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
  [`source_shape()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
  [`source_dtype()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
  [`source_chunks()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
  [`source_capabilities()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
  [`source_fingerprint()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
  [`source_open()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
  [`source_read()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
  [`source_read_native()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
  [`source_close()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
  : Serializable numerical array sources

- [`as.matrix(`*`<fmri_series>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/as.matrix.fmri_series.md)
  : Convert fmri_series to Matrix

- [`as.matrix_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/as.matrix_dataset.md)
  : Convert to Matrix Dataset

- [`as_delarr()`](https://bbuchsbaum.github.io/fmridataset/reference/as_delarr.md)
  : Convert backend to a delarr lazy matrix

- [`as_fmri_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/as_fmri_frame.md)
  : Coerce an object to the canonical frame protocol

- [`as_fmri_group()`](https://bbuchsbaum.github.io/fmridataset/reference/as_fmri_group.md)
  : Coerce a data frame into an fmri_group

- [`as_tibble(`*`<fmri_series>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/as_tibble.fmri_series.md)
  : Convert fmri_series to Tibble

- [`as_tibble(`*`<fmri_study_dataset>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/as_tibble.fmri_study_dataset.md)
  : Convert fmri_study_dataset to a tibble or lazy matrix

- [`axis_block()`](https://bbuchsbaum.github.io/fmridataset/reference/axis_block.md)
  [`axis_block_data()`](https://bbuchsbaum.github.io/fmridataset/reference/axis_block.md)
  [`block_components()`](https://bbuchsbaum.github.io/fmridataset/reference/axis_block.md)
  [`block_component_ids()`](https://bbuchsbaum.github.io/fmridataset/reference/axis_block.md)
  : Construct an axis-aligned multivariate block

- [`axis_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/axis_frame.md)
  [`axis_data()`](https://bbuchsbaum.github.io/fmridataset/reference/axis_frame.md)
  [`axis_blocks()`](https://bbuchsbaum.github.io/fmridataset/reference/axis_frame.md)
  [`axis_ids()`](https://bbuchsbaum.github.io/fmridataset/reference/axis_frame.md)
  : Construct an annotated axis

- [`basis_analysis()`](https://bbuchsbaum.github.io/fmridataset/reference/basis-operators.md)
  [`basis_synthesis()`](https://bbuchsbaum.github.io/fmridataset/reference/basis-operators.md)
  [`basis_projection_info()`](https://bbuchsbaum.github.io/fmridataset/reference/basis-operators.md)
  : Inspect basis-space operators

- [`basis_space()`](https://bbuchsbaum.github.io/fmridataset/reference/basis_space.md)
  : Construct a linear basis feature space

- [`basis_space_from_decoder()`](https://bbuchsbaum.github.io/fmridataset/reference/basis_space_from_decoder.md)
  : Construct a basis space from a synthesis dictionary

- [`basis_space_from_fmrilatent()`](https://bbuchsbaum.github.io/fmridataset/reference/basis_space_from_fmrilatent.md)
  : Adapt fmrilatent spatial loadings to a basis feature space

- [`bids_h5_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/bids_h5_dataset.md)
  : Open a BIDS HDF5 Study Archive

- [`bind_observations()`](https://bbuchsbaum.github.io/fmridataset/reference/bind_observations.md)
  : Bind frames along observations

- [`block_apply()`](https://bbuchsbaum.github.io/fmridataset/reference/block_apply.md)
  : Apply a function to bounded feature blocks

- [`block_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/block_manifest.md)
  : Inspect a frame block plan

- [`blockids()`](https://bbuchsbaum.github.io/fmridataset/reference/blockids.md)
  : Get Block IDs from Sampling Frame

- [`blocklens()`](https://bbuchsbaum.github.io/fmridataset/reference/blocklens.md)
  : Get Block Lengths from Objects

- [`collect_assay()`](https://bbuchsbaum.github.io/fmridataset/reference/collect_assay.md)
  : Collect one frame assay under an explicit memory budget

- [`collect_chunks()`](https://bbuchsbaum.github.io/fmridataset/reference/collect_chunks.md)
  : Collect all chunks from a chunk iterator

- [`collect_spatial_maps()`](https://bbuchsbaum.github.io/fmridataset/reference/collect_spatial_maps.md)
  : Collect spatial maps through native or reconstructed reads

- [`collection_frames()`](https://bbuchsbaum.github.io/fmridataset/reference/collection-accessors.md)
  [`collection_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/collection-accessors.md)
  [`collection_ids()`](https://bbuchsbaum.github.io/fmridataset/reference/collection-accessors.md)
  : Access frames in an fMRI collection

- [`collection_space_data()`](https://bbuchsbaum.github.io/fmridataset/reference/collection-spaces.md)
  [`collection_common_space()`](https://bbuchsbaum.github.io/fmridataset/reference/collection-spaces.md)
  : Summarize collection feature spaces

- [`collection_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/collection_digest.md)
  : Compute a deterministic collection digest

- [`composite_parts()`](https://bbuchsbaum.github.io/fmridataset/reference/composite-parts.md)
  [`composite_part_names()`](https://bbuchsbaum.github.io/fmridataset/reference/composite-parts.md)
  [`composite_part()`](https://bbuchsbaum.github.io/fmridataset/reference/composite-parts.md)
  : Inspect composite feature-space parts

- [`composite_space()`](https://bbuchsbaum.github.io/fmridataset/reference/composite_space.md)
  : Construct an ordered composite feature space

- [`compress_bids_study()`](https://bbuchsbaum.github.io/fmridataset/reference/compress_bids_study.md)
  : Compress a BIDS Study into a Single HDF5 Archive

- [`counting_source()`](https://bbuchsbaum.github.io/fmridataset/reference/counting_source.md)
  [`source_counts()`](https://bbuchsbaum.github.io/fmridataset/reference/counting_source.md)
  [`reset_source_counts()`](https://bbuchsbaum.github.io/fmridataset/reference/counting_source.md)
  : Instrument an array source

- [`create_backend()`](https://bbuchsbaum.github.io/fmridataset/reference/create_backend.md)
  : Create Backend Instance

- [`data_chunk()`](https://bbuchsbaum.github.io/fmridataset/reference/data_chunk.md)
  : Create a Data Chunk Object

- [`data_chunks()`](https://bbuchsbaum.github.io/fmridataset/reference/data_chunks.md)
  : Create Data Chunks for Processing

- [`data_chunks(`*`<fmri_file_dataset>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/data_chunks.fmri_file_dataset.md)
  : Create Data Chunks for fmri_file_dataset Objects

- [`data_chunks(`*`<fmri_mem_dataset>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/data_chunks.fmri_mem_dataset.md)
  : Create Data Chunks for fmri_mem_dataset Objects

- [`data_chunks(`*`<fmri_study_dataset>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/data_chunks.fmri_study_dataset.md)
  : Create Data Chunks for fmri_study_dataset Objects

- [`data_chunks(`*`<matrix_dataset>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/data_chunks.matrix_dataset.md)
  : Create Data Chunks for matrix_dataset Objects

- [`dim(`*`<fmri_series>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/dim.fmri_series.md)
  : Dimensions of fmri_series

- [`encoding_info()`](https://bbuchsbaum.github.io/fmridataset/reference/encoding_info.md)
  : Get Encoding Metadata from a Latent-Mode BIDS H5 Dataset

- [`entities()`](https://bbuchsbaum.github.io/fmridataset/reference/entity-accessors.md)
  [`entity()`](https://bbuchsbaum.github.io/fmridataset/reference/entity-accessors.md)
  [`entity_names()`](https://bbuchsbaum.github.io/fmridataset/reference/entity-accessors.md)
  : Access entities from a frame or registry

- [`entity_key()`](https://bbuchsbaum.github.io/fmridataset/reference/entity-frame-accessors.md)
  [`entity_ids()`](https://bbuchsbaum.github.io/fmridataset/reference/entity-frame-accessors.md)
  [`entity_data()`](https://bbuchsbaum.github.io/fmridataset/reference/entity-frame-accessors.md)
  [`entity_blocks()`](https://bbuchsbaum.github.io/fmridataset/reference/entity-frame-accessors.md)
  : Entity-frame accessors

- [`entity_feature_validity()`](https://bbuchsbaum.github.io/fmridataset/reference/entity_feature_validity.md)
  : Describe compressed entity-by-feature validity

- [`entity_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/entity_frame.md)
  : Construct a keyed entity frame

- [`entity_registry()`](https://bbuchsbaum.github.io/fmridataset/reference/entity_registry.md)
  [`validate_entity_registry()`](https://bbuchsbaum.github.io/fmridataset/reference/entity_registry.md)
  : Construct and validate an entity registry

- [`entity_registry_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/entity_registry_digest.md)
  : Compute a stable entity-registry digest

- [`event_data()`](https://bbuchsbaum.github.io/fmridataset/reference/event-accessors.md)
  [`event_key()`](https://bbuchsbaum.github.io/fmridataset/reference/event-accessors.md)
  : Event-table accessors

- [`event_table()`](https://bbuchsbaum.github.io/fmridataset/reference/event_table.md)
  : Construct a keyed event table

- [`exec_strategy()`](https://bbuchsbaum.github.io/fmridataset/reference/exec_strategy.md)
  : Create an Execution Strategy for Data Processing

- [`execute_block_plan()`](https://bbuchsbaum.github.io/fmridataset/reference/execute_block_plan.md)
  : Execute a bounded frame block plan

- [`execute_spatial()`](https://bbuchsbaum.github.io/fmridataset/reference/execute_spatial.md)
  : Stream an operation over spatial maps

- [`execution_path()`](https://bbuchsbaum.github.io/fmridataset/reference/execution_path.md)
  : Select a matrix or spatial execution path

- [`explain()`](https://bbuchsbaum.github.io/fmridataset/reference/explain.md)
  : Explain a frame without reading assay values

- [`fault_source()`](https://bbuchsbaum.github.io/fmridataset/reference/fault_source.md)
  : Inject deterministic source failures

- [`fds_frame_bindings()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_frame_bindings.md)
  : Extract physical array bindings from a frame

- [`fds_frame_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_frame_manifest.md)
  [`validate_fds_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_frame_manifest.md)
  : Construct and validate an FDS v1 frame manifest

- [`fds_manifest_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_manifest_digest.md)
  : Compute a canonical FDS manifest digest

- [`fds_schema()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_schema.md)
  [`fds_schema_version()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_schema.md)
  : FDS logical schema identity

- [`fds_study_bindings()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_study_bindings.md)
  : Extract shared study-level physical bindings

- [`fds_study_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_study_manifest.md)
  [`validate_fds_study_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_study_manifest.md)
  : Construct and validate an FDS v1 study manifest

- [`fds_study_manifest_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_study_manifest_digest.md)
  : Compute a canonical FDS study-manifest digest

- [`fds_study_representations()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_study_representations.md)
  : Extract canonical study representations for persistence

- [`validate_feature_map()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-map-accessors.md)
  [`feature_map_source_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-map-accessors.md)
  [`feature_map_target_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-map-accessors.md)
  [`feature_map_operator()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-map-accessors.md)
  [`feature_map_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-map-accessors.md)
  : Validate and inspect feature maps

- [`n_features()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
  [`feature_ids()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
  [`native_shape()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
  [`feature_data()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
  [`space_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
  [`restrict_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
  [`vectorize_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
  [`reconstruct_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
  [`adjacency()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
  [`compatible_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
  [`assert_compatible_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
  : Feature-space contract

- [`validate_entity_feature_validity()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-validity-accessors.md)
  [`validity_entity()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-validity-accessors.md)
  [`validity_entity_ids()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-validity-accessors.md)
  [`validity_mask_bank()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-validity-accessors.md)
  [`validity_space()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-validity-accessors.md)
  [`validity_matrix()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-validity-accessors.md)
  : Validate and inspect entity-feature validity

- [`feature_axis()`](https://bbuchsbaum.github.io/fmridataset/reference/feature_axis.md)
  : Construct a spatial feature axis

- [`feature_map()`](https://bbuchsbaum.github.io/fmridataset/reference/feature_map.md)
  : Describe an explicit transformation between feature spaces

- [`feature_map_from_target()`](https://bbuchsbaum.github.io/fmridataset/reference/feature_map_from_target.md)
  : Derive the canonical map owned by a parent-linked target space

- [`feature_mapped_source()`](https://bbuchsbaum.github.io/fmridataset/reference/feature_mapped_source.md)
  : Construct a lazy source transformed through a feature map

- [`filter_entities()`](https://bbuchsbaum.github.io/fmridataset/reference/filter_entities.md)
  : Filter every study representation through one shared entity
  selection

- [`filter_obs()`](https://bbuchsbaum.github.io/fmridataset/reference/filter_obs.md)
  : Filter frame observations using scalar metadata

- [`filter_subjects()`](https://bbuchsbaum.github.io/fmridataset/reference/filter_subjects.md)
  : Filter subjects in an fmri_group

- [`fmri_cache_info()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_cache_info.md)
  : Get cache information and statistics

- [`fmri_cache_resize()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_cache_resize.md)
  : Resize the fmridataset cache

- [`fmri_clear_cache()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_clear_cache.md)
  : Clear fmridataset cache

- [`fmri_collection()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_collection.md)
  : Construct a collection of semantically equivalent fMRI frames

- [`fmri_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_dataset.md)
  : Create an fMRI Dataset Object from a Set of Scans

- [`fmri_dataset_legacy()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_dataset_legacy.md)
  : Legacy fMRI Dataset Constructor

- [`fmri_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_frame.md)
  : Construct a spatially typed annotated matrix

- [`fmri_group()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_group.md)
  : Create an fmri_group (one row per subject)

- [`fmri_h5_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_h5_dataset.md)
  : Create an fMRI Dataset Object from H5 Files

- [`fmri_latent_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_latent_dataset.md)
  **\[deprecated\]** : Create an fMRI Dataset Object from LatentNeuroVec
  Files or Objects

- [`fmri_mem_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_mem_dataset.md)
  : Create an fMRI Memory Dataset Object

- [`fmri_series()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_series.md)
  : fmri_series: fMRI Time Series Container

- [`fmri_series_resolvers`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_series_resolvers.md)
  : Helpers for fmri_series spatial and temporal selection

- [`fmri_study()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_study.md)
  : Construct a linked fMRI study

- [`fmri_study_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_study_dataset.md)
  : Create an fmri_study_dataset

- [`fmri_zarr_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_zarr_dataset.md)
  : Create an fMRI Dataset from Zarr Arrays

- [`assays()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`assay()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`active_assay()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`observation_axis()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`observations()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`features()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`observation_ids()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`obs_blocks()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`feature_blocks()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`dim(`*`<fmri_frame>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`nrow.fmri_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`ncol.fmri_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`dim(`*`<fmri_view>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`nrow.fmri_view()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  [`ncol.fmri_view()`](https://bbuchsbaum.github.io/fmridataset/reference/frame-accessors.md)
  : Frame accessors

- [`frame_from_fds_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/frame_from_fds_manifest.md)
  : Reconstruct a frame from an FDS manifest and physical sources

- [`frame_link()`](https://bbuchsbaum.github.io/fmridataset/reference/frame_link.md)
  : Describe a typed link between study representations

- [`generics`](https://bbuchsbaum.github.io/fmridataset/reference/generics.md)
  : Generic Functions for fMRI Dataset Operations

- [`get_TR()`](https://bbuchsbaum.github.io/fmridataset/reference/get_TR.md)
  : Get TR (Repetition Time) from Sampling Frame

- [`get_backend_registry()`](https://bbuchsbaum.github.io/fmridataset/reference/get_backend_registry.md)
  : Get Registered Backend Information

- [`get_component_info()`](https://bbuchsbaum.github.io/fmridataset/reference/get_component_info.md)
  : Get Component Information

- [`get_confounds()`](https://bbuchsbaum.github.io/fmridataset/reference/get_confounds.md)
  : Get Confound Regressors from a BIDS H5 Dataset

- [`get_data()`](https://bbuchsbaum.github.io/fmridataset/reference/get_data.md)
  : Get Data from fMRI Dataset Objects

- [`get_data_matrix()`](https://bbuchsbaum.github.io/fmridataset/reference/get_data_matrix.md)
  : Get Data Matrix from fMRI Dataset Objects

- [`get_latent_scores()`](https://bbuchsbaum.github.io/fmridataset/reference/get_latent_scores.md)
  : Get Latent Scores from Dataset

- [`get_loadings()`](https://bbuchsbaum.github.io/fmridataset/reference/get_loadings.md)
  : Get Spatial Loadings from a Latent-Mode BIDS H5 Dataset

- [`get_mask()`](https://bbuchsbaum.github.io/fmridataset/reference/get_mask.md)
  : Get Mask from fMRI Dataset Objects

- [`get_run_duration()`](https://bbuchsbaum.github.io/fmridataset/reference/get_run_duration.md)
  : Get Run Duration from Sampling Frame

- [`get_run_lengths()`](https://bbuchsbaum.github.io/fmridataset/reference/get_run_lengths.md)
  : Get Run Lengths from Sampling Frame

- [`get_spatial_loadings()`](https://bbuchsbaum.github.io/fmridataset/reference/get_spatial_loadings.md)
  : Get Spatial Loadings from Dataset

- [`get_total_duration()`](https://bbuchsbaum.github.io/fmridataset/reference/get_total_duration.md)
  : Get Total Duration from Sampling Frame

- [`group_map()`](https://bbuchsbaum.github.io/fmridataset/reference/group_map.md)
  : Map a function over subjects in an fmri_group

- [`group_reduce()`](https://bbuchsbaum.github.io/fmridataset/reference/group_reduce.md)
  : Reduce over subjects in a single pass

- [`hierarchy_ids()`](https://bbuchsbaum.github.io/fmridataset/reference/hierarchy-accessors.md)
  [`hierarchy_groups()`](https://bbuchsbaum.github.io/fmridataset/reference/hierarchy-accessors.md)
  [`hierarchy_levels()`](https://bbuchsbaum.github.io/fmridataset/reference/hierarchy-accessors.md)
  [`hierarchy_relations()`](https://bbuchsbaum.github.io/fmridataset/reference/hierarchy-accessors.md)
  [`hierarchy_complete()`](https://bbuchsbaum.github.io/fmridataset/reference/hierarchy-accessors.md)
  : Access derived hierarchy index data

- [`hierarchy_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/hierarchy_digest.md)
  : Compute the deterministic digest of a hierarchy index

- [`hierarchy_index()`](https://bbuchsbaum.github.io/fmridataset/reference/hierarchy_index.md)
  : Derive stable observation hierarchy indices

- [`index_selector()`](https://bbuchsbaum.github.io/fmridataset/reference/index_selector.md)
  : Index-based Series Selector

- [`index_space()`](https://bbuchsbaum.github.io/fmridataset/reference/index_space.md)
  : Construct a generic indexed feature space

- [`is.fmri_series()`](https://bbuchsbaum.github.io/fmridataset/reference/is.fmri_series.md)
  : Check if object is an fmri_series

- [`is.sampling_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/is.sampling_frame.md)
  : Test if Object is a Sampling Frame

- [`is_backend_registered()`](https://bbuchsbaum.github.io/fmridataset/reference/is_backend_registered.md)
  : Check if Backend is Registered

- [`iter_subjects()`](https://bbuchsbaum.github.io/fmridataset/reference/iter_subjects.md)
  : Iterate subjects one-by-one (streaming)

- [`key_relation()`](https://bbuchsbaum.github.io/fmridataset/reference/key_relation.md)
  : Describe a symbolic foreign-key relation

- [`latent_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/latent_dataset.md)
  : Latent Dataset Interface

- [`left_join_subjects()`](https://bbuchsbaum.github.io/fmridataset/reference/left_join_subjects.md)
  : Left join additional subject metadata

- [`list_backend_names()`](https://bbuchsbaum.github.io/fmridataset/reference/list_backend_names.md)
  : List Registered Backend Names

- [`locate_source_rows()`](https://bbuchsbaum.github.io/fmridataset/reference/locate_source_rows.md)
  : Resolve logical observation rows to shards

- [`map_features()`](https://bbuchsbaum.github.io/fmridataset/reference/map_features.md)
  : Lazily transform a frame into a new feature domain

- [`validate_mask_bank()`](https://bbuchsbaum.github.io/fmridataset/reference/mask-bank-accessors.md)
  [`n_masks()`](https://bbuchsbaum.github.io/fmridataset/reference/mask-bank-accessors.md)
  [`mask_values()`](https://bbuchsbaum.github.io/fmridataset/reference/mask-bank-accessors.md)
  [`mask_bank_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/mask-bank-accessors.md)
  : Validate and inspect a mask bank

- [`mask_bank()`](https://bbuchsbaum.github.io/fmridataset/reference/mask_bank.md)
  : Construct a deduplicated, bit-packed bank of feature masks

- [`mask_selector()`](https://bbuchsbaum.github.io/fmridataset/reference/mask_selector.md)
  : Mask-based Series Selector

- [`matrix_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/matrix_dataset.md)
  : Matrix Dataset Constructor

- [`memory_source()`](https://bbuchsbaum.github.io/fmridataset/reference/memory_source.md)
  : Construct an in-memory array source

- [`mutate_subjects()`](https://bbuchsbaum.github.io/fmridataset/reference/mutate_subjects.md)
  : Mutate subject-level attributes

- [`n_runs()`](https://bbuchsbaum.github.io/fmridataset/reference/n_runs.md)
  : Get Number of Runs from Sampling Frame

- [`n_subjects()`](https://bbuchsbaum.github.io/fmridataset/reference/n_subjects.md)
  : Number of subjects in a group

- [`n_timepoints()`](https://bbuchsbaum.github.io/fmridataset/reference/n_timepoints.md)
  : Get Number of Timepoints from Sampling Frame

- [`ncol(`*`<fmri_series>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/ncol.fmri_series.md)
  : Number of columns in fmri_series

- [`nifti_array_source()`](https://bbuchsbaum.github.io/fmridataset/reference/nifti_array_source.md)
  : Construct a pushdown-aware NIfTI array source

- [`nifti_source_space()`](https://bbuchsbaum.github.io/fmridataset/reference/nifti_source_space.md)
  : Recover the spatial domain of a NIfTI source

- [`nrow(`*`<fmri_series>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/nrow.fmri_series.md)
  : Number of rows in fmri_series

- [`observation_validity()`](https://bbuchsbaum.github.io/fmridataset/reference/observation_validity.md)
  : Resolve validity onto frame observations

- [`parent_space()`](https://bbuchsbaum.github.io/fmridataset/reference/parcel-operators.md)
  [`parcel_membership()`](https://bbuchsbaum.github.io/fmridataset/reference/parcel-operators.md)
  [`parcel_aggregation()`](https://bbuchsbaum.github.io/fmridataset/reference/parcel-operators.md)
  : Inspect parent-linked feature spaces and parcel-space operators

- [`parcel_space()`](https://bbuchsbaum.github.io/fmridataset/reference/parcel_space.md)
  : Construct a parent-linked parcel feature space

- [`parcel_space_from_atlas()`](https://bbuchsbaum.github.io/fmridataset/reference/parcel_space_from_atlas.md)
  : Build a parcel space from a neuroatlas atlas

- [`parcellation_info()`](https://bbuchsbaum.github.io/fmridataset/reference/parcellation_info.md)
  : Get Parcellation Information from a BIDS H5 Dataset

- [`participants()`](https://bbuchsbaum.github.io/fmridataset/reference/participants.md)
  : Get Participant IDs from a Dataset

- [`plan_blocks()`](https://bbuchsbaum.github.io/fmridataset/reference/plan_blocks.md)
  : Plan bounded observation-by-feature blocks

- [`print(`*`<fmri_dataset>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/print.md)
  [`summary(`*`<fmri_dataset>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/print.md)
  [`print(`*`<chunkiter>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/print.md)
  [`print(`*`<data_chunk>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/print.md)
  [`print(`*`<matrix_dataset>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/print.md)
  : Print Methods for fmridataset Objects

- [`print(`*`<backend_registry>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/print.backend_registry.md)
  : Print Backend Registry

- [`print(`*`<fmri_series>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/print.fmri_series.md)
  : Print Method for fmri_series Objects

- [`print(`*`<series_selector>`*`)`](https://bbuchsbaum.github.io/fmridataset/reference/print.series_selector.md)
  : Print Methods for Series Selectors

- [`provenance_graph()`](https://bbuchsbaum.github.io/fmridataset/reference/provenance-graph.md)
  [`validate_provenance_graph()`](https://bbuchsbaum.github.io/fmridataset/reference/provenance-graph.md)
  [`provenance_records()`](https://bbuchsbaum.github.io/fmridataset/reference/provenance-graph.md)
  [`provenance_tips()`](https://bbuchsbaum.github.io/fmridataset/reference/provenance-graph.md)
  [`provenance_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/provenance-graph.md)
  [`append_provenance()`](https://bbuchsbaum.github.io/fmridataset/reference/provenance-graph.md)
  : Construct and inspect an immutable provenance graph

- [`provenance_record()`](https://bbuchsbaum.github.io/fmridataset/reference/provenance_record.md)
  : Create a content-addressed provenance record

- [`read_bids_bold()`](https://bbuchsbaum.github.io/fmridataset/reference/read_bids_bold.md)
  : Read one subject's preprocessed BIDS BOLD data as an fmri_frame

- [`read_fmri_config()`](https://bbuchsbaum.github.io/fmridataset/reference/read_fmri_config.md)
  : read a basic fMRI configuration file

- [`reconstruct_voxels()`](https://bbuchsbaum.github.io/fmridataset/reference/reconstruct_voxels.md)
  : Reconstruct Voxel-Space Data from a Latent-Mode BIDS H5 Dataset

- [`register_backend()`](https://bbuchsbaum.github.io/fmridataset/reference/register_backend.md)
  : Register a Storage Backend

- [`relations()`](https://bbuchsbaum.github.io/fmridataset/reference/relation-accessors.md)
  [`relation()`](https://bbuchsbaum.github.io/fmridataset/reference/relation-accessors.md)
  [`relation_names()`](https://bbuchsbaum.github.io/fmridataset/reference/relation-accessors.md)
  : Access frame relations

- [`relation_registry()`](https://bbuchsbaum.github.io/fmridataset/reference/relation_registry.md)
  : Construct a relation registry

- [`relation_registry_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/relation_registry_digest.md)
  : Compute a stable relation-registry digest

- [`resolve_indices()`](https://bbuchsbaum.github.io/fmridataset/reference/resolve_indices.md)
  : Resolve Indices from Series Selector

- [`roi_selector()`](https://bbuchsbaum.github.io/fmridataset/reference/roi_selector.md)
  : ROI-based Series Selector

- [`row_bound_source()`](https://bbuchsbaum.github.io/fmridataset/reference/row_bound_source.md)
  : Bind compatible sources along observations

- [`row_sharded_source()`](https://bbuchsbaum.github.io/fmridataset/reference/row_sharded_source.md)
  : Construct a manifest-backed row-sharded source

- [`sample_subjects()`](https://bbuchsbaum.github.io/fmridataset/reference/sample_subjects.md)
  : Sample subjects from an fmri_group

- [`samples()`](https://bbuchsbaum.github.io/fmridataset/reference/samples.md)
  : Get Sample Indices from Sampling Frame

- [`scan_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/scan_manifest.md)
  : Get Scan Manifest from a BIDS H5 Dataset

- [`select_features()`](https://bbuchsbaum.github.io/fmridataset/reference/select_features.md)
  : Select frame features using feature metadata

- [`series()`](https://bbuchsbaum.github.io/fmridataset/reference/series.md)
  :

  Deprecated alias for `fmri_series`

- [`series_selector`](https://bbuchsbaum.github.io/fmridataset/reference/series_selector.md)
  : Series Selector Classes for fMRI Data

- [`sessions()`](https://bbuchsbaum.github.io/fmridataset/reference/sessions.md)
  : Get Session Names from a Dataset

- [`shard_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/shard_manifest.md)
  : Inspect a row-sharded source manifest

- [`source_descriptor()`](https://bbuchsbaum.github.io/fmridataset/reference/source_descriptor.md)
  [`validate_array_source()`](https://bbuchsbaum.github.io/fmridataset/reference/source_descriptor.md)
  : Inspect and validate an array-source contract

- [`source_view()`](https://bbuchsbaum.github.io/fmridataset/reference/source_view.md)
  : Construct a lazy view over an array source

- [`space()`](https://bbuchsbaum.github.io/fmridataset/reference/space.md)
  : Feature-space accessor

- [`sparse_relation()`](https://bbuchsbaum.github.io/fmridataset/reference/sparse_relation.md)
  : Describe an explicit sparse or many-to-many relation

- [`spatial_map()`](https://bbuchsbaum.github.io/fmridataset/reference/spatial_map.md)
  : Recover a spatial map for one observation

- [`sphere_selector()`](https://bbuchsbaum.github.io/fmridataset/reference/sphere_selector.md)
  : Spherical ROI Series Selector

- [`stream_subjects()`](https://bbuchsbaum.github.io/fmridataset/reference/stream_subjects.md)
  : Stream subjects with optional ordering

- [`study_frames()`](https://bbuchsbaum.github.io/fmridataset/reference/study-accessors.md)
  [`study_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/study-accessors.md)
  [`study_ids()`](https://bbuchsbaum.github.io/fmridataset/reference/study-accessors.md)
  : Study representation accessors

- [`study_links()`](https://bbuchsbaum.github.io/fmridataset/reference/study-registries.md)
  [`study_link()`](https://bbuchsbaum.github.io/fmridataset/reference/study-registries.md)
  [`study_tables()`](https://bbuchsbaum.github.io/fmridataset/reference/study-registries.md)
  [`study_table()`](https://bbuchsbaum.github.io/fmridataset/reference/study-registries.md)
  [`events()`](https://bbuchsbaum.github.io/fmridataset/reference/study-registries.md)
  : Study link and table accessors

- [`study_backend()`](https://bbuchsbaum.github.io/fmridataset/reference/study_backend.md)
  : Study Backend

- [`study_digest()`](https://bbuchsbaum.github.io/fmridataset/reference/study_digest.md)
  : Compute a deterministic study digest

- [`study_from_fds_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/study_from_fds_manifest.md)
  : Reconstruct a study from semantic and physical components

- [`study_to_group()`](https://bbuchsbaum.github.io/fmridataset/reference/study_to_group.md)
  : Convert a BIDS H5 Study Dataset to an fmri_group

- [`subject_ids()`](https://bbuchsbaum.github.io/fmridataset/reference/subject_ids.md)
  : Get Subject IDs from Multi-Subject Dataset

- [`subjects()`](https://bbuchsbaum.github.io/fmridataset/reference/subjects.md)
  [`` `subjects<-`() ``](https://bbuchsbaum.github.io/fmridataset/reference/subjects.md)
  : Access the subjects tibble stored inside an fmri_group

- [`subset_bids_h5()`](https://bbuchsbaum.github.io/fmridataset/reference/subset_bids_h5.md)
  : Subset a BIDS H5 Study Dataset

- [`surface_space()`](https://bbuchsbaum.github.io/fmridataset/reference/surface_space.md)
  : Construct a packed cortical surface feature space

- [`surface_space_from_neurosurf()`](https://bbuchsbaum.github.io/fmridataset/reference/surface_space_from_neurosurf.md)
  : Adapt a neurosurf geometry to a surface feature space

- [`tasks()`](https://bbuchsbaum.github.io/fmridataset/reference/tasks.md)
  : Get Task Names from a Dataset

- [`unregister_backend()`](https://bbuchsbaum.github.io/fmridataset/reference/unregister_backend.md)
  : Unregister a Backend

- [`validate_event_table()`](https://bbuchsbaum.github.io/fmridataset/reference/validate_event_table.md)
  : Validate a keyed event table

- [`validate_fmri_collection()`](https://bbuchsbaum.github.io/fmridataset/reference/validate_fmri_collection.md)
  : Validate an fMRI collection

- [`validate_fmri_group()`](https://bbuchsbaum.github.io/fmridataset/reference/validate_fmri_group.md)
  : Validate an fmri_group object

- [`validate_fmri_study()`](https://bbuchsbaum.github.io/fmridataset/reference/validate_fmri_study.md)
  : Validate a study or filtered study view

- [`validate_relation_registry()`](https://bbuchsbaum.github.io/fmridataset/reference/validate_relation_registry.md)
  : Validate a relation registry

- [`validity_coverage()`](https://bbuchsbaum.github.io/fmridataset/reference/validity_coverage.md)
  : Summarize feature coverage without imposing an analysis policy

- [`validity_masked_source()`](https://bbuchsbaum.github.io/fmridataset/reference/validity_masked_source.md)
  : Lazily mask invalid observation-feature cells with missing values

- [`volume_space()`](https://bbuchsbaum.github.io/fmridataset/reference/volume_space.md)
  : Construct a packed volumetric feature space

- [`voxel_selector()`](https://bbuchsbaum.github.io/fmridataset/reference/voxel_selector.md)
  : Voxel Coordinate Series Selector

- [`with_rowData()`](https://bbuchsbaum.github.io/fmridataset/reference/with_rowData.md)
  : Attach rowData metadata to a lazy matrix

- [`write_fmri_config()`](https://bbuchsbaum.github.io/fmridataset/reference/write_fmri_config.md)
  : Write fMRI configuration file

- [`write_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/write_frame.md)
  [`open_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/write_frame.md)
  : Persist and reopen an fmri frame

- [`zarr_array_source()`](https://bbuchsbaum.github.io/fmridataset/reference/zarr_array_source.md)
  : Construct an experimental Zarr array source
