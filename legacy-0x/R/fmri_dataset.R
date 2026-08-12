# ========================================================================
# fMRI Dataset Package - Main Entry Point
# ========================================================================
#
# This file serves as the main entry point for the fmridataset package.
# The original fmri_dataset.R file has been refactored into multiple
# modular files for better maintainability:
#
# Code Organization:
# ------------------
#
# 📁 config.R             - Configuration and file reading functions
#    • default_config()
#    • read_fmri_config()
#
# 📁 dataset_constructors.R - Dataset creation functions
#    • matrix_dataset()
#    • fmri_mem_dataset()
#    • latent_dataset()
#    • fmri_dataset()
#
# 📁 data_access.R         - Data access and mask methods
#    • get_data.* methods
#    • get_data_matrix.* methods
#    • get_mask.* methods
#    • get_data_from_file()
#    • blocklens.* methods
#
# 📁 data_chunks.R         - Data chunking and iteration
#    • data_chunk()
#    • chunk_iter()
#    • data_chunks.* methods
#    • exec_strategy()
#    • collect_chunks()
#    • arbitrary_chunks()
#    • slicewise_chunks()
#    • one_chunk()
#
# 📁 print_methods.R       - Print and display methods
#    • print.fmri_dataset()
#    • print.latent_dataset()
#    • print.chunkiter()
#    • print.data_chunk()
#
# 📁 conversions.R         - Type conversion methods
#    • as.matrix_dataset()
#    • as.matrix_dataset.* methods
#
# ========================================================================

# Essential imports and operators that are used across multiple files
`%dopar%` <- foreach::`%dopar%`
`%do%` <- foreach::`%do%`

# ========================================================================
# Package-level documentation and imports
# ========================================================================
#
# This refactoring improves:
# 1. Code organization and readability
# 2. Maintainability - easier to find and modify specific functionality
# 3. Testing - can test individual modules in isolation
# 4. Development - multiple developers can work on different aspects
# 5. Documentation - clearer separation of concerns
#
# All original functionality is preserved - only the organization changed.
# ========================================================================
