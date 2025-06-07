#' Legacy fMRI Dataset Implementation
#'
#' @description
#' This file contains the original fmri_file_dataset implementation,
#' preserved for backwards compatibility and testing during the transition
#' to the new backend architecture.
#'
#' @name fmri_dataset_legacy
#' @keywords internal
NULL

#' @export
#' @keywords internal
fmri_dataset_legacy <- function(scans, mask, TR, 
                         run_length, 
                         event_table=data.frame(), 
                         base_path=".",
                         censor=NULL,
                         preload=FALSE,
                         mode=c("normal", "bigvec", "mmap", "filebacked")) {
  
  assert_that(is.character(mask), msg="'mask' should be the file name of the binary mask file")
  mode <- match.arg(mode)
  
  if (is.null(censor)) {
    censor <- rep(0, sum(run_length))
  }
  
  frame <- sampling_frame(run_length, TR)
  
  maskfile <- paste0(base_path, "/", mask)
  scans=paste0(base_path, "/", scans)

  maskvol <- if (preload) {
    assert_that(file.exists(maskfile))
    message(paste("preloading masks", maskfile))
    neuroim2::read_vol(maskfile)
  }
  
  vec <- if (preload) {
    message(paste("preloading scans", paste(scans, collapse = " ")))
    neuroim2::read_vec(scans, mode=mode,mask=maskvol)
  }
  
  
  ret <- list(
    scans=scans,
    vec=vec,
    mask_file=maskfile,
    mask=maskvol,
    nruns=length(run_length),
    event_table=suppressMessages(tibble::as_tibble(event_table,.name_repair="check_unique")),
    base_path=base_path,
    sampling_frame=frame,
    censor=censor,
    mode=mode,
    preload=preload
  )
  
  class(ret) <- c("fmri_file_dataset", "volumetric_dataset", "fmri_dataset", "list")
  ret
}