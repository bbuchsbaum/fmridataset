# fmri_dataset print snapshots

    Code
      print(dset_basic)
    Output
      
      === fMRI Dataset ===
      
      ** Dimensions:
        - Timepoints: 50 
        - Runs: 2  
        - Matrix: 50 x 100 (timepoints x voxels)
        - Voxels in mask: (lazy)
      
      ** Temporal Structure:
        - TR: 2 seconds
        - Run lengths: 25, 25 
      
      ** Event Table:
        - Empty event table
      

---

    Code
      print(dset_multi)
    Output
      
      === fMRI Dataset ===
      
      ** Dimensions:
        - Timepoints: 50 
        - Runs: 2  
        - Matrix: 50 x 100 (timepoints x voxels)
        - Voxels in mask: (lazy)
      
      ** Temporal Structure:
        - TR: 2 seconds
        - Run lengths: 25, 25 
      
      ** Event Table:
        - Empty event table
      

---

    Code
      print(dset_masked)
    Output
      
      === fMRI Dataset ===
      
      ** Dimensions:
        - Timepoints: 50 
        - Runs: 2  
        - Matrix: 50 x 100 (timepoints x voxels)
        - Voxels in mask: (lazy)
      
      ** Temporal Structure:
        - TR: 2 seconds
        - Run lengths: 25, 25 
      
      ** Event Table:
        - Empty event table
      

# backend print snapshots

    Code
      str(backend_mat, max.level = 1, give.attr = FALSE)
    Output
      List of 4
       $ data_matrix : num [1:50, 1:100] 1.371 -0.565 0.363 0.633 0.404 ...
       $ mask        : logi [1:100] TRUE TRUE TRUE TRUE TRUE TRUE ...
       $ spatial_dims: num [1:3] 100 1 1
       $ metadata    :List of 2

---

    Code
      str(backend_multi, max.level = 1, give.attr = FALSE)
    Output
      List of 4
       $ data_matrix : num [1:50, 1:100] 0.0712 0.9703 0.31 -0.1395 -0.3263 ...
       $ mask        : logi [1:100] TRUE TRUE TRUE TRUE TRUE TRUE ...
       $ spatial_dims: num [1:3] 100 1 1
       $ metadata    :List of 2

# sampling_frame print snapshots

    Code
      print(sframe_single)
    Output
      Sampling Frame
      ==============
      
      Structure:
        1 block
        Total scans: 100
      
      Timing:
        TR: 2 s
        Precision: 0.1 s
      
      Duration:
        Total time: 200.0 s

---

    Code
      print(sframe_multi)
    Output
      Sampling Frame
      ==============
      
      Structure:
        3 blocks
        Total scans: 370
      
      Timing:
        TR: 2.5 s
        Precision: 0.1 s
      
      Duration:
        Total time: 925.0 s

---

    Code
      list(frame = sframe_single, events = events)
    Output
      $frame
      Sampling Frame
      ==============
      
      Structure:
        1 block
        Total scans: 100
      
      Timing:
        TR: 2 s
        Precision: 0.1 s
      
      Duration:
        Total time: 200.0 s
      
      $events
         onset duration condition
      1      0        5         A
      2     10        5         B
      3     20        5         A
      4     30        5         B
      5     40        5         A
      6     50        5         B
      7     60        5         A
      8     70        5         B
      9     80        5         A
      10    90        5         B
      

# error message snapshots

    sum(run_length) not equal to nrow(datamat)

---

    TR values must be positive

---

    TR values must be positive

---

    Block lengths must be positive

---

    Precision must be positive and less than the minimum TR

# summary output snapshots

    Code
      summary(dset)
    Output
      
      === fMRI Dataset Summary ===
      
      ** Dimensions:
        - Timepoints: 50 
        - Runs: 2 
        - Matrix: 50 x 100 (timepoints x voxels)
        - Voxels in mask: (lazy)
      
      ** Temporal Structure:
        - TR: 2 seconds
        - Run lengths: 25, 25 
      
      ** Event Summary:
        - No events
      

# data chunk iterator snapshots

    Code
      print(chunk_info)
    Output
      [[1]]
      [[1]]$chunk_number
      [1] 1
      
      [[1]]$dimensions
      [1] 50 25
      
      [[1]]$first_value
      [1] 1.370958
      
      [[1]]$last_value
      [1] 0.03240079
      
      
      [[2]]
      [[2]]$chunk_number
      [1] 2
      
      [[2]]$dimensions
      [1] 50 25
      
      [[2]]$first_value
      [1] -1.551583
      
      [[2]]$last_value
      [1] 0.4985434
      
      
      [[3]]
      [[3]]$chunk_number
      [1] 3
      
      [[3]]$dimensions
      [1] 50 25
      
      [[3]]$first_value
      [1] 0.6173367
      
      [[3]]$last_value
      [1] -0.0170699
      
      

