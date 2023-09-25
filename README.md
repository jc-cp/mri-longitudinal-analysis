# Mri Longitudinal Analysis



## To Do's
- [ ] Check the whole BF correction properly as well the order, currently done in the registration
- [ ] Add instruction for HD-BET installation, considering adjustments and Aydans installment, e.g. branch
- [ ] Add instruction for nnUnet installation
- [ ] Check the segmentation quality in slicer

### _cfg/_ folder
- [ ] Open Todo

### _HDBET_Code/_ folder
- [ ] Open Todo

### _nnUNet/_ folder
- [ ] Open Todo

### _utils/_ folder
This folder contains several files with different purposes, from simple file checking utilities to evaluation or filtering scripts for the provided data. Below a list of the files and their corresponding utilities.

   <details>
   <summary>Files</summary>

   * **check_files.py**: Script that checks the completeness, faultiness between two directories to see if there is some misalignement in the number/ quality of data. 
   * **evaluation_t2w_files.py**: Script that reads in the annotations performed after the initial review by the user of the T2 sequences and outputs some basic data + an histogram as initial statistic.
   * **filter_clinical_data.py**: Script that reads in the clinical data extracted from the hospital containing the cohort of 89 patients (60 with no operatios + 29 with later surgery). 
   * **review_t2w.py**: Reviewing script for the data pipeline. Each of the flags and paths should be adjusted depeding on the stage of the review process. E.g., if the flag of MOVING2REVIEW is activated, the reviewed files are moved to subfolders for a second review by a trained radiologist. 

   </details>


### How to Run
