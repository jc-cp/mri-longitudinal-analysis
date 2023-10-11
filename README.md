# Mri Longitudinal Analysis

## Project Goals

## Project Structure


## Folder: _lib/_  
This folder contains all external libraries that need to be installed separately in order to make use of all functuionalitites in the repository. They are listed below.


### _HDBET_Code/_ 
- [ ] Open Todo

### _nnUNet/_ 
- [ ] Open Todo

## Folder: _mri longitudinal analysis/_ 
This is the main developed package which can also be installed by following the instructions:
- [ ] Open Todo

### _cfg/_ 
- [ ] Open Todo

### _src/_ 
- [ ] Open Todo

   <details>
   <summary>Scripts</summary>

   * **filter_clinical_data.py**: Script that reads in the clinical data extracted from the hospital containing the cohort of 89 patients (60 with no operatios + 29 with later surgery).

   </details>


### _templates/_ 
In this folder several template files are contained that are used thoughtout the package. An example is a standard brain MRI that can be used for registration purposes. 

### _utils/_ 
This folder contains several files with different purposes, from simple file checking utilities to evaluation or filtering scripts for the provided data. Below a list of the files and their corresponding utilities.

   <details>
   <summary>Scripts</summary>

   * **check_files.py**: Script that checks the completeness and faultiness between files located in two directories in order to see if there are some misalignements in the number/ quality of data. 
   * **evaluation_t2w_files.py**: Script that reads in the annotations performed after the initial review by the user of the T2 sequences and outputs some basic data + an histogram as initial statistic.
   * **helper_functions.py**: OPEN TODO 
   * **review_t2w.py**: Reviewing script for the data pipeline. Each of the flags and paths should be adjusted depeding on the stage of the review process. E.g., if the flag of MOVING2REVIEW is activated, the reviewed files are moved to subfolders for a second review by a trained radiologist. 

   </details>


## How to Run



## To Do's
- [ ] Check the whole BF correction properly as well the order, currently done in the registration
- [ ] Add instruction for HD-BET installation, considering adjustments and Aydans installment, e.g. branch
- [ ] Add instruction for nnUnet installation
- [ ] Check the segmentation quality in slicer
- [ ] Rethink the interpolation in volume estimation with splines and statistical imputation