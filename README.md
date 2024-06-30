# MRI pLGG Longitudinal Analysis
![Analysis Pipeline Overview](https://github.com/jc-cp/mri-longitudinal-analysis/assets/104212632/1f94122d-2dc9-4a1e-9f19-d1b2cf2134c2)


## Project Goals
Pediatric low-grade gliomas (pLGGs) have heterogeneous clinical presentations and prognoses.
Given the morbidity of treatment, some suspected pLGGs, especially those found incidentally,
are surveilled without treatment, though the natural histories of these tumors have yet
to be systematically studied. We leveraged deep learning and multi-institutional data to
methodically analyze longitudinal volumetric trajectories of pLGGs on surveillance, yielding
insights into their growth trajectories and clinical implications. The objective is two-fold:
first, to meticulously track and categorize the volumetric changes of pLGGs over time, and
second, to identify any potential risk factors associated with significant volumetric changes
that might need medical intervention.

## Project Structure

The project is made up of two main steps, a data preparation one and a data analysis one. **Data preparation** consists of cohort selection, patients classification (package [here](https://github.com/jc-cp/mri-sequence-classification)), MRI imaging pre-processing (package [here](https://github.com/jc-cp/mri_preprocessing)), a segmentation algorithm based on a nnU-Net architecture (package [here](https://github.com/BoydAidan/nnUNet_pLGG)) and the first block of the analysis, a _volumetric analysis_. In this analysis 3D segmentation extracted by the previous countouring algorithms are plotted over time yield insights into the trajectory of the individual patients and the overall cohort. In the **data analysis** step we perform three distinct analysis, a _tumor classification analysis_, where clinical risk factors for progression defined by a curve increase are obtained, a _survival anaylsis_ based on a Kaplan Meier and a Cox-Hazards model and a _prediction analysis_, where we conduct a volumetric prediction of the tumors based on the historical data provided by the curves as well  as progression prediction based on random forests and mixed-effects models. 


## Folder: _lib/_  
This folder contains all external libraries that need to be installed separately in order to make use of all functuionalitites in the repository. They are listed below.


### _HDBET_Code/_ 
Package needed for the [pre-processing](https://github.com/jc-cp/mri_preprocessing) of MRI sequences. The preprocessing is done using the package provided in the previous link or using the file mri_preprocess_3d.py in source. To use this program, you need to modify the code to enter the path to the scans (T2W_dir) and where you want to save the output (output_path). Use the respective configuration files provided in the _cfg/_ folder. It is assumed that your scans and ground truth masks are in the same folder and that it if a scan name is .nii.gz, the corresponding mask is named _mask.nii.gz. If you dont have ground truth segmentations, comment out all the lines regarding them. The output should be fully preprocessed scans in the specified <output_path>. These scans will be saved with the nnunet naming convention (_0000.nii.gz for the scan). Masks will be located in the <output_path> specified for the model (name will be .nii.gz for the mask).

This step may take a couple of hours to run depending on how many scans are in your folder.

### _nnUNet/_ 
Needed for the segmentation. The pLGGs were segmented using a T2W segmentation tool. The package and further instructions are [here](https://github.com/BoydAidan/nnUNet_pLGG).

## Folder: _mri longitudinal analysis/_ 
This is the main developed package which can also be installed by following the instructions:
````
git clone -b plgg_analysis https://github.com/jc-cp/mri-longitudinal-analysis
cd mri-longitudinal-analysis/mri_longitudinal_analysis
pip install -e .
`````
It ontains two main folders with the auxiliary files located in _utils/_, the main analysis files in _src/_ and the repective configuration files where all of the specific paths and variable are defined.

### _cfg/_ 
- [ ] Do list of all cfg files.

### _src/_ 
- [ ] Finish the list of src files.

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
If you want to run the whole pipeline in one go please follow the instructions below:

- [ ] TODO

## To Do's
- [ ] Add instruction for HD-BET installation, considering adjustments and segmentation installment, e.g. branch
- [ ] Add instruction for nnUnet installation
- [ ] Rethink the interpolation in volume estimation with splines and statistical imputation
- [ ]
- [ ]
