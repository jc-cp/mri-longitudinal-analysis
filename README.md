# mri-longitudinal-segmentation



## To Do's
- [ ] Check the whole BF correction properly as well the order, currently done in the registration
- [ ] Add instruction for HD-BET installation, considering adjustments and Aydans installment, e.g. branch
- [ ] Add instruction for nnUnet installation
- [ ] WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run consolidate_folds in the output folder of the model first! The folder you need to run this in is /mnt/kannlab_rfa/JuanCarlos/nnUnet_trained_models/nnUNet/3d_fullres/Task871_BrainTumourT2PedsPreTrainedEncoderFrozen/nnUNetTrainerV2__nnUNetPlansv2.1
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

   * **check_files.py**: Script that checks the completeness, faultiness between two directories to see if there is some misalignement in the number/ quality of data 
   * **evaluation_t2w_files.py**: 
   * **filter_clinical_data.py**: 

   </details>


### How to Run