# mri-longitudinal-segmentation

## To Do's
- [ ] Remember postprocessing being diabled in run.py
- [ ] Adjust preprocessing, e.g. preprocessed in name shouldnt be processed, labels
- [ ] Check the whole BF correction properly as well the order, currently done in the registration
- [x] Check the name suffix, currently always set to _reg.nii.gz
- [x] Check line 40 of hd_bet.py
- [ ] Add instruction for HD-BET installation, considering adjustments and Aydans installment, e.g. branch
- [ ] Add instruction for nnUnet installation
- [ ] WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run consolidate_folds in the output folder of the model first! The folder you need to run this in is /mnt/kannlab_rfa/JuanCarlos/nnUnet_trained_models/nnUNet/3d_fullres/Task871_BrainTumourT2PedsPreTrainedEncoderFrozen/nnUNetTrainerV2__nnUNetPlansv2.1
- [ ] Check the segmentation quality in slicer