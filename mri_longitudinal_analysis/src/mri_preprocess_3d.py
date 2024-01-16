"""
This script provides utilities to preprocess MRI data. 
It includes functions for bias field correction, brain extraction, and registration. 
"""
import glob
import os
import random
import sys

import SimpleITK as sitk
from cfg import preprocess_cfg
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from lib.HDBET_Code.HD_BET.hd_bet import hd_bet


def bf_correction(input_dir, output_dir):
    """
    Perform bias field correction on MRI data using SimpleITK.

    Args:
        input_dir (str): Path to the input directory containing MRI scans.
        output_dir (str): Path to the output directory where corrected scans will be saved.

    Returns:
        None. The corrected MRI scans are saved in the specified output directory.
    """

    for img_path in sorted(glob.glob(input_dir + "/*.nii.gz")):
        id_ = img_path.split("/")[-1].split(".")[0]
        if id_[-1] == "k":
            continue
        print(id_)
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        img = sitk.N4BiasFieldCorrection(img)
        id_ = img_path.split("/")[-1].split(".")[0]
        filename = id_ + "_bf_corrected.nii.gz"
        sitk.WriteImage(img, os.path.join(output_dir, filename))
    print("bias field correction complete!")


def brain_extraction(input_dir, output_dir):
    """
    Extract brain region from MRI data using the HDBET package.

    Args:
        input_dir (str): Path to the directory containing MRI scans.
        output_dir (str): Path to the output directory where brain-extracted images will be saved.

    Returns:
        None. The brain-extracted MRI scans are saved in the specified output directory.
    """
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    hd_bet(input_dir, output_dir, device="0,1", mode="fast", tta=0)
    print("Brain Extraction with HD-BET complete!")


def registration(
    input_data,
    output_dir,
    nnunet_dir,
    temp_img,
    interp_type="linear",
    save_tfm=True,
):
    """
    Register MRI scans to a template using SimpleITK.

    Args:
        input_data (list): List of paths to MRI scans.
        output_dir (str): Directory to save the registered scans.
        nnunet_dir (str): Directory for nnUNet segmentation outputs.
        temp_img (str): Path to the template image used for registration.
        interp_type (str, optional): Interpolation type used for registration. Defaults to 'linear'.
        save_tfm (bool, optional): If True, transformation files are saved. Defaults to True.

    Returns:
        Registered MRI scans and transformations are saved in the specified directories.
    """
    print("Registering test data...")

    fixed_img = sitk.ReadImage(temp_img, sitk.sitkFloat32)
    problematic_ids = []

    print("Preloading step...")

    for index, img_path in enumerate(tqdm(input_data)):
        try:
            _ = sitk.ReadImage(img_path, sitk.sitkFloat32)
        except IOError as error:
            problematic_id_ = os.path.splitext(img_path)[0]
            problematic_ids.append(problematic_id_)
            print(f"Could not preload image {problematic_id_}. Error: {error}")

    print("Problematic IDs: ", problematic_ids)

    random.shuffle(input_data)

    for index, img_path in enumerate(tqdm(input_data)):
        id_ = os.path.splitext(img_path)[0]
        if id_ in problematic_ids:
            print("problematic data!")
            continue

        print(index)
        print(id_)
        try:
            moving_img = sitk.ReadImage(img_path, sitk.sitkFloat32)
            # print('moving image:', moving_image.shape)

            # bias filed correction
            moving_img = sitk.N4BiasFieldCorrection(moving_img)

            # respace fixed img on z-direction
            # z_spacing = moving_img.GetSpacing()[2]
            old_size = fixed_img.GetSize()
            old_spacing = fixed_img.GetSpacing()
            new_spacing = (
                1,
                1,
                1,
            )  # CHANGED FROM ORIGINAL WHERE Z_SPACING WAS MAINTAINED
            new_size = [
                int(round((old_size[0] * old_spacing[0]) / float(new_spacing[0]))),
                int(round((old_size[1] * old_spacing[1]) / float(new_spacing[1]))),
                int(round((old_size[2] * old_spacing[2]) / float(new_spacing[2]))),
            ]
            # new_size =
            # [old_size[0], old_size[1], int(round((old_size[2] * 1) / float(z_spacing)))]
            # new_size = [old_size[0], old_size[1], old_size[2]]
            if interp_type == "linear":
                interp_type = sitk.sitkLinear
            elif interp_type == "bspline":
                interp_type = sitk.sitkBSpline
            elif interp_type == "nearest_neighbor":
                interp_type = sitk.sitkNearestNeighbor
            resample = sitk.ResampleImageFilter()
            resample.SetOutputSpacing(new_spacing)
            resample.SetSize(new_size)
            resample.SetOutputOrigin(fixed_img.GetOrigin())
            resample.SetOutputDirection(fixed_img.GetDirection())
            resample.SetInterpolator(interp_type)
            resample.SetDefaultPixelValue(fixed_img.GetPixelIDValue())
            resample.SetOutputPixelType(sitk.sitkFloat32)
            fixed_img = resample.Execute(fixed_img)
            # print(fixed_img.shape)
            transform = sitk.CenteredTransformInitializer(
                fixed_img,
                moving_img,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            # multi-resolution rigid registration using Mutual Information
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.01)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=1.0,
                numberOfIterations=100,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10,
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            registration_method.SetInitialTransform(transform)
            final_transform = registration_method.Execute(fixed_img, moving_img)

            ## WRITE MODIFIED SCAN
            moving_img_resampled = sitk.Resample(
                moving_img,
                fixed_img,
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_img.GetPixelID(),
            )

            # get the file names
            filename = os.path.basename(img_path)  # Get filename from the full path
            filename_parts = os.path.splitext(filename)[0].split(
                "_"
            )  # Split filename without extension on underscores
            patient_id = filename_parts[0]
            scan_id = filename_parts[1]
            new_filename = f"{patient_id}_{scan_id}_0000.nii.gz"
            out_path = os.path.join(output_dir, new_filename)
            sitk.WriteImage(moving_img_resampled, out_path)

            segmentation_loc = img_path.replace(".nii.gz", "_label.nii.gz")
            if not os.path.isfile(segmentation_loc):
                print(f"No corresponding label file for {img_path}")
                continue

            moving_label = sitk.ReadImage(segmentation_loc, sitk.sitkFloat32)
            moving_label_resampled = sitk.Resample(
                moving_label,
                fixed_img,
                final_transform,
                sitk.sitkNearestNeighbor,
                0.0,
                moving_img.GetPixelID(),
            )
            if not os.path.exists(os.path.join(nnunet_dir, "labelsTs")):
                os.makedirs(os.path.join(nnunet_dir, "labelsTs"))
            sitk.WriteImage(
                moving_label_resampled,
                os.path.join(nnunet_dir, "labelsTs", f"{id_}" + ".nii.gz"),
            )

            if save_tfm:
                sitk.WriteTransform(final_transform, os.path.join(output_dir, f"{id_}" + "_T2.tfm"))
        except IOError as error:
            print(f"Error with image {id_}: {error}")
    count = index + 1
    print("Registered", count, "scans.")


def get_image_files(base_dir):
    """
    Retrieve MRI scan file paths from a given directory.

    Args:
        base_dir (str): The directory from which MRI scan file paths are to be retrieved.

    Returns:
        list: A list of MRI scan file paths.
    """
    image_files_ = []
    for file in os.listdir(base_dir):
        full_path = os.path.join(base_dir, file)
        if os.path.isfile(full_path) and file.endswith(".nii.gz") and "label" not in file:
            image_files_.append(full_path)
            # Break when limit is reached
            if len(image_files_) >= preprocess_cfg.LIMIT_LOADING:
                return image_files_
    return image_files_


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # conditions
    REGISTRATION = preprocess_cfg.REGISTRATION
    EXTRACTION = preprocess_cfg.EXTRACTION
    BF_CORRECTION = preprocess_cfg.BF_CORRECTION

    input_data_dir = preprocess_cfg.INPUT_DIR
    output_path = preprocess_cfg.OUPUT_DIR

    reg_dir = preprocess_cfg.REG_DIR
    brain_dir = preprocess_cfg.BRAIN_EXTRACTION_DIR
    bf_correction_dir = preprocess_cfg.BF_CORRECTION_DIR
    segmentation_output_dir = preprocess_cfg.SEG_PRED_DIR

    image_files = get_image_files(input_data_dir)

    # create dirs
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(reg_dir, exist_ok=True)
    os.makedirs(brain_dir, exist_ok=True)
    # os.makedirs(bf_correction_dir, exist_ok=True)
    os.makedirs(segmentation_output_dir, exist_ok=True)

    if REGISTRATION:
        registration(
            input_data=image_files,
            output_dir=reg_dir,
            nnunet_dir=segmentation_output_dir,
            temp_img=preprocess_cfg.TEMP_IMG,
        )

    if EXTRACTION:
        brain_extraction(input_dir=reg_dir, output_dir=brain_dir)

    if BF_CORRECTION:
        bf_correction(input_dir=brain_dir, output_dir=bf_correction_dir)
