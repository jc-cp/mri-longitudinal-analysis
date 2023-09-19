import glob
import os
import random
import sys

import SimpleITK as sitk
from tqdm import tqdm

sys.path.append("./HDBET_Code/")
from HD_BET.hd_bet import hd_bet

from cfg.preprocess_cfg import (
    BF_CORRECTION,
    BF_CORRECTION_DIR,
    BRAIN_EXTRACTION_DIR,
    EXTRACTION,
    INPUT_DIR,
    LIMIT_LOADING,
    OUPUT_DIR,
    REG_DIR,
    REGISTRATION,
    SEG_PRED_DIR,
    TEMP_IMG,
)


def bf_correction(input_dir, output_dir):
    """
    Bias field correction with SimpleITK
    Args:
        input_dir {path} -- input directory
        output_dir {path} -- output directory
    Returns:
        Images in nii.gz format
    """

    for img_path in sorted(glob.glob(input_dir + "/*.nii.gz")):
        ID = img_path.split("/")[-1].split(".")[0]
        if ID[-1] == "k":
            continue
        else:
            print(ID)
            img = sitk.ReadImage(img_path, sitk.sitkFloat32)
            img = sitk.N4BiasFieldCorrection(img)
            ID = img_path.split("/")[-1].split(".")[0]
            fn = ID + "_bf_corrected.nii.gz"
            sitk.WriteImage(img, os.path.join(output_dir, fn))
    print("bias field correction complete!")


def brain_extraction(input_dir, output_dir):
    """
    Brain extraction using HDBET package (UNet based DL method)
    Args:
        T2W_dir {path} -- input dir;
        brain_dir {path} -- output dir;
    Returns:
        Brain images
    """
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    hd_bet(input_dir, output_dir, device="0", mode="fast", tta=0)
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
    MRI registration with SimpleITK
    Args:
        temp_img {str} -- registration image template
        output_dir {path} -- Path to folder where the registered nrrds will be saved.
    Returns:
        The sitk image object -- nii.gz
    Raises:
        Exception if an error occurs.
    """
    print("Registering test data...")

    fixed_img = sitk.ReadImage(temp_img, sitk.sitkFloat32)
    problematic_IDs = []

    print("Preloading step...")

    for index, img_path in enumerate(tqdm(input_data)):
        try:
            _ = sitk.ReadImage(img_path, sitk.sitkFloat32)
        except Exception as e:
            problematic_ID = os.path.splitext(img_path)[0]
            problematic_IDs.append(problematic_ID)
            print(f"Could not preload image {problematic_ID}. Error: {e}")

    print("Problematic IDs: ", problematic_IDs)

    random.shuffle(input_data)

    for index, img_path in enumerate(tqdm(input_data)):
        ID = os.path.splitext(img_path)[0]
        if ID in problematic_IDs:
            print("problematic data!")
            continue

        print(index)
        print(ID)
        try:
            moving_img = sitk.ReadImage(img_path, sitk.sitkFloat32)
            # print('moving image:', moving_image.shape)

            # bias filed correction
            moving_img = sitk.N4BiasFieldCorrection(moving_img)

            # respace fixed img on z-direction
            z_spacing = moving_img.GetSpacing()[2]
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
            # new_size = [old_size[0], old_size[1], int(round((old_size[2] * 1) / float(z_spacing)))]
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
            registration_method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50
            )
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
            output_path = os.path.join(output_dir, new_filename)
            sitk.WriteImage(moving_img_resampled, output_path)

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
                os.path.join(nnunet_dir, "labelsTs", f"{ID}" + ".nii.gz"),
            )

            if save_tfm:
                sitk.WriteTransform(
                    final_transform, os.path.join(output_dir, f"{ID}" + "_T2.tfm")
                )
        except Exception as e:
            print(f"Error with image {ID}: {e}")
    count = index + 1
    print("Registered", count, "scans.")


def get_image_files(base_dir):
    image_files = []
    for file in os.listdir(base_dir):
        full_path = os.path.join(base_dir, file)
        if (
            os.path.isfile(full_path)
            and file.endswith(".nii.gz")
            and "label" not in file
        ):
            image_files.append(full_path)
            # Break when limit is reached
            if len(image_files) >= LIMIT_LOADING:
                return image_files
    return image_files


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    register = REGISTRATION
    extraction = EXTRACTION
    bias_field_correction = BF_CORRECTION

    input_data_dir = INPUT_DIR
    output_path = OUPUT_DIR

    reg_dir = REG_DIR
    brain_dir = BRAIN_EXTRACTION_DIR
    bf_correction_dir = BF_CORRECTION_DIR
    segmentation_output_dir = SEG_PRED_DIR

    image_files = get_image_files(input_data_dir)

    # create dirs
    os.makedirs(reg_dir, exist_ok=True)
    os.makedirs(brain_dir, exist_ok=True)
    os.makedirs(bf_correction_dir, exist_ok=True)
    os.makedirs(segmentation_output_dir, exist_ok=True)

    if register:
        registration(
            input_data=image_files,
            output_dir=reg_dir,
            nnunet_dir=segmentation_output_dir,
            temp_img=TEMP_IMG,
        )

    if extraction:
        brain_extraction(input_dir=reg_dir, output_dir=brain_dir)

    if bias_field_correction:
        bf_correction(input_dir=brain_dir, output_dir=bf_correction_dir)
