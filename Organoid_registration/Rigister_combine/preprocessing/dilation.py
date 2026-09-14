import SimpleITK as sitk


MASK_IN = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/case2_FA_mask_LAS.nii.gz"

MASK_OUT = "/autofs/arch11/DATA/HOMES/yuchen/Minor/Organoids/Nifti_data/case2_young/data_final/case2_FA_mask_LAS_dilated_10vox.nii.gz"

mask = sitk.ReadImage(MASK_IN, sitk.sitkUInt8)

mask = sitk.BinaryThreshold(
    mask,
    lowerThreshold=1,
    upperThreshold=255,
    insideValue=1,
    outsideValue=0,
)

# dilation 2 voxels
mask_dilated = sitk.BinaryDilate(
    mask,
    [10, 10, 10],
    sitk.sitkBall,
    1,
)

sitk.WriteImage(mask_dilated, MASK_OUT)