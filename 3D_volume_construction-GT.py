""" Script to built 3D lesion segmentation mask generate from a 2D model including GT axial slices"""

## 3D nnUNET Laucher
## Step-1: nnUNet_plan_and_preprocess -t 115 --verify_dataset_integrity
## Step-2: nnUNet_train 3d_fullres nnUNetTrainerV2 115 0

## 3D nnUNET Laucher
## Step-1: nnUNet_plan_and_preprocess -t 115 -pl3d None
## Step-2: nnUNet_train 2d nnUNetTrainerV2 115 0


import os
import pandas as pd
from nnunet.dataset_conversion import utils

import glob
import nibabel as nib
import SimpleITK as sitk

from skimage import io, transform
import numpy as np


def read_nifti(filepath):
    '''' Reads .nii file and returns pixel array '''
    ct_scan = nib.load(filepath)
    img_array   = ct_scan.get_fdata()
    img_affine  = ct_scan.affine
    # array   = np.rot90(np.array(array))
    return (img_array, img_affine)


#####################################################################
## Dataset path definitions
nifti_folder = "/data/01_UB/2021-MedNeurIPS/train_Nifti-Data/"
lesion_folder = "/data/01_UB/2021-MedNeurIPS/train_Nifti-Seg-6-Classes/"
# nifti_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-Data/"
# lesion_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-Seg-6-Classes/"

metadata_file_path = "/data/01_UB/2021-MedNeurIPS/111_dataframe_axial_slices.csv"
axial_metadata_file_path = "/data/01_UB/2021-MedNeurIPS/111_dataframe_axial_slices.csv"

axial_lesion_folder = "/data/01_UB/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_COVIDSegChallenge/AAxial-labelsTr/"
# axial_lesion_folder = "/data/01_UB/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_COVIDSegChallenge/AAxial-labelsTs/"


#####################################################################
## Read the metadata
metadata_full = pd.read_csv(metadata_file_path, sep=',')
print("++++++++++++++++++++++++++++++++++++++")
# print("metadata: ", metadata_full.head())
print("++ metadata: ", metadata_full.shape)

## Using separate folder for training and test
metadata = metadata_full.query('split == "train"')
# metadata = metadata_full.query('split == "test"')
metadata = metadata.reset_index(drop=True)
# print("++ metadata:", metadata.head())
print("++ Metadata Train:", metadata.shape)
print("++++++++++++++++++++++++++++++++++++++")


#####################################################################
## Read the axial_metadata
axial_metadata = metadata

#####################################################################
## nn-UNet sandbox paths
dataset_folder = '/data/01_UB/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_COVIDSegChallenge/'

imagesTr_shape_path = os.path.join(dataset_folder, "3D-imagesTr-GT")
labelsTr_shape_path = os.path.join(dataset_folder, "3D-labelsTr-GT")


## Loop through dataset
for row in range(metadata.shape[0]):

    print('+ row: ', row)
    print('+ ct_file_name: ', metadata['ct_file_name'][row])

    ## locating the CT and Seg
    ct_scan_input_path = os.path.join(nifti_folder, metadata['ct_file_name'][row])
    lesion_scan_input_path = os.path.join(lesion_folder, metadata['lesion_file_name'][row])

    # print('+ ct_scan_input_path', ct_scan_input_path)
    # print('+ lesion_scan_input_path', lesion_scan_input_path)

    ## Reads .nii file and get the numpy image array
    ct = nib.load(ct_scan_input_path)
    ct_scan_array = ct.get_data()
    # print('+ ct_scan_array', ct.header)

    lesion = nib.load(lesion_scan_input_path)
    lesion_scan_array = lesion.get_data()
    print("lesion_scan_array: ", lesion_scan_array.shape)


    #####################################################################
    #####################################################################
    ## New
    new_3D_lesion_array = np.zeros_like(lesion_scan_array)
    print("new_3D_lesion_array: ", new_3D_lesion_array.shape)

    for slice in range(lesion_scan_array.shape[2]):

        slice_position = slice
        axial_lesion_filename, _ = str(metadata['lesion_file_name'][row]).split('-bilung.nii.gz')
        # print("axial_lesion_filename: ", axial_lesion_filename)


        ## Set file path of the 2D axial lesion
        axial_lesion_scan_input_path = os.path.join(axial_lesion_folder, str(axial_lesion_filename) + '-' + str(slice_position) + '.nii.gz')
        # print('axial_lesion_scan_input_path: ', axial_lesion_scan_input_path)

        ## Reads .nii file and get the numpy image array
        axial_lesion = nib.load(axial_lesion_scan_input_path)
        axial_lesion_array = axial_lesion.get_data()
        axial_lesion_affine = axial_lesion.affine

        # print('+ axial_lesion_array', axial_lesion.header)
        # print('+ axial_lesion_array', axial_lesion.shape)

        axial_lesion_reshape = np.reshape(axial_lesion_array, (512, 512))




        ##############################################################################
        ##############################################################################
        ## Added GT

        ## Get the GT axial slice positions
        loc_id_case = metadata['id_case'][row]
        # axial_loc = axial_metadata[axial_metadata['id_case'] == 'B0018_01_200410_CT_SK']
        gt_loc = axial_metadata[axial_metadata['id_case'] == loc_id_case]
        # print('axial_loc', axial_loc)

        ## Get the ground truht
        gt_slices = gt_loc['slice_position'].values
        gt_slices = gt_slices -1
        # print('axial_loc', gt_loc['slice_position'].values)
        # print('gt_slices', gt_slices)

        gt_lesions = gt_loc['slice_with_lesion'].values
        # print(gt_lesions)

        if slice in gt_slices:
            # print('slice', slice)
            gt_lesion_array = lesion_scan_array[:, :, slice]

            ## Adding slices
            new_3D_lesion_array[:, :, slice_position] = gt_lesion_array

        else:
            ## Adding slices
            # new_3D_lesion_array[slice_position] = np.append(new_3D_lesion_array, axial_lesion_reshape)
            new_3D_lesion_array[:, :, slice_position] = axial_lesion_reshape


        ## Added GT
        ##############################################################################
        ##############################################################################

    print("new_3D_lesion_array: ", new_3D_lesion_array.shape)
    # new_3D_lesion_array_reshape = new_3D_lesion_array.reshape((512, 512, lesion_scan_array.shape[2]))
    # print("new_3D_lesion_array_reshape: ", new_3D_lesion_array_reshape.shape)

    new_3D_lesion_array_reshape = new_3D_lesion_array

    ## Generate the 3D lesions images
    new_3D_lesion_nifti = nib.Nifti1Image(new_3D_lesion_array_reshape, lesion.affine)

    ## Set new name
    new_3D_lesion_filename = str(axial_lesion_filename) + '-3Dlesions.nii.gz'
    ct_slice_output_path = os.path.join(labelsTr_shape_path, new_3D_lesion_filename)

    ## and write the nifti files
    nib.save(new_3D_lesion_nifti, ct_slice_output_path)
