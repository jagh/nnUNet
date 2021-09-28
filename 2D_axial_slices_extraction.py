""" Script to extract all axial slices from a 3D CT scan.
    The axial slices are transformed from 2D shape (X, Y) to (1, X, Y).
    Then, is used nn-UNet to generate the 2D lesion segmentation for each axial slice.
"""

## 3D nnUNET Laucher
## Step-1: nnUNet_plan_and_preprocess -t 115 --verify_dataset_integrity
## Step-2: nnUNet_train 3d_fullres nnUNetTrainerV2 115 0

## 3D nnUNET Laucher
## Step-1: nnUNet_plan_and_preprocess -t 115 -pl3d None
## Step-2: nnUNet_train 2d nnUNetTrainerV2 115 0

## nohup nnUNet_predict -i /data/01_UB/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_COVIDSegChallenge/AAxial-imagesTs -o /data/01_UB/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_COVIDSegChallenge/output_AAxial-imagesTs -t 115 -m 2d -f 5 &


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
## Dataset location paths
nifti_folder = "/data/01_UB/2021-MedNeurIPS/train_Nifti-Data/"
lesion_folder = "/data/01_UB/2021-MedNeurIPS/train_Nifti-Seg-6-Classes/"
# nifti_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-Data/"
# lesion_folder = "/data/01_UB/2021-MedNeurIPS/test_Nifti-Seg-6-Classes/"
metadata_file_path = "/data/01_UB/2021-MedNeurIPS/111_dataframe_axial_slices.csv"


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
## nn-UNet sandbox paths
dataset_folder = '/data/01_UB/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_COVIDSegChallenge'

imagesTr_shape_path = os.path.join(dataset_folder, "AAxial-imagesTr")
# labelsTr_shape_path = os.path.join(dataset_folder, "All-labels")

#####################################################################
## Step-2: Convert the 2D shape (X, Y) to (1, X, Y) for deployment
for row in range(metadata.shape[0]):

    ## locating the CT and Seg
    ct_scan_input_path = os.path.join(nifti_folder, metadata['ct_file_name'][row])
    # print('+ ct_scan_input_path', ct_scan_input_path)

    ## Reads .nii file and get the numpy image array
    ct = nib.load(ct_scan_input_path)
    ct_scan_array = ct.get_data()
    # print("+ ct_scan_array: ", ct_scan_array.shape[2])
    print("+ ct_scan_array: ", ct_scan_array.shape)

    for slice in range(ct_scan_array.shape[2]):
        ## Get the segmented slice
        slice_position = slice
        ct_slice = ct_scan_array[:, :, slice_position]

        ## Convert 2D array to 3D numpy array
        ct_array_reshape = ct_slice.reshape((512, 512, 1))

        ## Generate the images
        ct_nifti = nib.Nifti1Image(ct_array_reshape, ct.affine)

        ## Output files
        ct_filename = ct_scan_input_path.split(os.path.sep)[-1]
        ct_first_part, _ = ct_filename.split('.nii.gz')
        ct_filename = str(ct_first_part + '-' + str(slice_position) + '_0000' + '.nii.gz')

        ct_slice_output_path = os.path.join(imagesTr_shape_path, ct_filename)
        # print('+ ct_filename:', ct_slice_output_path)
        # print('+ ct_nifti:', ct_nifti)

        ## and write the nifti files
        nib.save(ct_nifti, ct_slice_output_path)
