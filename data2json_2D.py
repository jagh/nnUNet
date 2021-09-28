## 3D nnUNET Laucher
## Step-1: nnUNet_plan_and_preprocess -t 115 --verify_dataset_integrity
## Step-2: nnUNet_train 3d_fullres nnUNetTrainerV2 115 0

## 3D nnUNET Laucher
## Step-1: nnUNet_plan_and_preprocess -t 115 -pl3d None
## Step-2: nnUNet_train 2d nnUNetTrainerV2 115 0


import os
import pandas as pd
from nnunet.dataset_conversion import utils


# ## Desktop folder
dataset_folder = '/data/01_UB/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_COVIDSegChallenge'
## Server folder
# dataset_folder = '/home/jgarcia/datasets/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_FullCOVIDSegChallenge'

## Folder directories for Json conversion
imagesTr_path = os.path.join(dataset_folder, "imagesTr")
labelsTr_path = os.path.join(dataset_folder, "labelsTr")
imagesTs_path = os.path.join(dataset_folder, "imagesTs")


#####################################################################
## Step-1: Json conversion

## Get dataset file names
imagesTr_example = utils.get_identifiers_from_covid_GC(imagesTr_path)
print("+ imagesTr: ",imagesTr_example)
print("+ imagesTr: ",len(imagesTr_example))

## Dataset conversion params
output_file             = os.path.join(dataset_folder, "dataset.json")
imagesTr_dir            = os.path.join(dataset_folder, "imagesTr")
imagesTs_dir            = os.path.join(dataset_folder, "imagesTs")
modalities              = ["CT"]
labels                  = {0: 'foreground', 1: 'GGO', 2: 'CON', 3: 'ATE',
                                            4: 'PLE', 5: 'BAN', 6: 'TBR'}
dataset_name            = "Task115_COVID-19",
license                 = "Hands on",
dataset_description     = "2D-Multiclass Lesion segmentation for covid-19",
dataset_reference       = "Multiomics 2D slices",
dataset_release         = '0.0'

utils.generate_dataset_json(output_file, imagesTr_dir, imagesTs_dir, modalities, labels,
                    dataset_name, license, dataset_description, dataset_reference, dataset_release)




# #####################################################################
# ## Step-2: Convert the 2D shape (X, Y) to (1, X, Y)
# import glob
# import nibabel as nib
# import SimpleITK as sitk
#
# from skimage import io, transform
# import numpy as np
#
#
# def read_nifti(filepath):
#     '''' Reads .nii file and returns pixel array '''
#     ct_scan = nib.load(filepath)
#     img_array   = ct_scan.get_fdata()
#     img_affine  = ct_scan.affine
#     # array   = np.rot90(np.array(array))
#     return (img_array, img_affine)
#
#
#
# # ## Desktop folder
# # dataset_folder = '/data/01_UB/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_COVIDSegChallenge'
# ## Server folder
# dataset_folder = '/home/jgarcia/datasets/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_FullCOVIDSegChallenge'
#
# ## Folder paths
# imagesTr_path = os.path.join(dataset_folder, "imagesTs")
# labelsTr_path = os.path.join(dataset_folder, "labelsTs")
#
# ## Set folder paths
# imagesTr_path_glob = glob.glob(imagesTr_path + '/*')
# # imagesTr_shape_path = os.path.join(dataset_folder, "imagesTr_shape")
# # labelsTr_shape_path = os.path.join(dataset_folder, "labelsTr_shape")
# imagesTr_shape_path = os.path.join(dataset_folder, "imagesTs_shape")
# labelsTr_shape_path = os.path.join(dataset_folder, "labelsTs_shape")
#
#
#
# ## Loop through dataset
# for filepath in imagesTr_path_glob[:]:
#
#     filename = filepath.split(os.path.sep)[-1]
#     ct_slice_input_path = filepath
#     lesion_slice_input_path = os.path.join(labelsTr_path, filename)
#
#     ## Reads .nii file and get the numpy image array
#     img_array, img_affine = read_nifti(ct_slice_input_path)
#     lesion_img_array, lesion_img_affine = read_nifti(lesion_slice_input_path)
#     # print("- img_array: ", img_array.shape)
#     # print("- img_affine: ", img_affine.shape)
#
#     ## Convert 2D array to 3D numpy array
#     img_array_reshape = img_array.reshape((img_array.shape[0], img_array.shape[1], 1))
#     lesion_img_array_reshape = lesion_img_array.reshape(
#                             (lesion_img_array.shape[0], lesion_img_array.shape[1], 1))
#     # print("+ img_array_reshape: ", img_array_reshape.shape)
#     # print("+ lesion_img_array: ", lesion_img_array.shape)
#
#     ## Generate the images
#     ct_nifti = nib.Nifti1Image(img_array_reshape, img_affine)
#     lesion_nifti = nib.Nifti1Image(lesion_img_array_reshape, lesion_img_affine)
#
#     ## Output files
#     ct_slice_output_path = os.path.join(imagesTr_shape_path, filename)
#     lesion_slice_output_path = os.path.join(labelsTr_shape_path, filename)
#
#     ## and write the nifti files
#     nib.save(ct_nifti, ct_slice_output_path)
#     nib.save(lesion_nifti, lesion_slice_output_path)
