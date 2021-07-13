

import os
import pandas as pd
from nnunet.dataset_conversion import utils



#dataset_folder = '/data/01_UB/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_COVIDSegChallenge'

dataset_folder = '/home/jgarcia/datasets/nnUNet_Sandbox/nnUNet_raw_data_base/nnUNet_raw_data/Task115_FullCOVIDSegChallenge'


imagesTr_path = os.path.join(dataset_folder, "imagesTr")
labelsTr_path = os.path.join(dataset_folder, "labelsTr")
imagesTs_path = os.path.join(dataset_folder, "imagesTs")
###
imagesTr_example = utils.get_identifiers_from_covid_GC(imagesTr_path)
print("+ imagesTr: ",imagesTr_example)
print("+ imagesTr: ",len(imagesTr_example))


# df_modality = pd.read_csv(os.path.join(radiomics_folder, "lesion_features-0.csv") , sep=',')


## Dataset conversion params
output_file             = os.path.join(dataset_folder, "dataset.json")
imagesTr_dir            = os.path.join(dataset_folder, "imagesTr")
imagesTs_dir            = os.path.join(dataset_folder, "imagesTs")
modalities              = ["CT"]
labels                  = {0: 'background', 1: 'GGO', 2: 'CON', 3: 'ATE', 4: 'PLE'}
dataset_name            = "Task115_COVID-19",
license                 = "Hands on",
dataset_description     = "General Lesion segmentation for covid+",
dataset_reference       = "COVID-19-20 - Grand Challenge & MICCAI",
dataset_release         = '0.0'


utils.generate_dataset_json(output_file, imagesTr_dir, imagesTs_dir, modalities, labels,
                    dataset_name, license, dataset_description, dataset_reference, dataset_release)


## nnUNET Laucher
## Step-1: nnUNet_plan_and_preprocess -t 115 --verify_dataset_integrity
## Step-2: CUDA_VISIBLE_DEVICES=4 nnUNet_train 3d_fullres nnUNetTrainerV2 115 0
