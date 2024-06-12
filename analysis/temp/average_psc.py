import os
import numpy as np
import json
import argparse
import glob
from nilearn.surface import load_surf_data
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
from nilearn.signal import clean
from nilearn.glm.first_level.design_matrix import _cosine_drift

import nibabel as nib

psc_savefolder = "/tank/shared/2024/visual/AOT/temp/prftest/psc"
psc_average_folder = "/tank/shared/2024/visual/AOT/temp/prftest/psc_average"

def average_on_psc_folder(folder):
    """
    Calculate average on a folder
    """
    data_list = []
    for file in glob.glob(os.path.join(folder, "*.nii.gz")):
        print("Processing file:", file)
        img = nib.load(file)
        data = img.get_fdata()
        data_list.append(data)
    
    data_list = np.array(data_list)
    print("data_list shape:", data_list.shape)

    average_data = np.mean(data_list, axis=0)
    print("average_data shape:", average_data.shape)
    #save the average data
    average_img = nib.Nifti1Image(average_data, img.affine)

    average_file = os.path.join(
        psc_average_folder, os.path.basename(file).replace("filtered", "filtered_psc_averageallruns")
    )
    nib.save(average_img, average_file)
    return average_file


if __name__ == "__main__":
    average_on_psc_folder(psc_savefolder)
