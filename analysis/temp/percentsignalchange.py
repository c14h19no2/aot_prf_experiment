# highpass
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
blank_TRs = 24

psc_savefolder = "/tank/shared/2024/visual/AOT/temp/prftest/new_psc"

def psc(file):
    """
    Calculate percent signal change
    """
    img = nib.load(file)
    data = img.get_fdata()
    print("original data shape:", data.shape)

    baselines_start = data[:,:,:,:blank_TRs]# the first 24 TRs are blank
    print("baselines_start shape:", baselines_start.shape)
    baselines_end = data[:,:,:,-blank_TRs:] # and the last 24 TRs are blank
    print("baselines_end shape:", baselines_end.shape)
    #concatenate the two baselines
    baselines = np.concatenate((baselines_start, baselines_end), axis=3)

    baseline = np.mean(baselines, axis=3)
    #duplicate the baseline to the same shape as data
    baseline = np.repeat(baseline[:, :, :, np.newaxis], data.shape[3], axis=3)

    psc_data = ((data - baseline) / baseline) * 100
    print("psc data shape:", psc_data.shape)

    #save the psc data
    psc_img = nib.Nifti1Image(psc_data, img.affine)
    psc_file = os.path.join(
        psc_savefolder, os.path.basename(file).replace("filtered", "filtered_psc")
    )
    nib.save(psc_img, psc_file)
    return psc_file


def psc_on_folder(folder):
    """
    Calculate percent signal change on a folder
    """
    for file in glob.glob(os.path.join(folder, "*.nii.gz")):
        print("Processing file:", file)
        psc(file)
    return


if __name__ == "__main__":
    psc_on_folder("/tank/shared/2024/visual/AOT/temp/prftest/new_filtering")
