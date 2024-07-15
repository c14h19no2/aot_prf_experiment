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

def highpass_dct(
    func,
    lb=0.01,
    TR=0.9,
    modes_to_remove=None,
    remove_constant=False,
):
    """highpass_dct

    Discrete cosine transform (DCT) is a basis set of cosine regressors of varying frequencies up to a filter cutoff of a specified number of seconds. Many software use 100s or 128s as a default cutoff, but we encourage caution that the filter cutoff isn't too short for your specific experimental design. Longer trials will require longer filter cutoffs. See this paper for a more technical treatment of using the DCT as a high pass filter in fMRI data analysis (https://canlab.github.io/_pages/tutorials/html/high_pass_filtering.html).

    Parameters
    ----------
    func: np.ndarray
        <n_voxels, n_timepoints> representing the functional data to be fitered
    lb: float, optional
        cutoff-frequency for low-pass (default = 0.01 Hz)
    TR: float, optional
        Repetition time of functional run, by default 0.9
    modes_to_remove: int, optional
        Remove first X cosines

    Returns
    ----------
    dct_data: np.ndarray
        array of shape(n_voxels, n_timepoints)
    cosine_drift: np.ndarray
        Cosine drifts of shape(n_scans, n_drifts) plus a constant regressor at cosine_drift[:, -1]

    Notes
    ----------
    * *High-pass* filters remove low-frequency (slow) noise and pass high-freqency signals.
    * Low-pass filters remove high-frequency noise and thus smooth the data.
    * Band-pass filters allow only certain frequencies and filter everything else out
    * Notch filters remove certain frequencies
    """
    #flatter first 3 dimention of func, keep the last dimention
    original_shape = func.shape
    x, y, z, t = func.shape
    n = x * y * z
    func = func.reshape(n, t)

    # Create high-pass filter and clean
    n_vol = func.shape[-1]
    st_ref = 0  # offset frametimes by st_ref * tr
    ft = np.linspace(st_ref * TR, (n_vol + st_ref) * TR, n_vol, endpoint=False)
    hp_set = _cosine_drift(lb, ft)

    # select modes
    if isinstance(modes_to_remove, int):
        hp_set[:, :modes_to_remove]
    else:
        # remove constant column
        if remove_constant:
            hp_set = hp_set[:, :-1]

    dct_data = clean(func.T, detrend=False, standardize=False, confounds=hp_set).T


    # reshape back to original shape
    dct_data = dct_data.reshape(original_shape)

    return dct_data, hp_set


def filter_on_file(sub,run):
    """
    Filter the functional data using highpass_dct
    """

    subtag = f"sub-{sub:03d}"
    run = str(run)
    # load the functional data
    func_file = f"/tank/shared/2024/visual/AOT/derivatives/fmripreps/aotfull_preprocs/fullpreproc_forcesyn_endfix/{subtag}/ses-pRF/func/{subtag}_ses-pRF_task-pRF_rec-nordicstc_run-{run}_space-T1w_desc-preproc_part-mag_bold.nii.gz"
    func = nib.load(func_file).get_fdata()
    print(func.shape)

    # filter the data
    filtered_data, hp_set = highpass_dct(func, lb=0.0033, TR=0.9)
    print(filtered_data.shape)
    print(hp_set.shape)

    # save the filtered data 
    filtered_img = nib.Nifti1Image(filtered_data, nib.load(func_file).affine, nib.load(func_file).header)
    nib.save(filtered_img, f"/tank/shared/2024/visual/AOT/temp/prftest/new_filtering/{subtag}_ses_pRF_run-{run.zfill(2)}_filtered_func.nii.gz")

    return


if __name__ == "__main__":
    filter_on_file(2,1)
    filter_on_file(2,2)
    filter_on_file(2,3)
    filter_on_file(2,4)
    filter_on_file(2,5)
    filter_on_file(2,6)
    filter_on_file(2,7)
    filter_on_file(2,8)
    filter_on_file(2,9)
    filter_on_file(2,10)
    

