import os
import numpy as np
import matplotlib.pyplot as plt
from utils.params_utils import compute_interaction_matrices
from utils.params_common_utils import generate_im_filename, prepare_interaction_matrix_params, compute_mmse_reconstructor, dm3d_to_2d
import specula
specula.init(device_idx=-1, precision=1)

from specula.data_objects.intmat import Intmat
from specula.lib.modal_base_generator import compute_ifs_covmat

# -------------------------------------------------------------------
# Get the path to the specula package's __init__.py file
specula_init_path = specula.__file__
# Navigate up to repository root
specula_package_dir = os.path.dirname(specula_init_path)
specula_repo_path = os.path.dirname(specula_package_dir)

# Path to the YAML configuration file and output directory
# The path to the YAML file is determined by the specula module
yaml_file = "/Users/guido/GitHub/SPECULA_scripts/params_morfeo_full2.yml"
root_dir = os.path.join(specula_repo_path, "main", "scao","calib","MCAO")
print(f"YAML file path: {yaml_file}")
#output directory is set to the caibration directory of the SPECULA repository
output_im_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "im")
output_rec_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "rec")
print(f"Output directory: {output_im_dir}")

paramsAll, matrices = compute_interaction_matrices(yaml_file, root_dir=root_dir, output_im_dir=output_im_dir, 
                                 wfs_type='ngs', overwrite=True, verbose=False, display=False)

# -------------------------------------------------------------------
# Load from disk the full set of interaction matrices
# then put them in a singla 2D array NXM
# where N is the number of modes, 2 for the first DM and 3 for the third DM
# and M is the number of slopes, 8 multiplied by 3 WFSs
N = 5
n_slopes = 2
M = 3*n_slopes
im_full = np.zeros((N,M)) 
for ii in range(3):
    for jj in range(3):
        if jj == 1:
            continue
        im_filename = generate_im_filename(yaml_file, wfs_type='ngs', wfs_index=ii+1, dm_index=jj+1)
        # Full paths for the files
        im_path = os.path.join(output_im_dir, im_filename)
        print(f"--> Generated IM filename: {im_filename}")
        # Load the interaction matrix
        intmat_obj = Intmat.restore(im_path)
        # Get the interaction matrix data
        if jj == 0:
            mode_idx = [0,1]
        if jj == 2:
            mode_idx = [2,3,4]
        print(f'size of intmat: {intmat_obj._intmat.shape}')
        im_full[mode_idx, n_slopes*ii:n_slopes*(ii+1)] = intmat_obj._intmat[mode_idx,:]
        
import pandas as pd
print("Full interaction matrix:")
df = pd.DataFrame(im_full)
print(df.to_string(float_format=lambda x: f"{x:.6e}"))

# Computation of rec matrix with MMSE
# 1 Compute atmospheric covariance matrix using the compute_ifs_covmat function
r0 = 0.2
L0 = 25
for i in range(3):
    params = prepare_interaction_matrix_params(paramsAll, wfs_type='ngs', 
                                               wfs_index=1, dm_index=i+1)
    dm2d = dm3d_to_2d(params['dm_array'],params['dm_mask'])
    C_atm = compute_ifs_covmat(
        params['dm_mask'], params['pup_diam_m'], dm2d, r0, L0, 
        oversampling=2, verbose=False
    )
    plt.figure(figsize=(10, 8))
    plt.imshow(C_atm, cmap='viridis')
    plt.colorbar()
    plt.title(f"Atmospheric Covariance Matrix for DM {i+1}")
    plt.tight_layout()
    plt.show()


# 2 Create the noise covariance matrix
n_slopes_total = interaction_matrix.shape[1]
n_wfs = len(noise_variance)
n_slopes_per_wfs = n_slopes_total // n_wfs

C_noise = np.zeros((n_slopes_total, n_slopes_total))
for i in range(n_wfs):
    # Set the diagonal elements for this WFS
    start_idx = i * n_slopes_per_wfs
    end_idx = (i + 1) * n_slopes_per_wfs
    C_noise[start_idx:end_idx, start_idx:end_idx] = noise_variance[i] * np.eye(n_slopes_per_wfs)

# 3 Compute the MMSE reconstructor
reconstructor = compute_mmse_reconstructor(interaction_matrix, C_atm, noise_variance=None, C_noise=C_noise, 
                        cinverse=False, verbose=False)

# Print some statistics about the matrices
print(f"Interaction matrix shape: {im_full.shape}")
print(f"Interaction matrix dtype: {im_full.dtype}")
print(f"Interaction matrix min: {im_full.min()}")
print(f"Interaction matrix max: {im_full.max()}")
print(f"Interaction matrix mean: {im_full.mean()}")
# -------------------------------------------------------------------

# Visualize the matrix
plt.figure(figsize=(10, 8))
plt.imshow(im_full, cmap='viridis')
plt.colorbar()
plt.title(f"Interaction Matrix")
plt.tight_layout()
plt.show()