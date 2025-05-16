import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from synim.params_manager import ParamsManager
from synim.utils import compute_mmse_reconstructor, dm3d_to_2d
import specula
specula.init(device_idx=-1, precision=1)

from specula.data_objects.intmat import Intmat
from specula.lib.modal_base_generator import compute_ifs_covmat

# -------------------------------------------------------------------
# Get the path to the specula package's __init__.py file
specula_init_path = specula.__file__
# Get the path to the synim package's __init__.py file
synim_init_path = os.path.dirname(__file__)
# Navigate up to repository root
specula_package_dir = os.path.dirname(specula_init_path)
specula_repo_path = os.path.dirname(specula_package_dir)

# Path to the YAML configuration file and output directory
# The path to the YAML file is determined by the specula module
yaml_file = os.path.join(synim_init_path, "params_morfeo.yml")
root_dir = os.path.join(specula_repo_path, "main", "scao","calib","MCAO")
print(f"YAML file path: {yaml_file}")

params_mgr = ParamsManager(yaml_file, root_dir=root_dir, verbose=True)
im_paths = params_mgr.compute_interaction_matrices(wfs_type='ngs', overwrite=False)

print('im_paths:', im_paths)

#output directory is set to the caibration directory of the SPECULA repository
output_im_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "im")
output_rec_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "rec")
print(f"Output directory: {output_im_dir}")

# -------------------------------------------------------------------
# Count NGS WFSs in the configuration
ngs_wfs_list = [wfs for wfs in params_mgr.wfs_list if 'ngs' in wfs['name']]
n_wfs = len(ngs_wfs_list)
print(f"Found {n_wfs} NGS WFSs")

im_full, n_slopes_per_wfs, mode_indices, dm_indices = params_mgr.assemble_interaction_matrices(
    wfs_type='ngs', output_im_dir=output_im_dir, save=False)

import pandas as pd
print("Full interaction matrix:")
df = pd.DataFrame(im_full)
print(df.to_string(float_format=lambda x: f"{x:.6e}"))

# Computation of rec matrix with MMSE
# 1 Compute atmospheric covariance matrix using the compute_ifs_covmat function
r0 = 0.2
L0 = 25
C_atm_full = np.zeros((im_full.shape[0], im_full.shape[0]))
weights = [1,0.15]
for i, dm in enumerate(dm_indices):
    params = params_mgr.prepare_interaction_matrix_params(wfs_type='ngs', 
                                                         wfs_index=1, dm_index=dm)
    dm2d = dm3d_to_2d(params['dm_array'],params['dm_mask'])
    dm2d = dm2d[mode_indices[i],:]   # Select only the first n modes
    print("dm2d shape", dm2d.shape)
    print("computing covariance matrix for DM", i+1)
    print("params['pup_diam_m']", params['pup_diam_m'])
    C_atm = compute_ifs_covmat(
        params['dm_mask'], params['pup_diam_m'], dm2d, r0, L0, 
        oversampling=2, verbose=False
    )
    print("Done.")
    # Create proper indexing for block assignment using np.ix_
    idx = np.ix_(mode_indices[i], mode_indices[i])
    C_atm_full[idx] = C_atm * weights[i] * (500/2/np.pi)**2

print("C_atm_full", C_atm_full.diagonal())

display_covmat = False
if display_covmat:
    plt.figure(figsize=(10, 8))
    plt.imshow(C_atm_full, cmap='viridis')
    plt.colorbar()
    plt.title(f"Atmospheric Covariance Matrix for all DMs")
    plt.tight_layout()
    plt.show()

# 2 Create the noise covariance matrix
# computes noise from magnitude and 0-magnitude flux
params = params_mgr.params
magnitude = np.array([params['source_ngs1']['magnitude'],params['source_ngs2']['magnitude'],params['source_ngs3']['magnitude']])
flux0 = 7.27e11
flux = flux0 * 10**(-0.4*magnitude) * 0.002
print("flux", flux)
sa_side_in_m = params['main']['pixel_pupil'] * params['main']['pixel_pitch'] / params['sh_ngs1']['subap_on_diameter']
sensor_fov = params['sh_ngs1']['sensor_pxscale'] * params['sh_ngs1']['subap_npx']
rad2arcsec = 3600.*180./np.pi
spot_size = 2 * params['sh_ngs1']['sensor_pxscale']
sigma2inArcsec2 = spot_size**2./2.35**2./flux
sigma2inNm2 = sigma2inArcsec2 * (1./rad2arcsec * sa_side_in_m / 4. * 1e9)**2.
sigma2inSlope = sigma2inArcsec2 * 1./(sensor_fov/2.)**2.
print("sigma2inSlope", sigma2inSlope)

noise_variance = sigma2inSlope
n_slopes_total = im_full.shape[1]

C_noise = np.zeros((n_slopes_total, n_slopes_total))
for i in range(n_wfs):
    # Set the diagonal elements for this WFS
    start_idx = i * n_slopes_per_wfs
    end_idx = (i + 1) * n_slopes_per_wfs
    C_noise[start_idx:end_idx, start_idx:end_idx] = noise_variance[i] * np.eye(n_slopes_per_wfs)

if display_covmat: 
    plt.figure(figsize=(10, 8))
    plt.imshow(C_noise, cmap='viridis')
    plt.colorbar()
    plt.title(f"Noise Covariance Matrix")
    plt.tight_layout()
    plt.show()

# 3 Compute the MMSE reconstructor
reconstructor = compute_mmse_reconstructor(im_full.T, C_atm_full, noise_variance=None, C_noise=C_noise, 
                        cinverse=False, verbose=False)

print("Full reconstructor matrix:")
df = pd.DataFrame(reconstructor)
print(df.to_string(float_format=lambda x: f"{x:.2f}"))
# -------------------------------------------------------------------

# Visualize the matrix
plt.figure(figsize=(10, 8))
plt.imshow(im_full, cmap='viridis')
plt.colorbar()
plt.title(f"Interaction Matrix")
plt.tight_layout()
plt.show()