import os
import numpy as np

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

# Path to the PRO configuration file and output directory
# The path to the PRO file is determined by the specula module
pro_file = os.path.join(synim_init_path, "params_maoryPhaseC_newpup_focus_filt.pro")
root_dir = "/raid1/guido/PASSATA/MAORYC/" #os.path.join(specula_repo_path, "main", "scao","calib","MCAO")
print(f"PRO file path: {pro_file}")

#output directory is set to the caibration directory of the SPECULA repository
output_im_dir = os.path.join(root_dir, "synim/") #os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "im")
output_rec_dir = os.path.join(root_dir, "synrec/") #os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "rec")
print(f"Output directory: {output_im_dir}")

params_mgr = ParamsManager(pro_file, root_dir=root_dir, verbose=True)
im_paths = params_mgr.compute_interaction_matrices(output_im_dir, output_rec_dir, wfs_type='lgs', overwrite=False)

print('im_paths:', im_paths)

# -------------------------------------------------------------------
# Count NGS WFSs in the configuration
out = params_mgr.count_mcao_stars()
n_wfs = out['n_lgs']
print(f"Found {n_wfs} NGS WFSs")

im_full, n_slopes_per_wfs, mode_indices, dm_indices = params_mgr.assemble_interaction_matrices(
    wfs_type='lgs', output_im_dir=output_im_dir, component_type='layer',save=False)

print('im_full shape:', im_full.shape)
print('n_slopes_per_wfs:', n_slopes_per_wfs)
print('dm_indices:', dm_indices)

# -------------------------------------------------------------------
# STEP 1: Compute/Load full covariance matrices (done once, saved)
# -------------------------------------------------------------------
r0 = 0.2
L0 = 25

cov_result = params_mgr.compute_covariance_matrices(
    r0=r0,
    L0=L0,
    component_type='layer',
    output_dir=os.path.join(root_dir, "covariance"),
    overwrite=False,
    full_modes=True,  # Compute for ALL modes
    verbose=True
)

# -------------------------------------------------------------------
# STEP 2: Assemble only the modes you need (fast, uses modal_combination)
# -------------------------------------------------------------------
weights = [1.0, 0.15]

C_atm_full = params_mgr.assemble_covariance_matrix(
    C_atm_blocks=cov_result['C_atm_blocks'],
    component_indices=cov_result['component_indices'],
    mode_indices=mode_indices,  # From assemble_interaction_matrices
    weights=weights,
    verbose=True
)

# Or equivalently, using wfs_type:
# C_atm_full = params_mgr.assemble_covariance_matrix(
#     C_atm_blocks=cov_result['C_atm_blocks'],
#     component_indices=cov_result['component_indices'],
#     wfs_type='lgs',
#     component_type='layer',
#     weights=weights,
#     verbose=True
# )

# Visualize
display_covmat = True
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
sa_side_in_m = params['main']['pixel_pupil'] * params['main']['pixel_pitch'] / params['sh_lgs1']['subap_on_diameter']
sensor_fov = params['sh_ngs1']['sensor_pxscale'] * params['sh_ngs1']['subap_npx']
rad2arcsec = 3600.*180./np.pi
sigma2inNm2 = 2e4
sigma2inArcsec2 = sigma2inNm2 / (1./rad2arcsec * sa_side_in_m / 4. * 1e9)**2.
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
reconstructor = compute_mmse_reconstructor(im_full.T, C_atm_full,
                                           noise_variance=None, C_noise=C_noise,
                                           cinverse=False, verbose=False)

# Visualize the matrix
plt.figure(figsize=(10, 8))
plt.imshow(im_full, cmap='viridis')
plt.colorbar()
plt.title(f"Interaction Matrix")
plt.tight_layout()
plt.show()
