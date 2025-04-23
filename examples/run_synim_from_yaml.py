import os
import numpy as np
import matplotlib.pyplot as plt
from synim.params_manager import ParamsManager
from synim.params_common_utils import parse_params_file, generate_im_filename

import specula
specula.init(device_idx=-1, precision=1)

from specula.data_objects.intmat import Intmat

# -------------------------------------------------------------------
# Get the path to the specula package's __init__.py file
specula_init_path = specula.__file__
# Navigate up to repository root
specula_package_dir = os.path.dirname(specula_init_path)
specula_repo_path = os.path.dirname(specula_package_dir)


# Path to the YAML configuration file and output directory
# The path to the YAML file is determined by the specula module
yaml_file = os.path.join(specula_repo_path, "main", "scao", "params_scao_sh.yml")
root_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "SCAO")
print(f"YAML file path: {yaml_file}")
#output directory is set to the caibration directory of the SPECULA repository
output_im_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "SCAO", "im")
output_rec_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "SCAO", "rec")
print(f"Output directory: {output_im_dir}")

# -------------------------------------------------------------------
# Make sure the output directory exists
os.makedirs(output_im_dir, exist_ok=True)
os.makedirs(output_rec_dir, exist_ok=True)

# ----------------------------------------------------------------------
# Initialize the ParamsManager with the YAML file and root directory
params_mgr = ParamsManager(yaml_file, root_dir=root_dir, verbose=True)

# -------------------------------------------------------------------
# Generate an appropriate filename for the matrix
im_filename = params_mgr.generate_im_filename()
rec_filename = im_filename.replace('IM', 'REC')
im_path = os.path.join(output_im_dir, im_filename)
rec_path = os.path.join(output_rec_dir, rec_filename)

print(f"Generated IM filename: {im_filename}")
print(f"Generated REC filename: {rec_filename}")
# -------------------------------------------------------------------

im = params_mgr.compute_interaction_matrix()
print('im.shape',im.shape)

# Create the Intmat object
# Extract WFS type info for pupdata_tag
wfs_params = params_mgr.get_wfs_params()
wfs_info = f"{wfs_params['wfs_type']}_{wfs_params['wfs_nsubaps']}"
config_name = os.path.basename(yaml_file).split('.')[0]
pupdata_tag = f"{config_name}_{wfs_info}"

# Create Intmat object and save it
intmat_obj = Intmat(
    im, 
    pupdata_tag=pupdata_tag,
    norm_factor=1.0,  # Default value
    target_device_idx=None,  # Use default device
    precision=None    # Use default precision
)

# Save the interaction matrix
intmat_obj.save(im_path)
print(f"Interaction matrix saved as: {im_path}")
# -------------------------------------------------------------------

# Generate and save the reconstruction matrix
dm_params = params_mgr.get_dm_params(0)
dm_array = dm_params['dm_array']
n_modes = dm_array.shape[2] if len(dm_array.shape) > 2 else 1
recmat_obj = intmat_obj.generate_rec(nmodes=n_modes)
recmat_obj.save(rec_path)
print(f"Reconstruction matrix saved as: {rec_path}")

# Print some statistics about the matrices
print(f"Interaction matrix shape: {im.shape}")
print(f"Interaction matrix dtype: {im.dtype}")
print(f"Interaction matrix min: {im.min()}")
print(f"Interaction matrix max: {im.max()}")
print(f"Interaction matrix mean: {im.mean()}")
# -------------------------------------------------------------------

# Visualize the matrix
plt.figure(figsize=(10, 8))
plt.imshow(im, cmap='viridis')
plt.colorbar()
plt.title(f"Interaction Matrix")
plt.tight_layout()

plt.figure(figsize=(8, 12))
imBig = None

wfs_params = params_mgr.get_wfs_params(0)
sa2D = np.zeros((wfs_params['wfs_nsubaps'],wfs_params['wfs_nsubaps']))
sa2D[wfs_params['idx_valid_sa'][:,0], wfs_params['idx_valid_sa'][:,1]] = 1
sa2D = np.transpose(sa2D)
idx_valid_sa_new = np.where(sa2D>0)
idx_valid_sa = wfs_params['idx_valid_sa']
idx_valid_sa[:,0] = idx_valid_sa_new[0]
idx_valid_sa[:,1] = idx_valid_sa_new[1]

for i in range(4):
    im2Dx = np.zeros((wfs_params['wfs_nsubaps'],wfs_params['wfs_nsubaps']), dtype=im.dtype)
    im2Dx[idx_valid_sa[:,0], idx_valid_sa[:,1]] = im[i,:idx_valid_sa.shape[0]]
    im2Dy = np.zeros((wfs_params['wfs_nsubaps'],wfs_params['wfs_nsubaps']), dtype=im.dtype)
    im2Dy[idx_valid_sa[:,0], idx_valid_sa[:,1]] = im[i,idx_valid_sa.shape[0]:]
    im2D = np.concatenate((im2Dx, im2Dy), axis=1)
    if imBig is None:
        imBig = im2D
    else:
        imBig = np.concatenate((imBig, im2D), axis=0)
plt.imshow(imBig, cmap='viridis')
plt.colorbar()
plt.title('Interaction Matrix 2D')
plt.tight_layout()

plt.show()