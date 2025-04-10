import os
import numpy as np
import matplotlib.pyplot as plt
from utils.params_utils import compute_interaction_matrix, prepare_interaction_matrix_params
from utils.filename_generator import generate_im_filenames

import specula
specula.init(device_idx=-1, precision=1)

from specula.data_objects.intmat import Intmat

# Get the path to the specula package's __init__.py file
specula_init_path = specula.__file__
# Navigate up to repository root
specula_package_dir = os.path.dirname(specula_init_path)
specula_repo_path = os.path.dirname(specula_package_dir)

# Path to the YAML configuration file and output directory
# The path to the YAML file is determined by the specula module
yaml_file = os.path.join(specula_repo_path, "main", "scao", "params_scao_sh.yml")
print(f"YAML file path: {yaml_file}")
#output directory is set to the caibration directory of the SPECULA repository
output_im_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "SCAO", "im")
output_rec_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "SCAO", "rec")
print(f"Output directory: {output_im_dir}")

# Make sure the output directory exists
os.makedirs(output_im_dir, exist_ok=True)
os.makedirs(output_rec_dir, exist_ok=True)

# Calculate the interaction matrix
params = prepare_interaction_matrix_params(yaml_file)
im = compute_interaction_matrix(params, verbose=True, display=True)

# Generate an appropriate filename for the matrix
filenames_by_type = generate_im_filenames(yaml_file)

# Default to NGS if available, otherwise use the first available type
if 'ngs' in filenames_by_type and filenames_by_type['ngs']:
    im_filename = filenames_by_type['ngs'][0]
elif 'lgs' in filenames_by_type and filenames_by_type['lgs']:
    im_filename = filenames_by_type['lgs'][0]
elif 'ref' in filenames_by_type and filenames_by_type['ref']:
    im_filename = filenames_by_type['ref'][0]
else:
    raise ValueError("No appropriate filename could be generated")

rec_filename = im_filename.replace('IM', 'REC')

# Full paths for the files
im_path = os.path.join(output_im_dir, im_filename)
rec_path = os.path.join(output_rec_dir, rec_filename)

print(f"Generated IM filename: {im_filename}")
print(f"Generated REC filename: {rec_filename}")

# Create the Intmat object
# Extract WFS type info for pupdata_tag
wfs_info = f"{params['wfs_type']}_{params['wfs_nsubaps']}"
pupdata_tag = f"{os.path.basename(yaml_file).split('.')[0]}_{wfs_info}"

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

# Generate and save the reconstruction matrix
n_modes = params['dm_array'].shape[2] if len(params['dm_array'].shape) > 2 else 1
recmat_obj = intmat_obj.generate_rec(nmodes=n_modes)
recmat_obj.save(rec_path)
print(f"Reconstruction matrix saved as: {rec_path}")

# Print some statistics about the matrices
print(f"Interaction matrix shape: {im.shape}")
print(f"Interaction matrix dtype: {im.dtype}")
print(f"Interaction matrix min: {im.min()}")
print(f"Interaction matrix max: {im.max()}")
print(f"Interaction matrix mean: {im.mean()}")

# Visualize the matrix
plt.figure(figsize=(10, 8))
plt.imshow(im, cmap='viridis')
plt.colorbar()
plt.title(f"Interaction Matrix")
plt.tight_layout()

plt.figure(figsize=(8, 12))
imBig = None
for i in range(4):
    im2Dx = np.zeros((params['wfs_nsubaps'],params['wfs_nsubaps']), dtype=im.dtype)
    im2Dx[params['idx_valid_sa'][:,0], params['idx_valid_sa'][:,1]] = im[:params['idx_valid_sa'].shape[0],i]
    im2Dy = np.zeros((params['wfs_nsubaps'],params['wfs_nsubaps']), dtype=im.dtype)
    im2Dy[params['idx_valid_sa'][:,0], params['idx_valid_sa'][:,1]] = im[params['idx_valid_sa'].shape[0]:,i]
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