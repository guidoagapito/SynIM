import os
import numpy as np
import matplotlib.pyplot as plt
from synim_utils import compute_interaction_matrix, prepare_interaction_matrix_params
from matrix_naming.filename_generator import generate_im_filenames

import specula
specula.init(device_idx=-1, precision=1)
# Get the path to the specula package's __init__.py file
specula_init_path = specula.__file__
# Navigate up to repository root
specula_package_dir = os.path.dirname(specula_init_path)
specula_repo_path = os.path.dirname(specula_package_dir)

# Path to the YAML configuration file and output directory
# The path to the YAML file is detemrined by the specula module
yaml_file = os.path.join(specula_repo_path, "main", "scao", "params_scao_sh.yml")
print(f"YAML file path: {yaml_file}")
#output directory is set to the caibration directory of the SPECULA repository
output_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "SCAO", "im")
print(f"Output directory: {output_dir}")

# Calculate the interaction matrix
im, params = compute_interaction_matrix(yaml_file, verbose=True, display=True)

# Generate an appropriate filename for the matrix
filename = generate_im_filenames(yaml_file)['ngs'][0]
print(f"Generated filename: {filename}")
filename = os.path.join(output_dir,filename)
print(f"Generated filename: {filename}")

print(f"Interaction matrix shape: {im.shape}")
print(f"Interaction matrix dtype: {im.dtype}")
print(f"Interaction matrix min: {im.min()}")
print(f"Interaction matrix max: {im.max()}")
print(f"Interaction matrix mean: {im.mean()}")
print(f"Interaction matrix std: {im.std()}")

# Save the matrix
from astropy.io import fits
fits.writeto(filename, im, overwrite=True)
print(f"Interaction matrix saved as: {filename}")

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

