import os
import numpy as np
import matplotlib.pyplot as plt
from utils.params_utils import compute_interaction_matrices
from utils.params_common_utils import generate_im_filename
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
yaml_file = "/Users/guido/GitHub/SPECULA_scripts/params_morfeo_full2.yml"
root_dir = os.path.join(specula_repo_path, "main", "scao","calib","MCAO")
print(f"YAML file path: {yaml_file}")
#output directory is set to the caibration directory of the SPECULA repository
output_im_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "im")
output_rec_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "rec")
print(f"Output directory: {output_im_dir}")

matrices = compute_interaction_matrices(yaml_file, root_dir=root_dir, output_im_dir=output_im_dir, 
                                 wfs_type='ngs', overwrite=True, verbose=True, display=False)

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

# TODO: computation of rec matrix with MMSE
# rec_filename = im_filename.replace('IM', 'REC')
# # Full paths for the files
# rec_path = os.path.join(output_rec_dir, rec_filename)
# print(f"Generated REC filename: {rec_filename}")
# # Generate and save the reconstruction matrix
# n_modes = params['dm_array'].shape[2] if len(params['dm_array'].shape) > 2 else 1
# recmat_obj = intmat_obj.generate_rec(nmodes=n_modes)
# recmat_obj.save(rec_path)
# print(f"Reconstruction matrix saved as: {rec_path}")

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