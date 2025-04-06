import numpy as np
import matplotlib.pyplot as plt
from synim_utils import compute_interaction_matrix, prepare_interaction_matrix_params
from matrix_naming.filename_generator import generate_im_filename

# Path to the YAML configuration file
yaml_file = r"C:\Users\guido\OneDrive\Documenti\GitHub\SPECULA\main\scao\params_scao_sh.yml"

# Calculate the interaction matrix
im, params = compute_interaction_matrix(yaml_file, verbose=True, display=True)

# Generate an appropriate filename for the matrix
filename = generate_im_filename(yaml_file)

# Save the matrix
from astropy.io import fits
fits.writeto(filename, im, overwrite=True)
print(f"Interaction matrix saved as: {filename}")

# Visualize the matrix
plt.figure(figsize=(10, 8))
plt.imshow(im, cmap='viridis')
plt.colorbar()
plt.title(f"Interaction Matrix: {filename}")
plt.tight_layout()
plt.show()