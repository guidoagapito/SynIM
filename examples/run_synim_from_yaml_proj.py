import os
import numpy as np
import matplotlib.pyplot as plt
from synim.params_manager import ParamsManager
from synim.utils import generate_pm_filename, extract_dm_list, extract_layer_list, extract_opt_list
import specula
specula.init(device_idx=-1, precision=1)

from specula.data_objects.intmat import Intmat

# -------------------------------------------------------------------
# Get paths
specula_init_path = specula.__file__
synim_init_path = os.path.dirname(__file__)
specula_package_dir = os.path.dirname(specula_init_path)
specula_repo_path = os.path.dirname(specula_package_dir)

# Set up file paths
yaml_file = os.path.join(synim_init_path, "params_morfeo_proj.yml")
root_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO")
output_pm_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "pm")
print(f"YAML file path: {yaml_file}")

# Compute projection matrices if needed
params_mgr = ParamsManager(yaml_file, root_dir=root_dir, verbose=True)
pm_paths = params_mgr.compute_projection_matrices(overwrite=False)

pm_full_dm, pm_full_layer, weights_array = params_mgr.assemble_projection_matrices(output_dir=output_pm_dir, save=False)


print(f"Weights array: {weights_array}")

# Visualize slices of the 4D matrices
if pm_full_dm is not None:
    # Visualize the first mode for first DM across all sources
    plt.figure(figsize=(10, 6))
    plt.imshow(pm_full_dm[0, :, :], cmap='viridis')
    plt.colorbar()
    plt.title(f"DM Projection Matrix - Mode 0, all DMs")
    plt.xlabel("DM Mode Index")
    plt.ylabel("Source Index")
    plt.tight_layout()

    # Visualize first source, first DM for all modes
    plt.figure(figsize=(10, 6))
    plt.imshow(pm_full_dm[:, 0, :], cmap='viridis')
    plt.colorbar()
    plt.title("DM Projection Matrix - Source 0, all DMs")
    plt.xlabel("DM Mode Index")
    plt.ylabel("Mode Index")
    plt.tight_layout()
    plt.show()

if pm_full_layer is not None:
    # Visualize the first mode for first layer across all sources
    plt.figure(figsize=(10, 6))
    plt.imshow(pm_full_layer[0, :, :], cmap='viridis')
    plt.colorbar()
    plt.title(f"Layer Projection Matrix - Mode 0, all Layers")
    plt.xlabel("Layer Mode Index")
    plt.ylabel("Source Index")
    plt.tight_layout()

    # Visualize first source, first layer for all modes
    plt.figure(figsize=(10, 6))
    plt.imshow(pm_full_layer[:, 0, :], cmap='viridis')
    plt.colorbar()
    plt.title("Layer Projection Matrix - Source 0, all Layers")
    plt.xlabel("Layer Mode Index")
    plt.ylabel("Mode Index")
    plt.tight_layout()
    plt.show()