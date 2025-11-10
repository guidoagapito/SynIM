import os
import numpy as np
import matplotlib.pyplot as plt

import specula
specula.init(device_idx=-1, precision=1)

import synim
synim.init(device_idx=-1, precision=1)
from synim.params_manager import ParamsManager

# ===================================================================
# Configuration
# ===================================================================
synim_init_path = os.path.dirname(__file__)
pro_file = os.path.join(synim_init_path, "params_maoryPhaseC_newpup_focus_filt.pro")
root_dir = "/raid1/guido/PASSATA/MAORYC/"

print(f"{'='*70}")
print(f"SYNIM - Atmospheric Covariance Matrix Computation")
print(f"{'='*70}")
print(f"PRO file: {pro_file}")
print(f"Root dir: {root_dir}")
print(f"{'='*70}\n")

# ===================================================================
# Initialize Parameters Manager
# ===================================================================
params_mgr = ParamsManager(pro_file, root_dir=root_dir, verbose=True)

# ===================================================================
# Atmospheric Parameters
# ===================================================================
r0 = 0.2    # Fried parameter [m]
L0 = 25.0   # Outer scale [m]

print(f"\n{'='*70}")
print(f"Atmospheric Parameters")
print(f"{'='*70}")
print(f"  r0 (Fried parameter): {r0} m")
print(f"  L0 (Outer scale): {L0} m")
print(f"{'='*70}\n")

# ===================================================================
# Step 1: Compute/Load Covariance Matrices for LAYERS
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 1: Computing/Loading Layer Covariance Matrices")
print(f"{'='*70}\n")

output_cov_dir = os.path.join(root_dir, "covariance")
os.makedirs(output_cov_dir, exist_ok=True)

cov_result_layers = params_mgr.compute_covariance_matrices(
    r0=r0,
    L0=L0,
    component_type='layer',
    output_dir=output_cov_dir,
    overwrite=False,      # Set to True to recompute
    full_modes=True,      # Compute for ALL modes
    verbose=True
)

print(f"\n✓ Layer covariance matrices ready:")
print(f"  Components: {cov_result_layers['component_indices']}")
print(f"  Files: {len(cov_result_layers['files'])}")

# ===================================================================
# Step 2: Compute/Load Covariance Matrices for DMs (if needed)
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 2: Computing/Loading DM Covariance Matrices")
print(f"{'='*70}\n")

cov_result_dms = params_mgr.compute_covariance_matrices(
    r0=r0,
    L0=L0,
    component_type='dm',
    output_dir=output_cov_dir,
    overwrite=False,      # Set to True to recompute
    full_modes=True,      # Compute for ALL modes
    verbose=True
)

print(f"\n✓ DM covariance matrices ready:")
print(f"  Components: {cov_result_dms['component_indices']}")
print(f"  Files: {len(cov_result_dms['files'])}")

# ===================================================================
# Step 3: Visualize Individual Covariance Matrices
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 3: Visualizing Individual Covariance Matrices")
print(f"{'='*70}\n")

display_individual = True

if display_individual:
    # Visualize first layer covariance
    if len(cov_result_layers['C_atm_blocks']) > 0:
        C_layer1 = cov_result_layers['C_atm_blocks'][0]
        layer_idx = cov_result_layers['component_indices'][0]

        plt.figure(figsize=(12, 5))

        # Covariance matrix
        plt.subplot(1, 2, 1)
        plt.imshow(C_layer1, cmap='viridis')
        plt.colorbar(label='Covariance [rad²]')
        plt.title(f"Layer {layer_idx} Covariance Matrix\n"
                  f"r0={r0}m, L0={L0}m")
        plt.xlabel("Mode Index")
        plt.ylabel("Mode Index")

        # Variance (diagonal)
        plt.subplot(1, 2, 2)
        variance = np.diag(C_layer1)
        plt.plot(variance, 'b-', linewidth=1.5)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Mode Index")
        plt.ylabel("Variance [rad²]")
        plt.title(f"Layer {layer_idx} Modal Variance\n"
                  f"RMS = {np.sqrt(variance.mean()):.4f} rad")

        plt.tight_layout()
        plt.savefig(os.path.join(output_cov_dir, f"covmat_layer{layer_idx}.png"), dpi=150)
        print(f"  ✓ Saved layer{layer_idx} visualization")

    # Visualize first DM covariance
    if len(cov_result_dms['C_atm_blocks']) > 0:
        C_dm1 = cov_result_dms['C_atm_blocks'][0]
        dm_idx = cov_result_dms['component_indices'][0]

        plt.figure(figsize=(12, 5))

        # Covariance matrix
        plt.subplot(1, 2, 1)
        plt.imshow(C_dm1, cmap='viridis')
        plt.colorbar(label='Covariance [rad²]')
        plt.title(f"DM {dm_idx} Covariance Matrix\n"
                  f"r0={r0}m, L0={L0}m")
        plt.xlabel("Mode Index")
        plt.ylabel("Mode Index")

        # Variance (diagonal)
        plt.subplot(1, 2, 2)
        variance = np.diag(C_dm1)
        plt.plot(variance, 'r-', linewidth=1.5)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Mode Index")
        plt.ylabel("Variance [rad²]")
        plt.title(f"DM {dm_idx} Modal Variance\n"
                  f"RMS = {np.sqrt(variance.mean()):.4f} rad")

        plt.tight_layout()
        plt.savefig(os.path.join(output_cov_dir, f"covmat_dm{dm_idx}.png"), dpi=150)
        print(f"  ✓ Saved dm{dm_idx} visualization")

# ===================================================================
# Step 4: Assemble Full Covariance Matrices (for specific WFS)
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 4: Assembling Full Covariance Matrices")
print(f"{'='*70}\n")

# Example: Assemble for NGS WFS
assemble_examples = True

if assemble_examples:
    # NGS layers
    try:
        C_atm_ngs_layers = params_mgr.assemble_covariance_matrix(
            C_atm_blocks=cov_result_layers['C_atm_blocks'],
            component_indices=cov_result_layers['component_indices'],
            wfs_type='ngs',
            component_type='layer',
            weights=None,  # Equal weights
            verbose=True
        )

        print(f"\n  ✓ NGS layers covariance assembled: {C_atm_ngs_layers.shape}")

        # Visualize
        plt.figure(figsize=(10, 8))
        plt.imshow(C_atm_ngs_layers, cmap='viridis')
        plt.colorbar(label='Covariance [rad²]')
        plt.title(f"NGS - Full Layer Covariance Matrix\n"
                  f"r0={r0}m, L0={L0}m")
        plt.xlabel("Mode Index")
        plt.ylabel("Mode Index")
        plt.tight_layout()
        plt.savefig(os.path.join(output_cov_dir, "covmat_ngs_layers_full.png"), dpi=150)
        print(f"  ✓ Saved NGS layers full covariance visualization")

    except Exception as e:
        print(f"  ⚠ Could not assemble NGS layers covariance: {e}")

    # LGS layers
    try:
        C_atm_lgs_layers = params_mgr.assemble_covariance_matrix(
            C_atm_blocks=cov_result_layers['C_atm_blocks'],
            component_indices=cov_result_layers['component_indices'],
            wfs_type='lgs',
            component_type='layer',
            weights=None,  # Equal weights
            verbose=True
        )

        print(f"\n  ✓ LGS layers covariance assembled: {C_atm_lgs_layers.shape}")

        # Visualize
        plt.figure(figsize=(10, 8))
        plt.imshow(C_atm_lgs_layers, cmap='viridis')
        plt.colorbar(label='Covariance [rad²]')
        plt.title(f"LGS - Full Layer Covariance Matrix\n"
                  f"r0={r0}m, L0={L0}m")
        plt.xlabel("Mode Index")
        plt.ylabel("Mode Index")
        plt.tight_layout()
        plt.savefig(os.path.join(output_cov_dir, "covmat_lgs_layers_full.png"), dpi=150)
        print(f"  ✓ Saved LGS layers full covariance visualization")

    except Exception as e:
        print(f"  ⚠ Could not assemble LGS layers covariance: {e}")

# ===================================================================
# Step 5: Statistics Summary
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 5: Statistics Summary")
print(f"{'='*70}\n")

print(f"LAYERS:")
for i, (idx, C_block, filepath) in enumerate(zip(
    cov_result_layers['component_indices'],
    cov_result_layers['C_atm_blocks'],
    cov_result_layers['files']
)):
    variance = np.diag(C_block)
    rms_rad = np.sqrt(variance.mean())

    # Convert to nm (at 500nm)
    wavelength_nm = cov_result_layers['wavelength_nm']
    rms_nm = rms_rad / (500/2/np.pi)

    print(f"  Layer {idx}:")
    print(f"    Modes: {C_block.shape[0]}")
    print(f"    RMS: {rms_rad:.4f} rad ({rms_nm:.2f} nm)")
    print(f"    File: {os.path.basename(filepath)}")
    print()

print(f"DMS:")
for i, (idx, C_block, filepath) in enumerate(zip(
    cov_result_dms['component_indices'],
    cov_result_dms['C_atm_blocks'],
    cov_result_dms['files']
)):
    variance = np.diag(C_block)
    rms_rad = np.sqrt(variance.mean())

    # Convert to nm (at 500nm)
    wavelength_nm = cov_result_dms['wavelength_nm']
    rms_nm = rms_rad / (500/2/np.pi)

    print(f"  DM {idx}:")
    print(f"    Modes: {C_block.shape[0]}")
    print(f"    RMS: {rms_rad:.4f} rad ({rms_nm:.2f} nm)")
    print(f"    File: {os.path.basename(filepath)}")
    print()

# ===================================================================
# Final Summary
# ===================================================================
print(f"\n{'='*70}")
print(f"COMPUTATION COMPLETE")
print(f"{'='*70}")
print(f"  Atmospheric parameters:")
print(f"    r0 = {r0} m")
print(f"    L0 = {L0} m")
print(f"  Output directory: {output_cov_dir}")
print(f"  Layer covariance files: {len(cov_result_layers['files'])}")
print(f"  DM covariance files: {len(cov_result_dms['files'])}")
print(f"{'='*70}\n")

# Show all plots
plt.show()