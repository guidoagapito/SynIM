import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import os

import specula
specula.init(device_idx=-1, precision=1)
# -------------------------------------------------------------------
# Get the path to the specula package's __init__.py file
specula_init_path = specula.__file__
# Navigate up to repository root
specula_package_dir = os.path.dirname(specula_init_path)
specula_repo_path = os.path.dirname(specula_package_dir)

recSynim_file = os.path.join(specula_repo_path, "main", "scao", "calib", "SCAO", "rec", "REC_syn_pd0.0a0_ps160p0.0500_sh20x20_wl750_fv1.2_np6_dmH0_nm54_zernike.fits")
recSpecula_file = os.path.join(specula_repo_path, "main", "scao", "calib", "SCAO", "rec", "scao_sh_rec5.fits")
subapSpecula_file = os.path.join(specula_repo_path, "main", "scao", "calib", "SCAO", "subapdata", "scao_subaps_n20_th0.5.fits")
# -------------------------------------------------------------------

#read the two rec files from the second extension of the fits file
recSynim = fits.open(recSynim_file)[1].data
recSpecula = fits.open(recSpecula_file)[1].data

print(f"recSynim shape: {recSynim.shape}")
print(f"recSpecula shape: {recSpecula.shape}")

#read the subaperture data from the second extension of the fits file
subapSpecula = fits.open(subapSpecula_file)[2].data
mask = np.zeros((20, 20))
mask.ravel()[subapSpecula.astype(int)] = 1
idx_valid = np.where(mask == 1)
n_subaps = int(idx_valid[0].shape[0])

print(f"subapSpecula shape: {subapSpecula.shape}")
print(f"mask shape: {mask.shape}")
print(f"idx_valid[0] shape: {idx_valid[0].shape}")
print(f"idx_valid[1] shape: {idx_valid[1].shape}")

# put the 2d rec in 3d matrices using the list of valid indices of the subapertures
recSynim_3d = np.zeros((recSynim.shape[0], 20, 20))
recSpecula_3d = np.zeros((recSynim.shape[0], 20, 20))

# scale the recSynim to the same max as recSpecula
ratio = recSpecula[0, :].max() / recSynim[0, :].max()
print(f"scale ratio: {ratio}")
for i in range(recSynim.shape[0]):
    recSynim_3d[i, idx_valid[0], idx_valid[1]] = recSynim[i, n_subaps:] * ratio
    recSpecula_3d[i, idx_valid[0], idx_valid[1]] = recSpecula[i, n_subaps:]
    
# plot 4 couples of modes fromt the two recs
plt.figure()
for i in range(4):
    plt.subplot(2, 4, i+1)
    plt.imshow(recSynim_3d[i], cmap='hot')
    plt.colorbar()
    plt.title(f'recSynim {i}')
    plt.subplot(2, 4, i+5)
    plt.imshow(recSpecula_3d[i], cmap='hot')
    plt.colorbar()
    plt.title(f'recSpecula {i}')
plt.show()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(recSynim_3d[0], cmap='hot')
plt.colorbar()
plt.title('recSynim')
plt.subplot(1, 2, 2)
plt.imshow(recSpecula_3d[0], cmap='hot')
plt.colorbar()
plt.title('recSpecula')
plt.show()
# plot the difference
plt.figure()
plt.imshow(recSynim_3d[0] - recSpecula_3d[0], cmap='hot')
plt.title('recSynim - recSpecula')
plt.colorbar()
plt.show()