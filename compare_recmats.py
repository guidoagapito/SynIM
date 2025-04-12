import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt

recSynim_file = '/Users/guido/GitHub/SPECULA/main/scao/calib/SCAO/rec/REC_syn_pd0.0a0_ps160p0.0500_sh20x20_wl750_fv1.2_np6_dmH0_nm54_zernike.fits'
recSpecula_file = '/Users/guido/GitHub/SPECULA/main/scao/calib/SCAO/rec/scao_sh_rec5.fits'

subapSpecula_file = '/Users/guido/GitHub/SPECULA/main/scao/calib/SCAO/subapdata/scao_subaps_n20_th0.5.fits'


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

print(f"subapSpecula shape: {subapSpecula.shape}")
print(f"mask shape: {mask.shape}")
print(f"idx_valid[0] shape: {idx_valid[0].shape}")
print(f"idx_valid[1] shape: {idx_valid[1].shape}")

# put the 2d rec in 3d matrices using the list of valid indices of the subapertures
recSynim_3d = np.zeros((recSynim.shape[0], 20, 20))
recSpecula_3d = np.zeros((recSynim.shape[0], 20, 20))
for i in range(recSynim.shape[0]):
    recSynim_3d[i, idx_valid[0], idx_valid[1]] = recSynim[i, :]
    recSpecula_3d[i, idx_valid[0], idx_valid[1]] = recSpecula[i, :]
    
# plot the two recs
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(recSynim_3d[0], cmap='gray')
plt.title('recSynim')
plt.subplot(1, 2, 2)
plt.imshow(recSpecula_3d[0], cmap='gray')
plt.title('recSpecula')
plt.show()
# plot the difference
plt.figure()
plt.imshow(recSynim_3d[0] - recSpecula_3d[0], cmap='gray')
plt.title('recSynim - recSpecula')
plt.colorbar()
plt.show()