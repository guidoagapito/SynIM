import numpy as np
import time
from numpy.testing import assert_array_equal, assert_almost_equal
import matplotlib.pyplot as plt

# Import the functions to test
from synim import extrapolate_edge_pixel, rotshiftzoom_array, rotshiftzoom_array_noaffine, compute_derivatives_with_extrapolation, zern2phi


def sample_mask(mask_size):
    """Generate a test mask with a central aperture"""
    mask = np.zeros((mask_size, mask_size), dtype=int)
    mask[1:mask_size-1, 1:mask_size-1] = 1  # Create a central "pupil"
    return mask

def sample_phase(mask_size,n_zernikes=16):
    """Generate a phase matrix with zernikes"""
    phase_array = zern2phi(mask_size, n_zernikes, mask=sample_mask(mask_size), no_round_mask=False, xsign=1, ysign=1, rot_angle=0, verbose=False)
    return phase_array

def test_rotshiftzoom_array(mask_size=256,n_zernikes=16):
    """Test the rotshiftzoom_array function"""
    input_array = sample_phase(mask_size,n_zernikes)
    dm_translation = (3, 0)
    dm_rotation = 45
    dm_magnification = (1, 1)
    wfs_translation = (0, 0)
    wfs_rotation = 0
    wfs_magnification = (1, 1)
    output_size = (mask_size, mask_size)

    #compute time of the computation
    start_time1 = time.time()

    transformed_array = rotshiftzoom_array(input_array, dm_translation=dm_translation, dm_rotation=dm_rotation, dm_magnification=dm_magnification,
                                           wfs_translation=wfs_translation, wfs_rotation=wfs_rotation, wfs_magnification=wfs_magnification,
                                           output_size=output_size)

    print("--- %s seconds ---" % (time.time() - start_time1))

    start_time2 = time.time()

    transformed_array_affine = rotshiftzoom_array_noaffine(input_array, dm_translation=dm_translation, dm_rotation=dm_rotation, dm_magnification=dm_magnification,
                                                  wfs_translation=wfs_translation, wfs_rotation=wfs_rotation, wfs_magnification=wfs_magnification,
                                                  output_size=output_size)
    
    print("--- %s seconds ---" % (time.time() - start_time2))

    print('input_array.shape:\n', input_array.shape)
    print('transformed_array.shape:\n', transformed_array.shape)
    print('transformed_array_affine.shape:\n', transformed_array_affine.shape)

    fig, axs = plt.subplots(1,2)
    im2 = axs[0].imshow(input_array[:,:,0])
    im2 = axs[1].imshow(transformed_array[:,:,0])
    axs[0].set_title('DM shapes before, after rotation')
    fig.colorbar(im2, ax=axs.ravel().tolist(),fraction=0.02)

    fig, axs = plt.subplots(1,2)
    im2 = axs[0].imshow(transformed_array[:,:,0])
    im2 = axs[1].imshow(transformed_array_affine[:,:,0])
    axs[0].set_title('DM shapes after rotation with two methods')
    fig.colorbar(im2, ax=axs.ravel().tolist(),fraction=0.02)

    plt.figure()
    plt.imshow(transformed_array[:,:,0]-transformed_array_affine[:,:,0])
    plt.colorbar()
    plt.title('Difference between the two methods')
    plt.show()

#run test_rotshiftzoom_array()
mask_size = 256
n_zernikes = 64
test_rotshiftzoom_array(mask_size,n_zernikes)
