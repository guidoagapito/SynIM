import numpy as np

def extrapolate_edge_pixel_mat_define(mask, doExt2Pix=False, twoSteps=False):
    """
    Defines the indices used to extrapolate the phase out of the edge of the pupil before interpolation.
    
    Parameters:
    mask (numpy.ndarray): Pupil mask (input).
    doExt2Pix (bool): If True, extrapolate 2 pixels.
    twoSteps (bool): If True, perform two-step extrapolation.
    
    Returns:
    numpy.ndarray: Matrices used for extrapolation.
    """
    smask = mask.shape
    float_mask = mask.astype(float)
    
    sum_1pix_extraX = np.full_like(float_mask, -1)
    sum_2pix_extraX = np.full_like(float_mask, -1)
    sum_1pix_extraY = np.full_like(float_mask, -1)
    sum_2pix_extraY = np.full_like(float_mask, -1)
    
    idx_mask_array = np.zeros_like(float_mask)
    idx_mask_array[mask > 0] = np.flatnonzero(mask > 0)
    
    find_1pix_extra = (np.roll(float_mask, 1, axis=0) + 
                       np.roll(float_mask, -1, axis=0) +
                       np.roll(float_mask, 1, axis=1) + 
                       np.roll(float_mask, -1, axis=1)) * (1.0 - float_mask)
    
    find_2pix_extra = (np.roll(float_mask, 2, axis=0) + 
                       np.roll(float_mask, -2, axis=0) +
                       np.roll(float_mask, 2, axis=1) + 
                       np.roll(float_mask, -2, axis=1)) * (1.0 - float_mask)
    
    idx_1pix_extra = np.where(find_1pix_extra > 0)
    idx_2pix_extra = np.where((find_2pix_extra > 0) & (find_1pix_extra == 0))
    
    for i, j in zip(*idx_1pix_extra):
        testX = -1
        if i + 1 < smask[0] - 1 and sum_1pix_extraX[i, j] == -1:
            if float_mask[i + 1, j] > 0:
                if i + 2 < smask[0] - 1 and float_mask[i + 2, j] > 0:
                    sum_2pix_extraX[i, j] = idx_mask_array[i + 2, j]
                    sum_1pix_extraX[i, j] = idx_mask_array[i + 1, j]
                else:
                    testX = idx_mask_array[i + 1, j]
        if i - 1 > 0 and sum_1pix_extraX[i, j] == -1:
            if float_mask[i - 1, j] > 0:
                if i - 2 > 0 and float_mask[i - 2, j] > 0:
                    sum_2pix_extraX[i, j] = idx_mask_array[i - 2, j]
                    sum_1pix_extraX[i, j] = idx_mask_array[i - 1, j]
                else:
                    testX = idx_mask_array[i - 1, j]
        if sum_1pix_extraX[i, j] == -1 and testX >= 0:
            sum_1pix_extraX[i, j] = testX
        
        testY = -1
        if j + 1 < smask[1] - 1 and sum_1pix_extraY[i, j] == -1:
            if float_mask[i, j + 1] > 0:
                if j + 2 < smask[1] - 1 and float_mask[i, j + 2] > 0:
                    sum_2pix_extraY[i, j] = idx_mask_array[i, j + 2]
                    sum_1pix_extraY[i, j] = idx_mask_array[i, j + 1]
                else:
                    testY = idx_mask_array[i, j + 1]
        if j - 1 > 0 and sum_1pix_extraY[i, j] == -1:
            if float_mask[i, j - 1] > 0:
                if j - 2 > 0 and float_mask[i, j - 2] > 0:
                    sum_2pix_extraY[i, j] = idx_mask_array[i, j - 2]
                    sum_1pix_extraY[i, j] = idx_mask_array[i, j - 1]
                else:
                    testY = idx_mask_array[i, j - 1]
        if sum_1pix_extraY[i, j] == -1 and testY >= 0:
            sum_1pix_extraY[i, j] = testY
    
    if doExt2Pix:
        for i, j in zip(*idx_2pix_extra):
            testX = -1
            if i + 2 < smask[0] - 1 and sum_2pix_extraX[i, j] == -1:
                if float_mask[i + 2, j] > 0:
                    if i + 3 < smask[0] - 1 and float_mask[i + 3, j] > 0:
                        sum_2pix_extraX[i, j] = idx_mask_array[i + 3, j]
                        sum_1pix_extraX[i, j] = idx_mask_array[i + 2, j]
                    else:
                        testX = idx_mask_array[i + 2, j]
            if i - 2 > 0 and sum_2pix_extraX[i, j] == -1:
                if float_mask[i - 2, j] > 0:
                    if i - 3 > 0 and float_mask[i - 3, j] > 0:
                        sum_2pix_extraX[i, j] = idx_mask_array[i - 3, j]
                        sum_1pix_extraX[i, j] = idx_mask_array[i - 2, j]
                    else:
                        testX = idx_mask_array[i - 2, j]
            if sum_1pix_extraX[i, j] == -1 and testX >= 0:
                sum_1pix_extraX[i, j] = testX

    if twoSteps:
        sum_pix_extra = np.stack((sum_1pix_extraX, sum_2pix_extraX, sum_1pix_extraY, sum_2pix_extraY), axis=-1)
    else:
        sum_pix_extra = np.stack((sum_1pix_extraX, sum_2pix_extraX), axis=-1)  # Fixed: using X not Y

    return sum_pix_extra


def extrapolate_edge_pixel(phase, sum_pix_extra):
    """
    Extrapolates the phase outside the pupil mask using defined indices.
    
    Parameters:
    phase (numpy.ndarray): The phase array (input).
    sum_pix_extra (numpy.ndarray): Matrices used for extrapolation.
    
    Returns:
    numpy.ndarray: The updated phase array after extrapolation.
    """
    # Make a copy to avoid modifying the original
    phase_copy = phase.copy()
    
    # Find indices where extrapolation should be applied
    idx_1pix = np.where(sum_pix_extra[:, :, 0] >= 0)
    
    # Determine extrapolation method based on matrix dimensions
    if sum_pix_extra.shape[2] > 2:  # Two-step extrapolation (X and Y directions)
        # Get X and Y reference indices
        idx_x = sum_pix_extra[:, :, 0][idx_1pix].astype(int)
        idx_y = sum_pix_extra[:, :, 2][idx_1pix].astype(int)
        
        # Get phase values
        vectExtraPolX = np.array([phase_copy.flat[i] for i in idx_x])
        vectExtraPolY = np.array([phase_copy.flat[i] for i in idx_y])
        
        # Initial extrapolation is average of X and Y directions
        vectExtraPol = 0.5 * (vectExtraPolX + vectExtraPolY)
        
        # Second level extrapolation
        # Get indices for second pixel references
        idx_x2 = sum_pix_extra[:, :, 1][idx_1pix]
        idx_y2 = sum_pix_extra[:, :, 3][idx_1pix]
        
        # Find valid indices
        valid_x2 = idx_x2 >= 0
        valid_y2 = idx_y2 >= 0
        
        # Convert valid indices to integers
        if np.any(valid_x2):
            idx_x2_valid = idx_x2[valid_x2].astype(int)
            vectExtraPol2X = np.array([phase_copy.flat[i] for i in idx_x2_valid])
            # Apply correction for X direction
            vectExtraPol[valid_x2] = vectExtraPol[valid_x2] + (vectExtraPolX[valid_x2] - vectExtraPol2X)
        
        if np.any(valid_y2):
            idx_y2_valid = idx_y2[valid_y2].astype(int)
            vectExtraPol2Y = np.array([phase_copy.flat[i] for i in idx_y2_valid])
            # Apply correction for Y direction
            vectExtraPol[valid_y2] = vectExtraPol[valid_y2] + (vectExtraPolY[valid_y2] - vectExtraPol2Y)
    else:  # Single-direction extrapolation
        # Get indices for references
        idx_1 = sum_pix_extra[:, :, 0][idx_1pix].astype(int)
        
        # Get phase values
        vectExtraPol = np.array([phase_copy.flat[i] for i in idx_1])
        
        # Second level extrapolation
        idx_2 = sum_pix_extra[:, :, 1][idx_1pix]
        valid_2 = idx_2 >= 0
        
        if np.any(valid_2):
            idx_2_valid = idx_2[valid_2].astype(int)
            vectExtraPol2 = np.array([phase_copy.flat[i] for i in idx_2_valid])
            # Apply linear extrapolation: 2*primary - secondary
            vectExtraPol[valid_2] = 2 * vectExtraPol[valid_2] - vectExtraPol2
    
    # Update phase values with extrapolated data
    for i, (row, col) in enumerate(zip(idx_1pix[0], idx_1pix[1])):
        phase_copy[row, col] = vectExtraPol[i]
    
    return phase_copy