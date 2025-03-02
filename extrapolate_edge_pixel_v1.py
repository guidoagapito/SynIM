import numpy as np

def extrapolate_edge_pixel_mat_define(mask, doExt2Pix=False):
    # defines the indices to be used to extrapolate the phase
    # out of the edge of the pupil before doing the interpolation
    # from Guido Agapito IDL function
    smask = mask.shape
    
    float_mask = mask.astype(float)
    sum_1pix_extra = np.full_like(float_mask, -1, dtype=int)
    sum_2pix_extra = np.full_like(float_mask, -1, dtype=int)
    
    idx_mask_array_1D = np.where(mask.flatten())
    idx_mask_array_2D = np.where(mask)
    idx_mask_array = np.full_like(mask, -1, dtype=int)
    idx_mask_array[idx_mask_array_2D[0],idx_mask_array_2D[1]] = idx_mask_array_1D
    
    # matrix which defines the pixel out of the pupil mask
    find_1pix_extra = np.roll(float_mask,shift=(1,0),axis=(0, 1)) + np.roll(float_mask,shift=(0,1),axis=(0, 1)) + \
                      np.roll(float_mask,shift=(-1,0),axis=(0, 1)) + np.roll(float_mask,shift=(0,-1),axis=(0, -1))
    find_1pix_extra *= (1.0-float_mask)
    
    # matrix which defines the pixel out of the pupil mask
    find_2pix_extra = np.roll(float_mask,shift=(2,0),axis=(0, 1)) + np.roll(float_mask,shift=(0,2),axis=(0, 1)) + \
                      np.roll(float_mask,shift=(-2,0),axis=(0, 1)) + np.roll(float_mask,shift=(0,-2),axis=(0, -1))
    find_2pix_extra *= (1.0-float_mask)
    
    idx_1pix_extra = np.ravel(np.array(np.where(find_1pix_extra.flatten() > 0)))
    idx_2pix_extra = np.ravel(np.array(np.where((find_2pix_extra.flatten() > 0) & (find_1pix_extra.flatten() == 0))) )
    
    test1 = float_mask * 0
    test1.flat[idx_1pix_extra] = 1
    test2 = float_mask * 0
    test2.flat[idx_2pix_extra] = 1
    
    for idx in idx_1pix_extra:
        ind = np.unravel_index(idx, float_mask.shape)
        test = -1
        if ((ind[0] + 1) < (smask[0] - 1)) and (sum_1pix_extra.flatten()[idx] == -1):
            if float_mask[ind[0] + 1, ind[1]] > 0:
                if ((ind[0] + 2) < (smask[0] - 1)) and (float_mask[ind[0] + 2, ind[1]] > 0):
                    sum_2pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0] + 2, ind[1]]
                    sum_1pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0] + 1, ind[1]]
                else:
                    test = idx_mask_array[ind[0] + 1, ind[1]]
        if ((ind[0] - 1) > 0) and (sum_1pix_extra.flatten()[idx] == -1):
            if float_mask[ind[0] - 1, ind[1]] > 0:
                if ((ind[0] - 2) > 0) and (float_mask[ind[0] - 2, ind[1]] > 0):
                    sum_2pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0] - 2, ind[1]]
                    sum_1pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0] - 1, ind[1]]
                else:
                    test = idx_mask_array[ind[0] - 1, ind[1]]
        if ((ind[1] + 1) < (smask[1] - 1)) and (sum_1pix_extra.flatten()[idx] == -1):
            if float_mask[ind[0], ind[1] + 1] > 0:
                if ((ind[1] + 2) < (smask[1] - 1)) and (float_mask[ind[0], ind[1] + 2] > 0):
                    sum_2pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0], ind[1] + 2]
                    sum_1pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0], ind[1] + 1]
                else:
                    test = idx_mask_array[ind[0], ind[1] + 1]
        if ((ind[1] - 1) > 0) and (sum_1pix_extra.flatten()[idx] == -1):
            if float_mask[ind[0], ind[1] - 1] > 0:
                if ((ind[1] - 2) > 0) and (float_mask[ind[0], ind[1] - 2] > 0):
                    sum_2pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0], ind[1] - 2]
                    sum_1pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0], ind[1] - 1]
                else:
                    test = idx_mask_array[ind[0], ind[1] - 1]
        if (sum_1pix_extra.flatten()[idx] == -1) and (test >= 0):
            sum_1pix_extra[ind[0], ind[1]] = test

    if doExt2Pix:
        for idx in idx_2pix_extra:
            ind = np.unravel_index(idx, float_mask.shape)
            test = -1
            if ((ind[0] + 2) < (smask[0] - 1)) and (sum_2pix_extra.flatten()[idx] == -1):
                if float_mask[ind[0] + 2, ind[1]] > 0:
                    if ((ind[0] + 3) < (smask[0] - 1)) and (float_mask[ind[0] + 3, ind[1]] > 0):
                        sum_2pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0] + 3, ind[1]]
                        sum_1pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0] + 2, ind[1]]
                    else:
                        test = idx_mask_array[ind[0] + 2, ind[1]]
            if ((ind[0] - 2) > 0) and (sum_2pix_extra.flatten()[idx] == -1):
                if float_mask[ind[0] - 2, ind[1]] > 0:
                    if ((ind[0] - 3) > 0) and (float_mask[ind[0] - 3, ind[1]] > 0):
                        sum_2pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0] - 3, ind[1]]
                        sum_1pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0] - 2, ind[1]]
                    else:
                        test = idx_mask_array[ind[0] - 2, ind[1]]
            if ((ind[1] + 2) < (smask[1] - 1 and sum_2pix_extra.flatten()[idx] == -1)):
                if float_mask[ind[0], ind[1] + 2] > 0:
                    if ((ind[1] + 3) < (smask[1] - 1)) and (float_mask[ind[0], ind[1] + 3] > 0):
                        sum_2pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0], ind[1] + 3]
                        sum_1pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0], ind[1] + 2]
                    else:
                        test = idx_mask_array[ind[0], ind[1] + 2]
            if ((ind[1] - 2) > 0) and (sum_2pix_extra.flatten()[idx] == -1):
                if float_mask[ind[0], ind[1] - 2] > 0:
                    if ((ind[1] - 3) > 0) and (float_mask[ind[0], ind[1] - 3] > 0):
                        sum_2pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0], ind[1] - 3]
                        sum_1pix_extra[ind[0], ind[1]] = idx_mask_array[ind[0], ind[1] - 2]
                    else:
                        test = idx_mask_array[ind[0], ind[1] - 2]
            if (sum_1pix_extra.flatten()[idx] == -1) and (test >= 0):
                sum_1pix_extra[ind[0], ind[1]] = test
    
    return sum_1pix_extra, sum_2pix_extra

def extrapolate_edge_pixel(phase, sum_1pix_extra, sum_2pix_extra):
    # extrapolate the phase out of the edge of the pupil before doing the interpolation
    # from Guido Agapito ILD function
    
    # Flatten the sum arrays
    flattened_sum_1pix_extra = sum_1pix_extra.flatten()
    flattened_sum_2pix_extra = sum_2pix_extra.flatten()
    
    # indices of pixel to be extrapolated
    idx_1pix = np.array(np.where(flattened_sum_1pix_extra >= 0))
        
    # Check if phase is 2D or 3D
    if len(phase.shape) == 3:
        # check size
        if phase.shape[2] > phase.shape[0] | phase.shape[2] > phase.shape[1]:
            raise ValueError('Error in input data, the input array third dimension must be smaller than the first two.')
        # Flatten the phase array
        flattened_phase = phase.reshape((-1,phase.shape[2]))
        # extrapolated values
        # Extract values using indices from sum_1pix_extra and sum_2pix_extra
        vectExtraPol = flattened_phase[np.ravel(flattened_sum_1pix_extra[idx_1pix]),:]
        vectExtraPol2 = flattened_phase[np.ravel(flattened_sum_2pix_extra[idx_1pix]),:]
        # Find indices where sum_2pix_extra is non-negative
        idxExtraPol2 = np.ravel(np.array(np.where(flattened_sum_2pix_extra[idx_1pix] >= 0)))
        # Apply the extrapolation formula
        vectExtraPol[idxExtraPol2,:] = 2 * vectExtraPol[idxExtraPol2,:] - vectExtraPol2[idxExtraPol2,:]
        # Update the phase array along both dimensions
        flattened_phase[idx_1pix,:] = vectExtraPol
    else:
        # Flatten the phase array
        flattened_phase = phase.flatten()
        # extrapolated values
        # Extract values using indices from sum_1pix_extra and sum_2pix_extra
        vectExtraPol = np.ravel(flattened_phase[flattened_sum_1pix_extra[idx_1pix]])
        vectExtraPol2 = np.ravel(flattened_phase[flattened_sum_2pix_extra[idx_1pix]])  
        # Find indices where sum_2pix_extra is non-negative      
        idxExtraPol2 = np.array(np.where(flattened_sum_2pix_extra[idx_1pix] >= 0)) 
        # Apply the extrapolation formula     
        vectExtraPol[idxExtraPol2] = 2 * vectExtraPol[idxExtraPol2] - vectExtraPol2[idxExtraPol2]
        # fill phase with extrapolated values
        flattened_phase[idx_1pix] = vectExtraPol
    
    # Reshape the updated flattened_phase back to the original shape   
    updated_phase = flattened_phase.reshape(phase.shape)

    return updated_phase