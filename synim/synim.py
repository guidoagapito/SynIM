import numpy as np
from scipy.ndimage import rotate, shift, zoom
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform, binary_dilation
from synim.utils import apply_mask, polar_to_xy, rebin

def compute_derivatives_with_extrapolation(data,mask=None):
    """
    Compute x and y derivatives using numpy.gradient on a 2D or 3D numpy array
    if mask is present does an extrapolation to avoid issue at the edges
    
    Parameters:
    - data: numpy 3D array
    - mask: optional, numpy 2D array, mask

    Returns:
    - dx: numpy 3D array, x derivative
    - dy: numpy 3D array, y derivative
    """

    if mask is not None:
        # Calculate indices and coefficients for extrapolation
        edge_pixels, reference_indices, coefficients = calculate_extrapolation_indices_coeffs(
            mask, debug=False, debug_pixels=None)
        for i in range(data.shape[2]):
            # Apply extrapolation
            data[:,:,i] = apply_extrapolation(data[:,:,i], edge_pixels, reference_indices, coefficients, 
                                              debug=True, problem_indices=None)

    # Compute x derivative
    dx = np.gradient(data, axis=(1), edge_order=1)
    
    # Compute y derivative
    dy = np.gradient(data, axis=(0), edge_order=1)
    
    if mask is not None:
        idx = np.ravel(np.array(np.where(mask.flatten() == 0)))
        dx2D = dx.reshape((-1,dx.shape[2]))
        dx2D[idx,:] = np.nan
        dy2D = dy.reshape((-1,dy.shape[2]))
        dy2D[idx,:] = np.nan
        dx = dx2D.reshape(dx.shape)
        dy = dy2D.reshape(dy.shape)
        
    return dx, dy

def integrate_derivatives(dx, dy):
    """
    Numerical integration of derivatives using numpy.cumsum
    along the x and y axes.

    Parameters:
        dx (ndarray): x derivative of the data.
        dy (ndarray): y derivative of the data.

    Returns:
        tuple: (integrated_x, integrated_y)
            - integrated_x: Integrated x derivative.
            - integrated_y: Integrated y derivative.
    """
    
    # Integrate x derivative along the x-axis
    integrated_x = np.cumsum(dx, axis=1)

    # Integrate y derivative along the y-axis
    integrated_y = np.cumsum(dy, axis=0)

    return integrated_x, integrated_y

# Labels for the extrapolation directions
directions_labels = ['Down (y+1)', 'Up (y-1)', 'Right (x+1)', 'Left (x-1)']

def calculate_extrapolation_indices_coeffs(mask, debug=False, debug_pixels=None):
    """
    Calculates indices and coefficients for extrapolating edge pixels of a mask.

    Parameters:
        mask (ndarray): Binary mask (True/1 inside, False/0 outside).
        debug (bool): If True, displays debug information and plots.
        debug_pixels (list): List of [y, x] coordinates for detailed debug output.

    Returns:
        tuple: (edge_pixels, reference_indices, coefficients)
            - edge_pixels: Linear indices of the edge pixels to extrapolate.
            - reference_indices: Array of reference pixel indices for extrapolation.
            - coefficients: Coefficients for linear extrapolation.
    """
    
    # Convert the mask to boolean
    binary_mask = mask.astype(bool)
    
    # Identify edge pixels (outside but adjacent to the mask) using binary dilation
    dilated_mask = binary_dilation(binary_mask)
    edge_pixels = np.where(dilated_mask & ~binary_mask)
    edge_pixels_linear = np.ravel_multi_index(edge_pixels, mask.shape)
    
    if debug:
        print(f"Found {len(edge_pixels[0])} edge pixels to extrapolate.")
        
        # Plot the original mask and the edge pixels
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(binary_mask, cmap='gray', interpolation='nearest')
        plt.title('Original Mask')
        
        plt.subplot(122)
        edge_mask = np.zeros_like(binary_mask)
        edge_mask[edge_pixels] = 1
        plt.imshow(binary_mask, cmap='gray', alpha=0.5, interpolation='nearest')
        plt.imshow(edge_mask, cmap='hot', alpha=0.5, interpolation='nearest')
        plt.title('Edge Pixels to Extrapolate (red)')
        plt.tight_layout()
        plt.show()
    
    # Preallocate arrays for reference indices and coefficients
    reference_indices = np.full((len(edge_pixels[0]), 8), -1, dtype=np.int32)
    coefficients = np.zeros((len(edge_pixels[0]), 8), dtype=np.float32)
    
    # Directions for extrapolation (y+1, y-1, x+1, x-1)
    directions = [
        (1, 0),  # y+1 (down)
        (-1, 0), # y-1 (up)
        (0, 1),  # x+1 (right)
        (0, -1)  # x-1 (left)
    ]
    
    # Iterate over each edge pixel
    problem_indices = []
    for i, (y, x) in enumerate(zip(*edge_pixels)):
        # Check if this pixel is in the debug list
        is_debug_pixel = False
        if debug_pixels is not None:
            for p in debug_pixels:
                if p[0] == y and p[1] == x:
                    is_debug_pixel = True
                    break
        
        valid_directions = 0
        
        if is_debug_pixel:
            print(f"\n[DEBUG] Detailed analysis for pixel [{y},{x}]:")
        
        # Examine the 4 directions
        for dir_idx, (dy, dx) in enumerate(directions):
            # Coordinates of reference points at distance 1 and 2
            y1, x1 = y + dy, x + dx
            y2, x2 = y + 2*dy, x + 2*dx

            # Check if the points are valid (inside the image and inside the mask)
            valid_ref1 = (0 <= y1 < mask.shape[0] and 
                          0 <= x1 < mask.shape[1] and 
                          binary_mask[y1, x1])
                          
            valid_ref2 = (0 <= y2 < mask.shape[0] and 
                          0 <= x2 < mask.shape[1] and 
                          binary_mask[y2, x2])

            if is_debug_pixel:
                print(f"  Direction {directions_labels[dir_idx]}: ")
                print(f"    Ref1 [{y1},{x1}] valid: {valid_ref1}")
                print(f"    Ref2 [{y2},{x2}] valid: {valid_ref2}")

            if valid_ref1:
                # Index of the first reference point (linear index)
                ref_idx1 = y1 * mask.shape[1] + x1
                reference_indices[i, 2*dir_idx] = ref_idx1

                if valid_ref2:
                    # Index of the second reference point (linear index)
                    ref_idx2 = y2 * mask.shape[1] + x2
                    reference_indices[i, 2*dir_idx + 1] = ref_idx2

                    # Coefficients for linear extrapolation: 2*P₁ - P₂
                    coefficients[i, 2*dir_idx] = 2.0
                    coefficients[i, 2*dir_idx + 1] = -1.0
                    valid_directions += 1

                    if is_debug_pixel:
                        print(f"    Using extrapolation: 2*{ref_idx1} - {ref_idx2}")
                else:
                    # If the second point is invalid, check if it's the only valid pixel
                    if valid_directions == 0:
                        coefficients[i, 2*dir_idx] = 1.0
                        valid_directions += 1
                        if is_debug_pixel:
                            print(f"    Using first ref value: {ref_idx1} (only valid pixel)")
                    else:
                        # Set coefficients to 0
                        coefficients[i, 2*dir_idx] = 0.0
                        coefficients[i, 2*dir_idx + 1] = 0.0
            else:
                # Set coefficients to 0 if the first reference is invalid
                coefficients[i, 2*dir_idx] = 0.0
                coefficients[i, 2*dir_idx + 1] = 0.0
        
        # Normalize coefficients based on the number of valid directions
        if valid_directions > 1:
            factor = 1.0 / valid_directions
            
            if is_debug_pixel:
                print(f"  Valid directions: {valid_directions}, factor: {factor}")
                print("  Coefficients before normalization:", coefficients[i])
            
            for dir_idx in range(4):
                if coefficients[i, 2*dir_idx] != 0:
                    coefficients[i, 2*dir_idx] *= factor
                    if coefficients[i, 2*dir_idx + 1] != 0:
                        coefficients[i, 2*dir_idx + 1] *= factor
            
            if is_debug_pixel:
                print("  Coefficients after normalization:", coefficients[i])
                problem_indices.append(i)

    if debug:
        print(f"Average valid directions per pixel: {np.sum(coefficients != 0) / (len(edge_pixels[0]) * 2):.2f}")
        
        # Display coefficient matrix for the first 10 pixels
        if len(edge_pixels[0]) >= 10:
            print("\nCoefficients for the first 10 pixels:")
            for i in range(min(10, len(edge_pixels[0]))):
                print(f"Pixel {i} ({edge_pixels[0][i]}, {edge_pixels[1][i]}): {coefficients[i]}")
                print(f"Indices: {reference_indices[i]}")
    
    return edge_pixels_linear, reference_indices, coefficients

def apply_extrapolation(data, edge_pixels, reference_indices, coefficients, debug=False, problem_indices=None):
    """
    Applies linear extrapolation to edge pixels using precalculated indices and coefficients.

    Parameters:
        data (ndarray): Input array to extrapolate.
        edge_pixels (ndarray): Linear indices of edge pixels to extrapolate.
        reference_indices (ndarray): Indices of reference pixels.
        coefficients (ndarray): Coefficients for linear extrapolation.
        debug (bool): If True, displays debug information.
        problem_indices (list): Indices of problematic pixels to analyze.

    Returns:
        ndarray: Array with extrapolated pixels.
    """
    # Create a copy of the input array
    result = data.copy()
    flat_result = result.ravel()
    flat_data = data.ravel()

    # Iterate over each edge pixel
    for i, edge_idx in enumerate(edge_pixels):
        is_problem_pixel = problem_indices is not None and i in problem_indices

        # Compute the 2D coordinates of the pixel
        edge_y = edge_idx // data.shape[1]
        edge_x = edge_idx % data.shape[1]

        if is_problem_pixel:
            print(f"\n[DEBUG] Calculating extrapolated value for pixel [{edge_y},{edge_x}]:")
            print(f"  Original value: {flat_data[edge_idx]}")

        # Initialize the extrapolated value
        extrap_value = 0.0

        # Sum contributions from all references
        for j in range(reference_indices.shape[1]):
            ref_idx = reference_indices[i, j]
            if ref_idx >= 0:  # If the index is valid
                ref_y = ref_idx // data.shape[1]
                ref_x = ref_idx % data.shape[1]
                contrib = coefficients[i, j] * flat_data[ref_idx]
                extrap_value += contrib

                if is_problem_pixel:
                    print(f"  Ref [{ref_y},{ref_x}] = {flat_data[ref_idx]} × {coefficients[i, j]:.4f} = {contrib:.4f}")

        # Assign the extrapolated value
        flat_result[edge_idx] = extrap_value

    return result


def shiftzoom_from_source_dm_params(source_pol_coo, source_height, dm_height, pixel_pitch):
    """
    Compute the shift and zoom parameters for a DM based on the source coordinates and heights.
    
    Parameters:
    - source_pol_coo: tuple, (radius, angle) in polar coordinates
    - source_height: float, height of the source
    - dm_height: float, height of the DM
    - pixel_pitch: float, pixel pitch in meters

    Returns:
    - shift: tuple, (x_shift, y_shift) in pixels
    - zoom: tuple, (x_zoom, y_zoom) magnification factors
    """

    arcsec2rad = np.pi/180/3600

    if np.isinf(source_height):
        mag_factor = 1.0
    else:
        mag_factor = source_height/(source_height-dm_height)
    source_rec_coo_asec = polar_to_xy(source_pol_coo[0],source_pol_coo[1]*np.pi/180)
    source_rec_coo_m = source_rec_coo_asec*dm_height*arcsec2rad
    # change sign to get the shift in the right direction considering the convention applied in rotshiftzoom_array
    source_rec_coo_pix = -1 * source_rec_coo_m / pixel_pitch

    shift = tuple(source_rec_coo_pix)
    zoom = (mag_factor, mag_factor)

    return shift, zoom

def rotshiftzoom_array_noaffine(input_array, dm_translation=(0.0, 0.0),  dm_rotation=0.0,   dm_magnification=(1.0, 1.0),
                                    wfs_translation=(0.0, 0.0), wfs_rotation=0.0, wfs_magnification=(1.0, 1.0), output_size=None):
    """
    This function applies magnification, rotation, shift and resize of a 2D or 3D numpy array
    
    Parameters:
    - input_array: numpy array, input data to be transformed
    - dm_translation: tuple, translation for DM (x, y)
    - dm_rotation: float, rotation angle for DM in degrees
    - dm_magnification: tuple, magnification factors for DM (x, y)
    - wfs_translation: tuple, translation for WFS (x, y)
    - wfs_rotation: float, rotation angle for WFS in degrees
    - wfs_magnification: tuple, magnification factors for WFS (x, y)
    - output_size: tuple, desired output size (height, width)

    Returns:
    - output: numpy array, transformed data
    """

    if np.isnan(input_array).any():
        np.nan_to_num(input_array, copy=False, nan=0.0, posinf=None, neginf=None)

    # Check if phase is 2D or 3D
    if len(input_array.shape) == 3:
        dm_translation_ = dm_translation + (0,)
        dm_magnification_ = dm_magnification + (1,)
        wfs_translation_ = wfs_translation + (0,)
        wfs_magnification_ = wfs_magnification + (1,)
    else:
        dm_translation_ = dm_translation
        dm_magnification_ = dm_magnification
        wfs_translation_ = wfs_translation
        wfs_magnification_ = wfs_magnification

    # resize
    if output_size == None:
        output_size = input_array.shape

    # (1) DM magnification
    if all(element == 1 for element in dm_magnification_):
        array_mag = input_array
    else:
        array_mag = zoom(input_array, dm_magnification_)

    # (2) DM rotation
    if dm_rotation == 0:
        array_rot = array_mag
    else:
        array_rot = rotate(array_mag, dm_rotation, axes=(1, 0), reshape=False)

    # (3) DM translation
    if all(element == 0 for element in dm_translation_):
        array_shi = array_rot
    else:
        array_shi = shift(array_rot, dm_translation_)

    # (4) WFS rotation
    if wfs_rotation == 0:
        array_rot = array_shi
    else:
        array_rot = rotate(array_shi, wfs_rotation, axes=(1, 0), reshape=False)

    # (5) WFS translation
    if all(element == 0 for element in wfs_translation_):
        array_shi = array_rot
    else:
        array_shi = shift(array_rot, wfs_translation_)

    # (6) WFS magnification
    if all(element == 1 for element in wfs_magnification_):
        array_mag = array_shi
    else:
        array_mag = zoom(array_shi, wfs_magnification_)

    if (array_mag.shape[0] > output_size[0]) | (array_mag.shape[1] > output_size[1]):
        # smaller output size
        if len(input_array.shape) == 3:
            output = array_mag[int(0.5*(array_mag.shape[0]-output_size[0])):int(0.5*(array_mag.shape[0]+output_size[0])), \
                               int(0.5*(array_mag.shape[1]-output_size[1])):int(0.5*(array_mag.shape[1]+output_size[1])),:]
        else:
            output = array_mag[int(0.5*(array_mag.shape[0]-output_size[0])):int(0.5*(array_mag.shape[0]+output_size[0])), \
                               int(0.5*(array_mag.shape[1]-output_size[1])):int(0.5*(array_mag.shape[1]+output_size[1]))]
    elif (array_mag.shape[0] < output_size[0]) | (array_mag.shape[1] < output_size[1]):
        # bigger output size
        if len(input_array.shape) == 3:
            output = np.zeros(output_size+(input_array.shape[2],))
            output[int(0.5*(output_size[0]-array_mag.shape[0])):int(0.5*(output_size[0]+array_mag.shape[0])), \
                   int(0.5*(output_size[1]-array_mag.shape[1])):int(0.5*(output_size[1]+array_mag.shape[1])),:] = array_mag
        else:
            output[int(0.5*(output_size[0]-array_mag.shape[0])):int(0.5*(output_size[0]+array_mag.shape[0])), \
                   int(0.5*(output_size[1]-array_mag.shape[1])):int(0.5*(output_size[1]+array_mag.shape[1]))] = array_mag
    else:
        output = array_mag

    return output

def rotshiftzoom_array(input_array, dm_translation=(0.0, 0.0), dm_rotation=0.0, dm_magnification=(1.0, 1.0),
                       wfs_translation=(0.0, 0.0), wfs_rotation=0.0, wfs_magnification=(1.0, 1.0), output_size=None):
    """
    This function applies magnification, rotation, shift and resize of a 2D or 3D numpy array using affine transformation.
    Rotation is applied in the same direction as the first function.

    Parameters:
    - input_array: numpy array, input data to be transformed
    - dm_translation: tuple, translation for DM (x, y)
    - dm_rotation: float, rotation angle for DM in degrees
    - dm_magnification: tuple, magnification factors for DM (x, y)
    - wfs_translation: tuple, translation for WFS (x, y)
    - wfs_rotation: float, rotation angle for WFS in degrees
    - wfs_magnification: tuple, magnification factors for WFS (x, y)
    - output_size: tuple, desired output size (height, width)

    Returns:
    - output: numpy array, transformed data
    """

    # Parameter handling: conversion of single values to tuples
    try:
        if not hasattr(dm_translation, '__len__') or len(dm_translation) != 2:
            dm_translation = (float(dm_translation), float(dm_translation))
    except (TypeError, ValueError):
        dm_translation = (0.0, 0.0)

    try:
        if not hasattr(wfs_translation, '__len__') or len(wfs_translation) != 2:
            wfs_translation = (float(wfs_translation), float(wfs_translation))
    except (TypeError, ValueError):
        wfs_translation = (0.0, 0.0)

    try:
        if not hasattr(dm_magnification, '__len__'):
            # If it is a single value, we create a tuple with two identical elements
            dm_magnification = (float(dm_magnification), float(dm_magnification))
        elif len(dm_magnification) != 2:
            # if it is a sequence but not of length 2
            dm_magnification = (float(dm_magnification[0]), float(dm_magnification[0]))
    except (TypeError, ValueError):
        dm_magnification = (1.0, 1.0)

    try:
        if not hasattr(wfs_magnification, '__len__'):
            # If it is a single value, we create a tuple with two identical elements
            wfs_magnification = (float(wfs_magnification), float(wfs_magnification))
        elif len(wfs_magnification) != 2:
            # if it is a sequence but not of length 2
            wfs_magnification = (float(wfs_magnification[0]), float(wfs_magnification[0]))
    except (TypeError, ValueError):
        wfs_magnification = (1.0, 1.0)

    if np.isnan(input_array).any():
        input_array = np.nan_to_num(input_array, copy=True, nan=0.0, posinf=None, neginf=None)

    # Check if array is 2D or 3D
    is_3d = len(input_array.shape) == 3

    # resize
    if output_size is None:
        output_size = input_array.shape[:2]  # Only take the first two dimensions

    # Center of the input array
    center = np.array(input_array.shape[:2]) / 2.0
    # Convert rotations to radians
    # Note: Inverting the sign of rotation to match the first function's direction
    dm_rot_rad = np.deg2rad(-dm_rotation)  # Negative sign to reverse direction
    wfs_rot_rad = np.deg2rad(-wfs_rotation)  # Negative sign to reverse direction

    # Initialize the output array
    if is_3d:
        output = np.zeros((output_size[0], output_size[1], input_array.shape[2]), dtype=input_array.dtype)
    else:
        output = np.zeros(output_size, dtype=input_array.dtype)

    # Create the transformation matrices
    # For DM transformation
    dm_scale_matrix = np.array([[1.0/dm_magnification[0], 0], [0, 1.0/dm_magnification[1]]])
    dm_rot_matrix = np.array([[np.cos(dm_rot_rad), -np.sin(dm_rot_rad)], [np.sin(dm_rot_rad), np.cos(dm_rot_rad)]])
    dm_matrix = np.dot(dm_rot_matrix, dm_scale_matrix)

    # For WFS transformation
    wfs_scale_matrix = np.array([[1.0/wfs_magnification[0], 0], [0, 1.0/wfs_magnification[1]]])
    wfs_rot_matrix = np.array([[np.cos(wfs_rot_rad), -np.sin(wfs_rot_rad)], [np.sin(wfs_rot_rad), np.cos(wfs_rot_rad)]])
    wfs_matrix = np.dot(wfs_rot_matrix, wfs_scale_matrix)

    # Combine transformations (first DM, then WFS)
    combined_matrix = np.dot(wfs_matrix, dm_matrix)

    # Calculate offset
    output_center = np.array(output_size) / 2.0
    offset = center - np.dot(combined_matrix, output_center) - np.dot(dm_matrix, dm_translation) - wfs_translation

    # Apply transformation
    if is_3d:
        # For 3D arrays, apply transformation to each slice
        for i in range(input_array.shape[2]):
            output[:, :, i] = affine_transform(
                input_array[:, :, i],
                combined_matrix,
                offset=offset,
                output_shape=output_size,
                order=1
            )
    else:
        # For 2D arrays
        output = affine_transform(
            input_array,
            combined_matrix,
            offset=offset,
            output_shape=output_size,
            order=1
        )

    return output

def projection_matrix(pup_diam_m, pup_mask, dm_array, dm_mask, base_inv_array, 
                      dm_height, dm_rotation, wfs_rotation, wfs_translation, wfs_magnification,
                      gs_pol_coo, gs_height, verbose=False, display=False, specula_convention=True):
    """
    Computes a projection matrix for DM modes onto a desired basis.

    Parameters:
    - pup_diam_m: float, size in m of the side of the pupil
    - pup_mask: numpy 2D array, pupil mask
    - dm_array: numpy 3D array, Deformable Mirror 2D shapes
    - dm_mask: numpy 2D array, DM mask
    - base_inv_array: numpy 2D array, inverted basis for projection (n_modes x n_valid_pixels)
    - dm_height: float, conjugation altitude of the Deformable Mirror
    - dm_rotation: float, rotation in deg of the Deformable Mirror with respect to the pupil
    - wfs_rotation: float, rotation of the wavefront sensor
    - wfs_translation: tuple, translation of the wavefront sensor
    - wfs_magnification: tuple, magnification of the wavefront sensor
    - gs_pol_coo: tuple, polar coordinates of the guide star radius in arcsec and angle in deg
    - gs_height: float, altitude of the guide star
    - verbose: bool, optional, display verbose output
    - display: bool, optional, display plots
    - specula_convention: bool, optional, use SPECULA convention

    Returns:
    - pm: numpy 2D array, projection matrix (n_base_modes x n_dm_modes)
    """

    pup_diam_pix = pup_mask.shape[0]
    pixel_pitch = pup_diam_m / pup_diam_pix
    dm_diam_pix = dm_mask.shape[0]
    if dm_mask.shape[0] != dm_array.shape[0]:
        raise ValueError('Error in input data, the dm and mask array must have the same dimensions.')

    pixel_pitch = pup_diam_m / pup_diam_pix

    dm_translation, dm_magnification = shiftzoom_from_source_dm_params(gs_pol_coo, gs_height, dm_height, pixel_pitch)
    output_size = (pup_diam_pix, pup_diam_pix)

    # Extraction of patch seen by GS and application of DM rotation
    trans_dm_array = rotshiftzoom_array(dm_array, 
                                       dm_translation=dm_translation, 
                                       dm_rotation=dm_rotation, 
                                       dm_magnification=dm_magnification,
                                       wfs_translation=wfs_translation, 
                                       wfs_rotation=wfs_rotation, 
                                       wfs_magnification=wfs_magnification,
                                       output_size=output_size)

    # Apply transformation to the DM mask
    trans_dm_mask = rotshiftzoom_array(dm_mask, 
                                      dm_translation=dm_translation, 
                                      dm_rotation=dm_rotation, 
                                      dm_magnification=dm_magnification,
                                      wfs_translation=(0,0), 
                                      wfs_rotation=0, 
                                      wfs_magnification=(1,1),
                                      output_size=output_size)
    trans_dm_mask[trans_dm_mask<0.5] = 0

    if np.max(trans_dm_mask) <= 0:
        raise ValueError('Error in input data, the rotated dm mask is empty.')

    # Apply transformation to the pupil mask
    trans_pup_mask = rotshiftzoom_array(pup_mask, 
                                       dm_translation=(0,0), 
                                       dm_rotation=0, 
                                       dm_magnification=(1,1),
                                       wfs_translation=wfs_translation, 
                                       wfs_rotation=wfs_rotation, 
                                       wfs_magnification=wfs_magnification,
                                       output_size=output_size)
    trans_pup_mask[trans_pup_mask<0.5] = 0

    if np.max(trans_pup_mask) <= 0:
        raise ValueError('Error in input data, the rotated pup mask is empty.')

    if verbose:
        print(f'DM rotation ({dm_rotation} deg), translation ({dm_translation} pixel), magnification ({dm_magnification})')
        print(f'WFS translation ({wfs_translation} pixel), wfs rotation ({wfs_rotation} deg), wfs magnification ({wfs_magnification})')
        print('done.')

    # Apply mask
    trans_dm_array = apply_mask(trans_dm_array, trans_dm_mask)
    if np.max(trans_dm_array) <= 0:
        raise ValueError('Error in input data, the rotated dm array is empty.')

    if verbose:
        print('Mask applied.')

    if specula_convention:
        # Transpose the DM array, mask and pupil mask to match the specula convention
        trans_dm_array = np.transpose(trans_dm_array, (1, 0, 2))
        trans_dm_mask = np.transpose(trans_dm_mask)
        trans_pup_mask = np.transpose(trans_pup_mask)

    # Apply pupil mask to the DM array
    trans_dm_array = apply_mask(trans_dm_array, trans_pup_mask)

    # Create mask for valid pixels (both in DM and pupil)
    valid_mask = trans_dm_mask * trans_pup_mask
    valid_pixels = valid_mask > 0.5

    # Extract valid pixels from dm_array
    n_valid_pixels = np.sum(valid_pixels)
    height, width, n_modes = trans_dm_array.shape
    dm_valid_values = np.zeros((n_valid_pixels, n_modes))

    for i in range(n_modes):
        dm_valid_values[:, i] = trans_dm_array[:, :, i][valid_pixels]

    # Perform matrix multiplication with base_inv_array to get projection coefficients
    projection = np.dot(base_inv_array, dm_valid_values)

    if verbose:
        print('Matrix multiplication done, projection shape:', projection.shape)

    if display:
        # Display valid pixels mask
        plt.figure()
        plt.imshow(valid_mask)
        plt.title('Valid pixels mask')
        plt.colorbar()
 
        # Display a couple of DM modes
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(trans_dm_array[:, :, 0], cmap='seismic')
        plt.title('DM Mode 0')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(trans_dm_array[:, :, 1], cmap='seismic')
        plt.title('DM Mode 1')
        plt.colorbar()

        # Display projection coefficients
        plt.figure()
        for i in range(min(5, projection.shape[0])):
            plt.plot(projection[i, :], label=f'Basis mode {i}')
        plt.legend()
        plt.title('Projection coefficients')
        plt.xlabel('DM mode index')
        plt.ylabel('Coefficient')
        plt.grid(True)
        plt.show()

    return projection

def interaction_matrix(pup_diam_m,pup_mask,dm_array,dm_mask,dm_height,dm_rotation,wfs_nsubaps,wfs_rotation,wfs_translation,wfs_magnification,
                       wfs_fov_arcsec,gs_pol_coo,gs_height,idx_valid_sa=None,verbose=False,display=False,specula_convention=True):
    """
    Computes a single interaction matrix.
    From Guido Agapito.

    Parameters:
    - pup_diam_m: float, size in m of the side of the pupil
    - pup_mask: numpy 2D array, mask
    - dm_array: numpy 3D array, Deformable Mirror 2D shapes
    - dm_mask: numpy 2D array, mask
    - dm_height: float, conjugation altitude of the Deformable Mirror
    - dm_rotation: float, rotation in deg of the Deformable Mirror with respect to the pupil
    - wfs_nsubaps: int, number of sub-aperture of the wavefront sensor
    - wfs_rotation
    - wfs_translation
    - wfs_magnification
    - wfs_fov_arcsec: float, field of view of the wavefront sensor in arcsec
    - gs_pol_coo: tuple, polar coordinates of the gudie star radius in arcsec and angle in deg
    - gs_height: float, altitude of the guide star
    - idx_valid_sa: numpy 1D array, indices of the valid sub-apertures
    - verbose, optional
    - display, optional
    - specula_convention, optional

    Returns:
    - im: numpy 2D array, set of signals
    """

    pup_diam_pix = pup_mask.shape[0]
    pixel_pitch = pup_diam_m/pup_diam_pix
    dm_diam_pix = dm_mask.shape[0]
    if dm_mask.shape[0] != dm_array.shape[0]:
        raise ValueError('Error in input data, the dm and mask array must have the same dimensions.')

    pixel_pitch = pup_diam_m / pup_diam_pix

    dm_translation, dm_magnification = shiftzoom_from_source_dm_params(gs_pol_coo, gs_height, dm_height, pixel_pitch)
    output_size = (pup_diam_pix,pup_diam_pix)

    #Extraction of patch seen by GS and application of DM rotation
    trans_dm_array = rotshiftzoom_array(dm_array, dm_translation=dm_translation, dm_rotation=dm_rotation, dm_magnification=dm_magnification,
                                        wfs_translation=wfs_translation, wfs_rotation=wfs_rotation, wfs_magnification=wfs_magnification,
                                        output_size=output_size)
    # apply transformation to the DM mask
    trans_dm_mask  = rotshiftzoom_array(dm_mask, dm_translation=dm_translation, dm_rotation=dm_rotation, dm_magnification=dm_magnification,
                                        wfs_translation=(0,0), wfs_rotation=0, wfs_magnification=(1,1),
                                        output_size=output_size)
    trans_dm_mask[trans_dm_mask<0.5] = 0
    if np.max(trans_dm_mask) <= 0:
        raise ValueError('Error in input data, the rotated dm mask is empty.')

    # apply transformation to the pupil mask
    trans_pup_mask  = rotshiftzoom_array(pup_mask, dm_translation=(0,0), dm_rotation=0, dm_magnification=(1,1),
                                        wfs_translation=wfs_translation, wfs_rotation=wfs_rotation, wfs_magnification=wfs_magnification,
                                        output_size=output_size)
    trans_pup_mask[trans_pup_mask<0.5] = 0

    if np.max(trans_pup_mask) <= 0:
        raise ValueError('Error in input data, the rotated pup mask is empty.')

    if verbose:
        print(f'DM rotation ({dm_rotation} deg), translation ({dm_translation} pixel), magnification ({dm_magnification})')
        print(f'WFS translation ({wfs_translation} pixel), wfs rotation ({wfs_rotation} deg), wfs magnification ({wfs_magnification})')
        print('done.')

    # apply mask
    trans_dm_array = apply_mask(trans_dm_array,trans_dm_mask)
    if np.max(trans_dm_array) <= 0:
        raise ValueError('Error in input data, the rotated dm array is empty.')

    if verbose:
        print('Mask applied.')

    if specula_convention:
        # transpose the DM array, mask and pupil mask to match the specula convention
        trans_dm_array = np.transpose(trans_dm_array, (1, 0, 2))
        trans_dm_mask = np.transpose(trans_dm_mask)
        trans_pup_mask = np.transpose(trans_pup_mask)

    # Derivative od DM modes shape
    der_dx, der_dy = compute_derivatives_with_extrapolation(trans_dm_array,mask=trans_dm_mask)

    if verbose:
        print('Derivatives done, size of der_dx and der_dy:', der_dx.shape, der_dy.shape)

    # estimate an array proportional to flux per sub-aperture from the mask   
    if np.isnan(trans_pup_mask).any():
        np.nan_to_num(trans_pup_mask, copy=False, nan=0.0, posinf=None, neginf=None)
    if np.isnan(trans_dm_mask).any():
        np.nan_to_num(trans_dm_mask, copy=False, nan=0.0, posinf=None, neginf=None)

    pup_mask_sa = rebin(trans_pup_mask, (wfs_nsubaps,wfs_nsubaps), method='sum')
    pup_mask_sa = pup_mask_sa * 1/np.max(pup_mask_sa)

    dm_mask_sa = rebin(trans_dm_mask, (wfs_nsubaps,wfs_nsubaps), method='sum')
    if np.max(dm_mask_sa) <= 0:
        raise ValueError('Error in input data, the dm mask is empty.')
    dm_mask_sa = dm_mask_sa * 1/np.max(dm_mask_sa)

    # rebin the array to get the correct signal size
    if np.isnan(der_dx).any():
        np.nan_to_num(der_dx, copy=False, nan=0.0, posinf=None, neginf=None)
    if np.isnan(der_dy).any():
        np.nan_to_num(der_dy, copy=False, nan=0.0, posinf=None, neginf=None)

    # apply pup mask
    der_dx = apply_mask(der_dx,trans_pup_mask)
    der_dy = apply_mask(der_dy,trans_pup_mask)

    scale_factor = (der_dx.shape[0]/wfs_nsubaps)/np.median(rebin(trans_pup_mask, (wfs_nsubaps,wfs_nsubaps), method='average'))
    WFS_signal_x = rebin(der_dx, (wfs_nsubaps,wfs_nsubaps), method='average')*scale_factor
    WFS_signal_y = rebin(der_dy, (wfs_nsubaps,wfs_nsubaps), method='average')*scale_factor

    debug_rebin_plot = False
    if debug_rebin_plot:
        # compare derivative before and after rebining
        idx_mode = 2 # you can change this value as you wish

        fig, axs = plt.subplots(3, 2, figsize=(12, 14))

        # first line: DM shape
        im0 = axs[0, 0].imshow(trans_dm_array[:, :, idx_mode], cmap='seismic')
        axs[0, 0].set_title(f'DM shape (mode {idx_mode})')
        fig.colorbar(im0, ax=axs[0, 0])
        axs[0, 1].axis('off') # empty cell

        # second line: Derivate
        im1 = axs[1, 0].imshow(der_dx[:, :, idx_mode], cmap='seismic')
        axs[1, 0].set_title(f'Derivative dx (mode {idx_mode})')
        fig.colorbar(im1, ax=axs[1, 0])
        im2 = axs[1, 1].imshow(der_dy[:, :, idx_mode], cmap='seismic')
        axs[1, 1].set_title(f'Derivative dy (mode {idx_mode})')
        fig.colorbar(im2, ax=axs[1, 1])

        # Third line: WFS signals
        im3 = axs[2, 0].imshow(WFS_signal_x[:, :, idx_mode], cmap='seismic')
        axs[2, 0].set_title(f'WFS signal x (mode {idx_mode})')
        fig.colorbar(im3, ax=axs[2, 0])
        im4 = axs[2, 1].imshow(WFS_signal_y[:, :, idx_mode], cmap='seismic')
        axs[2, 1].set_title(f'WFS signal y (mode {idx_mode})')
        fig.colorbar(im4, ax=axs[2, 1])

        fig.suptitle(f'DM, derivatives, and WFS signals (mode {idx_mode})')
        plt.tight_layout()
        plt.show()

    if verbose:
        print('Rebin done, size of WFS_signal_x and WFS_signal_y:', WFS_signal_x.shape, WFS_signal_y.shape)

    # normalize by dm_mask_sa to get the correct value at the edge of the dm
    # because at the edge the average of the rebin is done with pixel outside the
    # dm that have 0 values
    dm_mask_sa[dm_mask_sa<0.5] = 0
    WFS_signal_x = apply_mask(WFS_signal_x,dm_mask_sa,norm=True)
    WFS_signal_y = apply_mask(WFS_signal_y,dm_mask_sa,norm=True)
    
    # set to zero the signal outside the pupil
    pup_mask_sa[pup_mask_sa<0.5] = 0
    pup_mask_sa[pup_mask_sa>=0.5] = 1
    WFS_signal_x = apply_mask(WFS_signal_x,pup_mask_sa,norm=False)
    WFS_signal_y = apply_mask(WFS_signal_y,pup_mask_sa,norm=False)

    if verbose:
        print('Mask applied.')

    WFS_signal_x_2D = WFS_signal_x.reshape((-1,WFS_signal_x.shape[2]))
    WFS_signal_y_2D = WFS_signal_y.reshape((-1,WFS_signal_y.shape[2]))

    if idx_valid_sa is not None:

        if specula_convention:
            # transpose idx_valid_sa to match the specula convention
            sa2D = np.zeros((wfs_nsubaps,wfs_nsubaps))
            sa2D[idx_valid_sa[:,0], idx_valid_sa[:,1]] = 1
            sa2D = np.transpose(sa2D)
            idx_temp = np.where(sa2D>0)
            idx_valid_sa_new = idx_valid_sa*0.
            idx_valid_sa_new[:,0] = idx_temp[0]
            idx_valid_sa_new[:,1] = idx_temp[1]
        else:
            idx_valid_sa_new = idx_valid_sa

        if len(idx_valid_sa_new.shape) > 1 and idx_valid_sa_new.shape[1] == 2:
            # Convert 2D coordinates [y,x] to linear indices
            # Formula: linear_index = y * width + x
            width = wfs_nsubaps  # Width of the original 2D array
            linear_indices = idx_valid_sa_new[:,0] * width + idx_valid_sa_new[:,1]

            # Use these linear indices to select elements from flattened arrays
            WFS_signal_x_2D = WFS_signal_x_2D[linear_indices.astype(int),:]
            WFS_signal_y_2D = WFS_signal_y_2D[linear_indices.astype(int),:]
        else:
            # Use 1D array directly
            WFS_signal_x_2D = WFS_signal_x_2D[idx_valid_sa_new.astype(int),:]
            WFS_signal_y_2D = WFS_signal_y_2D[idx_valid_sa_new.astype(int),:]
        if verbose:
            print('Indices selected.')

    if specula_convention:
        im = np.concatenate((WFS_signal_y_2D, WFS_signal_x_2D))
    else:
        im = np.concatenate((WFS_signal_x_2D, WFS_signal_y_2D))

    # Here we consider that tilt give a 4nm/SA derivative
    # Conversion from 4nm tilt derivative to arcsec
    coeff = 1e-9/(pup_diam_m/wfs_nsubaps) * 206265
    # Conversion from arcsec to slope
    coeff *= 1/(0.5 * wfs_fov_arcsec)
    im = im * coeff

    if verbose:
        print('WFS signals reformed, IM size is:', im.shape)

    if display:
        fig, _ = plt.subplots()
        plt.imshow(pup_mask_sa)
        plt.title('Pupil masks rebinned on WFS sub-apertures')
        plt.colorbar()

        idx_plot = [2,5]

        fig, axs = plt.subplots(2,2)
        im3 = axs[0,0].imshow(trans_dm_array[:,:,idx_plot[0]], cmap='seismic')
        im3 = axs[0,1].imshow(trans_dm_array[:,:,idx_plot[0]], cmap='seismic')
        im3 = axs[1,0].imshow(trans_dm_array[:,:,idx_plot[1]], cmap='seismic')
        im3 = axs[1,1].imshow(trans_dm_array[:,:,idx_plot[1]], cmap='seismic')
        fig.suptitle('DM shapes seen on the WFS direction (idx {idx_plot[0]} and {idx_plot[1]})')
        fig.colorbar(im3, ax=axs.ravel().tolist(),fraction=0.02)

        fig, axs = plt.subplots(2,2)
        im4 = axs[0,0].imshow(der_dx[:,:,idx_plot[0]], cmap='seismic')
        im4 = axs[0,1].imshow(der_dy[:,:,idx_plot[0]], cmap='seismic')
        im4 = axs[1,0].imshow(der_dx[:,:,idx_plot[1]], cmap='seismic')
        im4 = axs[1,1].imshow(der_dy[:,:,idx_plot[1]], cmap='seismic')
        fig.suptitle('X and Y derivative of DM shapes seen on the WFS direction (idx {idx_plot[0]} and {idx_plot[1]})')
        fig.colorbar(im4, ax=axs.ravel().tolist(),fraction=0.02)

        fig, axs = plt.subplots(2,2)
        im5 = axs[0,0].imshow(WFS_signal_x[:,:,idx_plot[0]], cmap='seismic')
        im5 = axs[0,1].imshow(WFS_signal_y[:,:,idx_plot[0]], cmap='seismic')
        im5 = axs[1,0].imshow(WFS_signal_x[:,:,idx_plot[1]], cmap='seismic')
        im5 = axs[1,1].imshow(WFS_signal_y[:,:,idx_plot[1]], cmap='seismic')
        fig.suptitle('X and Y WFS signals')
        fig.colorbar(im5, ax=axs.ravel().tolist(),fraction=0.02)
        plt.show()

    return im