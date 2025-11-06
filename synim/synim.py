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
            temp = data[:,:,i].copy()
            data[:,:,i] = apply_extrapolation(data[:,:,i], edge_pixels, reference_indices, coefficients,
                                              debug=True, problem_indices=None)
            debug_extrapolation = False
            if i == 0 and debug_extrapolation:
                plt.figure(figsize=(8, 6))
                plt.imshow(temp, cmap='seismic', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Original data slice {i}')
                plt.figure(figsize=(8, 6))
                plt.imshow(data[:,:,i], cmap='seismic', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Extrapolated data slice {i}')
                plt.figure(figsize=(8, 6))
                plt.imshow(data[:,:,i] - temp, cmap='seismic', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Difference after extrapolation for slice {i}')
                plt.show()

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
        print(f"Average valid directions per pixel:"
              f" {np.sum(coefficients != 0) / (len(edge_pixels[0]) * 2):.2f}")

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

    if debug:
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
                        print(f"  Ref [{ref_y},{ref_x}] = {flat_data[ref_idx]} ×"
                              f"{coefficients[i, j]:.4f} = {contrib:.4f}")

            # Assign the extrapolated value
            flat_result[edge_idx] = extrap_value
    else:       
        # Create a mask for valid reference indices (>= 0)
        valid_ref_mask = reference_indices >= 0

        # Replace invalid indices with 0 to avoid indexing errors
        safe_ref_indices = np.where(valid_ref_mask, reference_indices, 0)

        # Get data values for all reference indices at once
        ref_data = flat_data[safe_ref_indices]  # Shape: (n_valid_edges, 8)

        # Zero out contributions from invalid references
        masked_coeffs = np.where(valid_ref_mask, coefficients, 0.0)

        # Compute all contributions at once and sum across reference positions
        contributions = masked_coeffs * ref_data  # Element-wise multiplication
        extrap_values = np.sum(contributions, axis=1)  # Sum across reference positions

        # Assign extrapolated values to edge pixels
        flat_result[edge_pixels] = extrap_values

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
        output = np.zeros((output_size[0], output_size[1], input_array.shape[2]),
                          dtype=input_array.dtype)
    else:
        output = np.zeros(output_size, dtype=input_array.dtype)

    # Create the transformation matrices
    # For DM transformation
    dm_scale_matrix = np.array(
        [[1.0/dm_magnification[0], 0], [0, 1.0/dm_magnification[1]]]
    )
    dm_rot_matrix = np.array(
        [[np.cos(dm_rot_rad), -np.sin(dm_rot_rad)], [np.sin(dm_rot_rad), np.cos(dm_rot_rad)]]
    )
    dm_matrix = np.dot(dm_rot_matrix, dm_scale_matrix)

    # For WFS transformation
    wfs_scale_matrix = np.array(
        [[1.0/wfs_magnification[0], 0], [0, 1.0/wfs_magnification[1]]]
    )
    wfs_rot_matrix = np.array(
        [[np.cos(wfs_rot_rad), -np.sin(wfs_rot_rad)],
         [np.sin(wfs_rot_rad), np.cos(wfs_rot_rad)]]
    )
    wfs_matrix = np.dot(wfs_rot_matrix, wfs_scale_matrix)

    # Combine transformations (first DM, then WFS)
    combined_matrix = np.dot(wfs_matrix, dm_matrix)

    # For 3D arrays, extend the transformation matrix to 3x3
    if is_3d:
        # Create a 3x3 identity matrix and insert the 2x2 transformation in the top-left
        combined_matrix_3d = np.eye(3)
        combined_matrix_3d[:2, :2] = combined_matrix
        combined_matrix = combined_matrix_3d

    # Calculate offset
    output_center = np.array(output_size) / 2.0
    if is_3d:
        # For 3D, calculate offset only for the first two dimensions
        offset_2d = center[:2] - np.dot(combined_matrix[:2, :2], output_center) \
            - np.dot(dm_matrix, dm_translation) - wfs_translation
        offset = np.array([offset_2d[0], offset_2d[1], 0])
    else:
        offset = center - np.dot(combined_matrix, output_center) \
            - np.dot(dm_matrix, dm_translation) - wfs_translation

    # Apply transformation
    output = affine_transform(
        input_array,
        combined_matrix,
        offset=offset,
        output_shape=output_size if not is_3d else output_size + (input_array.shape[2],),
        order=1
    )

    return output


def _has_transformations(rotation, translation, magnification):
    """
    Helper function to check if there are any non-trivial transformations.
    
    Returns:
    - bool: True if there are transformations, False otherwise
    """
    # Check rotation
    has_rotation = rotation != 0.0

    # Check translation
    if hasattr(translation, '__len__'):
        has_translation = not all(t == 0.0 for t in translation)
    else:
        has_translation = translation != 0.0

    # Check magnification
    if hasattr(magnification, '__len__'):
        has_magnification = not all(m == 1.0 for m in magnification)
    else:
        has_magnification = magnification != 1.0

    return has_rotation or has_translation or has_magnification


def apply_dm_transformations_separated(pup_diam_m, pup_mask, dm_array, dm_mask, 
                                       dm_height, dm_rotation,
                                       gs_pol_coo, gs_height, 
                                       verbose=False, specula_convention=True):
    """
    Apply ONLY DM transformations (for separated workflow).
    Returns derivatives that need WFS transformations applied separately.
    """

    if specula_convention:
        dm_array = np.transpose(dm_array, (1, 0, 2))
        dm_mask = np.transpose(dm_mask)
        pup_mask = np.transpose(pup_mask)

    pup_diam_pix = pup_mask.shape[0]
    pixel_pitch = pup_diam_m / pup_diam_pix

    if dm_mask.shape[0] != dm_array.shape[0]:
        raise ValueError('DM and mask arrays must have the same dimensions.')

    dm_translation, dm_magnification = shiftzoom_from_source_dm_params(
        gs_pol_coo, gs_height, dm_height, pixel_pitch
    )
    output_size = (pup_diam_pix, pup_diam_pix)

    if verbose:
        print(f'DM transformations (separated):')
        print(f'  Translation: {dm_translation} pixels')
        print(f'  Rotation: {dm_rotation} deg')
        print(f'  Magnification: {dm_magnification}')

    # Apply ONLY DM transformations
    trans_dm_array = rotshiftzoom_array(
        dm_array,
        dm_translation=dm_translation,
        dm_rotation=dm_rotation,
        dm_magnification=dm_magnification,
        wfs_translation=(0, 0),
        wfs_rotation=0,
        wfs_magnification=(1, 1),
        output_size=output_size
    )

    trans_dm_mask = rotshiftzoom_array(
        dm_mask,
        dm_translation=dm_translation,
        dm_rotation=dm_rotation,
        dm_magnification=dm_magnification,
        wfs_translation=(0, 0),
        wfs_rotation=0,
        wfs_magnification=(1, 1),
        output_size=output_size
    )
    trans_dm_mask[trans_dm_mask < 0.5] = 0

    if np.max(trans_dm_mask) <= 0:
        raise ValueError('Transformed DM mask is empty.')

    trans_dm_array = apply_mask(trans_dm_array, trans_dm_mask)

    # Compute derivatives on DM-transformed array
    derivatives_x, derivatives_y = compute_derivatives_with_extrapolation(
        trans_dm_array, mask=trans_dm_mask
    )

    if verbose:
        print(f'  ✓ DM array transformed, shape: {trans_dm_array.shape}')
        print(f'  ✓ Derivatives computed')

    # Return the ORIGINAL pupil mask (not transformed), so WFS transformations can be applied later
    return trans_dm_array, trans_dm_mask, pup_mask, derivatives_x, derivatives_y


def apply_dm_transformations_combined(pup_diam_m, pup_mask, dm_array, dm_mask, 
                                      dm_height, dm_rotation,
                                      wfs_rotation, wfs_translation, wfs_magnification,
                                      gs_pol_coo, gs_height, 
                                      verbose=False, specula_convention=True):
    """
    Apply DM and WFS transformations COMBINED (single interpolation step).
    This avoids cumulative interpolation errors when both DM and WFS have rotations.
    """

    if specula_convention:
        dm_array = np.transpose(dm_array, (1, 0, 2))
        dm_mask = np.transpose(dm_mask)
        pup_mask = np.transpose(pup_mask)

    pup_diam_pix = pup_mask.shape[0]
    pixel_pitch = pup_diam_m / pup_diam_pix

    if dm_mask.shape[0] != dm_array.shape[0]:
        raise ValueError('DM and mask arrays must have the same dimensions.')

    dm_translation, dm_magnification = shiftzoom_from_source_dm_params(
        gs_pol_coo, gs_height, dm_height, pixel_pitch
    )
    output_size = (pup_diam_pix, pup_diam_pix)

    if verbose:
        print(f'Combined DM+WFS transformations:')
        print(f'  DM translation: {dm_translation} pixels')
        print(f'  DM rotation: {dm_rotation} deg')
        print(f'  DM magnification: {dm_magnification}')
        print(f'  WFS translation: {wfs_translation} pixels')
        print(f'  WFS rotation: {wfs_rotation} deg')
        print(f'  WFS magnification: {wfs_magnification}')

    # Apply ALL transformations in one step
    trans_dm_array = rotshiftzoom_array(
        dm_array,
        dm_translation=dm_translation,
        dm_rotation=dm_rotation,
        dm_magnification=dm_magnification,
        wfs_translation=wfs_translation,  # Include WFS
        wfs_rotation=wfs_rotation,
        wfs_magnification=wfs_magnification,
        output_size=output_size
    )

    # DM mask (only DM transformations)
    trans_dm_mask = rotshiftzoom_array(
        dm_mask,
        dm_translation=dm_translation,
        dm_rotation=dm_rotation,
        dm_magnification=dm_magnification,
        wfs_translation=(0, 0),
        wfs_rotation=0,
        wfs_magnification=(1, 1),
        output_size=output_size
    )
    trans_dm_mask[trans_dm_mask < 0.5] = 0

    # Pupil mask (only WFS transformations)
    trans_pup_mask = rotshiftzoom_array(
        pup_mask,
        dm_translation=(0, 0),
        dm_rotation=0,
        dm_magnification=(1, 1),
        wfs_translation=wfs_translation,
        wfs_rotation=wfs_rotation,
        wfs_magnification=wfs_magnification,
        output_size=output_size
    )
    trans_pup_mask[trans_pup_mask < 0.5] = 0

    if np.max(trans_dm_mask) <= 0:
        raise ValueError('Transformed DM mask is empty.')
    if np.max(trans_pup_mask) <= 0:
        raise ValueError('Transformed pupil mask is empty.')

    trans_dm_array = apply_mask(trans_dm_array, trans_dm_mask)

    # Compute derivatives on already-transformed array
    derivatives_x, derivatives_y = compute_derivatives_with_extrapolation(
        trans_dm_array, mask=trans_dm_mask
    )

    if verbose:
        print(f'  ✓ Combined transformation applied, shape: {trans_dm_array.shape}')
        print(f'  ✓ Derivatives computed')

    return trans_dm_array, trans_dm_mask, trans_pup_mask, derivatives_x, derivatives_y


def apply_wfs_transformations_separated(derivatives_x, derivatives_y, pup_mask, dm_mask,
                                        wfs_nsubaps, wfs_rotation, wfs_translation, wfs_magnification,
                                        wfs_fov_arcsec, pup_diam_m, idx_valid_sa=None,
                                        verbose=False, specula_convention=True):
    """
    Apply WFS transformations to derivatives (for separated workflow).
    """

    output_size = pup_mask.shape

    # Transform pupil mask
    trans_pup_mask = rotshiftzoom_array(
        pup_mask,
        dm_translation=(0, 0),
        dm_rotation=0,
        dm_magnification=(1, 1),
        wfs_translation=wfs_translation,
        wfs_rotation=wfs_rotation,
        wfs_magnification=wfs_magnification,
        output_size=output_size
    )
    trans_pup_mask[trans_pup_mask < 0.5] = 0

    if np.max(trans_pup_mask) <= 0:
        raise ValueError('Transformed pupil mask is empty.')

    if verbose:
        print(f'WFS transformations (separated):')
        print(f'  Translation: {wfs_translation} pixels')
        print(f'  Rotation: {wfs_rotation} deg')
        print(f'  Magnification: {wfs_magnification}')

    # Transform derivatives
    trans_der_x = rotshiftzoom_array(
        derivatives_x,
        dm_translation=(0, 0),
        dm_rotation=0,
        dm_magnification=(1, 1),
        wfs_translation=wfs_translation,
        wfs_rotation=wfs_rotation,
        wfs_magnification=wfs_magnification,
        output_size=output_size
    )

    trans_der_y = rotshiftzoom_array(
        derivatives_y,
        dm_translation=(0, 0),
        dm_rotation=0,
        dm_magnification=(1, 1),
        wfs_translation=wfs_translation,
        wfs_rotation=wfs_rotation,
        wfs_magnification=wfs_magnification,
        output_size=output_size
    )

    # Continue with rebinning and slope computation
    return _compute_slopes_from_derivatives(
        trans_der_x, trans_der_y, trans_pup_mask, dm_mask,
        wfs_nsubaps, wfs_fov_arcsec, pup_diam_m, idx_valid_sa,
        verbose, specula_convention
    )


def apply_wfs_transformations_combined(derivatives_x, derivatives_y, trans_pup_mask, dm_mask,
                                       wfs_nsubaps, wfs_fov_arcsec, pup_diam_m, idx_valid_sa=None,
                                       verbose=False, specula_convention=True):
    """
    Compute slopes from pre-transformed derivatives (for combined workflow).
    No additional transformations needed.
    """

    # Derivatives are already transformed - just compute slopes
    return _compute_slopes_from_derivatives(
        derivatives_x, derivatives_y, trans_pup_mask, dm_mask,
        wfs_nsubaps, wfs_fov_arcsec, pup_diam_m, idx_valid_sa,
        verbose, specula_convention
    )


def _compute_slopes_from_derivatives(derivatives_x, derivatives_y, pup_mask, dm_mask,
                                     wfs_nsubaps, wfs_fov_arcsec, pup_diam_m, idx_valid_sa,
                                     verbose, specula_convention):
    """
    Common function to compute slopes from derivatives.
    Used by both separated and combined workflows.
    """

    # Clean up masks
    if np.isnan(pup_mask).any():
        np.nan_to_num(pup_mask, copy=False, nan=0.0)
    if np.isnan(dm_mask).any():
        np.nan_to_num(dm_mask, copy=False, nan=0.0)

    # Rebin masks to WFS resolution
    pup_mask_sa = rebin(pup_mask, (wfs_nsubaps, wfs_nsubaps), method='sum')
    pup_mask_sa = pup_mask_sa / np.max(pup_mask_sa) if np.max(pup_mask_sa) > 0 else pup_mask_sa

    dm_mask_sa = rebin(dm_mask, (wfs_nsubaps, wfs_nsubaps), method='sum')
    if np.max(dm_mask_sa) <= 0:
        raise ValueError('DM mask is empty after rebinning.')
    dm_mask_sa = dm_mask_sa / np.max(dm_mask_sa)

    # Clean derivatives
    if np.isnan(derivatives_x).any():
        np.nan_to_num(derivatives_x, copy=False, nan=0.0)
    if np.isnan(derivatives_y).any():
        np.nan_to_num(derivatives_y, copy=False, nan=0.0)

    # Apply pupil mask
    trans_der_x = apply_mask(derivatives_x, pup_mask, fill_value=np.nan)
    trans_der_y = apply_mask(derivatives_y, pup_mask, fill_value=np.nan)

    # Rebin derivatives
    scale_factor = (trans_der_x.shape[0] / wfs_nsubaps) / \
                   np.median(rebin(pup_mask, (wfs_nsubaps, wfs_nsubaps), method='average'))

    wfs_signal_x = rebin(trans_der_x, (wfs_nsubaps, wfs_nsubaps), method='nanmean') * scale_factor
    wfs_signal_y = rebin(trans_der_y, (wfs_nsubaps, wfs_nsubaps), method='nanmean') * scale_factor

    # Combined mask
    combined_mask_sa = (dm_mask_sa > 0.0) & (pup_mask_sa > 0.0)

    # Apply mask
    wfs_signal_x = apply_mask(wfs_signal_x, combined_mask_sa, fill_value=0)
    wfs_signal_y = apply_mask(wfs_signal_y, combined_mask_sa, fill_value=0)

    # Reshape
    wfs_signal_x_2D = wfs_signal_x.reshape((-1, wfs_signal_x.shape[2]))
    wfs_signal_y_2D = wfs_signal_y.reshape((-1, wfs_signal_y.shape[2]))

    # Select valid subapertures
    if idx_valid_sa is not None:
        if specula_convention and len(idx_valid_sa.shape) > 1 and idx_valid_sa.shape[1] == 2:
            sa2D = np.zeros((wfs_nsubaps, wfs_nsubaps))
            sa2D[idx_valid_sa[:, 0], idx_valid_sa[:, 1]] = 1
            sa2D = np.transpose(sa2D)
            idx_temp = np.where(sa2D > 0)
            idx_valid_sa_new = np.zeros_like(idx_valid_sa)
            idx_valid_sa_new[:, 0] = idx_temp[0]
            idx_valid_sa_new[:, 1] = idx_temp[1]
        else:
            idx_valid_sa_new = idx_valid_sa

        if len(idx_valid_sa_new.shape) > 1 and idx_valid_sa_new.shape[1] == 2:
            width = wfs_nsubaps
            linear_indices = idx_valid_sa_new[:, 0] * width + idx_valid_sa_new[:, 1]
            wfs_signal_x_2D = wfs_signal_x_2D[linear_indices.astype(int), :]
            wfs_signal_y_2D = wfs_signal_y_2D[linear_indices.astype(int), :]
        else:
            wfs_signal_x_2D = wfs_signal_x_2D[idx_valid_sa_new.astype(int), :]
            wfs_signal_y_2D = wfs_signal_y_2D[idx_valid_sa_new.astype(int), :]

    # Concatenate
    if specula_convention:
        im = np.concatenate((wfs_signal_y_2D, wfs_signal_x_2D))
    else:
        im = np.concatenate((wfs_signal_x_2D, wfs_signal_y_2D))

    # Convert to slope units
    pup_diam_pix = pup_mask.shape[0]
    pixel_pitch = pup_diam_m / pup_diam_pix
    coeff = 1e-9 / (pup_diam_m / wfs_nsubaps) * 206265
    coeff *= 1 / (0.5 * wfs_fov_arcsec)
    im = im * coeff

    if verbose:
        print(f'  ✓ Slopes computed, shape: {im.shape}')

    return im


def interaction_matrix(pup_diam_m, pup_mask, dm_array, dm_mask, dm_height, dm_rotation,
                       wfs_nsubaps, wfs_rotation, wfs_translation, wfs_magnification,
                       wfs_fov_arcsec, gs_pol_coo, gs_height, idx_valid_sa=None,
                       verbose=False, display=False, specula_convention=True):
    """
    Computes interaction matrix using intelligent workflow selection.
    
    Automatically chooses between:
    - Separated workflow: When transformations are only in DM OR WFS (2 interpolation steps)
    - Combined workflow: When both DM and WFS have transformations (1 interpolation step)
    """

    # Detect which transformations are present
    has_dm_transform = _has_transformations(dm_rotation, (0, 0), (1, 1)) or \
                       gs_pol_coo != (0, 0) or dm_height != 0
    has_wfs_transform = _has_transformations(wfs_rotation, wfs_translation, wfs_magnification)

    # Choose workflow
    use_combined = has_dm_transform and has_wfs_transform

    if verbose:
        print(f"\n{'='*60}")
        print(f"Interaction Matrix Computation")
        print(f"{'='*60}")
        print(f"DM transformations: {has_dm_transform}")
        print(f"WFS transformations: {has_wfs_transform}")
        print(f"Using {'COMBINED' if use_combined else 'SEPARATED'} workflow")
        print(f"{'='*60}\n")

    if use_combined:
        # Combined workflow: single interpolation
        trans_dm_array, trans_dm_mask, trans_pup_mask, derivatives_x, derivatives_y = \
            apply_dm_transformations_combined(
                pup_diam_m, pup_mask, dm_array, dm_mask,
                dm_height, dm_rotation,
                wfs_rotation, wfs_translation, wfs_magnification,
                gs_pol_coo, gs_height,
                verbose=verbose, specula_convention=specula_convention
            )

        im = apply_wfs_transformations_combined(
            derivatives_x, derivatives_y, trans_pup_mask, trans_dm_mask,
            wfs_nsubaps, wfs_fov_arcsec, pup_diam_m, idx_valid_sa=idx_valid_sa,
            verbose=verbose, specula_convention=specula_convention
        )
    else:
        # Separated workflow: two interpolation steps (more flexible)
        trans_dm_array, trans_dm_mask, pup_mask_conv, derivatives_x, derivatives_y = \
            apply_dm_transformations_separated(
                pup_diam_m, pup_mask, dm_array, dm_mask,
                dm_height, dm_rotation,
                gs_pol_coo, gs_height,
                verbose=verbose, specula_convention=specula_convention
            )

        im = apply_wfs_transformations_separated(
            derivatives_x, derivatives_y, pup_mask_conv, trans_dm_mask,
            wfs_nsubaps, wfs_rotation, wfs_translation, wfs_magnification,
            wfs_fov_arcsec, pup_diam_m, idx_valid_sa=idx_valid_sa,
            verbose=verbose, specula_convention=specula_convention
        )

    if display:
        idx_plot = [2, 5]
        fig, axs = plt.subplots(2, 2)
        im3 = axs[0, 0].imshow(trans_dm_array[:, :, idx_plot[0]], cmap='seismic')
        axs[0, 1].imshow(trans_dm_array[:, :, idx_plot[0]], cmap='seismic')
        axs[1, 0].imshow(trans_dm_array[:, :, idx_plot[1]], cmap='seismic')
        axs[1, 1].imshow(trans_dm_array[:, :, idx_plot[1]], cmap='seismic')
        fig.suptitle(f'DM shapes (modes {idx_plot[0]} and {idx_plot[1]})')
        fig.colorbar(im3, ax=axs.ravel().tolist(), fraction=0.02)
        plt.show()

    return im


def interaction_matrices_multi_wfs(pup_diam_m, pup_mask, dm_array, dm_mask, dm_height, dm_rotation,
                                   wfs_configs, gs_pol_coo=None, gs_height=None,
                                   verbose=False, specula_convention=True):
    """
    Computes interaction matrices for multiple WFS configurations.
    
    Each WFS can have its own guide star position (gs_pol_coo) and height (gs_height).
    
    Parameters:
    - pup_diam_m: float, pupil diameter in meters
    - pup_mask: numpy 2D array, pupil mask (n_pup x n_pup)
    - dm_array: numpy 3D array, DM modes (n x n x n_dm_modes)
    - dm_mask: numpy 2D array, DM mask (n x n)
    - dm_height: float, DM conjugation altitude
    - dm_rotation: float, DM rotation in degrees
    - wfs_configs: list of dict, each containing WFS parameters:
        {
            'nsubaps': int,
            'rotation': float (default 0.0),
            'translation': tuple (default (0.0, 0.0)),
            'magnification': tuple or float (default (1.0, 1.0)),
            'fov_arcsec': float,
            'idx_valid_sa': array or None,
            'gs_pol_coo': tuple (radius_arcsec, angle_deg) - REQUIRED if gs_pol_coo=None
            'gs_height': float - REQUIRED if gs_height=None
            'name': str (optional)
        }
    - gs_pol_coo: tuple or None (DEPRECATED - use wfs_config['gs_pol_coo'] instead)
        If provided, uses this for all WFS (backward compatibility)
    - gs_height: float or None (DEPRECATED - use wfs_config['gs_height'] instead)
        If provided, uses this for all WFS (backward compatibility)
    - verbose: bool, optional
    - specula_convention: bool, optional
    
    Returns:
    - im_dict: dict, interaction matrices keyed by WFS name or index
    - derivatives_info: dict with metadata about the computation
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"Computing interaction matrices for {len(wfs_configs)} WFS")
        print(f"{'='*60}")

    # Check if using deprecated global gs_pol_coo/gs_height
    use_global_gs = gs_pol_coo is not None and gs_height is not None

    if use_global_gs and verbose:
        print("WARNING: Using global gs_pol_coo and gs_height for all WFS (deprecated)")
        print("         Consider specifying gs_pol_coo and gs_height in each wfs_config")

    # Extract gs_pol_coo and gs_height for each WFS
    wfs_gs_info = []
    for i, wfs_config in enumerate(wfs_configs):
        if use_global_gs:
            # Backward compatibility: use global values
            wfs_gs_pol_coo = gs_pol_coo
            wfs_gs_height = gs_height
        else:
            # New method: get from wfs_config
            if 'gs_pol_coo' not in wfs_config:
                raise ValueError(f"WFS {i}: 'gs_pol_coo' must be"
                                 f" specified in wfs_config when gs_pol_coo=None")
            if 'gs_height' not in wfs_config:
                raise ValueError(f"WFS {i}: 'gs_height' must be"
                                 f"specified in wfs_config when gs_height=None")

            wfs_gs_pol_coo = wfs_config['gs_pol_coo']
            wfs_gs_height = wfs_config['gs_height']

        wfs_gs_info.append((wfs_gs_pol_coo, wfs_gs_height))

    # Check if all WFS see DM from the same direction
    all_gs_same = all(gs_info == wfs_gs_info[0] for gs_info in wfs_gs_info)

    # Detect transformations
    has_dm_transform_list = []
    wfs_transforms = []

    for gs_pol_coo_wfs, gs_height_wfs in wfs_gs_info:
        has_dm = _has_transformations(dm_rotation, (0, 0), (1, 1)) or \
                 gs_pol_coo_wfs != (0, 0) or gs_height_wfs != 0
        has_dm_transform_list.append(has_dm)

    for config in wfs_configs:
        wfs_rot = config.get('rotation', 0.0)
        wfs_trans = config.get('translation', (0.0, 0.0))
        wfs_mag = config.get('magnification', (1.0, 1.0))
        wfs_transforms.append((wfs_rot, wfs_trans, wfs_mag))

    # Check if all WFS transforms are identical
    all_wfs_same = all(t == wfs_transforms[0] for t in wfs_transforms)

    # Decide workflow:
    # SEPARATED: if all WFS see DM from same direction AND have same WFS transforms
    # COMBINED: otherwise (each WFS computed independently)
    use_separated = all_gs_same and all_wfs_same

    if verbose:
        print(f"All WFS see DM from same direction: {all_gs_same}")
        print(f"All WFS have same transforms: {all_wfs_same}")
        print(f"Using {'SEPARATED' if use_separated else 'COMBINED'} workflow")
        print(f"{'='*60}\n")

    im_dict = {}
    derivatives_info = {
        'workflow': 'separated' if use_separated else 'combined',
        'all_gs_same': all_gs_same,
        'all_wfs_same': all_wfs_same
    }

    if use_separated:
        # SEPARATED WORKFLOW: Compute DM transformations once
        if verbose:
            print("[Step 1/2] Computing DM transformations and derivatives...")

        # Use first WFS's gs_pol_coo and gs_height (they're all the same)
        gs_pol_coo_ref, gs_height_ref = wfs_gs_info[0]

        trans_dm_array, trans_dm_mask, pup_mask_conv, derivatives_x, derivatives_y = \
            apply_dm_transformations_separated(
                pup_diam_m, pup_mask, dm_array, dm_mask,
                dm_height, dm_rotation,
                gs_pol_coo_ref, gs_height_ref,
                verbose=verbose, specula_convention=specula_convention
            )

        if verbose:
            print(f"  ✓ DM transformed: {trans_dm_array.shape}")
            print(f"  ✓ Derivatives computed: {derivatives_x.shape}")
            print(f"\n[Step 2/2] Computing slopes for each WFS...")

        # Store derivatives for potential reuse
        derivatives_info['derivatives_x'] = derivatives_x
        derivatives_info['derivatives_y'] = derivatives_y
        derivatives_info['dm_array'] = trans_dm_array
        derivatives_info['dm_mask'] = trans_dm_mask

        # Apply WFS transformations for each WFS
        for i, wfs_config in enumerate(wfs_configs):
            wfs_name = wfs_config.get('name', f'wfs_{i}')
            wfs_nsubaps = wfs_config['nsubaps']
            wfs_rotation = wfs_config.get('rotation', 0.0)
            wfs_translation = wfs_config.get('translation', (0.0, 0.0))
            wfs_magnification = wfs_config.get('magnification', (1.0, 1.0))
            wfs_fov_arcsec = wfs_config['fov_arcsec']
            idx_valid_sa = wfs_config.get('idx_valid_sa', None)

            if verbose:
                print(f"\n  Processing {wfs_name}:")
                print(f"    Subapertures: {wfs_nsubaps}x{wfs_nsubaps}")
                print(f"    FOV: {wfs_fov_arcsec}''")

            im = apply_wfs_transformations_separated(
                derivatives_x, derivatives_y, pup_mask_conv, trans_dm_mask,
                wfs_nsubaps, wfs_rotation, wfs_translation, wfs_magnification,
                wfs_fov_arcsec, pup_diam_m, idx_valid_sa=idx_valid_sa,
                verbose=verbose, specula_convention=specula_convention
            )

            im_dict[wfs_name] = im

            if verbose:
                print(f"    ✓ IM shape: {im.shape}")

    else:
        # COMBINED WORKFLOW: Compute each WFS independently
        if verbose:
            print("Computing each WFS independently with combined DM+WFS transformations...")

        for i, wfs_config in enumerate(wfs_configs):
            wfs_name = wfs_config.get('name', f'wfs_{i}')
            wfs_nsubaps = wfs_config['nsubaps']
            wfs_rotation = wfs_config.get('rotation', 0.0)
            wfs_translation = wfs_config.get('translation', (0.0, 0.0))
            wfs_magnification = wfs_config.get('magnification', (1.0, 1.0))
            wfs_fov_arcsec = wfs_config['fov_arcsec']
            idx_valid_sa = wfs_config.get('idx_valid_sa', None)

            # Get WFS-specific gs_pol_coo and gs_height
            gs_pol_coo_wfs, gs_height_wfs = wfs_gs_info[i]

            if verbose:
                print(f"\n  [{i+1}/{len(wfs_configs)}] Processing {wfs_name}:")
                print(f"    Subapertures: {wfs_nsubaps}x{wfs_nsubaps}")
                print(f"    FOV: {wfs_fov_arcsec}''")
                print(f"    GS position: {gs_pol_coo_wfs}")
                print(f"    GS height: {gs_height_wfs} m")

            # Use interaction_matrix which already handles workflow selection
            im = interaction_matrix(
                pup_diam_m, pup_mask, dm_array, dm_mask,
                dm_height, dm_rotation,
                wfs_nsubaps, wfs_rotation, wfs_translation, wfs_magnification,
                wfs_fov_arcsec, gs_pol_coo_wfs, gs_height_wfs,
                idx_valid_sa=idx_valid_sa,
                verbose=verbose, display=False, specula_convention=specula_convention
            )

            im_dict[wfs_name] = im

            if verbose:
                print(f"    ✓ IM shape: {im.shape}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Completed {len(im_dict)} interaction matrices")
        print(f"Workflow: {derivatives_info['workflow'].upper()}")
        print(f"{'='*60}\n")

    return im_dict, derivatives_info


def projection_matrix(pup_diam_m, pup_mask, dm_array, dm_mask, base_inv_array,
                      dm_height, dm_rotation, base_rotation, base_translation, base_magnification,
                      gs_pol_coo, gs_height, verbose=False, display=False, specula_convention=True):
    """
    Computes a projection matrix for DM modes onto a desired basis.
    Uses intelligent workflow selection like interaction_matrix.

    Parameters:
    - pup_diam_m: float, size in m of the side of the pupil
    - pup_mask: numpy 2D array, pupil mask (n_pup x n_pup)
    - dm_array: numpy 3D array, Deformable Mirror 2D shapes (n x n x n_dm_modes)
    - dm_mask: numpy 2D array, DM mask (n x n)
    - base_inv_array: numpy 3D array, inverted basis for projection (n_pup x n_pup x n_base_modes)
    - dm_height: float, conjugation altitude of the Deformable Mirror
    - dm_rotation: float, rotation in deg of the Deformable Mirror with respect to the pupil
    - base_rotation: float, rotation of the basis in deg
    - base_translation: tuple, translation of the basis
    - base_magnification: tuple, magnification of the basis
    - gs_pol_coo: tuple, polar coordinates of the guide star radius in arcsec and angle in deg
    - gs_height: float, altitude of the guide star
    - verbose: bool, optional, display verbose output
    - display: bool, optional, display plots
    - specula_convention: bool, optional, use SPECULA convention

    Returns:
    - pm: numpy 2D array, projection matrix (n_base_modes x n_dm_modes)
    """

    # Detect which transformations are present
    has_dm_transform = _has_transformations(dm_rotation, (0, 0), (1, 1)) or \
                       gs_pol_coo != (0, 0) or dm_height != 0
    has_base_transform = _has_transformations(base_rotation, base_translation, base_magnification)

    # Choose workflow
    use_combined = has_dm_transform and has_base_transform

    if verbose:
        print(f"\n{'='*60}")
        print(f"Projection Matrix Computation")
        print(f"{'='*60}")
        print(f"DM transformations: {has_dm_transform}")
        print(f"Base transformations: {has_base_transform}")
        print(f"Using {'COMBINED' if use_combined else 'SEPARATED'} workflow")
        print(f"{'='*60}\n")

    if specula_convention:
        dm_array = np.transpose(dm_array, (1, 0, 2))
        dm_mask = np.transpose(dm_mask)
        pup_mask = np.transpose(pup_mask)

    pup_diam_pix = pup_mask.shape[0]
    pixel_pitch = pup_diam_m / pup_diam_pix

    if dm_mask.shape[0] != dm_array.shape[0]:
        raise ValueError('DM and mask arrays must have the same dimensions.')

    # Calculate DM transformations
    dm_translation, dm_magnification = shiftzoom_from_source_dm_params(
        gs_pol_coo, gs_height, dm_height, pixel_pitch
    )
    output_size = (pup_diam_pix, pup_diam_pix)

    if use_combined:
        # COMBINED: Apply DM + Base transformations together (single interpolation)
        if verbose:
            print(f'Combined DM+Base transformations:')
            print(f'  DM translation: {dm_translation} pixels')
            print(f'  DM rotation: {dm_rotation} deg')
            print(f'  DM magnification: {dm_magnification}')
            print(f'  Base translation: {base_translation} pixels')
            print(f'  Base rotation: {base_rotation} deg')
            print(f'  Base magnification: {base_magnification}')

        # Transform DM array with both DM and Base transformations
        trans_dm_array = rotshiftzoom_array(
            dm_array,
            dm_translation=dm_translation,
            dm_rotation=dm_rotation,
            dm_magnification=dm_magnification,
            wfs_translation=base_translation,
            wfs_rotation=base_rotation,
            wfs_magnification=base_magnification,
            output_size=output_size
        )

        # Transform DM mask (only DM transformations)
        trans_dm_mask = rotshiftzoom_array(
            dm_mask,
            dm_translation=dm_translation,
            dm_rotation=dm_rotation,
            dm_magnification=dm_magnification,
            wfs_translation=(0, 0),
            wfs_rotation=0,
            wfs_magnification=(1, 1),
            output_size=output_size
        )
        trans_dm_mask[trans_dm_mask < 0.5] = 0

        # Transform pupil mask (only Base transformations)
        trans_pup_mask = rotshiftzoom_array(
            pup_mask,
            dm_translation=(0, 0),
            dm_rotation=0,
            dm_magnification=(1, 1),
            wfs_translation=base_translation,
            wfs_rotation=base_rotation,
            wfs_magnification=base_magnification,
            output_size=output_size
        )
        trans_pup_mask[trans_pup_mask < 0.5] = 0

    else:
        # SEPARATED: Apply DM transformations only, then Base transformations
        if verbose:
            print(f'Separated workflow:')
            print(f'  DM translation: {dm_translation} pixels')
            print(f'  DM rotation: {dm_rotation} deg')
            print(f'  DM magnification: {dm_magnification}')

        # Transform DM array (only DM transformations)
        trans_dm_array = rotshiftzoom_array(
            dm_array,
            dm_translation=dm_translation,
            dm_rotation=dm_rotation,
            dm_magnification=dm_magnification,
            wfs_translation=(0, 0),
            wfs_rotation=0,
            wfs_magnification=(1, 1),
            output_size=output_size
        )

        # Transform DM mask (only DM transformations)
        trans_dm_mask = rotshiftzoom_array(
            dm_mask,
            dm_translation=dm_translation,
            dm_rotation=dm_rotation,
            dm_magnification=dm_magnification,
            wfs_translation=(0, 0),
            wfs_rotation=0,
            wfs_magnification=(1, 1),
            output_size=output_size
        )
        trans_dm_mask[trans_dm_mask < 0.5] = 0

        # Transform pupil mask (only Base transformations)
        trans_pup_mask = rotshiftzoom_array(
            pup_mask,
            dm_translation=(0, 0),
            dm_rotation=0,
            dm_magnification=(1, 1),
            wfs_translation=base_translation,
            wfs_rotation=base_rotation,
            wfs_magnification=base_magnification,
            output_size=output_size
        )
        trans_pup_mask[trans_pup_mask < 0.5] = 0

        if has_base_transform and verbose:
            print(f'  Base translation: {base_translation} pixels')
            print(f'  Base rotation: {base_rotation} deg')
            print(f'  Base magnification: {base_magnification}')

    # Check validity
    if np.max(trans_dm_mask) <= 0:
        raise ValueError('Transformed DM mask is empty.')
    if np.max(trans_pup_mask) <= 0:
        raise ValueError('Transformed pupil mask is empty.')

    # Apply DM mask
    trans_dm_array = apply_mask(trans_dm_array, trans_dm_mask)

    if verbose:
        print(f'  ✓ DM array transformed, shape: {trans_dm_array.shape}')

    # Create mask for valid pixels (both in DM and pupil)
    valid_mask = trans_dm_mask * trans_pup_mask
    flat_mask = valid_mask.flatten()
    valid_indices = np.where(flat_mask > 0.5)[0]
    n_valid_pixels = len(valid_indices)

    # Extract valid pixels from DM array (always 3D)
    height, width, n_modes = trans_dm_array.shape
    dm_valid_values = trans_dm_array[valid_indices]  # Shape: (n_valid_pixels, n_modes)

    # *** OPTIMIZED: Handle base_inv_array format ***
    if base_inv_array.ndim == 2:
        n_rows, n_cols = base_inv_array.shape

        # Determine format based on which dimension matches valid pixels
        if n_cols == n_valid_pixels:
            # Format: (nmodes, npixels_valid) - IFunc style
            n_modes_base = n_rows
            base_valid_values = base_inv_array.T  # (npixels_valid, nmodes)
            base_format = "IFunc (nmodes, npixels)"

        elif n_rows == n_valid_pixels:
            # Format: (npixels_valid, nmodes) - IFuncInv style
            n_modes_base = n_cols
            base_valid_values = base_inv_array  # No transpose!
            base_format = "IFuncInv (npixels, nmodes)"

        else:
            raise ValueError(
                f"Cannot determine base format: shape {base_inv_array.shape} "
                f"does not match valid pixels ({n_valid_pixels})"
            )

        if verbose:
            print(f'  Inverse basis format: {base_format}')
            print(f'  Shape: {base_inv_array.shape} → {base_valid_values.shape}')

    elif base_inv_array.ndim == 3:
        # 3D basis: shape is (height, width, n_modes)
        # Convert to 2D format internally
        height_base, width_base, n_modes_base = base_inv_array.shape

        if verbose:
            print(f'  Basis 3D shape: {base_inv_array.shape}')
            print(f'  Converting to 2D format...')

        # Extract valid pixels (same as before)
        base_valid_values = base_inv_array[valid_pixels]  # Shape: (n_valid_pixels, n_modes_base)
    else:
        raise ValueError(f"base_inv_array must be 2D or 3D, got {base_inv_array.ndim}D")

    # Compute projection (same for both cases)
    projection = np.dot(dm_valid_values.T, base_valid_values)

    if verbose:
        print(f'  ✓ Projection computed, shape: {projection.shape}')
        print(f'  ✓ Valid pixels used: {n_valid_pixels}')
        print(f'  ✓ Base format: {"2D (optimized)" if base_inv_array.ndim == 2 else "3D (converted)"}')

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
        for i in range(min(5, projection.shape[1])):
            plt.plot(projection[:, i], label=f'Basis mode {i}')
        plt.legend()
        plt.title('Projection coefficients')
        plt.xlabel('DM mode index')
        plt.ylabel('Coefficient')
        plt.grid(True)

        # Display projection array
        plt.figure()
        plt.imshow(projection, cmap='seismic', origin='lower', aspect='auto')
        plt.title('Projection matrix')
        plt.xlabel('Basis mode index')
        plt.ylabel('DM mode index')
        plt.colorbar()
        plt.grid(True)
        plt.show()

    return projection


def projection_matrices_multi_base(pup_diam_m, pup_mask, dm_array, dm_mask,
                                   dm_height, dm_rotation, base_configs,
                                   gs_pol_coo, gs_height,
                                   verbose=False, specula_convention=True):
    """
    Computes projection matrices for multiple basis configurations efficiently.
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"Computing projection matrices for {len(base_configs)} bases")
        print(f"{'='*60}")

    # Check if all bases have the same transformations
    base_transforms = []
    for config in base_configs:
        rot = config.get('rotation', 0.0)
        trans = config.get('translation', (0.0, 0.0))
        mag = config.get('magnification', (1.0, 1.0))
        base_transforms.append((rot, trans, mag))

    all_base_same = all(t == base_transforms[0] for t in base_transforms)

    # Detect DM transformations
    has_dm_transform = _has_transformations(dm_rotation, (0, 0), (1, 1)) or \
                       gs_pol_coo != (0, 0) or dm_height != 0

    # IMPORTANT: Even if all bases have same transforms, we need to check
    # if there are any base transforms at all
    has_base_transform = _has_transformations(base_transforms[0][0],
                                              base_transforms[0][1],
                                              base_transforms[0][2])

    # Use separated workflow ONLY if all bases have same transforms AND
    # either no DM transforms OR no base transforms (not both)
    use_separated = all_base_same and not (has_dm_transform and has_base_transform)

    if verbose:
        print(f"All bases have same transforms: {all_base_same}")
        print(f"DM has transformations: {has_dm_transform}")
        print(f"Base has transformations: {has_base_transform}")
        print(f"Using {'SEPARATED' if use_separated else 'COMBINED'} workflow")
        print(f"{'='*60}\n")

    pm_dict = {}
    transform_info = {
        'workflow': 'separated' if use_separated else 'combined',
        'all_base_same': all_base_same,
        'has_dm_transform': has_dm_transform,
        'has_base_transform': has_base_transform
    }

    if use_separated and all_base_same:
        # SEPARATED: Can only be used when there are NO base transforms
        # or NO DM transforms (not both)
        if verbose:
            print("[Step 1/2] Transforming DM array...")

        if specula_convention:
            dm_array = np.transpose(dm_array, (1, 0, 2))
            dm_mask = np.transpose(dm_mask)
            pup_mask = np.transpose(pup_mask)

        pup_diam_pix = pup_mask.shape[0]
        pixel_pitch = pup_diam_m / pup_diam_pix

        dm_translation, dm_magnification = shiftzoom_from_source_dm_params(
            gs_pol_coo, gs_height, dm_height, pixel_pitch
        )
        output_size = (pup_diam_pix, pup_diam_pix)

        # Transform DM (only DM transformations)
        trans_dm_array = rotshiftzoom_array(
            dm_array,
            dm_translation=dm_translation,
            dm_rotation=dm_rotation,
            dm_magnification=dm_magnification,
            wfs_translation=(0, 0),
            wfs_rotation=0,
            wfs_magnification=(1, 1),
            output_size=output_size
        )

        trans_dm_mask = rotshiftzoom_array(
            dm_mask,
            dm_translation=dm_translation,
            dm_rotation=dm_rotation,
            dm_magnification=dm_magnification,
            wfs_translation=(0, 0),
            wfs_rotation=0,
            wfs_magnification=(1, 1),
            output_size=output_size
        )
        trans_dm_mask[trans_dm_mask < 0.5] = 0
        trans_dm_array = apply_mask(trans_dm_array, trans_dm_mask)

        # Get base transformations (same for all)
        base_rot = base_configs[0].get('rotation', 0.0)
        base_trans = base_configs[0].get('translation', (0.0, 0.0))
        base_mag = base_configs[0].get('magnification', (1.0, 1.0))

        # Only transform pupil mask if base has transformations ***
        if has_base_transform:
            if verbose:
                print(f"  Applying base transformations to pupil mask:")
                print(f"    Rotation: {base_rot}°")
                print(f"    Translation: {base_trans}")
                print(f"    Magnification: {base_mag}")

            trans_pup_mask = rotshiftzoom_array(
                pup_mask,
                dm_translation=(0, 0),
                dm_rotation=0,
                dm_magnification=(1, 1),
                wfs_translation=base_trans,
                wfs_rotation=base_rot,
                wfs_magnification=base_mag,
                output_size=output_size
            )
            trans_pup_mask[trans_pup_mask < 0.5] = 0
        else:
            # No base transformations - use original pupil mask
            if verbose:
                print(f"  No base transformations - using original pupil mask")
            trans_pup_mask = pup_mask

        valid_mask = trans_dm_mask * trans_pup_mask
        flat_mask = valid_mask.flatten()
        valid_indices = np.where(flat_mask > 0.5)[0]
        n_valid_pixels = len(valid_indices)

        if verbose:
            print(f"  ✓ Valid pixels in trans_dm_mask: {np.sum(trans_dm_mask > 0.5)}")
            print(f"  ✓ Valid pixels in trans_pup_mask: {np.sum(trans_pup_mask > 0.5)}")
            print(f"  ✓ Valid pixels: {n_valid_pixels}")

        # Extract DM valid values once
        dm_valid_values = trans_dm_array[valid_indices]  # Shape: (n_valid_pixels, n_modes)

        if verbose:
            print(f"  ✓ DM transformed: {trans_dm_array.shape}")
            print(f"  ✓ Valid pixels: {n_valid_pixels}")
            print(f"\n[Step 2/2] Computing projections for each basis...")

        # Compute projection for each basis
        for i, base_config in enumerate(base_configs):
            base_name = base_config.get('name', f'base_{i}')
            base_inv_array = base_config['base_inv_array']

            if verbose:
                print(f"\n  [{i+1}/{len(base_configs)}] Processing {base_name}:")

            # *** OPTIMIZED: Handle 2D and 3D base formats ***
            # *** DETECT BASE FORMAT ***
            if base_inv_array.ndim == 2:
                n_rows, n_cols = base_inv_array.shape

                # Determine format based on which dimension matches valid pixels
                if n_cols == n_valid_pixels:
                    # Format: (nmodes, npixels_valid) - IFunc style
                    n_modes_base = n_rows
                    base_valid_values = base_inv_array.T  # (npixels_valid, nmodes)
                    if verbose:
                        print(f"    Inverse basis 2D (IFunc format): {base_inv_array.shape}")
                        print(f"    → Transposed to: {base_valid_values.shape}")
                        
                elif n_rows == n_valid_pixels:
                    # Format: (npixels_valid, nmodes) - IFuncInv style
                    n_modes_base = n_cols
                    base_valid_values = base_inv_array  # No transpose needed!
                    if verbose:
                        print(f"    Inverse basis 2D (IFuncInv format): {base_inv_array.shape}")
                        print(f"    → Direct use (no transpose)")
                        
                else:
                    raise ValueError(
                        f"Cannot determine base format: shape {base_inv_array.shape} "
                        f"does not match valid pixels ({n_valid_pixels})"
                    )

            elif base_inv_array.ndim == 3:
                # 3D basis: shape (height, width, n_modes)
                n_modes_base = base_inv_array.shape[2]

                if verbose:
                    print(f"    Basis 3D: {base_inv_array.shape}, converting...")

                # Extract valid pixels from each mode
                base_valid_values = base_inv_array[valid_indices]  # Shape: (n_valid_pixels, n_modes_base)
            else:
                raise ValueError(f"base_inv_array must be 2D or 3D, got {base_inv_array.ndim}D")

            plot_debug = True
            if plot_debug:
                # 2D plot to verify DM shape
                plt.figure()
                plt.imshow(trans_dm_array[:, :, 0], cmap='seismic')
                plt.colorbar()
                plt.title('Transformed DM Array (First Mode)')
                plt.xlabel('X Pixel')
                plt.ylabel('Y Pixel')
                plt.grid()
                plt.tight_layout()
                plt.show()
                print('trans_dm_array first mode multiplyed by base_valid_values')
                print('to verify correctness of the transformation.')
                print('Result:')
                print(np.dot(base_valid_values.T, dm_valid_values[:, 0]))
                print(np.dot(base_valid_values.T, dm_valid_values[:, 1]))
                print(np.dot(base_valid_values.T, dm_valid_values[:, 2]))
                plt.figure()
                temp_2d = np.zeros((pup_diam_pix, pup_diam_pix))
                temp_2d[valid_indices] = dm_valid_values[:, 0]
                plt.imshow(temp_2d, cmap='seismic')
                plt.colorbar()
                plt.title('Base Array (First Mode)')
                plt.xlabel('X Pixel')
                plt.ylabel('Y Pixel')
                plt.grid()
                plt.tight_layout()
                plt.figure()
                temp_inv_2d = np.zeros((pup_diam_pix, pup_diam_pix))
                temp_inv_2d[valid_indices] = base_valid_values[:, 0]
                plt.imshow(temp_inv_2d, cmap='seismic')
                plt.colorbar()
                plt.title('Base Inverse Array (First Mode)')
                plt.xlabel('X Pixel')
                plt.ylabel('Y Pixel')
                plt.grid()
                plt.tight_layout()
                plt.figure()
                plt.imshow(trans_dm_array[:, :, 0], cmap='seismic')
                plt.colorbar()
                plt.title('Transformed DM Array (First Mode)')
                plt.xlabel('X Pixel')
                plt.ylabel('Y Pixel')
                plt.grid()
                plt.tight_layout()
                plt.show()

            # Compute projection (same for both)
            projection = np.dot(base_valid_values.T, dm_valid_values)
            pm_dict[base_name] = projection

            if verbose:
                print(f"    ✓ PM shape: {projection.shape}")

    else:
        # COMBINED: Compute each basis independently
        if verbose:
            print("Computing each basis independently...")

        for i, base_config in enumerate(base_configs):
            base_name = base_config.get('name', f'base_{i}')
            base_inv_array = base_config['base_inv_array']
            base_rot = base_config.get('rotation', 0.0)
            base_trans = base_config.get('translation', (0.0, 0.0))
            base_mag = base_config.get('magnification', (1.0, 1.0))

            if verbose:
                print(f"\n  [{i+1}/{len(base_configs)}] Processing {base_name}...")

            pm = projection_matrix(
                pup_diam_m, pup_mask, dm_array, dm_mask, base_inv_array,
                dm_height, dm_rotation, base_rot, base_trans, base_mag,
                gs_pol_coo, gs_height,
                verbose=verbose, display=False, specula_convention=specula_convention
            )

            pm_dict[base_name] = pm

            if verbose:
                print(f"    ✓ PM shape: {pm.shape}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Completed {len(pm_dict)} projection matrices")
        print(f"Workflow: {transform_info['workflow'].upper()}")
        print(f"{'='*60}\n")

    return pm_dict, transform_info
