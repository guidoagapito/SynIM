
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import affine_transform, binary_dilation
from scipy.ndimage import rotate, shift, zoom


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


def rotshiftzoom_array(input_array, dm_translation=(0.0, 0.0),
                       dm_rotation=0.0, dm_magnification=(1.0, 1.0),
                       wfs_translation=(0.0, 0.0), wfs_rotation=0.0,
                       wfs_magnification=(1.0, 1.0), output_size=None):
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


def has_transformations(rotation, translation, magnification):
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


def rebin(array, new_shape, method='average'):
    """
    The rebin function resizes a vector or array to dimensions given by the parameters new_shape.
    In case of a 3D array the third dimension is not affected.

    Parameters:
    - array: numpy 2D or 3D array
    - new_shape: 2 elements tuple
    - method: 'sum' or 'average' or 'nanmean', used in the compression case

    Returns:
    - rebinned_array: numpy 2D or 3D array
    """

    if array.ndim == 1:
        array = array.reshape(array.shape[0], 1)

    shape = array.shape
    m, n = shape[0:2]
    M, N = new_shape

    if M > m or N > n:
        # Expansion case
        if m % M != 0 or n % N != 0:
            raise ValueError("New shape must be multiples of the input dimensions.")
        if array.ndim == 3:
            rebinned_array = np.tile(array, (M//m, N//n, 1))
        else:
            rebinned_array = np.tile(array, (M//m, N//n))
    else:    
        # Compression case
        if M == 0 or N == 0:
            raise ValueError("New shape dimensions must be greater than 0.")

        if array.ndim == 3:
            if method == 'sum':
                rebinned_array = np.sum(
                    array[:M*(m//M), :N*(n//N), :].reshape((M, m//M, N, n//N, shape[2])),
                    axis=(1, 3))
            elif method == 'average':
                rebinned_array = np.mean(
                    array[:M*(m//M), :N*(n//N), :].reshape((M, m//M, N, n//N, shape[2])),
                    axis=(1, 3))
            elif method == 'nanmean':
                with np.errstate(invalid='ignore'):
                    rebinned_array = np.nanmean(
                        array[:M*(m//M), :N*(n//N), :].reshape((M, m//M, N, n//N, shape[2])),
                        axis=(1, 3))
            else:
                raise ValueError(f"Unsupported method: {method}."
                                 " Use 'sum', 'average', or 'nanmean'.")
        else:
            if method == 'sum':
                rebinned_array = np.sum(
                    array[:M*(m//M), :N*(n//N)].reshape((M, m//M, N, n//N)),
                    axis=(1, 3))
            elif method == 'average':
                rebinned_array = np.mean(
                    array[:M*(m//M), :N*(n//N)].reshape((M, m//M, N, n//N)),
                    axis=(1, 3))
            elif method == 'nanmean':
                with np.errstate(invalid='ignore'):
                    rebinned_array = np.nanmean(
                        array[:M*(m//M), :N*(n//N)].reshape((M, m//M, N, n//N)),
                        axis=(1, 3))
            else:
                raise ValueError(f"Unsupported method: {method}."
                                 " Use 'sum', 'average', or 'nanmean'.")

    return rebinned_array


def apply_mask(array, mask, norm=False, fill_value=None):
    """
    Apply a 2D or 3D mask to a 2D or 3D array.
    
    Default behavior is to multiply the array by the mask (norm = False and fill_value = None).
    If norm is True, the mask is inverted (1/mask) before applying.
    If fill_value is provided, it replaces the masked values.

    Args:
        array (numpy.ndarray): Input 2D or 3D array to be masked.
        mask (numpy.ndarray): 2D mask to be applied.
        norm (bool, optional): If True, inverts the mask (1/mask) before applying. Defaults to False.
        fill_value (float, optional): Value to replace masked elements. If None, masked elements are set to 0. Defaults to None.

    Returns:
        numpy.ndarray: New array with the mask applied.
    """
    mask = np.asarray(mask)
    if norm:
        safe_mask = np.where(mask == 0, 1, mask)
        norm_mask = 1.0 / safe_mask
    else:
        norm_mask = mask

    # Broadcast mask for 3D arrays
    if array.ndim == 3 and mask.ndim == 2:
        norm_mask = norm_mask[:, :, np.newaxis]

    if fill_value is not None:
        return np.where(norm_mask, array, fill_value)
    else:
        result = array * norm_mask
        if norm and fill_value is None:
            # Set to 0 where mask was zero (to avoid inf)
            result = np.where(mask if mask.ndim == array.ndim else mask[:, :, np.newaxis], result, 0)
        return result


def polar_to_xy(r,theta):
    # conversion polar to rectangular coordinates
    # theta is in rad
    return np.array(( r * np.cos(theta),r * np.sin(theta) ))


def dm3d_to_2d(dm_array, mask):
    """
    Convert a 3D DM influence function to a 2D array using a mask.
    
    Args:
        dm_array (numpy.ndarray): 3D DM influence function array.
        mask (numpy.ndarray): 2D mask array.

    Returns:
        numpy.ndarray: 2D DM influence function array.
    """
    # Check if the mask is 2D
    if mask.ndim != 2:
        raise ValueError("The mask must be a 2D array.")
    # Check if the dm_array is 3D
    if dm_array.ndim != 3:
        raise ValueError("The dm_array must be a 3D array.")
    nmodes = dm_array.shape[2]
    idx = np.where(mask > 0)
    dm_array_2d = dm_array[idx[0], idx[1], :].transpose()
    for i in range(nmodes):
        dm_array_2d[i,:] /= np.sqrt(np.mean(dm_array_2d[i,:]**2))
        dm_array_2d[i,:] -= np.mean(dm_array_2d[i,:])

    return dm_array_2d


def dm2d_to_3d(dm_array, mask, normalize=True):
    """
    Convert a 2D DM influence function to a 3D array using a mask.
    
    Args:
        dm_array (numpy.ndarray): 2D DM influence function array.
        mask (numpy.ndarray): 2D mask array.

    Returns:
        numpy.ndarray: 3D DM influence function array.
    """
    # Check if the mask is 2D
    if mask.ndim != 2:
        raise ValueError("The mask must be a 2D array.")
    # Check if the dm_array is 2D
    if dm_array.ndim != 2:
        raise ValueError("The dm_array must be a 2D array.")
    npixels = mask.shape[0]
    nmodes = dm_array.shape[0]
    dm_array_3d = np.zeros((npixels, npixels, nmodes), dtype=float)
    for i in range(nmodes):
        idx = np.where(mask > 0)
        dm_i = dm_array[i]
        # normalize by the RMS
        if normalize:
            dm_i /= np.sqrt(np.mean(dm_i**2))
        dm_i_3d = np.zeros(mask.shape, dtype=float)
        dm_i_3d[idx] = dm_i
        dm_array_3d[:, :, i] = dm_i_3d

    return dm_array_3d
