import numpy as np
import matplotlib.pyplot as plt
from synim.synim import calculate_extrapolation_indices_coeffs

def apply_extrapolation(data,
                        edge_pixels,
                        reference_indices,
                        coefficients,
                        debug=False,
                        problem_indices=None):
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
        safe_ref_indices = xp.where(valid_ref_mask, reference_indices, 0)

        # Get data values for all reference indices at once
        ref_data = flat_data[safe_ref_indices]  # Shape: (n_valid_edges, 8)

        # Zero out contributions from invalid references
        masked_coeffs = xp.where(valid_ref_mask, coefficients, 0.0)

        # Compute all contributions at once and sum across reference positions
        contributions = masked_coeffs * ref_data  # Element-wise multiplication
        extrap_values = xp.sum(contributions, axis=1)  # Sum across reference positions

        # Assign extrapolated values to edge pixels
        flat_result[edge_pixels] = extrap_values

    return result

def test_extrapolation():
    """
    Test function for extrapolation routines.
    Creates a 10x10 array with increasing values in X and a circular mask,
    then applies extrapolation and compares the results.
    """
    # Create a 10x10 array with increasing values in X
    size = 10
    x = np.arange(size)
    test_array = np.tile(x, (size, 1))
    
    # Create a circular mask with radius 3
    y, x = np.ogrid[:size, :size]
    center = size // 2
    mask = ((x - center)**2 + (y - center)**2 <= 9)  # Radius 3
    
    # Identify problematic pixels for debugging
    problem_pixels = [[2, 6], [8, 6]]
    
    # Calculate indices and coefficients for extrapolation
    edge_pixels, reference_indices, coefficients = calculate_extrapolation_indices_coeffs(
        mask, debug=True, debug_pixels=problem_pixels)
    
    # Find the indices of problematic pixels in the edge_pixels array
    problem_indices = []
    for i, (y, x) in enumerate(zip(*np.unravel_index(edge_pixels, mask.shape))):
        if [y, x] in problem_pixels:
            problem_indices.append(i)
    
    print(f"Indices of problematic pixels: {problem_indices}")
    
    # Apply extrapolation
    result = apply_extrapolation(test_array, edge_pixels, reference_indices, coefficients, 
                                debug=True, problem_indices=problem_indices)
    
    # Create a larger mask for comparison
    bigger_mask = ((x - center)**2 + (y - center)**2 <= 16)  # Radius 4
    
    # Display comparison
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    masked_array = np.ma.masked_array(test_array, ~mask)
    plt.imshow(masked_array, cmap='viridis')
    plt.colorbar()
    plt.title('Original Array')
    
    plt.subplot(132)
    masked_original = np.ma.masked_array(test_array, ~bigger_mask)
    plt.imshow(masked_original, cmap='viridis')
    plt.colorbar()
    plt.title('Original Array (larger mask)')
    
    plt.subplot(133)
    masked_result = np.ma.masked_array(result, ~bigger_mask)
    plt.imshow(masked_result, cmap='viridis')
    plt.colorbar()
    plt.title('Extrapolated Array')
    plt.tight_layout()
    plt.show()
    
    return test_array, mask, result

# Run the test
test_array, mask, result = test_extrapolation()