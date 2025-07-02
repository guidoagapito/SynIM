import numpy as np
import matplotlib.pyplot as plt
from synim.synim import calculate_extrapolation_indices_coeffs, apply_extrapolation

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