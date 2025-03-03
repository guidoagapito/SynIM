import numpy as np
from scipy.ndimage import convolve
from functools import lru_cache

import time
import matplotlib.pyplot as plt

# Import the functions to test
from synim import zern2phi, make_mask, extrapolate_phase

def sample_phase(mask_size,mask=None,n_zernikes=16):
    """Generate a phase matrix with zernikes"""
    if mask is None:
        mask = make_mask(mask_size,diaratio=0.8)
    phase_array = zern2phi(mask_size, n_zernikes, mask=mask, no_round_mask=False, xsign=1, ysign=1, rot_angle=0, verbose=False)
    return phase_array

import numpy as np
from scipy.ndimage import convolve
from functools import lru_cache

# We need a hashable key for the mask, as arrays aren't hashable
def _make_mask_hashable(mask):
    """Convert a numpy array to a hashable tuple representation."""
    return (mask.shape, mask.tobytes())

@lru_cache(maxsize=128)
def _cached_edge_indices(mask_key):
    """
    Cached version that works with hashable input.
    
    Parameters:
    mask_key: Hashable representation of the mask (shape, bytes)
    
    Returns:
    tuple: Edge pixels and reference points
    """
    # Convert back to numpy array
    shape = mask_key[0]
    mask_bytes = mask_key[1]
    mask = np.frombuffer(mask_bytes, dtype=bool).reshape(shape)
    
    # Create float mask
    float_mask = mask.astype(float)
    height, width = mask.shape
    
    # Find edge pixels (first layer outside the mask)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    edge_mask = convolve(float_mask, kernel, mode='constant') * (1.0 - float_mask)
    edge_pixels = np.where(edge_mask > 0)
    
    # Initialize reference points
    x_first = np.full(mask.shape, -1)
    x_second = np.full(mask.shape, -1)
    y_first = np.full(mask.shape, -1)
    y_second = np.full(mask.shape, -1)
    
    # Create flat index array for the mask
    indices = np.zeros_like(float_mask, dtype=int)
    indices[mask > 0] = np.flatnonzero(mask > 0)
    
    # For each edge pixel, find reference points in x and y directions
    for i, j in zip(*edge_pixels):
        # Check x direction (horizontal)
        if i + 1 < height and float_mask[i + 1, j] > 0:
            x_first[i, j] = indices[i + 1, j]
            if i + 2 < height and float_mask[i + 2, j] > 0:
                x_second[i, j] = indices[i + 2, j]
        elif i - 1 >= 0 and float_mask[i - 1, j] > 0:
            x_first[i, j] = indices[i - 1, j]
            if i - 2 >= 0 and float_mask[i - 2, j] > 0:
                x_second[i, j] = indices[i - 2, j]
        
        # Check y direction (vertical)
        if j + 1 < width and float_mask[i, j + 1] > 0:
            y_first[i, j] = indices[i, j + 1]
            if j + 2 < width and float_mask[i, j + 2] > 0:
                y_second[i, j] = indices[i, j + 2]
        elif j - 1 >= 0 and float_mask[i, j - 1] > 0:
            y_first[i, j] = indices[i, j - 1]
            if j - 2 >= 0 and float_mask[i, j - 2] > 0:
                y_second[i, j] = indices[i, j - 2]
    
    # Convert references to tuples since we can't return mutable numpy arrays from a cached function
    return (
        edge_pixels,
        {
            'x_first': x_first.copy(),
            'x_second': x_second.copy(),
            'y_first': y_first.copy(),
            'y_second': y_second.copy()
        }
    )

def extrapolate_edge_indices(mask, use_cache=True):
    """
    Defines the indices and reference points for phase extrapolation outside the pupil mask.
    
    Parameters:
    mask (numpy.ndarray): Binary pupil mask (1 inside, 0 outside)
    use_cache (bool): Whether to use cached results
    
    Returns:
    tuple: (edge_pixels, reference_points)
    """
    if use_cache:
        # Convert mask to hashable form
        mask_key = _make_mask_hashable(mask.astype(bool))
        return _cached_edge_indices(mask_key)
    else:
        # Call the calculation function directly without caching
        return _calculate_edge_indices(mask)

def _calculate_edge_indices(mask):
    """
    Non-cached version of the edge indices calculation.
    """
    # Create float mask
    float_mask = mask.astype(float)
    height, width = mask.shape
    
    # Find edge pixels (first layer outside the mask)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    edge_mask = convolve(float_mask, kernel, mode='constant') * (1.0 - float_mask)
    edge_pixels = np.where(edge_mask > 0)
    
    # Initialize reference points
    reference_points = {
        'x_first': np.full(mask.shape, -1),
        'x_second': np.full(mask.shape, -1),
        'y_first': np.full(mask.shape, -1),
        'y_second': np.full(mask.shape, -1)
    }
    
    # Create flat index array for the mask
    indices = np.zeros_like(float_mask, dtype=int)
    indices[mask > 0] = np.flatnonzero(mask > 0)
    
    # For each edge pixel, find reference points in x and y directions
    for i, j in zip(*edge_pixels):
        # Check x direction (horizontal)
        if i + 1 < height and float_mask[i + 1, j] > 0:
            reference_points['x_first'][i, j] = indices[i + 1, j]
            if i + 2 < height and float_mask[i + 2, j] > 0:
                reference_points['x_second'][i, j] = indices[i + 2, j]
        elif i - 1 >= 0 and float_mask[i - 1, j] > 0:
            reference_points['x_first'][i, j] = indices[i - 1, j]
            if i - 2 >= 0 and float_mask[i - 2, j] > 0:
                reference_points['x_second'][i, j] = indices[i - 2, j]
        
        # Check y direction (vertical)
        if j + 1 < width and float_mask[i, j + 1] > 0:
            reference_points['y_first'][i, j] = indices[i, j + 1]
            if j + 2 < width and float_mask[i, j + 2] > 0:
                reference_points['y_second'][i, j] = indices[i, j + 2]
        elif j - 1 >= 0 and float_mask[i, j - 1] > 0:
            reference_points['y_first'][i, j] = indices[i, j - 1]
            if j - 2 >= 0 and float_mask[i, j - 2] > 0:
                reference_points['y_second'][i, j] = indices[i, j - 2]
    
    return edge_pixels, reference_points

def extrapolate_phase_linear(phase, mask, iterations=1, use_cache=True):
    """
    Extrapolates the phase outside the mask using linear extrapolation.
    
    Parameters:
    phase (numpy.ndarray): Phase array to be extrapolated
    mask (numpy.ndarray): Binary mask (1 inside, 0 outside)
    iterations (int): Number of iterations for the extrapolation
    use_cache (bool): Whether to use cached indices and reference points
    
    Returns:
    numpy.ndarray: Extrapolated phase array
    """
    result = phase.copy()
    current_mask = mask.copy()
    
    for _ in range(iterations):
        edge_pixels, references = extrapolate_edge_indices(current_mask, use_cache=use_cache)
        
        if len(edge_pixels[0]) == 0:
            break  # No more edge pixels to extrapolate
        
        # Create a mask for the current edge pixels
        edge_mask = np.zeros_like(mask, dtype=bool)
        edge_mask[edge_pixels] = True
        
        # Extrapolate values at edge pixels
        for i, j in zip(*edge_pixels):
            # Get valid references for x and y directions
            x_first = references['x_first'][i, j]
            x_second = references['x_second'][i, j]
            y_first = references['y_first'][i, j]
            y_second = references['y_second'][i, j]
            
            extrapolated_values = []
            
            # Linear extrapolation in x direction if we have both references
            if x_first >= 0 and x_second >= 0:
                x_value = 2 * result.flat[x_first] - result.flat[x_second]
                extrapolated_values.append(x_value)
            
            # Linear extrapolation in y direction if we have both references
            if y_first >= 0 and y_second >= 0:
                y_value = 2 * result.flat[y_first] - result.flat[y_second]
                extrapolated_values.append(y_value)
            
            # Use first reference as a fallback if we can't do linear extrapolation
            if not extrapolated_values:
                if x_first >= 0:
                    extrapolated_values.append(result.flat[x_first])
                if y_first >= 0:
                    extrapolated_values.append(result.flat[y_first])
            
            # Combine values from different directions
            if extrapolated_values:
                result[i, j] = sum(extrapolated_values) / len(extrapolated_values)
        
        # Update the mask to include the newly extrapolated pixels
        current_mask = current_mask | edge_mask
    
    return result

# Function to clear the LRU cache
def clear_extrapolation_cache():
    """Clears the cache of extrapolation indices and reference points."""
    _cached_edge_indices.cache_clear()

def extrapolate_phase_convolution(mask, phase, iterations=2):
    """
    Linear phase extrapolation outside the mask using convolution filters.
    This is an alternative implementation using scipy's convolve function.
    
    Parameters:
    mask (numpy.ndarray): Binary mask (1 inside, 0 outside)
    phase (numpy.ndarray): Phase array to be extrapolated
    iterations (int): Number of iterations for the extrapolation
    
    Returns:
    numpy.ndarray: Extrapolated phase array
    """
    # Make a copy of the phase
    result = phase.copy()
    
    # Define the kernel to find edge pixels
    kernel = np.array([[0, 1, 0], 
                       [1, 0, 1], 
                       [0, 1, 0]])
    
    # Make a copy of the mask because we will expand it
    current_mask = mask.copy()
    
    for _ in range(iterations):
        # Find the pixels at the edge (outside the mask but adjacent to valid pixels)
        edge = convolve(current_mask, kernel, mode='constant') * (1 - current_mask)
        edge = edge > 0
        
        if not np.any(edge):
            break
            
        # For each edge pixel, calculate the extrapolated value
        # using a weighted average of the valid neighbors
        for axis in range(2):  # Extrapolate separately in x and y directions
            # Create a kernel to take the neighbors in this direction
            dir_kernel = np.zeros((3, 3))
            if axis == 0:  # X direction
                dir_kernel[1, 0] = 1  # Left neighbor
                dir_kernel[1, 2] = 1  # Right neighbor
            else:  # Y direction
                dir_kernel[0, 1] = 1  # Top neighbor
                dir_kernel[2, 1] = 1  # Bottom neighbor
            
            # Find the values of the valid neighbors in the direction
            neighbor_vals = convolve(result * current_mask, dir_kernel, mode='constant')
            neighbor_count = convolve(current_mask, dir_kernel, mode='constant')
            
            # Where we have valid neighbors, update the edge pixels
            valid_neighbors = (neighbor_count > 0) & edge
            if np.any(valid_neighbors):
                result[valid_neighbors] = neighbor_vals[valid_neighbors] / neighbor_count[valid_neighbors]
        
        # Expand the mask to include the edge pixels
        current_mask = current_mask | edge
    
    return result

mask_size = 64
n_zernikes = 16
mask = make_mask(mask_size,diaratio=0.8)
input_array = sample_phase(mask_size,mask,n_zernikes)

t0 = time.time()
out_array1 = extrapolate_phase_convolution(mask, input_array[:,:,0], iterations=2)
print('time',time.time()-t0)
t1= time.time()
out_array2 = extrapolate_phase_linear(input_array[:,:,0], mask, iterations=2)
print('time',time.time()-t1)
t2 = time.time()
out_array3 = extrapolate_phase(mask, input_array[:,:,0], iterations=2)
print('time',time.time()-t2)

plt.figure()
plt.imshow(input_array[:,:,0])
plt.title('input phase')

plt.figure()
plt.imshow(out_array1-input_array[:,:,0])
plt.title('out phase 1')

plt.figure()
plt.imshow(out_array2-input_array[:,:,0])
plt.title('out phase 2')

plt.figure()
plt.imshow(out_array3-input_array[:,:,0])
plt.title('out phase 3')

plt.show()