import numpy as np

# Global variables for backend selection
xp = np  # Default to NumPy
scndI = None  # Will be set by init()
_use_cupy = False
_device_idx = -1

# Try to import cupy globally (but don't fail if not available)
try:
    import cupy as cp
except ImportError:
    cp = None

def init(device_idx=-1, use_cupy=None):
    """
    Initialize SynIM with either NumPy or CuPy backend
    
    Parameters:
    - device_idx: GPU device index (-1 for CPU, 0+ for GPU)
    - use_cupy: Force CuPy usage (None = auto-detect based on device_idx)
    """
    global xp, scndI, _use_cupy, _device_idx

    _device_idx = device_idx

    if use_cupy is None:
        _use_cupy = device_idx >= 0
    else:
        _use_cupy = use_cupy

    if _use_cupy and cp is not None:
        try:
            if device_idx >= 0:
                cp.cuda.Device(device_idx).use()
            xp = cp
            import cupyx.scipy.ndimage as scndI
            print(f"SynIM initialized with CuPy on device {device_idx}")
        except (ImportError, Exception) as e:
            print(f"Warning: CuPy initialization failed: {e}, falling back to NumPy")
            xp = np
            import scipy.ndimage as scndI
            _use_cupy = False
    else:
        xp = np
        import scipy.ndimage as scndI
        if _use_cupy:
            print("Warning: CuPy requested but not available, using NumPy")
        else:
            print("SynIM initialized with NumPy")

# *** AUTOMATIC INITIALISATION ***
# Called automatically when module is imported
init()

def get_array_module(arr=None):
    """Get appropriate array module (numpy or cupy)"""
    if arr is not None:
        if hasattr(arr, '__array_module__'):
            return arr.__array_module__
        elif 'cupy' in str(type(arr)):
            if cp is not None:
                return cp
            else:
                return np
    return xp

def cpuArray(v, dtype=None, force_copy=False):
    return to_xp(np, v, dtype=dtype, force_copy=force_copy)

def to_xp(target_xp, v, dtype=None, force_copy=False):
    '''
    Make sure that v is allocated as an array on the target backend.
    '''
    # If CuPy is requested but not available, fallback to NumPy
    if target_xp is cp and cp is None:
        target_xp = np

    if target_xp is cp:
        if isinstance(v, cp.ndarray) and not force_copy:
            retval = v
        else:
            retval = cp.array(v)
    else:
        if cp is not None and isinstance(v, cp.ndarray):
            retval = v.get()
        elif isinstance(v, np.ndarray) and not force_copy:
            retval = v
        else:
            retval = np.array(v)

    if dtype is None and not force_copy:
        return retval
    else:
        return retval.astype(dtype, copy=force_copy)
    
def set_backend(device_idx=None, use_cupy=None):
    """
        Optional: change backend after import
        
        Parameters:
        - device_idx: GPU device index (None = keep current)
        - use_cupy: Force CuPy usage (None = keep current)
    """
    global _device_idx, _use_cupy

    # Update only the specified parameters
    if device_idx is not None:
        _device_idx = device_idx
    if use_cupy is not None:
        _use_cupy = use_cupy

    # Re-initialise with new parameters
    init(_device_idx, _use_cupy)