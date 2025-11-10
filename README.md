# SynIM

This repository contains Python code for the computation of synthetic interaction matrices and projection matrices for adaptive optics systems equipped with Shack-Hartmann and Pyramid wavefront sensors.

## Main Features

- **GPU acceleration support** via CuPy for high-performance computation
- **Computation of synthetic interaction matrices** for various DM-WFS combinations, supporting both SCAO and MCAO configurations
- **Computation of projection matrices** for DM-Layer combinations in MCAO/LTAO configurations  
- **Computation of atmospheric covariance matrices** for MMSE reconstructor optimization
- **Tomographic projection matrices** for MCAO systems with multiple optical sources
- **Parametric management** via YAML or PRO configuration files, with automatic parsing of system, DM, WFS, and source parameters
- **Support for different sensor types** (Shack-Hartmann, Pyramid) and deformable mirrors
- **Intelligent workflow optimization** to minimize interpolation steps and maximize accuracy
- **Smart caching** of loaded data to avoid redundant file I/O
- **Utility functions** for mask manipulation, influence functions, Zernike polynomials, numerical derivatives, and array transformations
- **Saving and loading** of interaction matrices and auxiliary data in SPECULA-compatible FITS format

## Main File Structure

- `synim/synim.py`: Core functions for interaction matrix generation, derivatives, and multi-WFS optimization
- `synim/synpm.py`: Core functions for projection matrix generation with intelligent workflow selection
- `synim/params_manager.py`: The `ParamsManager` class for centralized parameter management and batch generation of matrices
- `synim/utils.py`: General utility functions (rebinning, masking, array transformations, etc.)
- `synim/params_utils.py`: Parameter parsing, filename generation, and MMSE reconstructor operators
- `synim/zernike.py`: Zernike polynomial generation and manipulation
- `examples/`: Example scripts demonstrating various use cases
- `test/`: Unit tests and integration tests

## GPU Support

SynIM supports GPU acceleration via [CuPy](https://cupy.dev/) for significant performance improvements:

### Installation

```bash
# Install CuPy (requires CUDA)
pip install cupy-cuda12x  # Replace 12x with your CUDA version

# Or install SynIM with GPU support
pip install .[gpu]
```

### Initialization

```python
import synim

# Use CPU (default)
synim.init(device_idx=-1, precision=1)

# Use GPU 0
synim.init(device_idx=0, precision=1)

# Use single precision (float32) for better GPU performance
synim.init(device_idx=0, precision=1)

# Use double precision (float64) for maximum accuracy
synim.init(device_idx=0, precision=0)
```

### Environment Control

```bash
# Disable GPU globally (forces CPU even if CuPy is installed)
export SYNIM_DISABLE_GPU=TRUE
```

### Performance Notes

- **GPU acceleration** is most effective for large arrays (>1000×1000 pixels)
- **Single precision** (`precision=1`) provides ~2× speedup on GPU vs double precision
- **CuPy** uses GPU-accelerated scipy functions when available (`cupyx.scipy.ndimage`)
- **Automatic fallback** to CPU if GPU operations are not available

### Memory Considerations

When using GPU acceleration, be aware that:
- GPU memory is typically more limited than system RAM (8-16 GB vs 32-128 GB)
- Large pupil sizes (>500 pixels) or many modes (>1000) may exceed available GPU memory
- Memory errors will trigger automatic fallback to CPU processing
- Consider using `precision=1` (single precision) to reduce memory usage by 50%
- Monitor GPU memory usage with `nvidia-smi` or similar tools

For very large computations that exceed GPU memory:
```python
# Force CPU computation for memory-intensive operations
import synim
synim.init(device_idx=-1, precision=0)  # CPU, double precision
```

## Example Usage

Complete examples are available in the `examples/` directory. See also the `test/` directory for additional usage patterns.

### General case

```python
from synim import synim

# Initialize (optional, defaults to CPU single precision)
import synim
synim.init(device_idx=0, precision=1)  # GPU 0, single precision

intmat = synim.interaction_matrix(
    pup_diam_m, pup_mask,
    dm_array, dm_mask,
    dm_height, dm_rotation,
    nsubaps, wfs_rotation,
    wfs_translation, wfs_magnification,
    wfs_fov_arcsec, gs_pol_coo,
    gs_height, idx_valid_sa=idx_valid_sa,
    verbose=True, display=True,
    specula_convention=True
)
```

###  Usage with SPECULA

```python
from synim.params_manager import ParamsManager
import synim

# Initialize GPU (optional)
synim.init(device_idx=0, precision=1)

# Initialize the manager with a YAML or PRO configuration file
pm = ParamsManager('config.yml', verbose=True)

# Compute an interaction matrix for a specific WFS-DM combination
im = pm.compute_interaction_matrix(
    wfs_type='lgs', 
    wfs_index=1, 
    dm_index=1, 
    display=True
)

# Compute and save all interaction matrices for all combinations
# Uses intelligent multi-WFS optimization when possible
saved_files = pm.compute_interaction_matrices(
    output_im_dir='output/im',
    output_rec_dir='output/rec', 
    overwrite=True
)

# Assemble full interaction matrix for MCAO
im_full, n_slopes_per_wfs, mode_indices, dm_indices = pm.assemble_interaction_matrices(
    wfs_type='ngs',
    output_im_dir='output/im',
    component_type='dm',
    save=True
)

# Compute projection matrices for all optical sources
saved_pm = pm.compute_projection_matrices(
    output_dir='output/pm',
    overwrite=True
)

# Compute tomographic projection matrix for MCAO
p_opt, pm_full_dm, pm_full_layer, info = pm.compute_tomographic_projection_matrix(
    reg_factor=1e-8,
    output_dir='output/pm',
    save=True
)

# Compute atmospheric covariance matrices
cov_data = pm.compute_covariance_matrices(
    r0=0.16,           # Fried parameter [m]
    L0=25.0,           # Outer scale [m]
    component_type='layer',
    output_dir='output/cov',
    overwrite=False    # Use cached files if available
)

# Assemble full covariance matrix for MMSE
C_atm_full = pm.assemble_covariance_matrix(
    C_atm_blocks=cov_data['C_atm_blocks'],
    component_indices=cov_data['component_indices'],
    wfs_type='ngs',
    component_type='layer'
)
```

## Key Functions and Classes

### ParamsManager
Handles parameter loading, DM/WFS selection, parameter preparation, and batch computation of matrices.

**Key methods:**
- `compute_interaction_matrix()`: Single IM computation
- `compute_interaction_matrices()`: Batch IM computation with multi-WFS optimization
- `assemble_interaction_matrices()`: Assemble full MCAO interaction matrix
- `compute_projection_matrices()`: Batch PM computation
- `compute_tomographic_projection_matrix()`: MCAO tomographic projection
- `compute_covariance_matrices()`: Atmospheric covariance with smart caching
- `assemble_covariance_matrix()`: Assemble full covariance for MMSE

### Core Functions

**`interaction_matrix()`**  
Main function in `synim.py` that computes synthetic interaction matrices with intelligent workflow selection:
- **SEPARATED workflow**: When transformations exist only in DM OR WFS (minimizes interpolation steps)
- **COMBINED workflow**: When both DM and WFS have transformations (avoids double interpolation)

**`interaction_matrices_multi_wfs()`**  
Optimized computation for multiple WFS viewing the same DM:
- Computes DM transformations and derivatives once
- Applies different WFS transformations to shared derivatives
- Significantly faster than computing each WFS independently

**`projection_matrix()`**  
Computes projection matrices in `synpm.py` with similar workflow optimization for DM-Layer projection.

**`compute_mmse_reconstructor()`**  
Computes Minimum Mean Square Error (MMSE) reconstructor from interaction matrix, atmospheric covariance, and noise covariance.

### Utility Functions
- `rebin()`, `make_mask()`, `apply_mask()`: Array manipulation (GPU-accelerated)
- `rotshiftzoom_array()`: Array transformations with automatic GPU support
- `compute_derivatives_with_extrapolation()`: Numerical derivatives with edge handling
- `zern()`, `zern2phi()`: Zernike polynomial generation
- `dm3d_to_2d()`, `dm2d_to_3d()`: DM array format conversions
- Parameter file parsing: `parse_params_file()`, `extract_wfs_list()`, `extract_dm_list()`
- Filename generation: `generate_im_filename()`, `generate_pm_filename()`

## Dependencies

- [numpy](https://numpy.org/) - Core numerical operations
- [scipy](https://scipy.org/) - Scientific computing (CPU fallback)
- [matplotlib](https://matplotlib.org/) - Plotting and visualization
- [astropy](https://www.astropy.org/) - FITS file I/O
- [pyyaml](https://pyyaml.org/) - YAML configuration parsing
- [specula](https://github.com/ArcetriAdaptiveOptics/SPECULA) - AO simulation framework (required for data management)
- [cupy](https://cupy.dev/) - GPU acceleration (optional, recommended for large systems)

## Installation

### From source

```bash
# Clone repository
git clone https://github.com/ArcetriAdaptiveOptics/SynIM.git
cd SynIM

# Basic installation (CPU only)
pip install .

# With GPU support
pip install .[gpu]

# Development installation
pip install -e .[dev]
```

### Requirements

- Python ≥ 3.8
- CUDA Toolkit (for GPU support)
- Compatible GPU with CUDA support (for GPU acceleration)

## Configuration Files

SynIM supports both YAML and PRO (IDL-style) configuration files:

### YAML Format (Recommended)
```yaml
main:
  root_dir: /path/to/data
  pixel_pupil: 240
  pixel_pitch: 0.03333

pupilstop:
  mask_diam: 1.0
  obs_diam: 0.14

dm:
  height: 0.0
  rotation: 0.0
  ifunc_tag: if_dm_tag
  nmodes: 50

sh_lgs1:
  subap_on_diameter: 40
  wavelengthInNm: 589
  rotation: 0.0

source_lgs1:
  polar_coordinate: [30.0, 0.0]
  height: 90000.0

# For MCAO: optical sources for tomography
source_opt1:
  polar_coordinate: [0.0, 0.0]
  height: .inf
  weight: 1.0

layer1:
  height: 5000.0
  ifunc_tag: if_layer1_tag
```

### PRO Format (Legacy)
```idl
{main, 
  root_dir: '/path/to/data',
  pixel_pupil: 240,
  pixel_pitch: 0.03333
}

{dm, 
  height: 0.0,
  ifunc_tag: 'if_dm_tag'
}
```

## Examples and Tests

The repository includes:

- **Examples** (`examples/`): Demonstration scripts showing various use cases.

- **Tests** (`test/`): Unit and integration tests:
  - `test_intmat.py`: Interaction matrix computation tests
  - `test_projection.py`: Projection matrix tests
  - `test_covariance.py`: Covariance matrix tests
  - `test_rebin.py`: Array rebinning tests
  - `test_sh_intmat.py`: Shack-Hartmann specific tests

Run tests with:
```bash
cd test
python test_intmat.py
python test_projection.py
# etc.
```

## Performance Optimization

### Multi-WFS Computation
When computing interaction matrices for multiple WFS viewing the same DM:
- Use `compute_interaction_matrices()` instead of individual calls
- Automatically detects when WFS see DM from same direction
- Computes DM derivatives once, applies different WFS transformations
- Can provide 2-5× speedup for systems with multiple WFS

### GPU Acceleration
- Best performance with `precision=1` (single precision)
- Most effective for large pupil sizes (>200 pixels)
- Automatic memory management and data transfer
- Falls back to CPU for incompatible operations
- Monitor GPU memory usage for very large systems

### Covariance Matrix Caching
- Covariance matrices cached on disk (FITS format)
- Filename includes r0, L0, and component parameters
- Reused across multiple reconstructor computations
- Compatible with IDL-generated covariance files

## Notes

- Functions designed for compatibility with the SPECULA framework
- All arrays support both CPU (numpy) and GPU (cupy) backends
- FITS files compatible with IDL/PASSATA formats
- For details on parameters, see example configurations and function docstrings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Guido Agapito (INAF - Osservatorio Astrofisico di Arcetri)

---
For questions or issues, please open an issue on GitHub.