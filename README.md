# SynIM

This repository contains Python code based on standard libraries for the computation of synthetic interaction matrices for adaptive optics systems equipped with Shack-Hartmann sensors.

## Main Features

- **Computation of synthetic interaction matrices** for various DM-WFS combinations, supporting both SCAO and MCAO configurations.
- - **Computation of projection matrices** for various DM-Layers combinations for MCAO / LTAO configurations.
- **Parametric management** via YAML or PRO configuration files, with automatic parsing of system, DM, WFS, and source parameters.
- **Support for different sensor types** (Shack-Hartmann, Pyramid) and deformable mirrors.
- **Utility functions** for mask manipulation, influence functions, Zernike polynomials, numerical derivatives, and rebinning operations.
- **Saving and loading** of interaction matrices and auxiliary data in SPECULA-compatible formats.

## Main File Structure

- `synim/synim.py`: Contains all low-level functions for interaction matrix generation, derivatives, etc.
- `synim/synpm.py`: Contains all low-level functions for projection matrix generation
- `synim/params_manager.py`: The `ParamsManager` class for centralized parameter management and batch generation of interaction matrices, projection matrices and covariance matrices. It can work with PASSATA or SPECULA parameters and calibration data.
- `synim/utils.py`: General utility functions.
- `synim/params_utils.py`: Utility functions for parsing, parameter extraction, filename generation, and MMSE reconstructor operators.

## Example Usage

### General case

```python
from synim.synim import synim

intmat = synim.interaction_matrix(pup_m,pup_mask,
                                  dm_array,dm_mask,
                                  dm_height,dm_rotation,
                                  nsubaps,wfs_rotation,
                                  wfs_translation,wfs_magnification,
                                  wfs_fov_arcsec,gs_pol_coo,
                                  gs_height,idx_valid_sa=idx_valid_sa,
                                  verbose=True,display=True,
                                  specula_convention=False)
```

###  Usage with SPECULA

```python
from synim.params_manager import ParamsManager

# Initialize the manager with a YAML or PRO configuration file
pm = ParamsManager('config.yml', verbose=True)

# Compute an interaction matrix for a specific WFS-DM combination
im = pm.compute_interaction_matrix(wfs_type='lgs', wfs_index=1, dm_index=1, display=True)

# Compute and save all interaction matrices for all combinations
pm.compute_interaction_matrices(output_im_dir='output/im', overwrite=True)
```

## Key Functions and Classes

- **ParamsManager**  
  Handles parameter loading, DM/WFS selection, parameter preparation for computation, and saving of interaction matrices.

- **interaction_matrix**  
  Main function in `synim.py` that computes the synthetic interaction matrix from system, DM, WFS, and source parameters.

- **Utility Functions**  
  - `rebin`, `make_mask`, `apply_mask`, `zern`, `zern2phi`, `compute_derivatives_with_extrapolation`, `rotshiftzoom_array`, etc.
  - Parameter file parsing: `parse_params_file`, `extract_wfs_list`, `extract_dm_list`, etc.
  - Filename generation: `generate_im_filename`, `generate_im_filenames`.


## Dependencies

- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [specula](https://github.com/SpecuLa-AO/specula) (required for data management and auxiliary classes)

## Notes

- The functions are designed to be compatible with the SPECULA framework.
- For details on supported parameters, see the example configuration files and inline documentation in the functions.

## Installation

You can install SynIM using pip (after cloning the repository):

```bash
pip install .
```

## Authors

- Guido Agapito.

---
For questions or issues, please open an issue on GitHub.
