Configuration Files
===================

SynIM supports both YAML and PRO (IDL-style) configuration files for defining AO system parameters.

Configuration Formats
---------------------

YAML Format (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

YAML files provide a clear, hierarchical structure:

.. code-block:: yaml

   # System parameters
   telescope:
     diameter: 8.0              # Telescope diameter [m]
     central_obstruction: 0.14  # Central obstruction ratio
   
   # Deformable mirrors
   dms:
     - tag: 'DM0'
       type: 'zonal'
       n_actuators: 41
       coupling: 0.2
       alt: 0.0                 # Conjugation altitude [m]
   
   # Wavefront sensors  
   wfss:
     - tag: 'WFS_LGS'
       type: 'SH'               # Shack-Hartmann
       n_subap: 40
       binning: 1
       sources:
         - type: 'LGS'
           altitude: 90000.0    # LGS altitude [m]
           zenith: 0.0
           azimuth: 0.0
           magnitude: 10.0

PRO Format (Legacy)
~~~~~~~~~~~~~~~~~~~

IDL-style parameter files are also supported:

.. code-block:: idl

   ; System parameters
   tel_diam = 8.0
   tel_cobs = 0.14
   
   ; Deformable mirror
   dm_tag = ['DM0']
   dm_type = ['zonal']
   dm_nacts = [41]
   dm_alt = [0.0]
   
   ; Wavefront sensor
   wfs_tag = ['WFS_LGS']
   wfs_type = ['SH']
   wfs_nsubap = [40]

Parameter Categories
--------------------

System Parameters
~~~~~~~~~~~~~~~~~

Global telescope and pupil configuration:

.. list-table::
   :header-rows: 1
   :widths: 20 50 15 15

   * - Parameter
     - Description
     - Unit
     - Required
   * - ``telescope.diameter``
     - Telescope diameter
     - meters
     - Yes
   * - ``telescope.central_obstruction``
     - Central obstruction ratio
     - fraction
     - No
   * - ``pupil.resolution``
     - Pupil sampling
     - pixels
     - Yes
   * - ``wavelength``
     - Reference wavelength
     - nm
     - Yes

Deformable Mirror Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration for each DM:

.. list-table::
   :header-rows: 1
   :widths: 20 50 15 15

   * - Parameter
     - Description
     - Unit
     - Required
   * - ``dms[].tag``
     - Unique identifier
     - string
     - Yes
   * - ``dms[].type``
     - DM type (zonal/modal)
     - string
     - Yes
   * - ``dms[].n_actuators``
     - Number of actuators
     - int
     - Yes
   * - ``dms[].coupling``
     - Actuator coupling
     - fraction
     - No
   * - ``dms[].alt``
     - Conjugation altitude
     - meters
     - Yes

Wavefront Sensor Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration for each WFS:

.. list-table::
   :header-rows: 1
   :widths: 20 50 15 15

   * - Parameter
     - Description
     - Unit
     - Required
   * - ``wfss[].tag``
     - Unique identifier
     - string
     - Yes
   * - ``wfss[].type``
     - Sensor type (SH/Pyramid)
     - string
     - Yes
   * - ``wfss[].n_subap``
     - Subapertures on diameter
     - int
     - Yes
   * - ``wfss[].pixel_scale``
     - Pixel scale
     - arcsec/pix
     - Yes
   * - ``wfss[].binning``
     - Detector binning
     - int
     - No

Source Parameters
~~~~~~~~~~~~~~~~~

Guide star configuration:

.. list-table::
   :header-rows: 1
   :widths: 20 50 15 15

   * - Parameter
     - Description
     - Unit
     - Required
   * - ``sources[].type``
     - Source type (NGS/LGS)
     - string
     - Yes
   * - ``sources[].altitude``
     - Source altitude
     - meters
     - LGS only
   * - ``sources[].zenith``
     - Zenith angle
     - degrees
     - Yes
   * - ``sources[].azimuth``
     - Azimuth angle
     - degrees
     - Yes
   * - ``sources[].magnitude``
     - Source magnitude
     - mag
     - Yes

Layer Parameters (MCAO)
~~~~~~~~~~~~~~~~~~~~~~~~

Atmospheric layer configuration:

.. list-table::
   :header-rows: 1
   :widths: 20 50 15 15

   * - Parameter
     - Description
     - Unit
     - Required
   * - ``layers[].altitude``
     - Layer height
     - meters
     - Yes
   * - ``layers[].cn2_fraction``
     - Turbulence fraction
     - fraction
     - Yes
   * - ``layers[].wind_speed``
     - Wind speed
     - m/s
     - No
   * - ``layers[].wind_direction``
     - Wind direction
     - degrees
     - No

Complete Examples
-----------------

SCAO Configuration
~~~~~~~~~~~~~~~~~~

Simple single-conjugate AO system:

.. code-block:: yaml

   # params_scao.yml
   telescope:
     diameter: 8.0
     central_obstruction: 0.14
   
   pupil:
     resolution: 240
   
   wavelength: 500.0  # nm
   
   dms:
     - tag: 'DM0'
       type: 'zonal'
       n_actuators: 41
       alt: 0.0
   
   wfss:
     - tag: 'WFS_NGS'
       type: 'SH'
       n_subap: 40
       pixel_scale: 0.5
       sources:
         - type: 'NGS'
           zenith: 0.0
           azimuth: 0.0
           magnitude: 8.0

MCAO Configuration
~~~~~~~~~~~~~~~~~~

Multi-conjugate AO with multiple WFS and DMs:

.. code-block:: yaml

   # params_mcao.yml
   telescope:
     diameter: 8.0
     central_obstruction: 0.14
   
   pupil:
     resolution: 480
   
   wavelength: 589.0  # LGS wavelength
   
   # Multiple DMs at different altitudes
   dms:
     - tag: 'DM0'
       type: 'zonal'
       n_actuators: 41
       alt: 0.0         # Ground layer
     - tag: 'DM4'
       type: 'zonal'
       n_actuators: 21
       alt: 12500.0     # High altitude
   
   # Multiple WFS with LGS constellation
   wfss:
     - tag: 'WFS_LGS1'
       type: 'SH'
       n_subap: 40
       sources:
         - type: 'LGS'
           altitude: 90000.0
           zenith: 30.0    # Off-axis
           azimuth: 0.0
           magnitude: 10.0
     
     - tag: 'WFS_LGS2'
       type: 'SH'
       n_subap: 40
       sources:
         - type: 'LGS'
           altitude: 90000.0
           zenith: 30.0
           azimuth: 90.0
           magnitude: 10.0
     
     - tag: 'WFS_NGS'
       type: 'SH'
       n_subap: 8
       sources:
         - type: 'NGS'
           zenith: 0.0
           azimuth: 0.0
           magnitude: 12.0   # TT star
   
   # Atmospheric layers for tomography
   layers:
     - altitude: 0.0
       cn2_fraction: 0.59
     - altitude: 500.0
       cn2_fraction: 0.02
     - altitude: 1000.0
       cn2_fraction: 0.04
     - altitude: 2000.0
       cn2_fraction: 0.06
     - altitude: 4000.0
       cn2_fraction: 0.01
     - altitude: 8000.0
       cn2_fraction: 0.05
     - altitude: 16000.0
       cn2_fraction: 0.23

Loading Configurations
----------------------

Using ParamsManager
~~~~~~~~~~~~~~~~~~~

The recommended way to load configurations:

.. code-block:: python

   from synim.params_manager import ParamsManager
   
   # Load YAML file
   pm = ParamsManager('params_mcao.yml', verbose=True)
   
   # Access configuration
   print(f"Telescope diameter: {pm.params['telescope']['diameter']} m")
   print(f"Number of DMs: {len(pm.dm_list)}")
   print(f"Number of WFS: {len(pm.wfs_list)}")

Direct Parsing
~~~~~~~~~~~~~~

For custom workflows:

.. code-block:: python

   from synim.params_utils import parse_params_file
   
   # Parse any supported format
   params = parse_params_file('config.yml')
   
   # Or PRO file
   params = parse_params_file('config.pro')

Validation
----------

SynIM automatically validates configurations:

.. code-block:: python

   from synim.params_utils import validate_opt_sources
   
   pm = ParamsManager('params.yml')
   
   # Validate optical sources for tomography
   validate_opt_sources(pm.params, verbose=True)
   # Output: Validating optical source configurations...
   #         ✓ All optical sources properly configured

Common validation checks:

- Required parameters present
- Valid ranges for physical quantities
- Consistent array dimensions
- Valid source types and positions
- DM-layer altitude compatibility

Best Practices
--------------

1. **Use YAML format** for new configurations (better readability)
2. **Comment your parameters** to document design choices
3. **Organize by subsystem** (telescope, DMs, WFSs, etc.)
4. **Include metadata** (date, author, system name)
5. **Version control** your configurations
6. **Validate early** before running expensive computations

.. code-block:: yaml

   # Good practice example
   # File: params_morfeo_v2.yml
   # Author: G. Agapito
   # Date: 2025-01-15
   # Description: MORFEO MCAO configuration for ELT
   
   metadata:
     system_name: "MORFEO"
     version: "2.0"
     telescope: "ELT"
     date: "2025-01-15"
   
   # ... rest of configuration ...

See Also
--------

- :doc:`interaction_matrices` - Computing interaction matrices
- :doc:`projection_matrices` - Computing projection matrices
- :doc:`covariance_matrices` - Computing covariance matrices
- :doc:`api/params_manager` - ParamsManager API referenceGeneral Information
===================

Overview
--------

SynIM (Synthetic Interaction Matrix) is a Python package designed for adaptive optics (AO) system analysis and design. It provides tools for computing:

- **Interaction matrices** between deformable mirrors (DMs) and wavefront sensors (WFS)
- **Projection matrices** for multi-conjugate adaptive optics (MCAO) tomography
- **Atmospheric covariance matrices** for optimal reconstructor design
- **MMSE reconstructors** with proper noise and turbulence covariance

Main Features
~~~~~~~~~~~~~

**GPU Acceleration**
   Built-in support for GPU acceleration via CuPy, with automatic fallback to CPU when GPU is unavailable or memory-limited.

**Multi-WFS Optimization**
   Intelligent batch processing that shares computations when multiple WFS view the same DM, providing 2-5× speedup.

**Flexible Configuration**
   Support for both YAML and PRO (IDL-style) configuration files with automatic parameter parsing and validation.

**SPECULA Integration**
   Compatible with the SPECULA adaptive optics simulation framework, using standard FITS file formats.

**Smart Caching**
   Automatic caching of computed matrices and intermediate results to avoid redundant computations.

**Workflow Optimization**
   Intelligent selection of computation workflows to minimize interpolation steps and maximize accuracy.

Architecture
------------

Core Components
~~~~~~~~~~~~~~~

SynIM is organized into several main modules:

**synim.py**
   Core functions for interaction matrix computation:
   
   - ``interaction_matrix()``: Main function with intelligent workflow selection
   - ``interaction_matrices_multi_wfs()``: Optimized multi-WFS computation
   - ``compute_derivatives_with_extrapolation()``: Numerical derivatives with edge handling

**synpm.py**
   Core functions for projection matrix computation:
   
   - ``projection_matrix()``: Main function for DM-Layer projection
   - Similar workflow optimization as interaction matrices

**params_manager.py**
   High-level interface via the ``ParamsManager`` class:
   
   - Centralized parameter management
   - Batch computation of matrices
   - Automatic file naming and organization
   - MMSE reconstructor computation

**params_utils.py**
   Parameter handling utilities:
   
   - Configuration file parsing (YAML/PRO)
   - Parameter validation
   - Filename generation
   - Array transformations

**utils.py**
   General utility functions:
   
   - Array rebinning and masking
   - Geometric transformations
   - Zernike polynomials
   - FITS I/O helpers

Computation Workflows
~~~~~~~~~~~~~~~~~~~~~

SynIM implements two main workflows for interaction matrix computation:

**SEPARATED Workflow**
   Used when transformations exist only in DM OR WFS:
   
   1. Apply DM transformations to influence function
   2. Compute numerical derivatives
   3. Apply WFS transformations to derivatives
   4. **Advantage**: Single interpolation step, maximum accuracy

**COMBINED Workflow**
   Used when both DM and WFS have transformations:
   
   1. Combine all transformations into single operation
   2. Apply combined transformation to influence function
   3. Compute derivatives on final grid
   4. **Advantage**: Avoids double interpolation artifacts

The workflow is automatically selected based on system geometry.

GPU Architecture
~~~~~~~~~~~~~~~~

GPU support is implemented through a flexible backend system:

**Automatic Backend Selection**
   .. code-block:: python
   
      import synim
      
      # CPU backend (numpy + scipy)
      synim.init(device_idx=-1, precision=1)
      
      # GPU backend (cupy + cupyx.scipy)
      synim.init(device_idx=0, precision=1)

**Memory Management**
   - Automatic data transfer between CPU and GPU
   - Graceful fallback to CPU on memory errors
   - Efficient caching of GPU arrays

**Precision Control**
   - ``precision=0``: Double precision (float64)
   - ``precision=1``: Single precision (float32, ~2× faster on GPU)

Configuration Files
-------------------

SynIM supports two configuration file formats for maximum flexibility.

YAML Format (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

YAML files provide a clear, hierarchical structure:

.. code-block:: yaml

   # System parameters
   telescope:
     diameter: 8.0              # Telescope diameter [m]
     central_obstruction: 0.14  # Central obstruction ratio
   
   pupil:
     resolution: 240            # Pupil sampling [pixels]
   
   wavelength: 589.0            # Reference wavelength [nm]
   
   # Deformable mirrors
   dms:
     - tag: 'DM0'
       type: 'zonal'
       n_actuators: 41
       coupling: 0.2
       alt: 0.0                 # Conjugation altitude [m]
   
   # Wavefront sensors  
   wfss:
     - tag: 'WFS_LGS'
       type: 'SH'               # Shack-Hartmann
       n_subap: 40
       binning: 1
       sources:
         - type: 'LGS'
           altitude: 90000.0    # LGS altitude [m]
           zenith: 0.0          # [degrees]
           azimuth: 0.0         # [degrees]
           magnitude: 10.0

PRO Format (Legacy)
~~~~~~~~~~~~~~~~~~~

IDL-style parameter files are also supported for backward compatibility:

.. code-block:: idl

   ; System parameters
   tel_diam = 8.0
   tel_cobs = 0.14
   pixel_pupil = 240
   
   ; Deformable mirror
   dm_tag = ['DM0']
   dm_type = ['zonal']
   dm_nacts = [41]
   dm_alt = [0.0]
   
   ; Wavefront sensor
   wfs_tag = ['WFS_LGS']
   wfs_type = ['SH']
   wfs_nsubap = [40]

Parameter Categories
~~~~~~~~~~~~~~~~~~~~

System Parameters
^^^^^^^^^^^^^^^^^

Global telescope and pupil configuration:

.. list-table::
   :header-rows: 1
   :widths: 25 45 15 15

   * - Parameter
     - Description
     - Unit
     - Required
   * - ``telescope.diameter``
     - Telescope diameter
     - meters
     - Yes
   * - ``telescope.central_obstruction``
     - Central obstruction ratio
     - fraction
     - No
   * - ``pupil.resolution``
     - Pupil sampling
     - pixels
     - Yes
   * - ``wavelength``
     - Reference wavelength
     - nm
     - Yes

Deformable Mirror Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configuration for each DM:

.. list-table::
   :header-rows: 1
   :widths: 25 45 15 15

   * - Parameter
     - Description
     - Unit
     - Required
   * - ``dms[].tag``
     - Unique identifier
     - string
     - Yes
   * - ``dms[].type``
     - DM type (zonal/modal)
     - string
     - Yes
   * - ``dms[].n_actuators``
     - Number of actuators
     - int
     - Yes
   * - ``dms[].coupling``
     - Actuator coupling
     - fraction
     - No
   * - ``dms[].alt``
     - Conjugation altitude
     - meters
     - Yes

Wavefront Sensor Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configuration for each WFS:

.. list-table::
   :header-rows: 1
   :widths: 25 45 15 15

   * - Parameter
     - Description
     - Unit
     - Required
   * - ``wfss[].tag``
     - Unique identifier
     - string
     - Yes
   * - ``wfss[].type``
     - Sensor type (SH/Pyramid)
     - string
     - Yes
   * - ``wfss[].n_subap``
     - Subapertures on diameter
     - int
     - Yes
   * - ``wfss[].pixel_scale``
     - Pixel scale
     - arcsec/pix
     - Yes
   * - ``wfss[].binning``
     - Detector binning
     - int
     - No

Source Parameters
^^^^^^^^^^^^^^^^^

Guide star configuration:

.. list-table::
   :header-rows: 1
   :widths: 25 45 15 15

   * - Parameter
     - Description
     - Unit
     - Required
   * - ``sources[].type``
     - Source type (NGS/LGS)
     - string
     - Yes
   * - ``sources[].altitude``
     - Source altitude (inf for NGS)
     - meters
     - Yes
   * - ``sources[].zenith``
     - Zenith angle
     - degrees
     - Yes
   * - ``sources[].azimuth``
     - Azimuth angle
     - degrees
     - Yes
   * - ``sources[].magnitude``
     - Source magnitude
     - mag
     - Yes

Layer Parameters (MCAO)
^^^^^^^^^^^^^^^^^^^^^^^^

Atmospheric layer configuration for tomography:

.. list-table::
   :header-rows: 1
   :widths: 25 45 15 15

   * - Parameter
     - Description
     - Unit
     - Required
   * - ``layers[].altitude``
     - Layer height
     - meters
     - Yes
   * - ``layers[].cn2_fraction``
     - Turbulence fraction
     - fraction
     - Yes
   * - ``layers[].wind_speed``
     - Wind speed
     - m/s
     - No
   * - ``layers[].wind_direction``
     - Wind direction
     - degrees
     - No

Complete Configuration Examples
--------------------------------

SCAO Configuration
~~~~~~~~~~~~~~~~~~

Simple single-conjugate AO system:

.. code-block:: yaml

   # params_scao.yml
   telescope:
     diameter: 8.0
     central_obstruction: 0.14
   
   pupil:
     resolution: 240
   
   wavelength: 500.0  # nm
   
   dms:
     - tag: 'DM0'
       type: 'zonal'
       n_actuators: 41
       coupling: 0.2
       alt: 0.0
   
   wfss:
     - tag: 'WFS_NGS'
       type: 'SH'
       n_subap: 40
       pixel_scale: 0.5
       sources:
         - type: 'NGS'
           altitude: .inf
           zenith: 0.0
           azimuth: 0.0
           magnitude: 8.0

MCAO Configuration
~~~~~~~~~~~~~~~~~~

Multi-conjugate AO with multiple WFS and DMs:

.. code-block:: yaml

   # params_mcao.yml
   telescope:
     diameter: 8.0
     central_obstruction: 0.14
   
   pupil:
     resolution: 480
   
   wavelength: 589.0  # LGS wavelength
   
   # Multiple DMs at different altitudes
   dms:
     - tag: 'DM0'
       type: 'zonal'
       n_actuators: 41
       coupling: 0.2
       alt: 0.0         # Ground layer
     
     - tag: 'DM4'
       type: 'zonal'
       n_actuators: 21
       coupling: 0.2
       alt: 12500.0     # High altitude
   
   # Multiple WFS with LGS constellation
   wfss:
     - tag: 'WFS_LGS1'
       type: 'SH'
       n_subap: 40
       pixel_scale: 0.5
       sources:
         - type: 'LGS'
           altitude: 90000.0
           zenith: 30.0    # Off-axis
           azimuth: 0.0
           magnitude: 10.0
     
     - tag: 'WFS_LGS2'
       type: 'SH'
       n_subap: 40
       pixel_scale: 0.5
       sources:
         - type: 'LGS'
           altitude: 90000.0
           zenith: 30.0
           azimuth: 90.0
           magnitude: 10.0
     
     - tag: 'WFS_NGS'
       type: 'SH'
       n_subap: 8
       pixel_scale: 1.0
       sources:
         - type: 'NGS'
           altitude: .inf
           zenith: 0.0
           azimuth: 0.0
           magnitude: 12.0   # TT star
   
   # Atmospheric layers for tomography
   layers:
     - altitude: 0.0
       cn2_fraction: 0.59
     - altitude: 500.0
       cn2_fraction: 0.02
     - altitude: 1000.0
       cn2_fraction: 0.04
     - altitude: 2000.0
       cn2_fraction: 0.06
     - altitude: 4000.0
       cn2_fraction: 0.01
     - altitude: 8000.0
       cn2_fraction: 0.05
     - altitude: 16000.0
       cn2_fraction: 0.23

Loading Configurations
----------------------

Using ParamsManager
~~~~~~~~~~~~~~~~~~~

The recommended way to load and use configurations:

.. code-block:: python

   from synim.params_manager import ParamsManager
   
   # Load YAML file
   pm = ParamsManager('params_mcao.yml', verbose=True)
   
   # Access configuration
   print(f"Telescope diameter: {pm.params['telescope']['diameter']} m")
   print(f"Number of DMs: {len(pm.dm_list)}")
   print(f"Number of WFS: {len(pm.wfs_list)}")
   
   # Access specific components
   dm0 = pm.dm_list[0]
   print(f"DM0 tag: {dm0['tag']}")
   print(f"DM0 altitude: {dm0['alt']} m")

Direct Parsing
~~~~~~~~~~~~~~

For custom workflows:

.. code-block:: python

   from synim.params_utils import parse_params_file
   
   # Parse any supported format
   params = parse_params_file('config.yml')
   
   # Or PRO file
   params = parse_params_file('config.pro')
   
   # Access parameters
   tel_diam = params['telescope']['diameter']

Validation
----------

SynIM automatically validates configurations during loading:

.. code-block:: python

   from synim.params_utils import validate_opt_sources
   from synim.params_manager import ParamsManager
   
   pm = ParamsManager('params.yml')
   
   # Validate optical sources for tomography
   validate_opt_sources(pm.params, verbose=True)
   # Output: Validating optical source configurations...
   #         ✓ All optical sources properly configured

Common validation checks include:

- Required parameters are present
- Values are within valid physical ranges
- Array dimensions are consistent
- Source types and positions are valid
- DM-layer altitude compatibility for MCAO

Best Practices
--------------

1. **Use YAML format** for new configurations (better readability and hierarchy)
2. **Comment your parameters** to document design choices and units
3. **Organize by subsystem** (telescope, DMs, WFSs, layers)
4. **Include metadata** at file beginning (date, author, system name)
5. **Version control** your configuration files
6. **Validate early** before running expensive computations
7. **Test incrementally** starting with simple SCAO before complex MCAO

Example with metadata:

.. code-block:: yaml

   # File: params_morfeo_v2.yml
   # Author: G. Agapito
   # Date: 2025-01-15
   # Description: MORFEO MCAO configuration for ELT
   
   metadata:
     system_name: "MORFEO"
     version: "2.0"
     telescope: "ELT"
     date: "2025-01-15"
     notes: "Updated LGS constellation geometry"
   
   telescope:
     diameter: 39.0  # ELT M1 diameter
     # ... rest of configuration ...

File Organization
-----------------

Recommended directory structure for SynIM projects:

.. code-block:: text

   project/
   ├── config/
   │   ├── params_scao.yml
   │   ├── params_mcao.yml
   │   └── params_ltao.yml
   ├── output/
   │   ├── im/          # Interaction matrices
   │   ├── pm/          # Projection matrices
   │   ├── rec/         # Reconstructors
   │   └── cov/         # Covariance matrices
   ├── calib/
   │   └── ifunc/       # Influence functions
   ├── scripts/
   │   ├── compute_im.py
   │   └── compute_rec.py
   └── analysis/
       └── results.ipynb

Performance Considerations
--------------------------

Computation Time
~~~~~~~~~~~~~~~~

Typical computation times on modern hardware:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Task
     - Pupil Size
     - CPU Time
     - GPU Time
   * - Single IM (SCAO)
     - 240 px
     - 2-5 sec
     - 0.5-1 sec
   * - Single IM (SCAO)
     - 480 px
     - 10-20 sec
     - 2-4 sec
   * - Multi-WFS IM (3 WFS)
     - 240 px
     - 8-15 sec
     - 2-3 sec
   * - Projection Matrix
     - 240 px
     - 5-10 sec
     - 1-2 sec
   * - Covariance Matrix
     - 240 px
     - 30-60 sec
     - 10-15 sec

*Times are approximate and depend on system complexity and hardware*

Memory Requirements
~~~~~~~~~~~~~~~~~~~

Approximate memory usage:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Array Type
     - Pupil 240px
     - Pupil 480px
     - Precision
   * - Pupil mask
     - 0.5 MB
     - 2 MB
     - float64
   * - Influence function
     - 1-5 MB
     - 4-20 MB
     - float64
   * - Interaction matrix
     - 10-50 MB
     - 40-200 MB
     - float64
   * - Covariance matrix
     - 50-200 MB
     - 200-800 MB
     - float64

**GPU Considerations:**
- GPU memory typically 8-24 GB
- Single precision reduces memory by 50%
- Large MCAO systems may require CPU processing

See Also
--------

- :doc:`installation` - Installation guide with GPU setup
- :doc:`user_guide/index` - Detailed user guides
- :doc:`api/index` - Complete API reference