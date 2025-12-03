SynIM Documentation
===================

**Synthetic Interaction Matrix generator for Adaptive Optics systems**

SynIM is a Python package for computing synthetic interaction matrices, projection matrices, and covariance matrices for adaptive optics (AO) systems. It supports both Single Conjugate AO (SCAO) and Multi-Conjugate AO (MCAO) configurations with optional GPU acceleration.

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Key Features
------------

- 🚀 **GPU acceleration** via CuPy for high-performance computation
- 🔧 **Interaction matrices** for DM-WFS combinations (SCAO/MCAO)  
- 📊 **Projection matrices** for DM-Layer tomography (MCAO/LTAO)
- 📈 **Covariance matrices** for MMSE reconstructor optimization
- ⚙️ **YAML/PRO configuration** with automatic parameter parsing
- 💾 **SPECULA-compatible** FITS format for data exchange
- 🎯 **Smart caching** to minimize redundant computations
- 🔀 **Multi-WFS optimization** for faster batch computation

Quick Example
-------------

.. code-block:: python

   import synim
   from synim.params_manager import ParamsManager

   # Initialize GPU (optional)
   synim.init(device_idx=0, precision=1)

   # Load configuration
   pm = ParamsManager('params_mcao.yml', verbose=True)

   # Compute interaction matrix
   im = pm.compute_interaction_matrix(
       wfs_type='lgs', 
       wfs_index=0,
       dm_index=0
   )

   # Compute all interaction matrices (batch processing)
   saved_files = pm.compute_interaction_matrices(
       output_im_dir='output/im',
       overwrite=False
   )

   # Assemble full MCAO interaction matrix
   im_full, n_slopes, modes, dms = pm.assemble_interaction_matrices(
       wfs_type='lgs',
       output_im_dir='output/im'
   )

   # Compute projection matrices for tomography
   pm_files = pm.compute_projection_matrices(
       output_dir='output/pm'
   )

   # Compute tomographic projection matrix
   p_opt, pm_full, info = pm.compute_tomographic_projection_matrix(
       reg_factor=1e-8,
       output_dir='output/pm'
   )

User Guide
----------

.. toctree::
   :maxdepth: 2

   installation
   general
   quickstart
   user_guide/index

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api/index

Examples
--------

Complete examples are available in the `examples/ <https://github.com/ArcetriAdaptiveOptics/SynIM/tree/main/examples>`_ directory:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example Script
     - Description
   * - ``run_synim_from_yaml_ngs.py``
     - NGS interaction matrices (SCAO)
   * - ``run_synim_from_yaml_lgs.py``
     - LGS interaction matrices (SCAO)
   * - ``run_synim_from_yaml_proj.py``
     - Projection matrices (MCAO)
   * - ``run_synim_covariance_matrices.py``
     - Atmospheric covariance matrices
   * - ``compare_recmats.py``
     - Compare different reconstructor types

Support
-------

- **Documentation**: https://synim.readthedocs.io
- **Source Code**: https://github.com/ArcetriAdaptiveOptics/SynIM
- **Issue Tracker**: https://github.com/ArcetriAdaptiveOptics/SynIM/issues

Citation
--------

If you use SynIM in your research, please cite:

.. code-block:: bibtex

   @software{synim2025,
     author = {Agapito, Guido},
     title = {SynIM: Synthetic Interaction Matrix Generator},
     year = {2025},
     url = {https://github.com/ArcetriAdaptiveOptics/SynIM}
   }

License
-------

SynIM is licensed under the MIT License. See the `LICENSE <https://github.com/ArcetriAdaptiveOptics/SynIM/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`