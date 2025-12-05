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


User Guide
----------

.. toctree::
   :maxdepth: 2

   installation
   general

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api/index

Examples
--------

Complete examples are available in the `examples/ <https://github.com/ArcetriAdaptiveOptics/SynIM/tree/main/examples>`_ directory:


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