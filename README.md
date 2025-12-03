# SynIM

**Synthetic Interaction Matrix generator for Adaptive Optics systems**

[![Documentation Status](https://readthedocs.org/projects/synim/badge/?version=latest)](https://synim.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

SynIM is a Python package for computing synthetic interaction matrices and projection matrices for adaptive optics (AO) systems. It supports both Single Conjugate AO (SCAO) and Multi-Conjugate AO (MCAO) configurations with GPU acceleration.

## Key Features

- 🚀 **GPU acceleration** via CuPy for high-performance computation
- 🔧 **Interaction matrices** for DM-WFS combinations (SCAO/MCAO)
- 📊 **Projection matrices** for DM-Layer tomography (MCAO/LTAO)
- 📈 **Covariance matrices** for MMSE reconstructor optimization
- ⚙️ **YAML/PRO configuration** with automatic parameter parsing
- 💾 **SPECULA-compatible** FITS format for data exchange
- 🎯 **Smart caching** to minimize redundant computations

## Quick Start

### Installation

```bash
pip install synim
```

### Basic Usage

```python
import synim
from synim.params_manager import ParamsManager

# Initialize GPU (optional)
synim.init(device_idx=0, precision=1)

# Load configuration and compute interaction matrix
pm = ParamsManager('config.yml')
im = pm.compute_interaction_matrix(wfs_type='lgs')
```

## Documentation

Full documentation available at: **[synim.readthedocs.io](https://synim.readthedocs.io)**

## Examples

See the [`examples/`](examples/) directory for complete working examples:

- `run_synim_from_yaml_ngs.py` - NGS interaction matrices
- `run_synim_from_yaml_lgs.py` - LGS interaction matrices  
- `run_synim_from_yaml_proj.py` - Projection matrices for MCAO
- `run_synim_covariance_matrices.py` - Atmospheric covariance

## GPU Performance

SynIM provides significant speedups on GPU for large problems:

```python
import synim

# Use GPU with single precision for best performance
synim.init(device_idx=0, precision=1)

# Automatic fallback to CPU if GPU memory exceeded
```

## Requirements

- Python ≥ 3.8
- numpy, scipy, matplotlib
- [specula](https://github.com/ArcetriAdaptiveOptics/SPECULA) - AO simulation framework
- [cupy](https://cupy.dev/) - GPU acceleration (optional, requires CUDA)

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Citation

If you use SynIM in your research, please cite:

```bibtex
@software{synim2025,
  author = {Agapito, Guido},
  title = {SynIM: Synthetic Interaction Matrix Generator},
  year = {2025},
  url = {https://github.com/ArcetriAdaptiveOptics/SynIM}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Authors

- **Guido Agapito** - INAF - Osservatorio Astrofisico di Arcetri

## Acknowledgments

Built for the [SPECULA](https://github.com/ArcetriAdaptiveOptics/SPECULA) adaptive optics simulation framework.