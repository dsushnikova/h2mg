# h2mg
This is a Python implementation of the H2-MG algorithm. The method combines the hierarchical structure of the H2 matrix with the multigrid approach, enabling efficient solutions of large systems with dense symmetric positive definite matrices.
## Requirenments

The minimum requirement by H2-MG is [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [Matplotlib](http://matplotlib.org/), [Numba](https://numba.pydata.org), and [Jupyter](https://jupyter.org).

## Quick Start

You can find Jupyter notebooks with 1D and 2D examples of the H2-MG algorithm for Gaussian and exponential kernels in the `notebooks` folder.

### Suggested Conda Environment

To ensure compatibility and avoid dependency issues, it is recommended to create a Conda environment for this project. Run the following commands:

```bash
conda create --name h2mg python=3.11 numpy scipy matplotlib numba jupyter -c conda-forge -y
conda activate h2mg'''
