# socu - Structured Optimization using CUDA

[![Preprint](https://img.shields.io/badge/Preprint-arXiv-blue.svg)](https://arxiv.org/abs/2601.03754)
[![Funding](https://img.shields.io/badge/Grant-NCCR%20Automation%20(51NF40__225155)-90e3dc.svg)](https://nccr-automation.ch/)
![License](https://img.shields.io/badge/License-BSD--2--Clause-brightgreen.svg)

## Overview

`socu` is a high-performance library for solving block tridiagonal linear systems using GPU-accelerated Cholesky factorization. These systems arise naturally in numerous real-time applications across engineering and scientific computing, including model predictive control (MPC), trajectory optimization, Kalman filtering, and robotics applications where temporal dynamics create structured sparsity patterns. The library is designed with extensibility in mind to support more general problem structures in the future.

Through a multi-stage permutation strategy based on nested dissection, `socu` reduces the computational complexity from **O(Nn³)** for sequential factorization to **O(log₂(N)n³)** when sufficient parallel resources are available, where `n` is the block size and `N` is the number of blocks.

The library is implemented using NVIDIA's Warp library as the computational backend and provides an interface to Jax.

## Installation

### Prerequisites

- Python 3.9 or later
- NVIDIA GPU with CUDA support

### Install from source

```bash
pip install git+https://github.com/PREDICT-EPFL/socu.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/PREDICT-EPFL/socu.git
cd socu
pip install -e .
```

## Usage

### Problem Setup

`socu` solves linear systems with block tridiagonal structure of the form:

$$\Psi x = b$$

where $\Psi$ is a symmetric positive definite block tridiagonal matrix:

$$
\Psi = \begin{bmatrix}
D_1 & E_1^T & & & \\
E_1 & D_2 & E_2^T & & \\
& E_2 & D_3 & \ddots & \\
& & \ddots & \ddots & E_{N-1}^T \\
& & & E_{N-1} & D_N
\end{bmatrix}
$$

with:
- $D_i \in \mathbb{R}^{n \times n}$ - symmetric diagonal blocks
- $E_i \in \mathbb{R}^{n \times n}$ - off-diagonal blocks
- $N$ - number of blocks
- $n$ - block size

### Data Layout

- **Diagonal blocks `L`**: shape $(N, n, n)$ - symmetric positive definite matrices
- **Off-diagonal blocks `E`**: shape $(N-1, n, n)$ for JAX, computed size for Warp
- **Right-hand side `b` / solution `x`**: shape $(N, n, 1)$ or $(N, n, m)$ for multiple RHS

### Basic Example with Warp Interface

```python
import numpy as np
import warp as wp
from socu.block_tridiag_solver import (
    create_cholesky_factor_launch,
    create_cholesky_solve_launch,
    create_cholesky_factor_and_solve_launch,
    create_cholesky_forward_substitution_launch,
    create_cholesky_backward_substitution_launch,
    calculate_off_diag_storage_len,
)

# Problem dimensions
n = 32  # block size
N = 100  # number of blocks

# Generate lower triangular Cholesky factors to ensure positive definiteness
D_chol = np.zeros((N, n, n))
E_chol = np.zeros((N-1, n, n))

for i in range(N):
    D_chol[i] = np.tril(np.random.randn(n, n)) + 10 * np.eye(n)
    if i < N-1:
        E_chol[i] = np.random.randn(n, n)

# Construct block tridiagonal matrix
L_np = np.zeros((N, n, n))
E_np = np.zeros((calculate_off_diag_storage_len(N), n, n)) # allocate correct size

for i in range(N):
    L_np[i] = D_chol[i] @ D_chol[i].T
    if i > 0:
        L_np[i] += E_chol[i-1] @ E_chol[i-1].T
    if i < N-1:
        E_np[i] = E_chol[i] @ D_chol[i].T

# Generate random right-hand side
b_np = np.random.randn(N, n, 1)

# Convert to Warp arrays
device = 'cuda'
dtype = wp.float64
L = wp.from_numpy(L_np, dtype=dtype, device=device)
E = wp.from_numpy(E_np, dtype=dtype, device=device)
x = wp.from_numpy(b_np, dtype=dtype, device=device)

# Create launch functions
cholesky_factor_launch = create_cholesky_factor_launch(
    L, E, device=device, dtype=dtype
)
cholesky_solve_launch = create_cholesky_solve_launch(
    L, E, x, device=device, dtype=dtype
)

# Solve: Factor then solve
cholesky_factor_launch()  # Compute Cholesky factorization
cholesky_solve_launch()   # Solve for x

# Get solution
x_solution = x.numpy()

# Verify solution correctness
# Reconstruct full matrix
Psi_full = np.zeros((N*n, N*n))
for i in range(N):
    Psi_full[i*n:(i+1)*n, i*n:(i+1)*n] = L_np[i]
    if i < N-1:
        Psi_full[(i+1)*n:(i+2)*n, i*n:(i+1)*n] = E_np[i]
        Psi_full[i*n:(i+1)*n, (i+1)*n:(i+2)*n] = E_np[i].T

# Check residual
residual = np.linalg.norm(b_np.flatten() - Psi_full @ x_solution.flatten())
assert residual < 1e-8, f"Solution error too large: {residual}"
print(f"Solution verified! Residual: {residual:.2e}")
```

### Combined Factor and Solve

For better performance, you can use the combined operation that interleaves factorization and forward substitution:

```python
# Reset x to b
x = wp.from_numpy(b_np, dtype=dtype, device=device)
L = wp.from_numpy(L_np, dtype=dtype, device=device)
E = wp.from_numpy(E_np, dtype=dtype, device=device)

# Combined factor and solve (faster)
cholesky_factor_and_solve_launch = create_cholesky_factor_and_solve_launch(
    L, E, x, device=device, dtype=dtype
)

cholesky_factor_and_solve_launch()
x_solution = x.numpy()

# Verify solution
residual = np.linalg.norm(b_np.flatten() - Psi_full @ x_solution.flatten())
assert residual < 1e-8, f"Solution error too large: {residual}"
```

### Separate Forward Substitution and Backward Substitution

In some cases, e.g., using `socu` to build your own project, you might need access to forward and backward substitutions separately:

```python
# Reset x to b, and L / E to the unfactored matrix
x = wp.from_numpy(b_np, dtype=dtype, device=device)
L = wp.from_numpy(L_np, dtype=dtype, device=device)
E = wp.from_numpy(E_np, dtype=dtype, device=device)

cholesky_factor_launch = create_cholesky_factor_launch(
    L, E, device=device, dtype=dtype
)
cholesky_forward_substitution_launch = create_cholesky_forward_substitution_launch(
    L, E, x, device=device, dtype=dtype
)
cholesky_backward_substitution_launch = create_cholesky_backward_substitution_launch(
    L, E, x, device=device, dtype=dtype
)

cholesky_factor_launch()
cholesky_forward_substitution_launch()
cholesky_backward_substitution_launch()

x_solution = x.numpy()

# Verify solution
residual = np.linalg.norm(b_np.flatten() - Psi_full @ x_solution.flatten())
assert residual < 1e-8, f"Solution error too large: {residual}"
```

### JAX Interface

```python
import jax.numpy as jnp
from jax import config
from socu.jax import (
    cholesky_factor,
    cholesky_solve,
    cholesky_factor_and_solve,
    forward_substitution,
    backward_substitution,
)

config.update("jax_enable_x64", True)

# Create JAX arrays
L_jax = jnp.array(L_np, dtype=jnp.float64)
E_jax = jnp.array(E_np, dtype=jnp.float64)
b_jax = jnp.array(b_np, dtype=jnp.float64)

# Separate factor and solve
L_factor, E_factor = cholesky_factor(L_jax, E_jax)
x_solution = cholesky_solve(L_factor, E_factor, b_jax)

# Verify solution
Psi_full_jax = jnp.array(Psi_full)
residual = jnp.linalg.norm(b_jax.flatten() - Psi_full_jax @ x_solution.flatten())
assert residual < 1e-8, f"Solution error too large: {residual}"

# Or combined
L_factor, E_factor, x_solution = cholesky_factor_and_solve(L_jax, E_jax, b_jax)

# Or with separate forward and backward substitution
y = forward_substitution(L_factor, E_factor, b_jax)
x_solution = backward_substitution(L_factor, E_factor, y)
```

### Performance Optimization

For optimal performance:

1. **Block size alignment**: Use block sizes $n$ that are multiples of 8 (for float64) or 16 (for float32). The JAX interface pads the input automatically by default (`pad_problem=True`), i.e., manual alignment is not needed, but the Warp interface doesn't.
2. **CUDA graphs**: Enable CUDA graphs for reduced kernel launch overhead:
   ```python
   cholesky_factor_launch = create_cholesky_factor_launch(
       L, E, device=device, dtype=dtype, use_cuda_graph=True
   )
   ```
3. **Precision selection**: Use `wp.float32` / `jnp.float32` for up to 4x speedup when precision allows

## Citing our Work

If you found socu useful in your scientific work, we encourage you to cite our preprint:
```
@misc{schwan2026socu,
  author = {Roland Schwan and Daniel Kuhn and Colin N. Jones},
  title = {{GPU}-Accelerated {Cholesky} Factorization of Block Tridiagonal Matrices},
  year = {2026},
  eprint = {arXiv:2601.03754},
}
```

## License

`socu` is released under the BSD 2-Clause License.
