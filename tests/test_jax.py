import pytest
import numpy as np
import scipy.linalg as la
import jax
import jax.numpy as jnp
from functools import partial
from jax import config

# Enable 64-bit precision in JAX
config.update("jax_enable_x64", True)

from socu.jax import (
    backward_substitution,
    cholesky_factor,
    cholesky_factor_and_solve,
    cholesky_solve,
    forward_substitution,
)


TOLERANCE = {
    jnp.float32: 1e-4,
    jnp.float64: 1e-10,
}


class TestJaxBlockTridiagSolver:
    
    def generate_random_psd_block_tridiag(self, n: int, N: int, seed: int = 42):
        """Generate a random positive semi-definite block tridiagonal matrix"""
        np.random.seed(seed)
        L = np.zeros((N * n, N * n))
        for i in range(N):
            L[i*n:(i+1)*n, i*n:(i+1)*n] = np.tril(np.random.rand(n, n)) + 10 * np.eye(n)
            if i < N - 1:
                L[(i+1)*n:(i+2)*n, i*n:(i+1)*n] = np.random.rand(n, n)
        A = L @ L.T
        return L, A
    
    def prepare_test_data(self, n: int, N: int, dtype):
        """Prepare test data for given n and N"""
        _, A = self.generate_random_psd_block_tridiag(n, N)
        
        # Extract diagonal blocks (L)
        L_np = np.zeros((N, n, n), dtype=np.float64)
        for i in range(N):
            L_np[i, :, :] = A[i*n:(i+1)*n, i*n:(i+1)*n]
        
        # Extract off-diagonal blocks (E)
        E_np = np.zeros((N - 1, n, n), dtype=np.float64)
        for i in range(N - 1):
            E_np[i, :, :] = A[(i+1)*n:(i+2)*n, i*n:(i+1)*n]
        
        # Generate random RHS
        np.random.seed(42)
        b_np = np.random.rand(N * n)
        
        # Compute reference solution using scipy
        Lc_np = la.cholesky(A, lower=True)
        y_np = la.solve_triangular(Lc_np, b_np, lower=True)
        x_ref = la.solve_triangular(Lc_np.T, y_np, lower=False)
        
        # Convert to JAX arrays with appropriate dtype
        jax_dtype = jnp.float32 if dtype == jnp.float32 else jnp.float64
        L_jax = jnp.array(L_np, dtype=jax_dtype)
        E_jax = jnp.array(E_np, dtype=jax_dtype)
        b_jax = jnp.array(b_np.reshape(N, n, 1), dtype=jax_dtype)
        
        return A, L_jax, E_jax, b_jax, x_ref, Lc_np
    
    def run_solver_test(self, n: int, N: int, pad_problem: bool, dtype: jnp.dtype):
        """Run solver test for given parameters"""
        tolerance = TOLERANCE[dtype]
        
        A, L_jax, E_jax, b_jax, x_ref, Lc_np = self.prepare_test_data(n, N, dtype)
        
        # Test separate factor and solve
        L_factor, E_factor = cholesky_factor(L_jax, E_jax, pad_problem=pad_problem)
        x_result = cholesky_solve(L_factor, E_factor, b_jax, pad_problem=pad_problem)
        
        # Convert result to numpy for comparison
        x_result_np = np.array(x_result).flatten()
        
        # Compute errors
        x_error = la.norm(x_ref - x_result_np)
        residual_error = la.norm(np.array(b_jax).flatten() - A @ x_result_np)
        
        assert x_error < tolerance, \
            f"Solution error too large for n={n}, N={N}, dtype={dtype}, pad_problem={pad_problem}: {x_error}"
        assert residual_error < tolerance, \
            f"Residual error too large for n={n}, N={N}, dtype={dtype}, pad_problem={pad_problem}: {residual_error}"
        
        # Test combined factor and solve
        L_combined, E_combined, x_result_combined = cholesky_factor_and_solve(L_jax, E_jax, b_jax, pad_problem=pad_problem)
        
        # Convert combined result to numpy for comparison
        x_result_combined_np = np.array(x_result_combined).flatten()
        
        # Compute errors for combined method
        x_error_combined = la.norm(x_ref - x_result_combined_np)
        residual_error_combined = la.norm(np.array(b_jax).flatten() - A @ x_result_combined_np)
        
        assert x_error_combined < tolerance, \
            f"Combined solution error too large for n={n}, N={N}, dtype={dtype}, pad_problem={pad_problem}: {x_error_combined}"
        assert residual_error_combined < tolerance, \
            f"Combined residual error too large for n={n}, N={N}, dtype={dtype}, pad_problem={pad_problem}: {residual_error_combined}"
        
        # Verify both methods give same result
        consistency_error = la.norm(x_result_np - x_result_combined_np)
        assert consistency_error < tolerance, \
            f"Inconsistent results between separate and combined methods for n={n}, N={N}, dtype={dtype}, pad_problem={pad_problem}: {consistency_error}"

        # Test separate forward and backward substitution
        y_result = forward_substitution(L_factor, E_factor, b_jax, pad_problem=pad_problem)
        x_result_split = backward_substitution(L_factor, E_factor, y_result, pad_problem=pad_problem)

        x_result_split_np = np.array(x_result_split).flatten()
        split_consistency_error = la.norm(x_result_np - x_result_split_np)
        assert split_consistency_error < tolerance, \
            f"Inconsistent results between solve and forward+backward substitution for n={n}, N={N}, dtype={dtype}, pad_problem={pad_problem}: {split_consistency_error}"
    
    
    @pytest.mark.parametrize("N", [1, 2, 3, 4, 7, 8, 20, 100])
    @pytest.mark.parametrize("pad_problem", [True, False])
    @pytest.mark.parametrize("n", [1, 10, 32, 36, 64, 68])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_jax_cholesky_solver(self, n, N, pad_problem, dtype):
        """Test JAX Cholesky solver with various parameters"""
        self.run_solver_test(n, N, pad_problem, dtype)

    @pytest.mark.parametrize("pad_problem", [True, False])
    def test_jax_cholesky_factor_and_solve_vmap(self, pad_problem):
        """Test that vmap uses the native batched JAX/Warp path."""
        n = 4
        N = 3
        batch_size = 2

        Ls = []
        Es = []
        bs = []
        x_refs = []
        for seed in range(batch_size):
            A, L_jax, E_jax, b_jax, x_ref, _ = self.prepare_test_data(n, N, jnp.float64)
            Ls.append(L_jax + 0.1 * seed * jnp.eye(n, dtype=jnp.float64)[None, :, :])
            Es.append(E_jax)
            bs.append(b_jax)

            A_shifted = A.copy()
            for i in range(N):
                A_shifted[i*n:(i+1)*n, i*n:(i+1)*n] += 0.1 * seed * np.eye(n)
            x_refs.append(la.solve(A_shifted, np.array(b_jax).reshape(N * n, 1)).reshape(N, n, 1))

        L_batch = jnp.stack(Ls)
        E_batch = jnp.stack(Es)
        b_batch = jnp.stack(bs)
        x_ref_batch = np.stack(x_refs)

        _, _, x_result = jax.vmap(
            partial(cholesky_factor_and_solve, pad_problem=pad_problem),
            in_axes=(0, 0, 0),
        )(
            L_batch,
            E_batch,
            b_batch,
        )

        assert la.norm(np.array(x_result) - x_ref_batch) < 1e-10

    @pytest.mark.parametrize("pad_problem", [True, False])
    def test_jax_cholesky_solve_vmap_rhs_only(self, pad_problem):
        """Test vmap over multiple RHS batches with shared factors."""
        n = 4
        N = 3
        batch_size = 2

        A, L_jax, E_jax, b_jax, _, _ = self.prepare_test_data(n, N, jnp.float64)
        L_factor, E_factor = cholesky_factor(L_jax, E_jax, pad_problem=pad_problem)

        b_batch = jnp.stack([b_jax, 2.0 * b_jax])
        x_result = jax.vmap(
            partial(cholesky_solve, pad_problem=pad_problem),
            in_axes=(None, None, 0),
        )(
            L_factor,
            E_factor,
            b_batch,
        )

        x_ref = la.solve(A, np.array(b_jax).reshape(N * n, 1)).reshape(N, n, 1)
        x_ref_batch = np.stack([x_ref, 2.0 * x_ref])

        assert la.norm(np.array(x_result) - x_ref_batch) < 1e-10
