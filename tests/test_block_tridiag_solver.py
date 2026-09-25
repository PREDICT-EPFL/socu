import pytest
import numpy as np
import scipy.linalg as la
import warp as wp

from socu.block_tridiag_solver import *


TOLERANCE = {
    wp.float32: 1e-4,
    wp.float64: 1e-10,
}


class TestBlockTridiagSolver:
    
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
    
    def prepare_test_data(self, n: int, N: int):
        """Prepare test data for given n and N"""
        _, A = self.generate_random_psd_block_tridiag(n, N)
        
        # Extract diagonal blocks (L)
        L_np = np.zeros((N, n, n), dtype=np.float64)
        for i in range(N):
            L_np[i, :, :] = A[i*n:(i+1)*n, i*n:(i+1)*n]
        
        # Extract off-diagonal blocks (E)
        n_E = calculate_off_diag_storage_len(N)
        E_np = np.zeros((n_E, n, n), dtype=np.float64)
        for i in range(N - 1):
            E_np[i, :, :] = A[(i+1)*n:(i+2)*n, i*n:(i+1)*n]
        
        # Generate random RHS
        np.random.seed(42)
        b_np = np.random.rand(N * n)
        
        # Compute reference solution using scipy
        Lc_np = la.cholesky(A, lower=True)
        y_np = la.solve_triangular(Lc_np, b_np, lower=True)
        x_ref = la.solve_triangular(Lc_np.T, y_np, lower=False)
        
        return A, L_np, E_np, b_np, x_ref
    
    def run_solver_test(self, n: int, N: int, use_cuda_graph: bool, dtype, device):
        """Run solver test for given parameters"""
        tolerance = TOLERANCE[dtype]
        
        A, L_np, E_np, b_np, x_ref = self.prepare_test_data(n, N)
        
        # Create warp arrays
        L = wp.from_numpy(L_np, dtype=dtype, device=device)
        E = wp.from_numpy(E_np, dtype=dtype, device=device)
        x = wp.from_numpy(b_np.reshape(N, n, 1), dtype=dtype, device=device)
        
        # Create launch functions
        cholesky_factor_launch = create_cholesky_factor_launch(
            L, E, device=device, use_cuda_graph=use_cuda_graph, dtype=dtype
        )
        cholesky_solve_launch = create_cholesky_solve_launch(
            L, E, x, device=device, use_cuda_graph=use_cuda_graph, dtype=dtype
        )
        cholesky_factor_and_solve_launch = create_cholesky_factor_and_solve_launch(
            L, E, x, device=device, use_cuda_graph=use_cuda_graph, dtype=dtype
        )
        forward_substitution_launch = create_cholesky_forward_substitution_launch(
            L, E, x, device=device, use_cuda_graph=use_cuda_graph, dtype=dtype
        )
        backward_substitution_launch = create_cholesky_backward_substitution_launch(
            L, E, x, device=device, use_cuda_graph=use_cuda_graph, dtype=dtype
        )

        # Test separate factor and solve
        wp.copy(L, wp.from_numpy(L_np, dtype=dtype))
        wp.copy(E, wp.from_numpy(E_np, dtype=dtype))
        wp.copy(x, wp.from_numpy(b_np.reshape(N, n, 1), dtype=dtype))
        
        cholesky_factor_launch()
        cholesky_solve_launch()
        
        x_result = x.numpy().flatten()
        x_error = la.norm(x_ref - x_result)
        residual_error = la.norm(b_np - A @ x_result)
        
        assert x_error < tolerance, \
            f"Solution error too large for n={n}, N={N}, dtype={dtype}, cuda_graph={use_cuda_graph}: {x_error}"
        assert residual_error < tolerance, \
            f"Residual error too large for n={n}, N={N}, dtype={dtype}, cuda_graph={use_cuda_graph}: {residual_error}"
        
        # Test combined factor and solve
        wp.copy(L, wp.from_numpy(L_np, dtype=dtype))
        wp.copy(E, wp.from_numpy(E_np, dtype=dtype))
        wp.copy(x, wp.from_numpy(b_np.reshape(N, n, 1), dtype=dtype))
        
        cholesky_factor_and_solve_launch()
        
        x_result_combined = x.numpy().flatten()
        x_error_combined = la.norm(x_ref - x_result_combined)
        residual_error_combined = la.norm(b_np - A @ x_result_combined)
        
        assert x_error_combined < tolerance, \
            f"Combined solution error too large for n={n}, N={N}, dtype={dtype}, cuda_graph={use_cuda_graph}: {x_error_combined}"
        assert residual_error_combined < tolerance, \
            f"Combined residual error too large for n={n}, N={N}, dtype={dtype}, cuda_graph={use_cuda_graph}: {residual_error_combined}"
        
        # Verify both methods give same result
        consistency_error = la.norm(x_result - x_result_combined)
        assert consistency_error < tolerance, \
            f"Inconsistent results between separate and combined methods for n={n}, N={N}, dtype={dtype}, cuda_graph={use_cuda_graph}: {consistency_error}"

        # Test separate forward and backward substitution (L, E hold the factor
        # from the combined run above)
        wp.copy(x, wp.from_numpy(b_np.reshape(N, n, 1), dtype=dtype))

        forward_substitution_launch()
        backward_substitution_launch()

        x_result_split = x.numpy().flatten()
        split_consistency_error = la.norm(x_result - x_result_split)
        assert split_consistency_error < tolerance, \
            f"Inconsistent results between solve and forward+backward substitution for n={n}, N={N}, dtype={dtype}, cuda_graph={use_cuda_graph}: {split_consistency_error}"

    @pytest.mark.parametrize("N", [1, 2, 3, 4, 7, 8, 20, 100])
    @pytest.mark.parametrize("use_cuda_graph", [False, True])
    @pytest.mark.parametrize("n", [1, 10, 32, 36, 64, 68])
    @pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
    def test_cholesky_solver(self, n, N, use_cuda_graph, dtype, device):
        """Test Cholesky solver with various parameters"""
        if use_cuda_graph and not device.is_cuda:
            pytest.skip("CUDA graphs require a CUDA device")
        self.run_solver_test(n, N, use_cuda_graph, dtype, device)
