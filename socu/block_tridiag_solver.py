import math
from functools import lru_cache

import warp as wp

from socu.utils import create_cuda_graph_callback
from socu.blas.gemm import create_gemm_nn_blocked_func, create_gemm_tn_blocked_func
from socu.blas.potrf import create_potrf_l_blocked_func
from socu.blas.syrk import create_syrk_ln_blocked_func, create_syrk_lt_blocked_func
from socu.blas.trsm import create_trsm_rltn_blocked_func, create_trsm_llnn_blocked_func, create_trsm_lltn_blocked_func

def calculate_recursive_iterations(horizon: int) -> int:
    assert horizon > 0
    # equivalent to floor(log2(horizon)) + 1
    return horizon.bit_length()


def calculate_off_diag_storage_len(horizon: int) -> int:
    iterations = calculate_recursive_iterations(horizon)
    n_off_diag = 0
    for i in range(iterations):
        n_off_diag += horizon // (2 ** i) - 1
    return n_off_diag


def optimal_problem_settings(n: int, dtype):
    if dtype == wp.float64:
        factor_fuse_threshold = 16
        solve_fuse_threshold = 32
        pad_multiple_fuse = 2
        pad_multiple_blocked = 8
        
    elif dtype == wp.float32:
        factor_fuse_threshold = 32
        solve_fuse_threshold = 32
        pad_multiple_fuse = 4
        pad_multiple_blocked = 16
    else:
        raise ValueError('dtype must be wp.float64 or wp.float32')
    
    settings = {
        'block_dim': {
            'factor': 64,
            'solve': 64,
        },
        'block_size': {
            'factor': 32 if n % 32 == 0 else 16,
            'solve': 32 if n % 32 == 0 else 16,
        },
        'pad_multiple': {
            'factor': pad_multiple_fuse if n < factor_fuse_threshold else pad_multiple_blocked,
            'solve': pad_multiple_fuse if n < solve_fuse_threshold else pad_multiple_blocked,
        }
    }
    if n < factor_fuse_threshold:
        settings['block_size']['factor'] = factor_fuse_threshold
    if n < solve_fuse_threshold:
        settings['block_size']['solve'] = solve_fuse_threshold

    return settings


def create_cholesky_factor_forward_substituition_iteration_func(with_forward_sub: bool,
                                                                n: int,
                                                                n_rhs=1,
                                                                dtype=wp.float64):

    @wp.func
    def cholesky_forward_substitution_iteration_func(batch_id: int,
                                                     i: int,
                                                     stride: int,
                                                     prev_off_diag_offset: int,
                                                     curr_off_diag_offset: int,
                                                     next_off_diag_offset: int,
                                                     horizon: int,
                                                     L: wp.array4d[dtype], # type: ignore
                                                     E: wp.array4d[dtype], # type: ignore
                                                     y: wp.array4d[dtype]): # type: ignore

        L_i = wp.tile_load(L[batch_id, i], shape=(n, n))
        if i + stride < horizon:
            L_ii = wp.tile_load(L[batch_id, i+stride], shape=(n, n))

        if wp.static(with_forward_sub):
            y_i = wp.tile_load(y[batch_id, i], shape=(n, n_rhs))

        if stride > 1:
            
            if i < horizon - (stride // 2):
                # left looking
                prev_od_idx = i // (stride // 2)
                E_prev_od = wp.tile_load(E[batch_id, prev_off_diag_offset+prev_od_idx], shape=(n, n))
                E_prev_od_T = wp.tile_transpose(E_prev_od)
                wp.tile_matmul(E_prev_od_T, E_prev_od, L_i, alpha=-1.0)

            if i + stride < horizon - (stride // 2):
                # left-right looking
                prev_od_next_idx = (i + stride) // (stride // 2)
                E_prev_od_next = wp.tile_load(E[batch_id, prev_off_diag_offset+prev_od_next_idx], shape=(n, n))
                E_prev_od_next_T = wp.tile_transpose(E_prev_od_next)
                wp.tile_matmul(E_prev_od_next_T, E_prev_od_next, L_ii, alpha=-1.0)

        # current column
        wp.tile_cholesky_inplace(L_i)
        wp.tile_store(L[batch_id, i], L_i)

        if wp.static(with_forward_sub):
            wp.tile_lower_solve_inplace(L_i, y_i)
            wp.tile_store(y[batch_id, i], y_i)

        if i + stride < horizon:
            # current column
            curr_od_idx = i // stride
            E_curr_od = wp.tile_load(E[batch_id, curr_off_diag_offset+curr_od_idx], shape=(n, n))
            E_curr_od_T = wp.tile_transpose(E_curr_od)
            wp.tile_lower_solve_inplace(L_i, E_curr_od_T)
            E_curr_od = wp.tile_transpose(E_curr_od_T)
            wp.tile_store(E[batch_id, curr_off_diag_offset+curr_od_idx], E_curr_od)

            # right looking
            wp.tile_matmul(E_curr_od, E_curr_od_T, L_ii, alpha=-1.0)

            if wp.static(with_forward_sub):
                tmp1 = wp.tile_matmul(E_curr_od, y_i)
                wp.tile_atomic_add(y[batch_id, i+stride], -tmp1)

        if i + stride < horizon:
            wp.tile_store(L[batch_id, i+stride], L_ii)

        if i >= stride:
            # current column
            curr_od_prev_idx = (i - stride) // stride
            E_curr_od_prev = wp.tile_load(E[batch_id, curr_off_diag_offset+curr_od_prev_idx], shape=(n, n))
            wp.tile_lower_solve_inplace(L_i, E_curr_od_prev)
            wp.tile_store(E[batch_id, curr_off_diag_offset+curr_od_prev_idx], E_curr_od_prev)

            if wp.static(with_forward_sub):
                E_curr_od_prev_T = wp.tile_transpose(E_curr_od_prev)
                tmp2 = wp.tile_matmul(E_curr_od_prev_T, y_i)
                wp.tile_atomic_add(y[batch_id, i-stride], -tmp2)

            if i + stride < horizon:
                # right looking
                next_od_idx = (i - stride) // (stride * 2)
                E_next_od = wp.tile_load(E[batch_id, next_off_diag_offset+next_od_idx], shape=(n, n))
                wp.tile_matmul(E_curr_od, E_curr_od_prev, E_next_od, alpha=-1.0)
                wp.tile_store(E[batch_id, next_off_diag_offset+next_od_idx], E_next_od)

    return cholesky_forward_substitution_iteration_func


@lru_cache(maxsize=1)
def create_cholesky_factor_iteration_kernel(n: int, dtype=wp.float64):

    module = wp.Module('cholesky_factor_iteration_kernel', None)
    module.options['enable_backward'] = False

    @wp.kernel(module=module)
    def cholesky_factor_iteration_kernel(stride: int,
                                         prev_off_diag_offset: int,
                                         curr_off_diag_offset: int,
                                         next_off_diag_offset: int,
                                         horizon: int,
                                         L: wp.array4d[dtype], # type: ignore
                                         E: wp.array4d[dtype]): # type: ignore

        batch_id, tid, _ = wp.tid()
        
        i = stride - 1 + tid * 2 * stride

        wp.static(create_cholesky_factor_forward_substituition_iteration_func(
            with_forward_sub=False,
            n=n,
            dtype=dtype,
        ))(
            batch_id,
            i,
            stride,
            prev_off_diag_offset,
            curr_off_diag_offset,
            next_off_diag_offset,
            horizon,
            L,
            E,
            E, # dummpy value for y which is not used
        )

    return cholesky_factor_iteration_kernel


@lru_cache(maxsize=1)
def create_cholesky_factor_potrf_l_blocked_kernel(n: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('cholesky_factor_potrf_l_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def cholesky_factor_potrf_l_blocked_kernel(stride: int, L: wp.array4d[dtype]): # type: ignore
        
        batch_id, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride
        wp.static(create_potrf_l_blocked_func(n, block_size, dtype))(batch_id, L, i)

    return cholesky_factor_potrf_l_blocked_kernel


@lru_cache(maxsize=1)
def create_cholesky_factor_trsm_rltn_blocked_kernel(n: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('cholesky_factor_trsm_rltn_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def cholesky_factor_trsm_rltn_blocked_kernel(stride: int,
                                                 curr_off_diag_offset: int,
                                                 horizon: int,
                                                 L: wp.array4d[dtype], # type: ignore
                                                 E: wp.array4d[dtype]): # type: ignore
        
        batch_id, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride
        curr_od_idx = i // stride

        if i + stride < horizon:
            wp.static(create_trsm_rltn_blocked_func(n, n, block_size, dtype))(
                batch_id,
                L, i,
                E, curr_off_diag_offset + curr_od_idx
            )

    return cholesky_factor_trsm_rltn_blocked_kernel


@lru_cache(maxsize=1)
def create_cholesky_factor_trsm_llnn_blocked_kernel(n: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('cholesky_factor_trsm_llnn_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def cholesky_factor_trsm_llnn_blocked_kernel(stride: int,
                                                 curr_off_diag_offset: int,
                                                 L: wp.array4d[dtype], # type: ignore
                                                 E: wp.array4d[dtype]): # type: ignore
        
        batch_id, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride
        curr_od_prev_idx = (i - stride) // stride

        if i >= stride:
            wp.static(create_trsm_llnn_blocked_func(n, n, block_size, dtype))(
                batch_id,
                L, i,
                E, curr_off_diag_offset + curr_od_prev_idx
            )

    return cholesky_factor_trsm_llnn_blocked_kernel


@lru_cache(maxsize=1)
def create_cholesky_factor_syrk_ln_blocked_kernel(n: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('cholesky_factor_syrk_ln_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def cholesky_factor_syrk_ln_blocked_kernel(stride: int,
                                               curr_off_diag_offset: int,
                                               horizon: int,
                                               L: wp.array4d[dtype], # type: ignore
                                               E: wp.array4d[dtype]): # type: ignore
        
        batch_id, k, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride
        curr_od_idx = i // stride

        if i + stride < horizon:
            wp.static(create_syrk_ln_blocked_func(n, block_size, atomic=True, dtype=dtype))(
                batch_id, k,
                E, curr_off_diag_offset + curr_od_idx,
                L, i + stride
            )

    return cholesky_factor_syrk_ln_blocked_kernel


@lru_cache(maxsize=1)
def create_cholesky_factor_syrk_lt_blocked_kernel(n: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('cholesky_factor_syrk_lt_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def cholesky_factor_syrk_lt_blocked_kernel(stride: int,
                                               curr_off_diag_offset: int,
                                               L: wp.array4d[dtype], # type: ignore
                                               E: wp.array4d[dtype]): # type: ignore
        
        batch_id, k, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride
        curr_od_prev_idx = (i - stride) // stride

        if i >= stride:
            wp.static(create_syrk_lt_blocked_func(n, block_size, atomic=True, dtype=dtype))(
                batch_id, k,
                E, curr_off_diag_offset + curr_od_prev_idx,
                L, i - stride
            )

    return cholesky_factor_syrk_lt_blocked_kernel


@lru_cache(maxsize=1)
def create_cholesky_factor_gemm_nn_blocked_kernel(n: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('cholesky_factor_gemm_nn_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def cholesky_factor_gemm_nn_blocked_kernel(stride: int,
                                               curr_off_diag_offset: int,
                                               next_off_diag_offset: int,
                                               horizon: int,
                                               E: wp.array4d[dtype]): # type: ignore
        
        batch_id, p, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride
        curr_od_prev_idx = (i - stride) // stride
        curr_od_idx = i // stride
        next_od_idx = (i - stride) // (stride * 2)

        if i + stride < horizon and i >= stride:
            wp.static(create_gemm_nn_blocked_func(n, n, n, block_size, atomic=False, dtype=dtype))(
                batch_id, p,
                E, curr_off_diag_offset + curr_od_idx,
                E, curr_off_diag_offset + curr_od_prev_idx,
                E, next_off_diag_offset + next_od_idx
            )

    return cholesky_factor_gemm_nn_blocked_kernel


@lru_cache(maxsize=1)
def create_cholesky_factor_forward_substituition_iteration_kernel(n: int, n_rhs=1, dtype=wp.float64):

    module = wp.Module('cholesky_factor_forward_substituition_iteration_kernel', None)
    module.options['enable_backward'] = False
    
    @wp.kernel(module=module)
    def cholesky_factor_forward_substituition_iteration_kernel(stride: int,
                                                               prev_off_diag_offset: int,
                                                               curr_off_diag_offset: int,
                                                               next_off_diag_offset: int,
                                                               horizon: int,
                                                               L: wp.array4d[dtype], # type: ignore
                                                               E: wp.array4d[dtype], # type: ignore
                                                               y: wp.array4d[dtype]): # type: ignore

        batch_id, tid, _ = wp.tid()
        
        i = stride - 1 + tid * 2 * stride

        wp.static(create_cholesky_factor_forward_substituition_iteration_func(
            with_forward_sub=True,
            n=n,
            n_rhs=n_rhs,
            dtype=dtype,
        ))(
            batch_id,
            i,
            stride,
            prev_off_diag_offset,
            curr_off_diag_offset,
            next_off_diag_offset,
            horizon,
            L,
            E,
            y,
        )

    return cholesky_factor_forward_substituition_iteration_kernel


@lru_cache(maxsize=1)
def create_forward_substitution_iteration_kernel(n: int, n_rhs=1, dtype=wp.float64):

    module = wp.Module('forward_substitution_iteration_kernel', None)
    module.options['enable_backward'] = False
    
    @wp.kernel(module=module)
    def forward_substitution_iteration_kernel(stride: int,
                                              curr_off_diag_offset: int,
                                              horizon: int,
                                              L: wp.array4d[dtype], # type: ignore
                                              E: wp.array4d[dtype], # type: ignore
                                              y: wp.array4d[dtype]): # type: ignore

        batch_id, tid, _ = wp.tid()

        i = stride - 1 + tid * 2 * stride
        
        L_i = wp.tile_load(L[batch_id, i], shape=(n, n))
        y_i = wp.tile_load(y[batch_id, i], shape=(n, n_rhs))
        wp.tile_lower_solve_inplace(L_i, y_i)
        wp.tile_store(y[batch_id, i], y_i)

        if i // stride < horizon // stride - 1:
            curr_od_idx = i // stride
            E_curr_od = wp.tile_load(E[batch_id, curr_off_diag_offset+curr_od_idx], shape=(n, n))
            tmp1 = wp.tile_matmul(E_curr_od, y_i)
            wp.tile_atomic_add(y[batch_id, i+stride], -tmp1)

        if i >= stride:
            curr_od_prev_idx = (i - stride) // stride
            E_curr_od_prev = wp.tile_load(E[batch_id, curr_off_diag_offset+curr_od_prev_idx], shape=(n, n))
            E_curr_od_prev_T = wp.tile_transpose(E_curr_od_prev)
            tmp2 = wp.tile_matmul(E_curr_od_prev_T, y_i)
            wp.tile_atomic_add(y[batch_id, i-stride], -tmp2)

    return forward_substitution_iteration_kernel


@lru_cache(maxsize=1)
def create_forward_substitution_trsm_llnn_blocked_kernel(n: int, n_rhs: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('forward_substitution_trsm_llnn_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def forward_substitution_trsm_llnn_blocked_kernel(stride: int,
                                                      L: wp.array4d[dtype], # type: ignore
                                                      y: wp.array4d[dtype]): # type: ignore
        
        batch_id, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride

        wp.static(create_trsm_llnn_blocked_func(n, n_rhs, block_size, dtype))(
            batch_id,
            L, i,
            y, i
        )

    return forward_substitution_trsm_llnn_blocked_kernel


@lru_cache(maxsize=1)
def create_forward_substitution_gemm_nn_blocked_kernel(n: int, n_rhs: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('forward_substitution_gemm_nn_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def forward_substitution_gemm_nn_blocked_kernel(stride: int,
                                                    curr_off_diag_offset: int,
                                                    horizon: int,
                                                    E: wp.array4d[dtype], # type: ignore
                                                    y: wp.array4d[dtype]): # type: ignore
        
        batch_id, p, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride
        curr_od_idx = i // stride

        if i // stride < horizon // stride - 1:
            wp.static(create_gemm_nn_blocked_func(n, n_rhs, n, block_size, atomic=True, dtype=dtype))(
                batch_id, p,
                E, curr_off_diag_offset + curr_od_idx,
                y, i,
                y, i + stride
            )

    return forward_substitution_gemm_nn_blocked_kernel


@lru_cache(maxsize=1)
def create_forward_substitution_gemm_tn_blocked_kernel(n: int, n_rhs: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('forward_substitution_gemm_tn_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def forward_substitution_gemm_tn_blocked_kernel(stride: int,
                                                    curr_off_diag_offset: int,
                                                    E: wp.array4d[dtype], # type: ignore
                                                    y: wp.array4d[dtype]): # type: ignore
        
        batch_id, p, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride
        curr_od_prev_idx = (i - stride) // stride

        if i >= stride:
            wp.static(create_gemm_tn_blocked_func(n, n_rhs, n, block_size, atomic=True, dtype=dtype))(
                batch_id, p,
                E, curr_off_diag_offset + curr_od_prev_idx,
                y, i,
                y, i - stride
            )

    return forward_substitution_gemm_tn_blocked_kernel


@lru_cache(maxsize=1)
def create_backward_substitution_iteration_kernel(n: int, n_rhs=1, dtype=wp.float64):

    module = wp.Module('backward_substitution_iteration_kernel', None)
    module.options['enable_backward'] = False
    
    @wp.kernel(module=module)
    def backward_substitution_iteration_kernel(stride: int,
                                               curr_off_diag_offset: int,
                                               horizon: int,
                                               L: wp.array4d[dtype], # type: ignore
                                               E: wp.array4d[dtype], # type: ignore
                                               x: wp.array4d[dtype]): # type: ignore

        batch_id, tid, _ = wp.tid()

        i = stride - 1 + tid * 2 * stride

        x_i = wp.tile_load(x[batch_id, i], shape=(n, n_rhs))

        if i + stride < horizon:
            curr_od_idx = i // stride
            E_curr_od = wp.tile_load(E[batch_id, curr_off_diag_offset+curr_od_idx], shape=(n, n))
            E_curr_od_T = wp.tile_transpose(E_curr_od)
            x_next = wp.tile_load(x[batch_id, i+stride], shape=(n, n_rhs))
            wp.tile_matmul(E_curr_od_T, x_next, x_i, alpha=-1.0)

        if i >= stride:
            curr_od_prev_idx = (i - stride) // stride
            E_curr_od_prev = wp.tile_load(E[batch_id, curr_off_diag_offset+curr_od_prev_idx], shape=(n, n))
            x_prev = wp.tile_load(x[batch_id, i-stride], shape=(n, n_rhs))
            wp.tile_matmul(E_curr_od_prev, x_prev, x_i, alpha=-1.0)

        L_i = wp.tile_load(L[batch_id, i], shape=(n, n))
        L_i_T = wp.tile_transpose(L_i)
        wp.tile_upper_solve_inplace(L_i_T, x_i)
        wp.tile_store(x[batch_id, i], x_i)

    return backward_substitution_iteration_kernel


@lru_cache(maxsize=1)
def create_backward_substitution_gemm_tn_blocked_kernel(n: int, n_rhs: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('backward_substitution_gemm_tn_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def backward_substitution_gemm_tn_blocked_kernel(stride: int,
                                                     curr_off_diag_offset: int,
                                                     horizon: int,
                                                     E: wp.array4d[dtype], # type: ignore
                                                     y: wp.array4d[dtype]): # type: ignore
        
        batch_id, p, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride
        curr_od_idx = i // stride

        if i + stride < horizon:
            wp.static(create_gemm_tn_blocked_func(n, n_rhs, n, block_size, atomic=True, dtype=dtype))(
                batch_id, p,
                E, curr_off_diag_offset + curr_od_idx,
                y, i + stride,
                y, i
            )

    return backward_substitution_gemm_tn_blocked_kernel


@lru_cache(maxsize=1)
def create_backward_substitution_gemm_nn_blocked_kernel(n: int, n_rhs: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('backward_substitution_gemm_nn_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def backward_substitution_gemm_nn_blocked_kernel(stride: int,
                                                     curr_off_diag_offset: int,
                                                     E: wp.array4d[dtype], # type: ignore
                                                     y: wp.array4d[dtype]): # type: ignore
        
        batch_id, p, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride
        curr_od_prev_idx = (i - stride) // stride

        if i >= stride:
            wp.static(create_gemm_nn_blocked_func(n, n_rhs, n, block_size, atomic=True, dtype=dtype))(
                batch_id, p,
                E, curr_off_diag_offset + curr_od_prev_idx,
                y, i - stride,
                y, i
            )

    return backward_substitution_gemm_nn_blocked_kernel


@lru_cache(maxsize=1)
def create_backward_substitution_trsm_lltn_blocked_kernel(n: int, n_rhs: int, block_size: int, dtype=wp.float64):
    
    module = wp.Module('backward_substitution_trsm_lltn_blocked_kernel', None)
    module.options['enable_backward'] = False
    module.options['max_unroll'] = 0

    @wp.kernel(module=module)
    def backward_substitution_trsm_lltn_blocked_kernel(stride: int,
                                                       L: wp.array4d[dtype], # type: ignore
                                                       y: wp.array4d[dtype]): # type: ignore
        
        batch_id, tid, _ = wp.tid()
        i = stride - 1 + tid * 2 * stride

        wp.static(create_trsm_lltn_blocked_func(n, n_rhs, block_size, dtype))(
            batch_id,
            L, i,
            y, i
        )

    return backward_substitution_trsm_lltn_blocked_kernel


def create_cholesky_factor_launch(L: wp.array,
                                  E: wp.array,
                                  block_dim=None,
                                  block_size=None,
                                  dtype=wp.float64,
                                  device=None,
                                  stream=None,
                                  use_cuda_graph=False):
    
    assert E.ndim == L.ndim, "L and E must have the same number of dimensions"
    
    if L.ndim == 4:
        batch_dim = L.shape[0]
    elif L.ndim == 3:
        batch_dim = 1
        L = L.reshape((1, *L.shape))
        E = E.reshape((1, *E.shape))
    else:
        raise ValueError("L must have 4 or 3 dimensions")
    
    assert E.shape[0] == batch_dim, "L and E must have same batch dimension"

    horizon = L.shape[1]
    n = L.shape[2]
    assert L.shape[3] == n, "The last two dimensions of L must be the same"

    assert E.shape[2] == n and E.shape[3] == n, "L and E must have same block size"
    n_E = calculate_off_diag_storage_len(horizon)
    assert E.shape[1] == n_E, "E has incorrect size, use calculate_off_diag_storage_len to calculate the correct number of blocks"
    
    iterations = calculate_recursive_iterations(horizon)

    opt_settings = optimal_problem_settings(n, dtype)
    if block_dim is None:
        block_dim = opt_settings['block_dim']['factor']
    if block_size is None:
        block_size = opt_settings['block_size']['factor']

    if n < block_size:
        launch = wp.launch_tiled(create_cholesky_factor_iteration_kernel(n, dtype),
                                 record_cmd=True,
                                 device=device,
                                 stream=stream,
                                 dim=[batch_dim, 1],
                                 inputs=[0, 0, 0, 0, horizon, L, E],
                                 block_dim=block_dim)
        
        def callback():
            stride = 1
            prev_off_diag_offset = 0
            curr_off_diag_offset = 0
            next_off_diag_offset = horizon - 1
            for _ in range(iterations):
                dim = (horizon + stride) // (2 * stride)

                launch.set_dim([batch_dim, dim, block_dim])
                launch.set_params([
                    stride,
                    prev_off_diag_offset,
                    curr_off_diag_offset,
                    next_off_diag_offset,
                    # horizon, L, E are already set
                ])
                launch.launch()
                
                stride *= 2
                prev_off_diag_offset = curr_off_diag_offset
                curr_off_diag_offset = next_off_diag_offset
                next_off_diag_offset += horizon // stride - 1
        
    else:
        if stream is None:
            device = wp.get_device(device)
            stream = device.stream
        else:
            device = stream.device
        
        stream2 = wp.Stream(device)
        stream3 = wp.Stream(device)

        potrf_launch = wp.launch_tiled(create_cholesky_factor_potrf_l_blocked_kernel(n, block_size, dtype),
                                       record_cmd=True,
                                       device=device,
                                       dim=[batch_dim, 1],
                                       inputs=[0, L],
                                       block_dim=block_dim)

        trsm_rltn_launch = wp.launch_tiled(create_cholesky_factor_trsm_rltn_blocked_kernel(n, block_size, dtype),
                                           record_cmd=True,
                                           device=device,
                                           dim=[batch_dim, 1],
                                           inputs=[0, 0, horizon, L, E],
                                           block_dim=block_dim)
        
        trsm_llnn_launch = wp.launch_tiled(create_cholesky_factor_trsm_llnn_blocked_kernel(n, block_size, dtype),
                                           record_cmd=True,
                                           device=device,
                                           dim=[batch_dim, 1],
                                           inputs=[0, 0, L, E],
                                           block_dim=block_dim)
        
        syrk_ln_launch = wp.launch_tiled(create_cholesky_factor_syrk_ln_blocked_kernel(n, block_size, dtype),
                                         record_cmd=True,
                                         device=device,
                                         dim=[batch_dim, 1, 1],
                                         inputs=[0, 0, horizon, L, E],
                                         block_dim=block_dim)
        
        syrk_lt_launch = wp.launch_tiled(create_cholesky_factor_syrk_lt_blocked_kernel(n, block_size, dtype),
                                         record_cmd=True,
                                         device=device,
                                         dim=[batch_dim, 1, 1],
                                         inputs=[0, 0, L, E],
                                         block_dim=block_dim)
        
        gemm_nn_launch = wp.launch_tiled(create_cholesky_factor_gemm_nn_blocked_kernel(n, block_size, dtype),
                                         record_cmd=True,
                                         device=device,
                                         dim=[batch_dim, 1, 1],
                                         inputs=[0, 0, 0, horizon, E],
                                         block_dim=block_dim)

        def callback():
            num_blocks_per_row = math.ceil(n / block_size)
            num_tril_blocks = num_blocks_per_row * (num_blocks_per_row + 1) // 2
            num_gemm_blocks = num_blocks_per_row * num_blocks_per_row

            stride = 1
            curr_off_diag_offset = 0
            next_off_diag_offset = horizon - 1
            for i in range(iterations):
                dim = (horizon + stride) // (2 * stride)

                if i > 0:
                    stream.wait_stream(stream2)

                potrf_launch.set_dim([batch_dim, dim, block_dim])
                potrf_launch.set_params([stride]) # L is already set
                potrf_launch.launch(stream=stream)

                if i < iterations - 1:
                    stream2.wait_stream(stream)

                    if i > 0:
                        stream.wait_stream(stream3)
                        stream2.wait_stream(stream3)
                    
                    trsm_rltn_launch.set_dim([batch_dim, dim, block_dim])
                    trsm_rltn_launch.set_params([stride, curr_off_diag_offset]) # horizon, L, E are already set
                    trsm_rltn_launch.launch(stream=stream)

                    trsm_llnn_launch.set_dim([batch_dim, dim, block_dim])
                    trsm_llnn_launch.set_params([stride, curr_off_diag_offset]) # L, E are already set
                    trsm_llnn_launch.launch(stream=stream2)

                    stream3.wait_stream(stream)
                    stream3.wait_stream(stream2)

                    syrk_ln_launch.set_dim([batch_dim, num_tril_blocks, dim, block_dim])
                    syrk_ln_launch.set_params([stride, curr_off_diag_offset]) # horizon, L, E are already set
                    syrk_ln_launch.launch(stream=stream)

                    syrk_lt_launch.set_dim([batch_dim, num_tril_blocks, dim, block_dim])
                    syrk_lt_launch.set_params([stride, curr_off_diag_offset]) # L, E are already set
                    syrk_lt_launch.launch(stream=stream2)

                    gemm_nn_launch.set_dim([batch_dim, num_gemm_blocks, dim, block_dim])
                    gemm_nn_launch.set_params([stride, curr_off_diag_offset, next_off_diag_offset]) # horizon, E are already set
                    gemm_nn_launch.launch(stream=stream3)
                
                stride *= 2
                curr_off_diag_offset = next_off_diag_offset
                next_off_diag_offset += horizon // stride - 1

            if iterations > 1:
                stream.wait_stream(stream2)
                stream.wait_stream(stream3)

    if use_cuda_graph:
        return create_cuda_graph_callback(callback, device, stream)

    return callback


def create_cholesky_factor_and_solve_launch(L: wp.array,
                                            E: wp.array,
                                            x: wp.array,
                                            block_dim=None,
                                            block_size=None,
                                            dtype=wp.float64,
                                            device=None,
                                            stream=None,
                                            use_cuda_graph=False):
    
    assert E.ndim == L.ndim and x.ndim == L.ndim, "L, E, and x must have the same number of dimensions"
    
    if L.ndim == 4:
        batch_dim = L.shape[0]
    elif L.ndim == 3:
        batch_dim = 1
        L = L.reshape((1, *L.shape))
        E = E.reshape((1, *E.shape))
        x = x.reshape((1, *x.shape))
    else:
        raise ValueError("L must have 4 or 3 dimensions")
    
    assert E.shape[0] == batch_dim and x.shape[0] == batch_dim, "L, E, and x must have same batch dimension"

    horizon = L.shape[1]
    n = L.shape[2]
    assert L.shape[3] == n, "The last two dimensions of L must be the same"

    assert E.shape[2] == n and E.shape[3] == n, "L and E must have same block size"
    n_E = calculate_off_diag_storage_len(horizon)
    assert E.shape[1] == n_E, "E has incorrect size, use calculate_off_diag_storage_len to calculate the correct number of blocks"

    n_rhs = x.shape[3]
    assert x.shape[1] == horizon and x.shape[2] == n, "x has incorrect dimensions"
    
    iterations = calculate_recursive_iterations(horizon)

    opt_settings = optimal_problem_settings(n, dtype)
    if block_dim is None:
        block_dim = opt_settings['block_dim']['factor']
    if block_size is None:
        block_size = opt_settings['block_size']['factor']

    if n < block_size:
        fac_fwd_launch = wp.launch_tiled(create_cholesky_factor_forward_substituition_iteration_kernel(n, n_rhs, dtype),
                                     record_cmd=True,
                                     device=device,
                                     stream=stream,
                                     dim=[batch_dim, 1],
                                     inputs=[0, 0, 0, 0, horizon, L, E, x],
                                     block_dim=block_dim)
    
        back_launch = wp.launch_tiled(create_backward_substitution_iteration_kernel(n, n_rhs, dtype),
                                    record_cmd=True,
                                    device=device,
                                    stream=stream,
                                    dim=[batch_dim, 1],
                                    inputs=[0, 0, horizon, L, E, x],
                                    block_dim=block_dim)
    
        def callback():
            stride = 1
            prev_off_diag_offset = 0
            curr_off_diag_offset = 0
            next_off_diag_offset = horizon - 1

            for _ in range(iterations):
                dim = (horizon + stride) // (2 * stride)

                fac_fwd_launch.set_dim([batch_dim, dim, block_dim])
                fac_fwd_launch.set_params([
                    stride,
                    prev_off_diag_offset,
                    curr_off_diag_offset,
                    next_off_diag_offset,
                    # horizon, L, E, x are already set
                ])
                fac_fwd_launch.launch()
                
                stride *= 2
                prev_off_diag_offset = curr_off_diag_offset
                curr_off_diag_offset = next_off_diag_offset
                next_off_diag_offset += horizon // stride - 1

            for _ in reversed(range(iterations)):
                stride //= 2
                curr_off_diag_offset -= horizon // stride - 1

                dim = (horizon + stride) // (2 * stride)

                back_launch.set_dim([batch_dim, dim, block_dim])
                back_launch.set_params([
                    stride,
                    curr_off_diag_offset,
                    # horizon, L, E, x are already set
                ])
                back_launch.launch()

    else:
        if stream is None:
            device = wp.get_device(device)
            stream = device.stream
        else:
            device = stream.device
        
        stream2 = wp.Stream(device)
        stream3 = wp.Stream(device)

        stream4 = wp.Stream(device)
        stream5 = wp.Stream(device)

        fac_potrf_launch = wp.launch_tiled(create_cholesky_factor_potrf_l_blocked_kernel(n, block_size, dtype),
                                           record_cmd=True,
                                           device=device,
                                           dim=[batch_dim, 1],
                                           inputs=[0, L],
                                           block_dim=block_dim)

        fac_trsm_rltn_launch = wp.launch_tiled(create_cholesky_factor_trsm_rltn_blocked_kernel(n, block_size, dtype),
                                               record_cmd=True,
                                               device=device,
                                               dim=[batch_dim, 1],
                                               inputs=[0, 0, horizon, L, E],
                                               block_dim=block_dim)
        
        fac_trsm_llnn_launch = wp.launch_tiled(create_cholesky_factor_trsm_llnn_blocked_kernel(n, block_size, dtype),
                                               record_cmd=True,
                                               device=device,
                                               dim=[batch_dim, 1],
                                               inputs=[0, 0, L, E],
                                               block_dim=block_dim)
        
        fac_syrk_ln_launch = wp.launch_tiled(create_cholesky_factor_syrk_ln_blocked_kernel(n, block_size, dtype),
                                             record_cmd=True,
                                             device=device,
                                             dim=[batch_dim, 1, 1],
                                             inputs=[0, 0, horizon, L, E],
                                             block_dim=block_dim)
        
        fac_syrk_lt_launch = wp.launch_tiled(create_cholesky_factor_syrk_lt_blocked_kernel(n, block_size, dtype),
                                             record_cmd=True,
                                             device=device,
                                             dim=[batch_dim, 1, 1],
                                             inputs=[0, 0, L, E],
                                             block_dim=block_dim)
        
        fac_gemm_nn_launch = wp.launch_tiled(create_cholesky_factor_gemm_nn_blocked_kernel(n, block_size, dtype),
                                             record_cmd=True,
                                             device=device,
                                             dim=[batch_dim, 1, 1],
                                             inputs=[0, 0, 0, horizon, E],
                                             block_dim=block_dim)

        forward_trsm_llnn_launch = wp.launch_tiled(create_forward_substitution_trsm_llnn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                   record_cmd=True,
                                                   device=device,
                                                   dim=[batch_dim, 1],
                                                   inputs=[0, L, x],
                                                   block_dim=block_dim)

        forward_gemm_nn_launch = wp.launch_tiled(create_forward_substitution_gemm_nn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                 record_cmd=True,
                                                 device=device,
                                                 dim=[batch_dim, 1, 1],
                                                 inputs=[0, 0, horizon, E, x],
                                                 block_dim=block_dim)

        forward_gemm_tn_launch = wp.launch_tiled(create_forward_substitution_gemm_tn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                 record_cmd=True,
                                                 device=device,
                                                 dim=[batch_dim, 1, 1],
                                                 inputs=[0, 0, E, x],
                                                 block_dim=block_dim)

        backward_gemm_tn_launch = wp.launch_tiled(create_backward_substitution_gemm_tn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                  record_cmd=True,
                                                  device=device,
                                                  dim=[batch_dim, 1, 1],
                                                  inputs=[0, 0, horizon, E, x],
                                                  block_dim=block_dim)
        
        backward_gemm_nn_launch = wp.launch_tiled(create_backward_substitution_gemm_nn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                  record_cmd=True,
                                                  device=device,
                                                  dim=[batch_dim, 1, 1],
                                                  inputs=[0, 0, E, x],
                                                  block_dim=block_dim)
        
        backward_trsm_lltn_launch = wp.launch_tiled(create_backward_substitution_trsm_lltn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                    record_cmd=True,
                                                    device=device,
                                                    dim=[batch_dim, 1],
                                                    inputs=[0, L, x],
                                                    block_dim=block_dim)

        def callback():
            num_blocks_per_row = math.ceil(n / block_size)
            num_tril_blocks = num_blocks_per_row * (num_blocks_per_row + 1) // 2
            num_fac_gemm_blocks = num_blocks_per_row * num_blocks_per_row

            num_blocks_per_rhs_col = math.ceil(n_rhs / block_size)
            num_solve_gemm_blocks = num_blocks_per_row * num_blocks_per_rhs_col

            stride = 1
            curr_off_diag_offset = 0
            next_off_diag_offset = horizon - 1
            for i in range(iterations):
                dim = (horizon + stride) // (2 * stride)

                if i > 0:
                    stream.wait_stream(stream2)

                fac_potrf_launch.set_dim([batch_dim, dim, block_dim])
                fac_potrf_launch.set_params([stride]) # L is already set
                fac_potrf_launch.launch(stream=stream)

                stream4.wait_stream(stream)
                if i > 0:
                    stream4.wait_stream(stream5)

                forward_trsm_llnn_launch.set_dim([batch_dim, dim, block_dim])
                forward_trsm_llnn_launch.set_params([stride]) # L, x are already set
                forward_trsm_llnn_launch.launch(stream=stream4)

                if i < iterations - 1:
                    stream2.wait_stream(stream)
                    stream5.wait_stream(stream4)

                    if i > 0:
                        stream.wait_stream(stream3)
                        stream2.wait_stream(stream3)
                    
                    fac_trsm_rltn_launch.set_dim([batch_dim, dim, block_dim])
                    fac_trsm_rltn_launch.set_params([stride, curr_off_diag_offset]) # horizon, L, E are already set
                    fac_trsm_rltn_launch.launch(stream=stream)

                    fac_trsm_llnn_launch.set_dim([batch_dim, dim, block_dim])
                    fac_trsm_llnn_launch.set_params([stride, curr_off_diag_offset]) # L, E are already set
                    fac_trsm_llnn_launch.launch(stream=stream2)

                    stream4.wait_stream(stream)
                    stream5.wait_stream(stream2)

                    forward_gemm_nn_launch.set_dim([batch_dim, num_solve_gemm_blocks, dim, block_dim])
                    forward_gemm_nn_launch.set_params([stride, curr_off_diag_offset]) # horizon, E, x are already set
                    forward_gemm_nn_launch.launch(stream=stream4)

                    forward_gemm_tn_launch.set_dim([batch_dim, num_solve_gemm_blocks, dim, block_dim])
                    forward_gemm_tn_launch.set_params([stride, curr_off_diag_offset]) # E, x are already set
                    forward_gemm_tn_launch.launch(stream=stream5)

                    stream3.wait_stream(stream)
                    stream3.wait_stream(stream2)

                    fac_syrk_ln_launch.set_dim([batch_dim, num_tril_blocks, dim, block_dim])
                    fac_syrk_ln_launch.set_params([stride, curr_off_diag_offset]) # horizon, L, E are already set
                    fac_syrk_ln_launch.launch(stream=stream)

                    fac_syrk_lt_launch.set_dim([batch_dim, num_tril_blocks, dim, block_dim])
                    fac_syrk_lt_launch.set_params([stride, curr_off_diag_offset]) # L, E are already set
                    fac_syrk_lt_launch.launch(stream=stream2)

                    fac_gemm_nn_launch.set_dim([batch_dim, num_fac_gemm_blocks, dim, block_dim])
                    fac_gemm_nn_launch.set_params([stride, curr_off_diag_offset, next_off_diag_offset]) # horizon, E are already set
                    fac_gemm_nn_launch.launch(stream=stream3)
                
                stride *= 2
                curr_off_diag_offset = next_off_diag_offset
                next_off_diag_offset += horizon // stride - 1

            if iterations > 1:
                stream.wait_stream(stream2)
                stream.wait_stream(stream3)

            for i in reversed(range(iterations)):
                stride //= 2
                curr_off_diag_offset -= horizon // stride - 1

                dim = (horizon + stride) // (2 * stride)

                if i < iterations - 1:
                    stream5.wait_stream(stream4)

                    backward_gemm_tn_launch.set_dim([batch_dim, num_solve_gemm_blocks, dim, block_dim])
                    backward_gemm_tn_launch.set_params([stride, curr_off_diag_offset]) # horizon, E, x are already set
                    backward_gemm_tn_launch.launch(stream=stream4)

                    backward_gemm_nn_launch.set_dim([batch_dim, num_solve_gemm_blocks, dim, block_dim])
                    backward_gemm_nn_launch.set_params([stride, curr_off_diag_offset]) # E, x are already set
                    backward_gemm_nn_launch.launch(stream=stream5)

                    stream4.wait_stream(stream5)

                backward_trsm_lltn_launch.set_dim([batch_dim, dim, block_dim])
                backward_trsm_lltn_launch.set_params([stride]) # L, x are already set
                backward_trsm_lltn_launch.launch(stream=stream4)

            stream.wait_stream(stream4)
            if iterations > 1:
                stream.wait_stream(stream5)

    if use_cuda_graph:
        return create_cuda_graph_callback(callback, device, stream)

    return callback


def create_cholesky_solve_launch(L: wp.array,
                                 E: wp.array,
                                 x: wp.array,
                                 block_dim=None,
                                 block_size=None,
                                 dtype=wp.float64,
                                 device=None,
                                 stream=None,
                                 use_cuda_graph=False):

    assert E.ndim == L.ndim and x.ndim == L.ndim, "L, E, and x must have the same number of dimensions"
    
    if L.ndim == 4:
        batch_dim = L.shape[0]
    elif L.ndim == 3:
        batch_dim = 1
        L = L.reshape((1, *L.shape))
        E = E.reshape((1, *E.shape))
        x = x.reshape((1, *x.shape))
    else:
        raise ValueError("L must have 4 or 3 dimensions")
    
    assert E.shape[0] == batch_dim and x.shape[0] == batch_dim, "L, E, and x must have same batch dimension"

    horizon = L.shape[1]
    n = L.shape[2]
    assert L.shape[3] == n, "The last two dimensions of L must be the same"

    assert E.shape[2] == n and E.shape[3] == n, "L and E must have same block size"
    n_E = calculate_off_diag_storage_len(horizon)
    assert E.shape[1] == n_E, "E has incorrect size, use calculate_off_diag_storage_len to calculate the correct number of blocks"

    n_rhs = x.shape[3]
    assert x.shape[1] == horizon and x.shape[2] == n, "x has incorrect dimensions"
    
    iterations = calculate_recursive_iterations(horizon)

    opt_settings = optimal_problem_settings(n, dtype)
    if block_dim is None:
        block_dim = opt_settings['block_dim']['solve']
    if block_size is None:
        block_size = opt_settings['block_size']['solve']

    if n < block_size:
        fwd_launch = wp.launch_tiled(create_forward_substitution_iteration_kernel(n, n_rhs, dtype),
                                     record_cmd=True,
                                     device=device,
                                     stream=stream,
                                     dim=[batch_dim, 1],
                                     inputs=[0, 0, horizon, L, E, x],
                                     block_dim=block_dim)
        
        back_launch = wp.launch_tiled(create_backward_substitution_iteration_kernel(n, n_rhs, dtype),
                                      record_cmd=True,
                                      device=device,
                                      stream=stream,
                                      dim=[batch_dim, 1],
                                      inputs=[0, 0, horizon, L, E, x],
                                      block_dim=block_dim)
        
        def callback():
            stride = 1
            curr_off_diag_offset = 0

            for _ in range(iterations):
                dim = (horizon + stride) // (2 * stride)

                fwd_launch.set_dim([batch_dim, dim, block_dim])
                fwd_launch.set_params([
                    stride,
                    curr_off_diag_offset,
                    # horizon, L, E, x are already set
                ])
                fwd_launch.launch()
                
                curr_off_diag_offset += horizon // stride - 1
                stride *= 2

            for _ in reversed(range(iterations)):
                stride //= 2
                curr_off_diag_offset -= horizon // stride - 1

                dim = (horizon + stride) // (2 * stride)

                back_launch.set_dim([batch_dim, dim, block_dim])
                back_launch.set_params([
                    stride,
                    curr_off_diag_offset,
                    # horizon, L, E, x are already set
                ])
                back_launch.launch()
    
    else:
        if stream is None:
            device = wp.get_device(device)
            stream = device.stream
        else:
            device = stream.device
        
        stream2 = wp.Stream(device)

        forward_trsm_llnn_launch = wp.launch_tiled(create_forward_substitution_trsm_llnn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                   record_cmd=True,
                                                   device=device,
                                                   dim=[batch_dim, 1],
                                                   inputs=[0, L, x],
                                                   block_dim=block_dim)

        forward_gemm_nn_launch = wp.launch_tiled(create_forward_substitution_gemm_nn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                 record_cmd=True,
                                                 device=device,
                                                 dim=[batch_dim, 1, 1],
                                                 inputs=[0, 0, horizon, E, x],
                                                 block_dim=block_dim)

        forward_gemm_tn_launch = wp.launch_tiled(create_forward_substitution_gemm_tn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                 record_cmd=True,
                                                 device=device,
                                                 dim=[batch_dim, 1, 1],
                                                 inputs=[0, 0, E, x],
                                                 block_dim=block_dim)

        backward_gemm_tn_launch = wp.launch_tiled(create_backward_substitution_gemm_tn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                  record_cmd=True,
                                                  device=device,
                                                  dim=[batch_dim, 1, 1],
                                                  inputs=[0, 0, horizon, E, x],
                                                  block_dim=block_dim)
        
        backward_gemm_nn_launch = wp.launch_tiled(create_backward_substitution_gemm_nn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                  record_cmd=True,
                                                  device=device,
                                                  dim=[batch_dim, 1, 1],
                                                  inputs=[0, 0, E, x],
                                                  block_dim=block_dim)
        
        backward_trsm_lltn_launch = wp.launch_tiled(create_backward_substitution_trsm_lltn_blocked_kernel(n, n_rhs, block_size, dtype),
                                                    record_cmd=True,
                                                    device=device,
                                                    dim=[batch_dim, 1],
                                                    inputs=[0, L, x],
                                                    block_dim=block_dim)

        def callback():
            num_blocks_per_row = math.ceil(n / block_size)
            num_blocks_per_col = math.ceil(n_rhs / block_size)
            num_gemm_blocks = num_blocks_per_row * num_blocks_per_col
            
            stride = 1
            curr_off_diag_offset = 0

            for i in range(iterations):
                dim = (horizon + stride) // (2 * stride)

                if i > 0:
                    stream.wait_stream(stream2)

                forward_trsm_llnn_launch.set_dim([batch_dim, dim, block_dim])
                forward_trsm_llnn_launch.set_params([stride]) # L, x are already set
                forward_trsm_llnn_launch.launch(stream=stream)

                if i < iterations - 1:
                    stream2.wait_stream(stream)

                    forward_gemm_nn_launch.set_dim([batch_dim, num_gemm_blocks, dim, block_dim])
                    forward_gemm_nn_launch.set_params([stride, curr_off_diag_offset]) # horizon, E, x are already set
                    forward_gemm_nn_launch.launch(stream=stream)

                    forward_gemm_tn_launch.set_dim([batch_dim, num_gemm_blocks, dim, block_dim])
                    forward_gemm_tn_launch.set_params([stride, curr_off_diag_offset]) # E, x are already set
                    forward_gemm_tn_launch.launch(stream=stream2)
                
                curr_off_diag_offset += horizon // stride - 1
                stride *= 2

            for i in reversed(range(iterations)):
                stride //= 2
                curr_off_diag_offset -= horizon // stride - 1

                dim = (horizon + stride) // (2 * stride)

                if i < iterations - 1:
                    stream2.wait_stream(stream)

                    backward_gemm_tn_launch.set_dim([batch_dim, num_gemm_blocks, dim, block_dim])
                    backward_gemm_tn_launch.set_params([stride, curr_off_diag_offset]) # horizon, E, x are already set
                    backward_gemm_tn_launch.launch(stream=stream)

                    backward_gemm_nn_launch.set_dim([batch_dim, num_gemm_blocks, dim, block_dim])
                    backward_gemm_nn_launch.set_params([stride, curr_off_diag_offset]) # E, x are already set
                    backward_gemm_nn_launch.launch(stream=stream2)

                    stream.wait_stream(stream2)

                backward_trsm_lltn_launch.set_dim([batch_dim, dim, block_dim])
                backward_trsm_lltn_launch.set_params([stride]) # L, x are already set
                backward_trsm_lltn_launch.launch(stream=stream)

            if iterations > 1:
                stream.wait_stream(stream2)

    if use_cuda_graph:
        return create_cuda_graph_callback(callback, device, stream)

    return callback
