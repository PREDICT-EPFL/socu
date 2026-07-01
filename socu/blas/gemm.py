import math

import warp as wp

def create_gemm_nn_blocked_func(m: int, n: int, k: int, block_size: int, atomic=False, dtype=wp.float64):

    tail_m = m % block_size
    tail_n = n % block_size
    tail_k = k % block_size

    @wp.func
    def gemm_nn_bb(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   B: wp.array4d[dtype], # type: ignore
                   B_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(block_size <= m and block_size <= n):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(block_size, block_size), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(block_size, block_size), offset=(i, j))

            if wp.static(block_size <= k):
                for kk in range(0, A.shape[3] - tail_k, block_size):
                    A_ik = wp.tile_load(A[batch_id, A_offset], shape=(block_size, block_size), offset=(i, kk))
                    B_kj = wp.tile_load(B[batch_id, B_offset], shape=(block_size, block_size), offset=(kk, j))
                    wp.tile_matmul(A_ik, B_kj, C_ij, alpha=-1.0)

            if wp.static(tail_k > 0):
                kk = A.shape[3] - tail_k
                A_ik_tail = wp.tile_load(A[batch_id, A_offset], shape=(block_size, tail_k), offset=(i, kk))
                B_kj_tail = wp.tile_load(B[batch_id, B_offset], shape=(tail_k, block_size), offset=(kk, j))
                wp.tile_matmul(A_ik_tail, B_kj_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))

    
    @wp.func
    def gemm_nn_bt(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   B: wp.array4d[dtype], # type: ignore
                   B_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(block_size <= m and tail_n > 0):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(block_size, tail_n), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(block_size, tail_n), offset=(i, j))

            if wp.static(block_size <= k):
                for kk in range(0, A.shape[3] - tail_k, block_size):
                    A_ik = wp.tile_load(A[batch_id, A_offset], shape=(block_size, block_size), offset=(i, kk))
                    B_kj = wp.tile_load(B[batch_id, B_offset], shape=(block_size, tail_n), offset=(kk, j))
                    wp.tile_matmul(A_ik, B_kj, C_ij, alpha=-1.0)

            if wp.static(tail_k > 0):
                kk = A.shape[3] - tail_k
                A_ik_tail = wp.tile_load(A[batch_id, A_offset], shape=(block_size, tail_k), offset=(i, kk))
                B_kj_tail = wp.tile_load(B[batch_id, B_offset], shape=(tail_k, tail_n), offset=(kk, j))
                wp.tile_matmul(A_ik_tail, B_kj_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def gemm_nn_tb(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   B: wp.array4d[dtype], # type: ignore
                   B_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(tail_m > 0 and block_size <= n):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(tail_m, block_size), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(tail_m, block_size), offset=(i, j))

            if wp.static(block_size <= k):
                for kk in range(0, A.shape[3] - tail_k, block_size):
                    A_ik = wp.tile_load(A[batch_id, A_offset], shape=(tail_m, block_size), offset=(i, kk))
                    B_kj = wp.tile_load(B[batch_id, B_offset], shape=(block_size, block_size), offset=(kk, j))
                    wp.tile_matmul(A_ik, B_kj, C_ij, alpha=-1.0)

            if wp.static(tail_k > 0):
                kk = A.shape[3] - tail_k
                A_ik_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_m, tail_k), offset=(i, kk))
                B_kj_tail = wp.tile_load(B[batch_id, B_offset], shape=(tail_k, block_size), offset=(kk, j))
                wp.tile_matmul(A_ik_tail, B_kj_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def gemm_nn_tt(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   B: wp.array4d[dtype], # type: ignore
                   B_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(tail_m > 0 and tail_n > 0):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(tail_m, tail_n), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(tail_m, tail_n), offset=(i, j))

            if wp.static(block_size <= k):
                for kk in range(0, A.shape[3] - tail_k, block_size):
                    A_ik = wp.tile_load(A[batch_id, A_offset], shape=(tail_m, block_size), offset=(i, kk))
                    B_kj = wp.tile_load(B[batch_id, B_offset], shape=(block_size, tail_n), offset=(kk, j))
                    wp.tile_matmul(A_ik, B_kj, C_ij, alpha=-1.0)

            if wp.static(tail_k > 0):
                kk = A.shape[3] - tail_k
                A_ik_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_m, tail_k), offset=(i, kk))
                B_kj_tail = wp.tile_load(B[batch_id, B_offset], shape=(tail_k, tail_n), offset=(kk, j))
                wp.tile_matmul(A_ik_tail, B_kj_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def gemm_nn_blocked(batch_id: int,
                        p: int,
                        A: wp.array4d[dtype], # type: ignore
                        A_offset: int,
                        B: wp.array4d[dtype], # type: ignore
                        B_offset: int,
                        C: wp.array4d[dtype], # type: ignore
                        C_offset: int):
        '''
        Batched blocked matrix matrix multiplication.

        It performs a matrix matrix multiplication, i.e.,
        C[batch_id, offset, :, :] -= A[batch_id, offset, :, :] * B[batch_id, offset, :, :]
        for the entry C[batch_id, offset, i, j] where (i,j) corresponds to the coordinates which are mapped from
        p in [0, nb*nb-1] with nb = ceil(n / block_size) to the C matrix
        [0  1    2    3    ... nb-1  ]
        [nb nb+1 nb+2 nb+3 ... 2*nb-1]
        [                  ...       ]
        where every block has size block_size x block_size (except for potentially last row and column)

        A is of shape (batch_dim, A_elements, m, k),
        where batch_dim is the number of batches,
              A_elements is the number of blockes stored,
              m x k is the size of a single block

        B is of shape (batch_dim, B_elements, k, n),
        where batch_dim is the number of batches,
              B_elements is the number of blockes stored,
              k x n is the size of a single block

        C is of shape (batch_dim, C_elements, m, n),
        where batch_dim is the number of batches,
              C_elements is the number of blockes stored,
              m x n is the size of a single block
        '''

        nb = (B.shape[3] + block_size - 1) // block_size # math.ceil(n / block_size)
        i = (p // nb) * block_size
        j = (p % nb) * block_size

        if i < A.shape[2] - tail_m and j < B.shape[3] - tail_n:
            gemm_nn_bb(batch_id, i, j, A, A_offset, B, B_offset, C, C_offset)
        if i < A.shape[2] - tail_m and j >= B.shape[3] - tail_n:
            gemm_nn_bt(batch_id, i, j, A, A_offset, B, B_offset, C, C_offset)
        if i >= A.shape[2] - tail_m and j < B.shape[3] - tail_n:
            gemm_nn_tb(batch_id, i, j, A, A_offset, B, B_offset, C, C_offset)
        if i >= A.shape[2] - tail_m and j >= B.shape[3] - tail_n:
            gemm_nn_tt(batch_id, i, j, A, A_offset, B, B_offset, C, C_offset)

    return gemm_nn_blocked


def create_gemm_tn_blocked_func(m: int, n: int, k: int, block_size: int, atomic=False, dtype=wp.float64):

    tail_m = m % block_size
    tail_n = n % block_size
    tail_k = k % block_size

    @wp.func
    def gemm_tn_bb(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   B: wp.array4d[dtype], # type: ignore
                   B_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(block_size <= m and block_size <= n):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(block_size, block_size), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(block_size, block_size), offset=(i, j))

            if wp.static(block_size <= k):
                for kk in range(0, A.shape[2] - tail_k, block_size):
                    A_ki = wp.tile_load(A[batch_id, A_offset], shape=(block_size, block_size), offset=(kk, i))
                    A_ki_T = wp.tile_transpose(A_ki)
                    B_kj = wp.tile_load(B[batch_id, B_offset], shape=(block_size, block_size), offset=(kk, j))
                    wp.tile_matmul(A_ki_T, B_kj, C_ij, alpha=-1.0)

            if wp.static(tail_k > 0):
                kk = A.shape[2] - tail_k
                A_ki_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_k, block_size), offset=(kk, i))
                A_ki_tail_T = wp.tile_transpose(A_ki_tail)
                B_kj_tail = wp.tile_load(B[batch_id, B_offset], shape=(tail_k, block_size), offset=(kk, j))
                wp.tile_matmul(A_ki_tail_T, B_kj_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))

    
    @wp.func
    def gemm_tn_bt(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   B: wp.array4d[dtype], # type: ignore
                   B_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(block_size <= m and tail_n > 0):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(block_size, tail_n), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(block_size, tail_n), offset=(i, j))

            if wp.static(block_size <= k):
                for kk in range(0, A.shape[2] - tail_k, block_size):
                    A_ki = wp.tile_load(A[batch_id, A_offset], shape=(block_size, block_size), offset=(kk, i))
                    A_ki_T = wp.tile_transpose(A_ki)
                    B_kj = wp.tile_load(B[batch_id, B_offset], shape=(block_size, tail_n), offset=(kk, j))
                    wp.tile_matmul(A_ki_T, B_kj, C_ij, alpha=-1.0)

            if wp.static(tail_k > 0):
                kk = A.shape[2] - tail_k
                A_ki_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_k, block_size), offset=(kk, i))
                A_ki_tail_T = wp.tile_transpose(A_ki_tail)
                B_kj_tail = wp.tile_load(B[batch_id, B_offset], shape=(tail_k, tail_n), offset=(kk, j))
                wp.tile_matmul(A_ki_tail_T, B_kj_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def gemm_tn_tb(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   B: wp.array4d[dtype], # type: ignore
                   B_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(tail_m > 0 and block_size <= n):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(tail_m, block_size), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(tail_m, block_size), offset=(i, j))

            if wp.static(block_size <= k):
                for kk in range(0, A.shape[2] - tail_k, block_size):
                    A_ki = wp.tile_load(A[batch_id, A_offset], shape=(block_size, tail_m), offset=(kk, i))
                    A_ki_T = wp.tile_transpose(A_ki)
                    B_kj = wp.tile_load(B[batch_id, B_offset], shape=(block_size, block_size), offset=(kk, j))
                    wp.tile_matmul(A_ki_T, B_kj, C_ij, alpha=-1.0)

            if wp.static(tail_k > 0):
                kk = A.shape[2] - tail_k
                A_ki_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_k, tail_m), offset=(kk, i))
                A_ki_tail_T = wp.tile_transpose(A_ki_tail)
                B_kj_tail = wp.tile_load(B[batch_id, B_offset], shape=(tail_k, block_size), offset=(kk, j))
                wp.tile_matmul(A_ki_tail_T, B_kj_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def gemm_tn_tt(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   B: wp.array4d[dtype], # type: ignore
                   B_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(tail_m > 0 and tail_n > 0):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(tail_m, tail_n), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(tail_m, tail_n), offset=(i, j))

            if wp.static(block_size <= k):
                for kk in range(0, A.shape[2] - tail_k, block_size):
                    A_ki = wp.tile_load(A[batch_id, A_offset], shape=(block_size, tail_m), offset=(kk, i))
                    A_ki_T = wp.tile_transpose(A_ki)
                    B_kj = wp.tile_load(B[batch_id, B_offset], shape=(block_size, tail_n), offset=(kk, j))
                    wp.tile_matmul(A_ki_T, B_kj, C_ij, alpha=-1.0)

            if wp.static(tail_k > 0):
                kk = A.shape[2] - tail_k
                A_ki_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_k, tail_m), offset=(kk, i))
                A_ki_tail_T = wp.tile_transpose(A_ki_tail)
                B_kj_tail = wp.tile_load(B[batch_id, B_offset], shape=(tail_k, tail_n), offset=(kk, j))
                wp.tile_matmul(A_ki_tail_T, B_kj_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def gemm_tn_blocked(batch_id: int,
                        p: int,
                        A: wp.array4d[dtype], # type: ignore
                        A_offset: int,
                        B: wp.array4d[dtype], # type: ignore
                        B_offset: int,
                        C: wp.array4d[dtype], # type: ignore
                        C_offset: int):
        '''
        Batched blocked matrix matrix multiplication.

        It performs a matrix matrix multiplication, i.e.,
        C[batch_id, offset, :, :] -= A[batch_id, offset, :, :]^T * B[batch_id, offset, :, :]
        for the entry C[batch_id, offset, i, j] where (i,j) corresponds to the coordinates which are mapped from
        p in [0, nb*nb-1] with nb = ceil(n / block_size) to the C matrix
        [0  1    2    3    ... nb-1  ]
        [nb nb+1 nb+2 nb+3 ... 2*nb-1]
        [                  ...       ]
        where every block has size block_size x block_size (except for potentially last row and column)

        A is of shape (batch_dim, A_elements, k, m),
        where batch_dim is the number of batches,
              A_elements is the number of blockes stored,
              k x m is the size of a single block

        B is of shape (batch_dim, B_elements, k, n),
        where batch_dim is the number of batches,
              A_elements is the number of blockes stored,
              k x n is the size of a single block

        C is of shape (batch_dim, C_elements, m, n),
        where batch_dim is the number of batches,
              C_elements is the number of blockes stored,
              m x n is the size of a single block
        '''

        nb = (B.shape[3] + block_size - 1) // block_size # math.ceil(n / block_size)
        i = (p // nb) * block_size
        j = (p % nb) * block_size

        if i < A.shape[3] - tail_m and j < B.shape[3] - tail_n:
            gemm_tn_bb(batch_id, i, j, A, A_offset, B, B_offset, C, C_offset)
        if i < A.shape[3] - tail_m and j >= B.shape[3] - tail_n:
            gemm_tn_bt(batch_id, i, j, A, A_offset, B, B_offset, C, C_offset)
        if i >= A.shape[3] - tail_m and j < B.shape[3] - tail_n:
            gemm_tn_tb(batch_id, i, j, A, A_offset, B, B_offset, C, C_offset)
        if i >= A.shape[3] - tail_m and j >= B.shape[3] - tail_n:
            gemm_tn_tt(batch_id, i, j, A, A_offset, B, B_offset, C, C_offset)

    return gemm_tn_blocked
