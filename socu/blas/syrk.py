import warp as wp

def create_syrk_ln_blocked_func(n: int, block_size: int, atomic=False, dtype=wp.float64):

    tail_size = n % block_size

    @wp.func
    def syrk_ln_bb(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(atomic):
            C_ij = wp.tile_zeros(shape=(block_size, block_size), dtype=dtype)
        else:
            C_ij = wp.tile_load(C[batch_id, C_offset], shape=(block_size, block_size), offset=(i, j))

        for k in range(0, A.shape[2] - tail_size, block_size):
            A_jk = wp.tile_load(A[batch_id, A_offset], shape=(block_size, block_size), offset=(j, k))
            A_jk_T = wp.tile_transpose(A_jk)
            A_ik = wp.tile_load(A[batch_id, A_offset], shape=(block_size, block_size), offset=(i, k))
            wp.tile_matmul(A_ik, A_jk_T, C_ij, alpha=-1.0)

        if wp.static(tail_size > 0):
            k = A.shape[2] - tail_size    
            A_jk_tail = wp.tile_load(A[batch_id, A_offset], shape=(block_size, tail_size), offset=(j, k))
            A_jk_T_tail = wp.tile_transpose(A_jk_tail)
            A_ik_tail = wp.tile_load(A[batch_id, A_offset], shape=(block_size, tail_size), offset=(i, k))
            wp.tile_matmul(A_ik_tail, A_jk_T_tail, C_ij, alpha=-1.0)
        
        if wp.static(atomic):
            wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
        else:
            wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def syrk_ln_tb(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(tail_size > 0):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(tail_size, block_size), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(tail_size, block_size), offset=(i, j))

            for k in range(0, A.shape[2] - tail_size, block_size):
                A_jk = wp.tile_load(A[batch_id, A_offset], shape=(block_size, block_size), offset=(j, k))
                A_jk_T = wp.tile_transpose(A_jk)
                A_ik = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, block_size), offset=(i, k))
                wp.tile_matmul(A_ik, A_jk_T, C_ij, alpha=-1.0)

            k = A.shape[2] - tail_size    
            A_jk_tail = wp.tile_load(A[batch_id, A_offset], shape=(block_size, tail_size), offset=(j, k))
            A_jk_T_tail = wp.tile_transpose(A_jk_tail)
            A_ik_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, tail_size), offset=(i, k))
            wp.tile_matmul(A_ik_tail, A_jk_T_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def syrk_ln_tt(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(tail_size > 0):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(tail_size, tail_size), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(tail_size, tail_size), offset=(i, j))

            for k in range(0, A.shape[2] - tail_size, block_size):
                A_jk = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, block_size), offset=(j, k))
                A_jk_T = wp.tile_transpose(A_jk)
                A_ik = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, block_size), offset=(i, k))
                wp.tile_matmul(A_ik, A_jk_T, C_ij, alpha=-1.0)

            k = A.shape[2] - tail_size    
            A_jk_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, tail_size), offset=(j, k))
            A_jk_T_tail = wp.tile_transpose(A_jk_tail)
            A_ik_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, tail_size), offset=(i, k))
            wp.tile_matmul(A_ik_tail, A_jk_T_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def syrk_ln_blocked(batch_id: int,
                        k: int,
                        A: wp.array4d[dtype], # type: ignore
                        A_offset: int,
                        C: wp.array4d[dtype], # type: ignore
                        C_offset: int):
        '''
        Batched blocked symmetric rank update of a matrix. Only the lower triangular part is calculated.

        It performs a symmetric rank update, i.e.,
        C[batch_id, offset, :, :] -= A[batch_id, offset, :, :] * A[batch_id, offset, :, :]^T
        for the entry C[batch_id, offset, i, j] where (i,j) corresponds to the coordinates which are mapped from
        k in [0, ceil(n / block_size) - 1] to the C matrix
        [0       ]
        [1 2     ]
        [3 4 5   ]
        [6 7 8 9 ]
        [  ...   ]
        where every block has size block_size x block_size (except for potentially last row and column)

        A is of shape (batch_dim, A_elements, n, n),
        where batch_dim is the number of batches,
              A_elements is the number of blockes stored,
              n x n is the size of a single block,
              

        C is of shape (batch_dim, C_elements, n, n),
        where batch_dim is the number of batches,
              C_elements is the number of blockes stored,
              n x n is the size of a single block,
        '''

        i = int(0)
        row_start = int(0)
        next_row_start = int(1)
        while k >= next_row_start:
            i = i + 1
            row_start = next_row_start
            next_row_start = row_start + i + 1
        j = k - row_start

        i *= block_size
        j *= block_size

        if i < A.shape[2] - tail_size and j < A.shape[2] - tail_size:
            syrk_ln_bb(batch_id, i, j, A, A_offset, C, C_offset)
        if i >= A.shape[2] - tail_size and j < A.shape[2] - tail_size:
            syrk_ln_tb(batch_id, i, j, A, A_offset, C, C_offset)
        if i >= A.shape[2] - tail_size and j >= A.shape[2] - tail_size:
            syrk_ln_tt(batch_id, i, j, A, A_offset, C, C_offset)

    return syrk_ln_blocked


def create_syrk_lt_blocked_func(n: int, block_size: int, atomic=False, dtype=wp.float64):

    tail_size = n % block_size

    @wp.func
    def syrk_lt_bb(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(atomic):
            C_ij = wp.tile_zeros(shape=(block_size, block_size), dtype=dtype)
        else:
            C_ij = wp.tile_load(C[batch_id, C_offset], shape=(block_size, block_size), offset=(i, j))

        for k in range(0, A.shape[2] - tail_size, block_size):
            A_jk_T = wp.tile_load(A[batch_id, A_offset], shape=(block_size, block_size), offset=(k, j))
            A_ik_T = wp.tile_load(A[batch_id, A_offset], shape=(block_size, block_size), offset=(k, i))
            A_ik = wp.tile_transpose(A_ik_T)
            wp.tile_matmul(A_ik, A_jk_T, C_ij, alpha=-1.0)

        if wp.static(tail_size > 0):
            k = A.shape[2] - tail_size    
            A_jk_T_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, block_size), offset=(k, j))
            A_ik_T_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, block_size), offset=(k, i))
            A_ik_tail = wp.tile_transpose(A_ik_T_tail)
            wp.tile_matmul(A_ik_tail, A_jk_T_tail, C_ij, alpha=-1.0)
        
        if wp.static(atomic):
            wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
        else:
            wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def syrk_lt_tb(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(tail_size > 0):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(tail_size, block_size), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(tail_size, block_size), offset=(i, j))

            for k in range(0, A.shape[2] - tail_size, block_size):
                A_jk_T = wp.tile_load(A[batch_id, A_offset], shape=(block_size, block_size), offset=(k, j))
                A_ik_T = wp.tile_load(A[batch_id, A_offset], shape=(block_size, tail_size), offset=(k, i))
                A_ik = wp.tile_transpose(A_ik_T)
                wp.tile_matmul(A_ik, A_jk_T, C_ij, alpha=-1.0)

            k = A.shape[2] - tail_size    
            A_jk_T_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, block_size), offset=(k, j))
            A_ik_T_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, tail_size), offset=(k, i))
            A_ik_tail = wp.tile_transpose(A_ik_T_tail)
            wp.tile_matmul(A_ik_tail, A_jk_T_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def syrk_lt_tt(batch_id: int,
                   i: int,
                   j: int,
                   A: wp.array4d[dtype], # type: ignore
                   A_offset: int,
                   C: wp.array4d[dtype], # type: ignore
                   C_offset: int):
        
        if wp.static(tail_size > 0):
            if wp.static(atomic):
                C_ij = wp.tile_zeros(shape=(tail_size, tail_size), dtype=dtype)
            else:
                C_ij = wp.tile_load(C[batch_id, C_offset], shape=(tail_size, tail_size), offset=(i, j))

            for k in range(0, A.shape[2] - tail_size, block_size):
                A_jk_T = wp.tile_load(A[batch_id, A_offset], shape=(block_size, tail_size), offset=(k, j))
                A_ik_T = wp.tile_load(A[batch_id, A_offset], shape=(block_size, tail_size), offset=(k, i))
                A_ik = wp.tile_transpose(A_ik_T)
                wp.tile_matmul(A_ik, A_jk_T, C_ij, alpha=-1.0)

            k = A.shape[2] - tail_size    
            A_jk_T_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, tail_size), offset=(k, j))
            A_ik_T_tail = wp.tile_load(A[batch_id, A_offset], shape=(tail_size, tail_size), offset=(k, i))
            A_ik_tail = wp.tile_transpose(A_ik_T_tail)
            wp.tile_matmul(A_ik_tail, A_jk_T_tail, C_ij, alpha=-1.0)
            
            if wp.static(atomic):
                wp.tile_atomic_add(C[batch_id, C_offset], C_ij, offset=(i, j))
            else:
                wp.tile_store(C[batch_id, C_offset], C_ij, offset=(i, j))


    @wp.func
    def syrk_lt_blocked(batch_id: int,
                        k: int,
                        A: wp.array4d[dtype], # type: ignore
                        A_offset: int,
                        C: wp.array4d[dtype], # type: ignore
                        C_offset: int):
        '''
        Batched blocked symmetric rank update of a matrix. Only the lower triangular part is calculated.

        It performs a symmetric rank update, i.e.,
        C[batch_id, offset, :, :] -= A[batch_id, offset, :, :]^T * A[batch_id, offset, :, :]
        for the entry C[batch_id, offset, i, j] where (i,j) corresponds to the coordinates which are mapped from
        k in [0, ceil(n / block_size) - 1] to the C matrix
        [0       ]
        [1 2     ]
        [3 4 5   ]
        [6 7 8 9 ]
        [  ...   ]
        where every block has size block_size x block_size (except for potentially last row and column)

        A is of shape (batch_dim, n * A_elements, n),
        where batch_dim is the number of batches,
              A_elements is the number of blockes stored,
              n x n is the size of a single block

        C is of shape (batch_dim, n * C_elements, n),
        where batch_dim is the number of batches,
              C_elements is the number of blockes stored,
              n x n is the size of a single block
        '''

        i = int(0)
        row_start = int(0)
        next_row_start = int(1)
        while k >= next_row_start:
            i = i + 1
            row_start = next_row_start
            next_row_start = row_start + i + 1
        j = k - row_start

        i *= block_size
        j *= block_size

        if i < A.shape[2] - tail_size and j < A.shape[2] - tail_size:
            syrk_lt_bb(batch_id, i, j, A, A_offset, C, C_offset)
        if i >= A.shape[2] - tail_size and j < A.shape[2] - tail_size:
            syrk_lt_tb(batch_id, i, j, A, A_offset, C, C_offset)
        if i >= A.shape[2] - tail_size and j >= A.shape[2] - tail_size:
            syrk_lt_tt(batch_id, i, j, A, A_offset, C, C_offset)

    return syrk_lt_blocked
