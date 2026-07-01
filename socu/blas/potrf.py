import warp as wp

def create_potrf_l_blocked_func(n: int, block_size: int, dtype=wp.float64):

    tail_size = n % block_size

    @wp.func
    def potrf_l(batch_id: int,
                k: int,
                L: wp.array4d[dtype], # type: ignore
                offset: int):
        
        L_kk = wp.tile_load(L[batch_id, offset], shape=(block_size, block_size), offset=(k, k))

        # look left
        for j in range(0, k, block_size):
            L_kj = wp.tile_load(L[batch_id, offset], shape=(block_size, block_size), offset=(k, j))
            L_kj_T = wp.tile_transpose(L_kj)
            wp.tile_matmul(L_kj, L_kj_T, L_kk, alpha=-1.0)

        # cholesky of block
        wp.tile_cholesky_inplace(L_kk)
        wp.tile_store(L[batch_id, offset], L_kk, offset=(k, k))


    @wp.func
    def potrf_l_trsm(batch_id: int,
                     k: int,
                     L: wp.array4d[dtype], # type: ignore
                     offset: int):
        
        L_kk = wp.tile_load(L[batch_id, offset], shape=(block_size, block_size), offset=(k, k))

        # process blocks below
        for i in range(k + block_size, L.shape[2] - tail_size, block_size):
            L_ik = wp.tile_load(L[batch_id, offset], shape=(block_size, block_size), offset=(i, k))

            # look left
            for j in range(0, k, block_size):
                L_ij = wp.tile_load(L[batch_id, offset], shape=(block_size, block_size), offset=(i, j))
                L_kj = wp.tile_load(L[batch_id, offset], shape=(block_size, block_size), offset=(k, j))
                L_kj_T = wp.tile_transpose(L_kj)
                wp.tile_matmul(L_ij, L_kj_T, L_ik, alpha=-1.0)
            
            L_ik_T = wp.tile_transpose(L_ik)
            wp.tile_lower_solve_inplace(L_kk, L_ik_T)
            L_ik = wp.tile_transpose(L_ik_T)
            wp.tile_store(L[batch_id, offset], L_ik, offset=(i, k))


    @wp.func
    def potrf_l_trsm_tail(batch_id: int,
                          k: int,
                          L: wp.array4d[dtype], # type: ignore
                          offset: int):
        
        if wp.static(tail_size > 0):
            L_kk = wp.tile_load(L[batch_id, offset], shape=(block_size, block_size), offset=(k, k))

            # process tail
            i = L.shape[2] - tail_size
            L_ik = wp.tile_load(L[batch_id, offset], shape=(tail_size, block_size), offset=(i, k))

            # look left
            for j in range(0, k, block_size):
                L_ij = wp.tile_load(L[batch_id, offset], shape=(tail_size, block_size), offset=(i, j))
                L_kj = wp.tile_load(L[batch_id, offset], shape=(block_size, block_size), offset=(k, j))
                L_kj_T = wp.tile_transpose(L_kj)
                wp.tile_matmul(L_ij, L_kj_T, L_ik, alpha=-1.0)
            
            L_ik_T = wp.tile_transpose(L_ik)
            wp.tile_lower_solve_inplace(L_kk, L_ik_T)
            L_ik = wp.tile_transpose(L_ik_T)
            wp.tile_store(L[batch_id, offset], L_ik, offset=(i, k))


    @wp.func
    def potrf_l_tail(batch_id: int,
                     L: wp.array4d[dtype], # type: ignore
                     offset: int):
        
        if wp.static(tail_size > 0):
            k = L.shape[2] - tail_size
            L_kk = wp.tile_load(L[batch_id, offset], shape=(tail_size, tail_size), offset=(k, k))

            # look left
            for j in range(0, k, block_size):
                L_kj = wp.tile_load(L[batch_id, offset], shape=(tail_size, block_size), offset=(k, j))
                L_kj_T = wp.tile_transpose(L_kj)
                wp.tile_matmul(L_kj, L_kj_T, L_kk, alpha=-1.0)

            # cholesky of block
            wp.tile_cholesky_inplace(L_kk)
            wp.tile_store(L[batch_id, offset], L_kk, offset=(k, k))


    @wp.func
    def potrf_blocked(batch_id: int,
                      L: wp.array4d[dtype], # type: ignore
                      offset: int):
        '''
        Batched blocked in-place cholesky factorization. Only the lower triangular part is used.

        It performs an inline cholesky factorization for a specific entry in L, i.e.,
        cholesky_inplace(L[batch_id, offset, :, :])

        L is of shape (batch_dim, N, n, n),
        where batch_dim is the number of batches,
              N is the number of blockes stored,
              n x n is the size of a single block,
        '''

        for k in range(0, L.shape[2] - tail_size, block_size):
            potrf_l(batch_id, k, L, offset)
            potrf_l_trsm(batch_id, k, L, offset)
            potrf_l_trsm_tail(batch_id, k, L, offset)
        potrf_l_tail(batch_id, L, offset)

    return potrf_blocked
