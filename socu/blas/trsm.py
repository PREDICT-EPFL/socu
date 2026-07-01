import warp as wp

def create_trsm_rltn_blocked_func(m: int, n: int, block_size: int, dtype=wp.float64):

    tail_m = m % block_size
    tail_n = n % block_size

    @wp.func
    def trsm_rltn(batch_id: int,
                  L: wp.array4d[dtype], # type: ignore
                  L_offset: int,
                  E: wp.array4d[dtype], # type: ignore
                  E_offset: int):
        
        if wp.static(block_size <= n):
            for k in range(0, L.shape[2] - tail_n, block_size):
                L_kk = wp.tile_load(L[batch_id, L_offset], shape=(block_size, block_size), offset=(k, k))

                if wp.static(block_size <= m):
                    for i in range(0, E.shape[2] - tail_m, block_size):    
                        E_ik = wp.tile_load(E[batch_id, E_offset], shape=(block_size, block_size), offset=(i, k))
                        
                        for j in range(0, k, block_size):
                            E_ij = wp.tile_load(E[batch_id, E_offset], shape=(block_size, block_size), offset=(i, j))
                            L_kj = wp.tile_load(L[batch_id, L_offset], shape=(block_size, block_size), offset=(k, j))
                            L_kj_T = wp.tile_transpose(L_kj)
                            wp.tile_matmul(E_ij, L_kj_T, E_ik, alpha=-1.0)

                        E_ik_T = wp.tile_transpose(E_ik)
                        wp.tile_lower_solve_inplace(L_kk, E_ik_T)
                        E_ik = wp.tile_transpose(E_ik_T)
                        wp.tile_store(E[batch_id, E_offset], E_ik, offset=(i, k))

                if wp.static(tail_m > 0):
                    i = E.shape[2] - tail_m
                    E_ik_tail = wp.tile_load(E[batch_id, E_offset], shape=(tail_m, block_size), offset=(i, k))
                    
                    for j in range(0, k, block_size):
                        E_ij_tail = wp.tile_load(E[batch_id, E_offset], shape=(tail_m, block_size), offset=(i, j))
                        L_kj_tail = wp.tile_load(L[batch_id, L_offset], shape=(block_size, block_size), offset=(k, j))
                        L_kj_T_tail = wp.tile_transpose(L_kj_tail)
                        wp.tile_matmul(E_ij_tail, L_kj_T_tail, E_ik_tail, alpha=-1.0)

                    L_ik_T_tail = wp.tile_transpose(E_ik_tail)
                    wp.tile_lower_solve_inplace(L_kk, L_ik_T_tail)
                    E_ik_tail = wp.tile_transpose(L_ik_T_tail)
                    wp.tile_store(E[batch_id, E_offset], E_ik_tail, offset=(i, k))


    @wp.func
    def trsm_rltn_tail(batch_id: int,
                       L: wp.array4d[dtype], # type: ignore
                       L_offset: int,
                       E: wp.array4d[dtype], # type: ignore
                       E_offset: int):
        
        if wp.static(tail_n > 0):
            k = L.shape[2] - tail_n
            L_kk = wp.tile_load(L[batch_id, L_offset], shape=(tail_n, tail_n), offset=(k, k))

            if wp.static(block_size <= m):
                for i in range(0, E.shape[2] - tail_m, block_size):    
                    E_ik = wp.tile_load(E[batch_id, E_offset], shape=(block_size, tail_n), offset=(i, k))
                    
                    if wp.static(block_size <= n):
                        for j in range(0, k, block_size):
                            E_ij = wp.tile_load(E[batch_id, E_offset], shape=(block_size, block_size), offset=(i, j))
                            L_kj = wp.tile_load(L[batch_id, L_offset], shape=(tail_n, block_size), offset=(k, j))
                            L_kj_T = wp.tile_transpose(L_kj)
                            wp.tile_matmul(E_ij, L_kj_T, E_ik, alpha=-1.0)

                    E_ik_T = wp.tile_transpose(E_ik)
                    wp.tile_lower_solve_inplace(L_kk, E_ik_T)
                    E_ik = wp.tile_transpose(E_ik_T)
                    wp.tile_store(E[batch_id, E_offset], E_ik, offset=(i, k))

            if wp.static(tail_m > 0):
                i = E.shape[2] - tail_m
                E_ik_tail = wp.tile_load(E[batch_id, E_offset], shape=(tail_m, tail_n), offset=(i, k))
                
                if wp.static(block_size <= n):
                    for j in range(0, k, block_size):
                        E_ij_tail = wp.tile_load(E[batch_id, E_offset], shape=(tail_m, block_size), offset=(i, j))
                        L_kj_tail = wp.tile_load(L[batch_id, L_offset], shape=(tail_m, block_size), offset=(k, j))
                        L_kj_T_tail = wp.tile_transpose(L_kj_tail)
                        wp.tile_matmul(E_ij_tail, L_kj_T_tail, E_ik_tail, alpha=-1.0)

                L_ik_T_tail = wp.tile_transpose(E_ik_tail)
                wp.tile_lower_solve_inplace(L_kk, L_ik_T_tail)
                E_ik_tail = wp.tile_transpose(L_ik_T_tail)
                wp.tile_store(E[batch_id, E_offset], E_ik_tail, offset=(i, k))


    @wp.func
    def trsm_rltn_blocked(batch_id: int,
                          L: wp.array4d[dtype], # type: ignore
                          L_offset: int,
                          E: wp.array4d[dtype], # type: ignore
                          E_offset: int):
        '''
        Batched blocked in-place triangular solve. Only the lower triangular part of L is used.

        It performs an inline in-place triangular solve for a specific entry in L and E, i.e.,
        E[batch_id, E_offset, :, :] = E[batch_id, E_offset, :, :] * L[batch_id, L_offset, :, :]^{-T}

        L is of shape (batch_dim, L_elements, n, n),
        where batch_dim is the number of batches,
              L_elements is the number of blockes stored,
              n x n is the size of a single block

        E is of shape (batch_dim, E_elements, m, n),
        where batch_dim is the number of batches,
              E_elements is the number of blockes stored,
              m x n is the size of a single block
        '''

        trsm_rltn(batch_id, L, L_offset, E, E_offset)
        trsm_rltn_tail(batch_id, L, L_offset, E, E_offset)

    return trsm_rltn_blocked


def create_trsm_llnn_blocked_func(m: int, n: int, block_size: int, dtype=wp.float64):

    tail_m = m % block_size
    tail_n = n % block_size

    @wp.func
    def trsm_llnn(batch_id: int,
                  L: wp.array4d[dtype], # type: ignore
                  L_offset: int,
                  E: wp.array4d[dtype], # type: ignore
                  E_offset: int):
        
        if wp.static(block_size <= m):
            for k in range(0, L.shape[2] - tail_m, block_size):
                L_kk = wp.tile_load(L[batch_id, L_offset], shape=(block_size, block_size), offset=(k, k))

                if wp.static(block_size <= n):
                    for j in range(0, E.shape[3] - tail_n, block_size):    
                        E_kj = wp.tile_load(E[batch_id, E_offset], shape=(block_size, block_size), offset=(k, j))
                        
                        for i in range(0, k, block_size):
                            E_ij = wp.tile_load(E[batch_id, E_offset], shape=(block_size, block_size), offset=(i, j))
                            L_ki = wp.tile_load(L[batch_id, L_offset], shape=(block_size, block_size), offset=(k, i))
                            wp.tile_matmul(L_ki, E_ij, E_kj, alpha=-1.0)

                        wp.tile_lower_solve_inplace(L_kk, E_kj)
                        wp.tile_store(E[batch_id, E_offset], E_kj, offset=(k, j))

                if wp.static(tail_n > 0):
                    j = E.shape[3] - tail_n
                    E_kj_tail = wp.tile_load(E[batch_id, E_offset], shape=(block_size, tail_n), offset=(k, j))
                    
                    for i in range(0, k, block_size):
                        L_ij_tail = wp.tile_load(E[batch_id, E_offset], shape=(block_size, tail_n), offset=(i, j))
                        L_ki_tail = wp.tile_load(L[batch_id, L_offset], shape=(block_size, block_size), offset=(k, i))
                        wp.tile_matmul(L_ki_tail, L_ij_tail, E_kj_tail, alpha=-1.0)

                    wp.tile_lower_solve_inplace(L_kk, E_kj_tail)
                    wp.tile_store(E[batch_id, E_offset], E_kj_tail, offset=(k, j))


    @wp.func
    def trsm_llnn_tail(batch_id: int,
                       L: wp.array4d[dtype], # type: ignore
                       L_offset: int,
                       E: wp.array4d[dtype], # type: ignore
                       E_offset: int):
        
        if wp.static(tail_m > 0):
            k = L.shape[2] - tail_m
            L_kk = wp.tile_load(L[batch_id, L_offset], shape=(tail_m, tail_m), offset=(k, k))

            if wp.static(block_size <= n):
                for j in range(0, E.shape[3] - tail_n, block_size):    
                    E_kj = wp.tile_load(E[batch_id, E_offset], shape=(tail_m, block_size), offset=(k, j))
                    
                    if wp.static(block_size <= m):
                        for i in range(0, k, block_size):
                            E_ij = wp.tile_load(E[batch_id, E_offset], shape=(block_size, block_size), offset=(i, j))
                            L_ki = wp.tile_load(L[batch_id, L_offset], shape=(tail_m, block_size), offset=(k, i))
                            wp.tile_matmul(L_ki, E_ij, E_kj, alpha=-1.0)

                    wp.tile_lower_solve_inplace(L_kk, E_kj)
                    wp.tile_store(E[batch_id, E_offset], E_kj, offset=(k, j))

            if wp.static(tail_n > 0):
                j = E.shape[3] - tail_n
                E_kj_tail = wp.tile_load(E[batch_id, E_offset], shape=(tail_m, tail_n), offset=(k, j))
                
                if wp.static(block_size <= m):
                    for i in range(0, k, block_size):
                        E_ij_tail = wp.tile_load(E[batch_id, E_offset], shape=(block_size, tail_n), offset=(i, j))
                        L_ki_tail = wp.tile_load(L[batch_id, L_offset], shape=(tail_m, block_size), offset=(k, i))
                        wp.tile_matmul(L_ki_tail, E_ij_tail, E_kj_tail, alpha=-1.0)

                wp.tile_lower_solve_inplace(L_kk, E_kj_tail)
                wp.tile_store(E[batch_id, E_offset], E_kj_tail, offset=(k, j))


    @wp.func
    def trsm_llnn_blocked(batch_id: int,
                          L: wp.array4d[dtype], # type: ignore
                          L_offset: int,
                          E: wp.array4d[dtype], # type: ignore
                          E_offset: int):
        '''
        Batched blocked in-place triangular solve. Only the lower triangular part of L is used.

        It performs an inline in-place triangular solve for a specific entry in L and E, i.e.,
        E[batch_id, E_offset, :, :] = L[batch_id, L_offset, :, :]^{-1} * E[batch_id, E_offset, :, :]

        L is of shape (batch_dim, L_elements, m, m),
        where batch_dim is the number of batches,
              L_elements is the number of blockes stored,
              m x m is the size of a single block

        E is of shape (batch_dim, E_elements, m, n),
        where batch_dim is the number of batches,
              E_elements is the number of blockes stored,
              m x n is the size of a single block
        '''

        trsm_llnn(batch_id, L, L_offset, E, E_offset)
        trsm_llnn_tail(batch_id, L, L_offset, E, E_offset)

    return trsm_llnn_blocked


def create_trsm_lltn_blocked_func(m: int, n: int, block_size: int, dtype=wp.float64):

    tail_m = m % block_size
    tail_n = n % block_size

    @wp.func
    def trsm_lltn(batch_id: int,
                  L: wp.array4d[dtype], # type: ignore
                  L_offset: int,
                  E: wp.array4d[dtype], # type: ignore
                  E_offset: int):
        
        if wp.static(block_size <= m):
            for k in reversed(range(0, L.shape[2] - tail_m, block_size)):
                L_kk = wp.tile_load(L[batch_id, L_offset], shape=(block_size, block_size), offset=(k, k))
                L_kk_T = wp.tile_transpose(L_kk)

                if wp.static(block_size <= n):
                    for j in range(0, E.shape[3] - tail_n, block_size):
                        E_kj = wp.tile_load(E[batch_id, E_offset], shape=(block_size, block_size), offset=(k, j))
                        
                        for i in range(k + block_size, L.shape[2] - tail_m, block_size):
                            E_ij = wp.tile_load(E[batch_id, E_offset], shape=(block_size, block_size), offset=(i, j))
                            L_ik = wp.tile_load(L[batch_id, L_offset], shape=(block_size, block_size), offset=(i, k))
                            L_ik_T = wp.tile_transpose(L_ik)
                            wp.tile_matmul(L_ik_T, E_ij, E_kj, alpha=-1.0)

                        if wp.static(tail_m > 0):
                            i = L.shape[2] - tail_m
                            E_ij_2 = wp.tile_load(E[batch_id, E_offset], shape=(tail_m, block_size), offset=(i, j))
                            L_ik_2 = wp.tile_load(L[batch_id, L_offset], shape=(tail_m, block_size), offset=(i, k))
                            L_ik_T_2 = wp.tile_transpose(L_ik_2)
                            wp.tile_matmul(L_ik_T_2, E_ij_2, E_kj, alpha=-1.0)

                        wp.tile_upper_solve_inplace(L_kk_T, E_kj)
                        wp.tile_store(E[batch_id, E_offset], E_kj, offset=(k, j))

                if wp.static(tail_n > 0):
                    j = E.shape[3] - tail_n
                    E_kj_tail = wp.tile_load(E[batch_id, E_offset], shape=(block_size, tail_n), offset=(k, j))
                        
                    for i in range(k + block_size, L.shape[2] - tail_m, block_size):
                        E_ij_3 = wp.tile_load(E[batch_id, E_offset], shape=(block_size, tail_n), offset=(i, j))
                        L_ik_3 = wp.tile_load(L[batch_id, L_offset], shape=(block_size, block_size), offset=(i, k))
                        L_ik_T_3 = wp.tile_transpose(L_ik_3)
                        wp.tile_matmul(L_ik_T_3, E_ij_3, E_kj_tail, alpha=-1.0)

                    if wp.static(tail_m > 0):
                        i = L.shape[2] - tail_m
                        E_ij_4 = wp.tile_load(E[batch_id, E_offset], shape=(tail_m, tail_n), offset=(i, j))
                        L_ik_4 = wp.tile_load(L[batch_id, L_offset], shape=(tail_m, block_size), offset=(i, k))
                        L_ik_T_4 = wp.tile_transpose(L_ik_4)
                        wp.tile_matmul(L_ik_T_4, E_ij_4, E_kj_tail, alpha=-1.0)

                    wp.tile_upper_solve_inplace(L_kk_T, E_kj_tail)
                    wp.tile_store(E[batch_id, E_offset], E_kj_tail, offset=(k, j))


    @wp.func
    def trsm_lltn_tail(batch_id: int,
                       L: wp.array4d[dtype], # type: ignore
                       L_offset: int,
                       E: wp.array4d[dtype], # type: ignore
                       E_offset: int):
        
        if wp.static(tail_m > 0):
            k = L.shape[2] - tail_m
            L_kk = wp.tile_load(L[batch_id, L_offset], shape=(tail_m, tail_m), offset=(k, k))
            L_kk_T = wp.tile_transpose(L_kk)

            if wp.static(block_size <= n):
                for j in range(0, E.shape[3] - tail_n, block_size):    
                    E_kj = wp.tile_load(E[batch_id, E_offset], shape=(tail_m, block_size), offset=(k, j))
                    wp.tile_upper_solve_inplace(L_kk_T, E_kj)
                    wp.tile_store(E[batch_id, E_offset], E_kj, offset=(k, j))

            if wp.static(tail_n > 0):
                j = E.shape[3] - tail_n
                E_kj_tail = wp.tile_load(E[batch_id, E_offset], shape=(tail_m, tail_n), offset=(k, j))
                wp.tile_upper_solve_inplace(L_kk_T, E_kj_tail)
                wp.tile_store(E[batch_id, E_offset], E_kj_tail, offset=(k, j))


    @wp.func
    def trsm_lltn_blocked(batch_id: int,
                          L: wp.array4d[dtype], # type: ignore
                          L_offset: int,
                          E: wp.array4d[dtype], # type: ignore
                          E_offset: int):
        '''
        Batched blocked in-place triangular solve. Only the lower triangular part of L is used.

        It performs an inline in-place triangular solve for a specific entry in L and E, i.e.,
        E[batch_id, E_offset, :, :] = L[batch_id, L_offset, :, :]^{-T} * E[batch_id, E_offset, :, :]

        L is of shape (batch_dim, L_elements, m, m),
        where batch_dim is the number of batches,
              L_elements is the number of blockes stored,
              m x m is the size of a single block

        E is of shape (batch_dim, E_elements, m, n),
        where batch_dim is the number of batches,
              E_elements is the number of blockes stored,
              m x n is the size of a single block
        '''

        trsm_lltn_tail(batch_id, L, L_offset, E, E_offset)
        trsm_lltn(batch_id, L, L_offset, E, E_offset)

    return trsm_lltn_blocked
