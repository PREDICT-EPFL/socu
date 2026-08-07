import jax
import jax.numpy as jnp
import warp as wp

from socu.block_tridiag_solver import (
    calculate_off_diag_storage_len,
    create_backward_substitution_launch,
    create_cholesky_factor_and_solve_launch,
    create_cholesky_factor_launch,
    create_cholesky_solve_launch,
    create_forward_substitution_launch,
    optimal_problem_settings,
)


def to_wp_dtype(dtype: jnp.dtype):
    if dtype == jnp.float32:
        return wp.float32
    elif dtype == jnp.float64:
        return wp.float64
    else:
        raise RuntimeError("dtype must be float32 or float64")


def _validate_problem_shapes(L: jnp.ndarray, E: jnp.ndarray, b: jnp.ndarray = None):
    if L.ndim not in (3, 4):
        raise ValueError(f"L must be 3- or 4-dimensional, got shape {L.shape}")
    if E.ndim != L.ndim:
        raise ValueError(f"E must have the same rank as L, got L: {L.shape}, E: {E.shape}")

    horizon_axis = L.ndim - 3
    block_axis = L.ndim - 2

    N = L.shape[horizon_axis]
    n = L.shape[block_axis]
    if L.shape[-1] != n:
        raise ValueError(f"L must be square in last two dimensions, got shape {L.shape}")

    if E.shape[horizon_axis + 1 :] != L.shape[block_axis:]:
        raise ValueError(f"E block dimensions must match L's n={n}, got E shape {E.shape}")

    if L.ndim == 4 and E.shape[0] != L.shape[0]:
        raise ValueError(f"L and E must have the same batch size, got L: {L.shape}, E: {E.shape}")

    if b is not None:
        if b.ndim != L.ndim:
            raise ValueError(f"b must have the same rank as L, got L: {L.shape}, b: {b.shape}")
        if b.shape[horizon_axis] != N:
            raise ValueError(f"b horizon dimension must be {N}, got b shape {b.shape}")
        if b.shape[block_axis] != n:
            raise ValueError(f"b block dimension must be {n}, got b shape {b.shape}")
        if L.ndim == 4 and b.shape[0] != L.shape[0]:
            raise ValueError(f"L and b must have the same batch size, got L: {L.shape}, b: {b.shape}")
        if L.dtype != b.dtype:
            raise ValueError(f"L and b must have same dtype, got L: {L.dtype}, b: {b.dtype}")

    if L.dtype != E.dtype:
        raise ValueError(f"L and E must have same dtype, got L: {L.dtype}, E: {E.dtype}")


def _to_batched(L: jnp.ndarray, E: jnp.ndarray, b: jnp.ndarray = None):
    # Internally all solver calls use an explicit leading batch axis.
    was_unbatched = L.ndim == 3
    if was_unbatched:
        L = L[None, ...]
        E = E[None, ...]
        if b is not None:
            b = b[None, ...]
    return L, E, b, was_unbatched


def _ensure_E_size(horizon: int, E: jnp.ndarray):
    E_elements = calculate_off_diag_storage_len(horizon)
    if E.shape[1] < E_elements:
        return jnp.pad(
            E,
            pad_width=((0, 0), (0, E_elements - E.shape[1]), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        return E


def _pad_factor_data(
    L: jnp.ndarray,
    E: jnp.ndarray,
    b: jnp.ndarray = None,
    factor: bool = True,
):
    # Pad block dimensions to Warp's preferred tile size.
    n = L.shape[2]
    opt_settings = optimal_problem_settings(n, to_wp_dtype(L.dtype))
    if factor:
        pad_multiple = opt_settings["pad_multiple"]["factor"]
    else:
        pad_multiple = opt_settings["pad_multiple"]["solve"]
    n_padded = ((n + pad_multiple - 1) // pad_multiple) * pad_multiple

    if n_padded != n:
        pad_width = n_padded - n

        L = jnp.pad(
            L,
            pad_width=((0, 0), (0, 0), (0, pad_width), (0, pad_width)),
            mode="constant",
            constant_values=0.0,
        )
        identity_pad = jnp.eye(n_padded, dtype=L.dtype)
        identity_pad = identity_pad.at[:n, :n].set(0)
        # Padded diagonal blocks must stay SPD but leave the real block unchanged.
        L = L + identity_pad[None, None, :, :]

        E = jnp.pad(
            E,
            pad_width=((0, 0), (0, 0), (0, pad_width), (0, pad_width)),
            mode="constant",
            constant_values=0.0,
        )

        if b is not None:
            b = jnp.pad(
                b,
                pad_width=((0, 0), (0, 0), (0, pad_width), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )

    return L, E, b


def _cholesky_factor_and_solve_impl(
    L: jnp.ndarray,  # (B, N, n, n)
    E: jnp.ndarray,  # (B, n_E, n, n)
    b: jnp.ndarray = None,  # (B, N, n, n_rhs)
    pad_problem: bool = True,
    mode: str = "factor",
):
    dtype = to_wp_dtype(L.dtype)
    horizon = L.shape[1]
    n = L.shape[2]

    E = _ensure_E_size(horizon, E)

    if pad_problem:
        L, E, b = _pad_factor_data(L, E, b, factor=mode in ("factor", "factor_and_solve"))

    if mode in ("solve", "forward", "backward"):
        assert b is not None

        if mode == "solve":
            create_launch = create_cholesky_solve_launch
        elif mode == "forward":
            create_launch = create_forward_substitution_launch
        else:
            create_launch = create_backward_substitution_launch

        def func(
            L_wp: wp.array4d[dtype],  # type: ignore
            E_wp: wp.array4d[dtype],  # type: ignore
            x_wp: wp.array4d[dtype],  # type: ignore
        ):
            create_launch(L_wp, E_wp, x_wp, dtype=dtype)()

        jax_func = wp.jax_callable(func, num_outputs=1, in_out_argnames=["x_wp"])
        x = jax_func(L, E, b)[0]

        return x[:, :, :n, :]

    if b is None:

        def func(
            L_wp: wp.array4d[dtype],  # type: ignore
            E_wp: wp.array4d[dtype],  # type: ignore
        ):
            create_cholesky_factor_launch(L_wp, E_wp, dtype=dtype)()

        jax_func = wp.jax_callable(func, num_outputs=2, in_out_argnames=["L_wp", "E_wp"])
        L, E = jax_func(L, E)

        return L[:, :, :n, :n], E[:, :, :n, :n]

    else:

        def func(
            L_wp: wp.array4d[dtype],  # type: ignore
            E_wp: wp.array4d[dtype],  # type: ignore
            x_wp: wp.array4d[dtype],  # type: ignore
        ):
            create_cholesky_factor_and_solve_launch(L_wp, E_wp, x_wp, dtype=dtype)()

        jax_func = wp.jax_callable(func, num_outputs=3, in_out_argnames=["L_wp", "E_wp", "x_wp"])
        L, E, x = jax_func(L, E, b)

        return L[:, :, :n, :n], E[:, :, :n, :n], x[:, :, :n, :]


def _cholesky_factor_and_solve(
    L: jnp.ndarray,
    E: jnp.ndarray,
    b: jnp.ndarray = None,
    pad_problem: bool = True,
    mode: str = "factor",
):
    _validate_problem_shapes(L, E, b)
    L, E, b, was_unbatched = _to_batched(L, E, b)

    out = _cholesky_factor_and_solve_impl(L, E, b, pad_problem=pad_problem, mode=mode)

    if not was_unbatched:
        return out

    if mode in ("solve", "forward", "backward"):
        return out[0]
    if b is None:
        L_out, E_out = out
        return L_out[0], E_out[0]
    else:
        L_out, E_out, x_out = out
        return L_out[0], E_out[0], x_out[0]


def _make_cholesky_custom_vmap(pad_problem: bool, mode: str, has_b: bool):
    # Route vmap to one 4D Warp call instead of letting FFI flatten (B, N).
    if has_b:

        @jax.custom_batching.custom_vmap
        def op(L: jnp.ndarray, E: jnp.ndarray, b: jnp.ndarray):
            return _cholesky_factor_and_solve(L, E, b, pad_problem=pad_problem, mode=mode)

        @op.def_vmap
        def op_vmap(axis_size, in_batched, L, E, b):
            L_batched = L if in_batched[0] else jnp.broadcast_to(L, (axis_size,) + L.shape)
            E_batched = E if in_batched[1] else jnp.broadcast_to(E, (axis_size,) + E.shape)
            b_batched = b if in_batched[2] else jnp.broadcast_to(b, (axis_size,) + b.shape)
            out = _cholesky_factor_and_solve_impl(
                L_batched,
                E_batched,
                b_batched,
                pad_problem=pad_problem,
                mode=mode,
            )
            return out, (True, True, True) if mode == "factor_and_solve" else True

    else:

        @jax.custom_batching.custom_vmap
        def op(L: jnp.ndarray, E: jnp.ndarray):
            return _cholesky_factor_and_solve(L, E, pad_problem=pad_problem, mode=mode)

        @op.def_vmap
        def op_vmap(axis_size, in_batched, L, E):
            L_batched = L if in_batched[0] else jnp.broadcast_to(L, (axis_size,) + L.shape)
            E_batched = E if in_batched[1] else jnp.broadcast_to(E, (axis_size,) + E.shape)
            out = _cholesky_factor_and_solve_impl(
                L_batched,
                E_batched,
                pad_problem=pad_problem,
                mode=mode,
            )
            return out, (True, True)

    return op


_cholesky_factor_padded = _make_cholesky_custom_vmap(pad_problem=True, mode="factor", has_b=False)
_cholesky_factor_unpadded = _make_cholesky_custom_vmap(pad_problem=False, mode="factor", has_b=False)


def cholesky_factor(
    L: jnp.ndarray,  # (N, n, n) or (B, N, n, n)
    E: jnp.ndarray,  # (N - 1, n, n) or (B, N - 1, n, n)
    pad_problem: bool = True,
):
    if pad_problem:
        return _cholesky_factor_padded(L, E)
    return _cholesky_factor_unpadded(L, E)


_cholesky_solve_padded = _make_cholesky_custom_vmap(pad_problem=True, mode="solve", has_b=True)
_cholesky_solve_unpadded = _make_cholesky_custom_vmap(pad_problem=False, mode="solve", has_b=True)


def cholesky_solve(
    L: jnp.ndarray,  # (N, n, n) or (B, N, n, n)
    E: jnp.ndarray,  # (N - 1, n, n) or (B, N - 1, n, n)
    b: jnp.ndarray,  # (N, n, n_rhs) or (B, N, n, n_rhs)
    pad_problem: bool = True,
):
    if pad_problem:
        return _cholesky_solve_padded(L, E, b)
    return _cholesky_solve_unpadded(L, E, b)


_cholesky_factor_and_solve_padded = _make_cholesky_custom_vmap(pad_problem=True, mode="factor_and_solve", has_b=True)
_cholesky_factor_and_solve_unpadded = _make_cholesky_custom_vmap(pad_problem=False, mode="factor_and_solve", has_b=True)


def cholesky_factor_and_solve(
    L: jnp.ndarray,  # (N, n, n) or (B, N, n, n)
    E: jnp.ndarray,  # (N - 1, n, n) or (B, N - 1, n, n)
    b: jnp.ndarray,  # (N, n, n_rhs) or (B, N, n, n_rhs)
    pad_problem: bool = True,
):
    if pad_problem:
        return _cholesky_factor_and_solve_padded(L, E, b)
    return _cholesky_factor_and_solve_unpadded(L, E, b)


_forward_substitution_padded = _make_cholesky_custom_vmap(pad_problem=True, mode="forward", has_b=True)
_forward_substitution_unpadded = _make_cholesky_custom_vmap(pad_problem=False, mode="forward", has_b=True)


def forward_substitution(
    L: jnp.ndarray,  # (N, n, n) or (B, N, n, n), factor from cholesky_factor
    E: jnp.ndarray,  # (N - 1, n, n) or (B, N - 1, n, n), factor from cholesky_factor
    b: jnp.ndarray,  # (N, n, n_rhs) or (B, N, n, n_rhs)
    pad_problem: bool = True,
):
    if pad_problem:
        return _forward_substitution_padded(L, E, b)
    return _forward_substitution_unpadded(L, E, b)


_backward_substitution_padded = _make_cholesky_custom_vmap(pad_problem=True, mode="backward", has_b=True)
_backward_substitution_unpadded = _make_cholesky_custom_vmap(pad_problem=False, mode="backward", has_b=True)


def backward_substitution(
    L: jnp.ndarray,  # (N, n, n) or (B, N, n, n), factor from cholesky_factor
    E: jnp.ndarray,  # (N - 1, n, n) or (B, N - 1, n, n), factor from cholesky_factor
    y: jnp.ndarray,  # (N, n, n_rhs) or (B, N, n, n_rhs)
    pad_problem: bool = True,
):
    if pad_problem:
        return _backward_substitution_padded(L, E, y)
    return _backward_substitution_unpadded(L, E, y)
