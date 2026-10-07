"""JAX zoom / shift transforms applied to the base beam."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import vmap
from jax.scipy.ndimage import map_coordinates


def zoom_and_shift_one_array(in_array, factor, shift_x, shift_y, order=1):
    """Apply zoom and shift to a single array (input and output same shape)."""
    in_array = jnp.asarray(in_array, dtype=jnp.float64)
    factor = jnp.asarray(factor, dtype=jnp.float64)
    shift_x = jnp.asarray(shift_x, dtype=jnp.float64)
    shift_y = jnp.asarray(shift_y, dtype=jnp.float64)

    ny, nx = in_array.shape
    center = jnp.array([ny - 1, nx - 1], dtype=jnp.float64) / 2.0

    out_coos = jnp.indices(in_array.shape).astype(jnp.float64)
    out_coos_T = out_coos.T.reshape(-1, 2)

    out_coos_T = out_coos_T - center
    out_coos_T = out_coos_T * factor
    out_coos_T = out_coos_T + jnp.array([shift_y, shift_x], dtype=jnp.float64)
    out_coos_T = out_coos_T + center

    out_coos = out_coos_T.reshape(out_coos.T.shape).T

    return map_coordinates(in_array, out_coos, order=order, mode='constant', cval=0.0)


def zoom_and_shift(in_array, factor, shift_x, shift_y, order=1):
    """Vectorized zoom+shift over the frequency axis.

    `in_array` is the 2D base beam (shared across frequencies).
    `factor`, `shift_x`, `shift_y` are per-frequency 1D arrays (or scalars).
    Returns a stack of transformed beams shaped (n_freq, ny, nx), or (ny, nx)
    for scalar parameters.
    """
    in_array = jnp.asarray(in_array, dtype=jnp.float64)
    factor = jnp.asarray(factor, dtype=jnp.float64)
    shift_x = jnp.asarray(shift_x, dtype=jnp.float64)
    shift_y = jnp.asarray(shift_y, dtype=jnp.float64)

    if factor.ndim == 0:
        return zoom_and_shift_one_array(in_array, factor, shift_x, shift_y, order)
    else:
        vmap_func = vmap(zoom_and_shift_one_array, in_axes=(None, 0, 0, 0, None))
        return vmap_func(in_array, factor, shift_x, shift_y, order)
