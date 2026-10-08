"""Zernike decomposition of the base beam and its evaluation at any resolution.

`ZernikeDecomposer` is the user's original class (same basis, same unit-disk
convention, same pseudo-inverse fit). The only changes are that the covariance
matrix is built with one matrix product instead of a double Python loop, and
that matplotlib is imported lazily so it is not a hard dependency.

`zernike_image` evaluates a set of coefficients on an arbitrary grid that
covers the same field of view as the native beam, quickly (modes grouped by m,
Jacobi recurrence, JAX). `zernike_image_direct` is the straightforward
reference; at the native resolution it reproduces
`ZernikeDecomposer.reconstruct` exactly.
"""

import functools
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

fac = lambda x: math.factorial(int(x))


def noll_to_zern(j):
    j += 1  # shift to 1-based
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return (n, m)


def zernike_rad(m, n, rho):
    if (n < 0 or m < 0 or abs(m) > n):
        raise ValueError
    if ((n - m) % 2):
        return rho * 0.0
    result = 0
    for k in range((n - m) // 2 + 1):
        num = (-1)**k * fac(n - k)
        denom = fac(k) * fac((n + m) // 2 - k) * fac((n - m) // 2 - k)
        result += num / denom * rho**(n - 2 * k)
    return result


def zernike(m, n, rho, phi):
    if m > 0:
        return zernike_rad(m, n, rho) * np.cos(m * phi)
    elif m < 0:
        return zernike_rad(-m, n, rho) * np.sin(-m * phi)
    else:
        return zernike_rad(0, n, rho)


def zernikel(j, rho, phi):
    n, m = noll_to_zern(j)
    return zernike(m, n, rho, phi)


def unit_disk(img_shape):
    """The decomposer's native unit disk: pixel i maps to (i - N/2) / (N/2)."""
    nx, ny = img_shape
    grid = (np.indices((nx, ny), dtype=float) - nx / 2) / (nx * 1.0 / 2)
    rho = np.sqrt(grid[0]**2 + grid[1]**2)
    phi = np.arctan2(grid[0], grid[1])
    mask = rho <= 1
    return rho, phi, mask


def _resampled_axes(native_size, out_shape):
    """Row (y) and column (x) unit-disk coordinates of the resampled grid."""
    n = float(native_size)
    ny, nx = out_shape
    iy = (np.arange(ny) + 0.5) * n / ny - 0.5
    ix = (np.arange(nx) + 0.5) * n / nx - 0.5
    return (iy - n / 2) / (n / 2), (ix - n / 2) / (n / 2)


def resampled_disk(native_size, out_shape):
    """Unit-disk coordinates of an `out_shape` grid with the native field of view.

    The native grid has `native_size` pixels per side; native pixel `i` sits at
    u = (i - N/2) / (N/2), exactly as in `unit_disk`. The output grid covers the
    same field of view (the same outer pixel edges) with `out_shape` pixels, so
    output pixel `k` has its centre at native index (k + 0.5) * N / N' - 0.5.
    For `out_shape == (N, N)` this is identical to `unit_disk((N, N))`.
    """
    gy, gx = np.meshgrid(*_resampled_axes(native_size, out_shape), indexing="ij")
    rho = np.sqrt(gy**2 + gx**2)
    phi = np.arctan2(gy, gx)
    mask = rho <= 1
    return rho, phi, mask


def zernike_image_direct(coeffs, native_size, out_shape=None, block_pixels=1 << 18):
    """Evaluate `sum_j coeffs[j] * Z_j` on a grid of `out_shape` pixels.

    Parameters
    ----------
    coeffs : array (Nmodes,)
        Noll-ordered coefficients (index 0 = piston), as returned by
        `ZernikeDecomposer.decompose`.
    native_size : int
        Side length in pixels of the beam the coefficients were fitted on.
    out_shape : int or (ny, nx), optional
        Output grid. Defaults to the native resolution.
    block_pixels : int
        Rows are processed in blocks of about this many pixels, so memory stays
        bounded at any resolution.

    Direct evaluation with the factorial formula of `zernike_rad` / `zernike`
    (powers and cos/sin factors shared within a block). At the native
    resolution it equals `ZernikeDecomposer.reconstruct` bit for bit. Kept as
    the reference; `zernike_image` is the fast version used by the model.
    """
    if out_shape is None:
        out_shape = (native_size, native_size)
    elif np.isscalar(out_shape):
        out_shape = (int(out_shape), int(out_shape))
    coeffs = np.asarray(coeffs, dtype=np.float64)
    modes = [(j, c) + noll_to_zern(j) for j, c in enumerate(coeffs) if c != 0.0]
    nmax = max((n for _, _, n, _ in modes), default=0)

    rho, phi, mask = resampled_disk(native_size, out_shape)
    recon = np.zeros(out_shape, dtype=np.float64)
    rows = max(1, block_pixels // out_shape[1])
    for r0 in range(0, out_shape[0], rows):
        rb, pb = rho[r0:r0 + rows], phi[r0:r0 + rows]
        powers = [rb**p for p in range(nmax + 1)]
        ang = {}
        acc = np.zeros_like(rb)
        for j, c, n, m in modes:
            am = abs(m)
            if (n - am) % 2:
                continue
            R = 0
            for k in range((n - am) // 2 + 1):
                num = (-1)**k * fac(n - k)
                denom = fac(k) * fac((n + am) // 2 - k) * fac((n - am) // 2 - k)
                R += num / denom * powers[n - 2 * k]
            if m != 0:
                if m not in ang:
                    ang[m] = np.cos(m * pb) if m > 0 else np.sin(-m * pb)
                R = R * ang[m]
            acc += c * R
        recon[r0:r0 + rows] = acc
    return recon * mask


def _group_by_m(coeffs):
    """Coefficient tables A[m, k] (cos / m >= 0) and B[m, k] (sin) for R_{m+2k}^m."""
    groups = [(noll_to_zern(j), c) for j, c in enumerate(np.asarray(coeffs, dtype=np.float64))]
    mmax = max(abs(m) for (n, m), _ in groups)
    kmax = max((n - abs(m)) // 2 for (n, m), _ in groups)
    A = np.zeros((mmax + 1, kmax + 1))
    B = np.zeros((mmax + 1, kmax + 1))
    for (n, m), c in groups:
        if (n - abs(m)) % 2:
            continue
        (A if m >= 0 else B)[abs(m), (n - abs(m)) // 2] += c
    used = (A != 0) | (B != 0)
    kmax_m = np.array([max([k for k in range(kmax + 1) if used[m, k]] + [1]) for m in range(mmax + 1)])
    return A, B, kmax_m


@functools.partial(jax.jit, static_argnums=4)
def _zernike_block(x, y, A, B, mmax, kmax_m):
    """sum_m [Re(w^m) a_m(t) + Im(w^m) b_m(t)] on a block, w = x + i y, t = 2 rho^2 - 1.

    Uses R_{m+2k}^m(rho) = rho^m P_k^(0,m)(2 rho^2 - 1) with the three-term Jacobi
    recurrence, and rho^m cos(m phi) = Re(w^m), rho^m sin(m phi) = Im(w^m).
    """
    r2 = x * x + y * y
    t = 2 * r2 - 1

    def per_m(m, carry):
        img, wr, wi = carry
        mf = m.astype(jnp.float64)
        P0 = jnp.ones_like(t)
        P1 = 1 + (mf + 2) * (t - 1) / 2
        a = A[m, 0] * P0 + A[m, 1] * P1
        b = B[m, 0] * P0 + B[m, 1] * P1

        def per_k(k, c):
            P0, P1, a, b = c
            kf = k.astype(jnp.float64)
            cc = 2 * kf + mf
            P2 = ((cc - 1) * (cc * (cc - 2) * t - mf * mf) * P1
                  - 2 * (kf - 1) * (kf + mf - 1) * cc * P0) / (2 * kf * (kf + mf) * (cc - 2))
            return P1, P2, a + A[m, k] * P2, b + B[m, k] * P2

        _, _, a, b = jax.lax.fori_loop(2, kmax_m[m] + 1, per_k, (P0, P1, a, b))
        img = img + wr * a + wi * b
        return img, wr * x - wi * y, wr * y + wi * x

    img, _, _ = jax.lax.fori_loop(0, mmax + 1, per_m,
                                  (jnp.zeros_like(x), jnp.ones_like(x), jnp.zeros_like(x)))
    return jnp.where(r2 <= 1, img, 0.0)


def zernike_image(coeffs, native_size, out_shape=None, block_pixels=1 << 20):
    """Evaluate `sum_j coeffs[j] * Z_j` on a grid of `out_shape` pixels (fast).

    Same grid and basis as `zernike_image_direct`, computed faster and more
    accurately: modes are grouped by azimuthal order m (about 40 groups for 800
    modes), the radial polynomials come from the stable Jacobi recurrence instead
    of the factorial sums, and the per-block sum is compiled with JAX. Rows are
    processed in blocks of about `block_pixels` pixels to bound memory.
    """
    if out_shape is None:
        out_shape = (native_size, native_size)
    elif np.isscalar(out_shape):
        out_shape = (int(out_shape), int(out_shape))
    A, B, kmax_m = _group_by_m(coeffs)
    A, B, kmax_m = jnp.asarray(A), jnp.asarray(B), jnp.asarray(kmax_m)
    mmax = A.shape[0] - 1
    gy, gx = _resampled_axes(native_size, out_shape)
    out = np.empty(out_shape, dtype=np.float64)
    rows = max(1, block_pixels // out_shape[1])
    xx = np.broadcast_to(gx, (rows, out_shape[1]))
    for r0 in range(0, out_shape[0], rows):
        r1 = min(r0 + rows, out_shape[0])
        yy = np.zeros((rows, 1))
        yy[: r1 - r0, 0] = gy[r0:r1]   # pad the last block so every block has one shape
        out[r0:r1] = np.asarray(_zernike_block(xx, np.broadcast_to(yy, xx.shape),
                                               A, B, mmax, kmax_m))[: r1 - r0]
    return out


class ZernikeDecomposer:
    def __init__(self, Nmodes=50):
        self.Nmodes = Nmodes

    def zernike_rad(self, m, n, rho):
        return zernike_rad(m, n, rho)

    def zernike(self, m, n, rho, phi):
        return zernike(m, n, rho, phi)

    def noll_to_zern(self, j):
        return noll_to_zern(j)

    def zernikel(self, j, rho, phi):
        return zernikel(j, rho, phi)

    def unit_disk(self, img_shape):
        return unit_disk(img_shape)

    def decompose(self, img):
        img = np.asarray(img, dtype=np.float64)
        self.rho, self.phi, self.mask = self.unit_disk(img.shape)
        self.basis = [self.zernikel(j, self.rho, self.phi) * self.mask for j in range(self.Nmodes)]
        # Same as [[sum(b1*b2) for b1] for b2] and [sum(img*b) for b], computed
        # as matrix products over the pixels inside the disk.
        B = np.array([b[self.mask] for b in self.basis])
        cov_mat = B @ B.T
        cov_mat_inv = np.linalg.pinv(cov_mat)
        innerprod = B @ img[self.mask]
        self.coeffs = np.dot(cov_mat_inv, innerprod)
        return self.coeffs

    def reconstruct(self, coeffs=None):
        if coeffs is None:
            coeffs = self.coeffs
        recon = sum(c * b for c, b in zip(coeffs, self.basis))
        return recon * self.mask

    def keep_top_n(self, coeffs, n=20):
        """Keep the top-N coefficients by absolute value, zero out the rest."""
        coeffs_copy = np.copy(coeffs)
        idx_sorted = np.argsort(np.abs(coeffs_copy))[::-1]
        idx_to_zero = idx_sorted[n:]
        coeffs_copy[idx_to_zero] = 0
        return coeffs_copy

    def plot_basis(self, indices=None, coeffs=None, cmap='RdBu_r', savepath=None):
        """
        Plot selected Zernike basis polynomials with (n, m) in the title.

        Parameters
        ----------
        indices : list or None
            Indices of basis polynomials to plot. If None, plot all.
        coeffs : array-like or None
            Optional coefficients to scale the basis functions.
        cmap : str
            Colormap for visualization.
        savepath : str or None
            If provided, save the figure to this path (e.g. 'zernike_basis.pdf').
        """
        import matplotlib.pyplot as plt

        if not hasattr(self, 'basis'):
            raise ValueError("You must run decompose() first.")

        if indices is None:
            indices = range(len(self.basis))

        n = len(indices)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

        fig = plt.figure(figsize=(3.5 * ncols, 3.5 * nrows), dpi=120)

        for i, idx in enumerate(indices):
            n_val, m_val = noll_to_zern(idx)

            data = self.basis[idx]
            if coeffs is not None:
                data = data * coeffs[idx]

            vmax = np.max(np.abs(data))

            ax = plt.subplot(nrows, ncols, i + 1)
            im = ax.imshow(
                data,
                origin='lower',
                interpolation='nearest',
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"$Z_{{{n_val}}}^{{{m_val}}}$ (Idx {idx+1})", fontsize=25)
            ax.axis('off')

        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
