"""Zernike decomposition of the base beam and its evaluation at any resolution.

`ZernikeDecomposer` is the user's original class (same basis, same unit-disk
convention, same pseudo-inverse fit). The only changes are that the covariance
matrix is built with one matrix product instead of a double Python loop, and
that matplotlib is imported lazily so it is not a hard dependency.

`zernike_image` evaluates a set of coefficients on an arbitrary grid that
covers the same field of view as the native beam. At the native resolution it
reproduces `ZernikeDecomposer.reconstruct` exactly.
"""

import math

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


def resampled_disk(native_size, out_shape):
    """Unit-disk coordinates of an `out_shape` grid with the native field of view.

    The native grid has `native_size` pixels per side; native pixel `i` sits at
    u = (i - N/2) / (N/2), exactly as in `unit_disk`. The output grid covers the
    same field of view (the same outer pixel edges) with `out_shape` pixels, so
    output pixel `k` has its centre at native index (k + 0.5) * N / N' - 0.5.
    For `out_shape == (N, N)` this is identical to `unit_disk((N, N))`.
    """
    n = float(native_size)
    ny, nx = out_shape
    iy = (np.arange(ny) + 0.5) * n / ny - 0.5
    ix = (np.arange(nx) + 0.5) * n / nx - 0.5
    gy = (iy - n / 2) / (n / 2)
    gx = (ix - n / 2) / (n / 2)
    gy, gx = np.meshgrid(gy, gx, indexing="ij")
    rho = np.sqrt(gy**2 + gx**2)
    phi = np.arctan2(gy, gx)
    mask = rho <= 1
    return rho, phi, mask


def zernike_image(coeffs, native_size, out_shape=None, block_pixels=1 << 18):
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

    Within a block the powers rho**p and the cos/sin(m*phi) factors are computed
    once and shared by all modes; the arithmetic is otherwise that of
    `zernike_rad` / `zernike`, so the native-resolution result equals
    `ZernikeDecomposer.reconstruct`.
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
