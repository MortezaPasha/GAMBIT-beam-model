"""MeerKAT array-average beam model: base beam + per-channel zoom/shift transformations."""

from pathlib import Path

import numpy as np

from .transforms import zoom_and_shift
from .zernike import zernike_image

#: Folder inside the package holding beam_mean.npy, transformations.npy and
#: zernike_coeffs.npz.
DATA_DIR = Path(__file__).resolve().parent / "data"

#: Native (base) resolution of beam_mean and of all stored parameters, in pixels.
NATIVE_SIZE = 128

#: MeerKAT L-band channel frequencies in MHz (900 channels).
FREQS = np.arange(856, 1712, 0.8359375)[:900]

SOURCES = ("mean", "zernike")


def _load_coeffs(path):
    with np.load(path) as f:
        key = "coeffs" if "coeffs" in f.files else f.files[0]
        return np.asarray(f[key], dtype=np.float64).ravel()


def _load_beam(path):
    """Load the base beam as a NATIVE_SIZE x NATIVE_SIZE array.

    Extra length-1 axes are dropped, and a flattened beam of 128*128 values is
    reshaped to (128, 128).
    """
    beam = np.squeeze(np.load(path)).astype(np.float64)
    if beam.size == NATIVE_SIZE * NATIVE_SIZE:
        beam = beam.reshape(NATIVE_SIZE, NATIVE_SIZE)
    if beam.shape != (NATIVE_SIZE, NATIVE_SIZE):
        raise ValueError(f"{path} should hold a {NATIVE_SIZE}x{NATIVE_SIZE} beam, "
                         f"got shape {beam.shape}")
    return beam


class MeerkatModel:
    """MeerKAT antenna-array average beam.

    There is one 128 x 128 base beam with no frequency axis (``base_beam``).
    Frequency only selects the zoom/shift parameters: the beam at a channel is
    the base beam transformed with that channel's parameters (``beam``).

    The data are read from the package's data folder:

    - ``beam_mean.npy``: original base beam (square 2D array)
    - ``transformations.npy``: [factor_mean, factor_std, x_shift_mean,
      x_shift_std, y_shift_mean, y_shift_std], shape (6, 900), one column per
      channel of ``FREQS`` (shape (6,) is also accepted and used for every channel)
    - ``zernike_coeffs.npz``: Zernike coefficients of the base beam (Noll order)

    Everything is defined on the native 128 x 128 grid of beam_mean: shifts are
    in native pixels and are rescaled when another resolution is requested; the
    zoom factor is dimensionless.
    """

    freqs = FREQS

    def __init__(self, order=1):
        self.order = order
        self.beam_mean = _load_beam(DATA_DIR / "beam_mean.npy")
        self.native_size = NATIVE_SIZE
        t = np.load(DATA_DIR / "transformations.npy").astype(np.float64)
        if t.ndim == 1:
            t = np.repeat(t[:, None], len(FREQS), axis=1)
        if t.shape != (6, len(FREQS)):
            raise ValueError(f"transformations.npy must have shape (6, {len(FREQS)}), got {t.shape}")
        self.transformations = t
        (self.factor_mean, self.factor_std,
         self.shift_x_mean, self.shift_x_std,
         self.shift_y_mean, self.shift_y_std) = t
        self.zernike_coeffs = _load_coeffs(DATA_DIR / "zernike_coeffs.npz")
        self._zernike_cache = {}

    def channels(self, channels=None, freqs=None):
        """Channel indices: given directly, nearest to `freqs` (MHz), or all 900."""
        if freqs is not None:
            freqs = np.asarray(freqs, dtype=np.float64)
            return np.abs(FREQS[:, None] - freqs.ravel()[None, :]).argmin(0).reshape(freqs.shape)
        if channels is None:
            return np.arange(len(FREQS))
        return np.asarray(channels, dtype=int)

    def base_beam(self, source="mean", resolution=None):
        """The untransformed base beam.

        ``source="mean"``: the original beam_mean (native resolution only).
        ``source="zernike"``: the Zernike expansion on a ``resolution x resolution``
        grid covering the same field of view as beam_mean.
        """
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        res = self.native_size if resolution is None else int(resolution)
        if source == "mean":
            if res != self.native_size:
                raise ValueError(f'source="mean" is only available at the native '
                                 f'resolution ({self.native_size}); use source="zernike"')
            return self.beam_mean
        if res not in self._zernike_cache:
            self._zernike_cache[res] = zernike_image(self.zernike_coeffs, self.native_size, res)
        return self._zernike_cache[res]

    def transform(self, factor, shift_x, shift_y, source="mean", resolution=None):
        """Base beam zoomed and shifted by explicit parameters (scalars or 1D arrays).

        Shifts are given in native pixels and rescaled to the requested resolution.
        """
        factor, shift_x, shift_y = np.broadcast_arrays(
            np.asarray(factor, dtype=np.float64),
            np.asarray(shift_x, dtype=np.float64),
            np.asarray(shift_y, dtype=np.float64))
        base = self.base_beam(source, resolution)
        s = base.shape[0] / self.native_size
        return zoom_and_shift(base, factor, shift_x * s, shift_y * s, self.order)

    def beam(self, channels=None, freqs=None, source="mean", resolution=None):
        """Base beam transformed with the mean zoom/shift of the selected channels.

        Select channels by index (`channels`) or by frequency in MHz (`freqs`,
        nearest channel); by default all 900. Returns (res, res) for a single
        channel, otherwise (n_channels, res, res).
        """
        ch = self.channels(channels, freqs)
        return self.transform(self.factor_mean[ch], self.shift_x_mean[ch],
                              self.shift_y_mean[ch], source, resolution)

    def sample_params(self, channels=None, freqs=None, seed=None):
        """Draw (factor, shift_x, shift_y) per channel from the stored Gaussian means / stds."""
        ch = self.channels(channels, freqs)
        rng = np.random.default_rng(seed)
        return (rng.normal(self.factor_mean[ch], self.factor_std[ch]),
                rng.normal(self.shift_x_mean[ch], self.shift_x_std[ch]),
                rng.normal(self.shift_y_mean[ch], self.shift_y_std[ch]))

    def sample(self, channels=None, freqs=None, seed=None, source="mean", resolution=None):
        """One random beam per selected channel; returns (beams, (factor, shift_x, shift_y))."""
        params = self.sample_params(channels, freqs, seed)
        return self.transform(*params, source=source, resolution=resolution), params
