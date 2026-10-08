"""Reconstruct MeerKAT beams both ways from the files in the package's data folder."""

import numpy as np

from gambit import FREQS, MeerkatModel

model = MeerkatModel()

# The base beam itself has no frequency: one 128x128 image, or its Zernike version at any size
base = model.base_beam(source="mean")
base_hr = model.base_beam(source="zernike", resolution=512)
print("base beam", base.shape, "| Zernike base beam", base_hr.shape)

# Frequency only selects the zoom/shift parameters applied to that base beam

# 1) from the original base beam mean, one channel
beam_from_mean = np.asarray(model.beam(freqs=1284.0, source="mean"))

# 2) from the Zernike base beam, same channel, native and higher resolution
beam_from_zernike = np.asarray(model.beam(freqs=1284.0, source="zernike"))
beam_hires = np.asarray(model.beam(freqs=1284.0, source="zernike", resolution=4 * model.native_size))

diff = np.abs(beam_from_mean - beam_from_zernike).max() / np.abs(beam_from_mean).max()
print("native size", model.native_size, "| max |mean - zernike| / peak =", f"{diff:.2e}")
print("high-res beam shape", beam_hires.shape)

# All 900 channels at a chosen resolution
cube = model.beam(source="zernike", resolution=128)
print("beam cube", cube.shape, "from", FREQS[0], "to", FREQS[-1], "MHz")

# One random beam per channel, drawn from the stored means / stds
beams, (factor, shift_x, shift_y) = model.sample(channels=np.arange(0, 900, 100), seed=0)
print("sampled beams", beams.shape)
