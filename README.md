# gambit

MeerKAT antenna-array average **Stokes I** beam model.

There is **one base beam**, with no frequency axis. It comes either

1. **from the original base beam posterior mean** (`source="mean"`, 128 × 128 only), or
2. **from the Zernike expansion of that base beam** (`source="zernike"`, any resolution).

Frequency enters only through the transformations: for each of the 900 L-band
channels (`FREQS = np.arange(856, 1712, 0.8359375)[:900]`, MHz),
`transformations.npy` holds the mean and std of the geometric transformations (scaling and the l/m centroid
shifts). The beam at a channel is the base beam scaled and shifted with that
channel's parameters (JAX `map_coordinates` / `vmap`).

<<<<<<< HEAD
## Data

The package reads its data from `src/gambit/data/`, which is shipped with it:

| File | Content |
|---|---|
| `beam_mean.npy` | Stokes I base beam, 128 × 128 (the native resolution of all files) |
| `transformations.npy` | `[scale_mean, scale_std, shift_l_mean, shift_l_std, shift_m_mean, shift_m_std]`, shape `(6, 900)`, one column per channel |
| `zernike_coeffs.npz` | Zernike coefficients of the base beam, Noll order (key `coeffs`, or the only array in the file) |
=======
>>>>>>> 96509a75b8abf3d4bb0bd729290e523a9f5301f3

## Install

```bash
pip install git+https://github.com/MortezaPasha/GAMBIT-beam-model.git
```

Dependencies: `numpy`, `jax`

## Usage

```python
from gambit import MeerkatModel, FREQS

model = MeerkatModel()

# The base beam: no frequency
base_mean = model.base_beam(source="mean")                       # 1) original base beam, (128, 128)
base_zern = model.base_beam(source="zernike", resolution=1024)   # 2) Zernike base beam, (1024, 1024)

# Beam at a channel = base beam zoomed and shifted with that channel's mean parameters
b1 = model.beam(freqs=1284.0, source="mean")
# nearest channel to 1284 MHz, for accurate results, use the exact frequency channel

b2 = model.beam(freqs=1284.0, source="zernike", resolution=1024)
some = model.beam(channels=[0, 100, 899])                        # by channel index: (3, 128, 128)
cube = model.beam(source="zernike", resolution=256)              # all 900 channels: (900, 256, 256)

# The scale and shifts themselves, at given frequencies or channels
scale, shift_l, shift_m = model.params(freqs=[900.0, 1284.0])            # means
(s_mean, s_std), (sl_mean, sl_std), (sm_mean, sm_std) = model.params(channels=range(10), std=True)

# Beams at given channels with your own parameters (scalar or one value per channel)
own = model.beam(freqs=[900.0, 1284.0], scale=[1.01, 1.03], shift_l=0.2, shift_m=-0.1)

# Your own parameters, no channel involved
custom = model.transform(scale=1.02, shift_l=0.3, shift_m=-0.2, source="zernike", resolution=512)
```

### High resolution (e.g. for imaging)

Building a Zernike base beam at high resolution is the slow step, and a full cube
is large (e.g. 900 × 1024² float64 ≈ 7.5 GB). So:

- The high-res base beam is **built once and saved** to `~/.cache/gambit/`
  (or `$GAMBIT_CACHE`). Later calls, also in new sessions, just load it.
- `iter_beams` then applies **only the geometric transform** per channel, one
  channel at a time, so memory stays at about two images.

```python
for ch, beam in model.iter_beams(freqs=[900.0, 1284.0, 1600.0], resolution=8192):
    ...   # use the (8192, 8192) beam for this channel, e.g. in imaging
```

<<<<<<< HEAD
The base beam is evaluated with the modes grouped by azimuthal order m, the
stable Jacobi recurrence for the radial polynomials, and a JAX-compiled sum.
On a 4-core test machine (CPU) with 800 modes, building it took 4.5 s at
2048 × 2048 and 18 s at 4096 × 4096 (the direct formula took 48 s at 2048); a
saved one loads in about 0.01 s, and each channel then takes about 0.1 s at 2048.
A GPU build of JAX speeds the first build up further.
=======
On a test machine (CPU 32.0 GiB) , building a 8k × 8k base beam took less than one minute the first time and
0.01 s to load afterwards; each channel then took about 0.1 s.
A GPU build of JAX will be faster.
>>>>>>> 96509a75b8abf3d4bb0bd729290e523a9f5301f3


<<<<<<< HEAD
- **Transform.** Output pixel `(m, l)` (row, column) is read from input position
  `center + scale * ((m, l) - center) + (shift_m, shift_l)`, with
  `center = ((ny-1)/2, (nx-1)/2)`, linear interpolation, and zeros outside.
- **Native grid.** All files are defined on the 128 × 128 grid of `beam_mean`
  (`gambit.NATIVE_SIZE`); `resolution=None` means 128.
- **Other resolutions.** `scale` is used as is. Shifts are in native (128-grid)
  pixels and are multiplied by `resolution / 128`.
- **Zernike grid.** At the native size `N` the grid is the one used by
  `ZernikeDecomposer.unit_disk`. A grid of `N'` pixels covers the same field of
  view. The beam is zero outside the unit disk.
- **Zernike accuracy.** `zernike_image` (fast, used by the model) agrees with the
  direct factorial formula `zernike_image_direct` (which equals
  `ZernikeDecomposer.reconstruct` exactly) to about 3e-8 of the peak with 800
  modes. Checked against 50-digit arithmetic, the fast version is the more accurate
  of the two: the factorial sums lose digits near the disk edge at high order.
- `source="mean"` is only available at the native resolution.

## Publishing on GitHub

```bash
cd gambit
# put beam_mean.npy, transformations.npy and zernike_coeffs.npz in src/gambit/data/
pytest                                   # optional check
git init -b main
git add .
git commit -m "MeerkatModel: MeerKAT array-average beam model"
# repository: https://github.com/MortezaPasha/GAMBIT-beam-model
git remote add origin https://github.com/MortezaPasha/GAMBIT-beam-model.git
git push -u origin main
git tag v0.1.0 && git push origin v0.1.0  # optional, for pinned installs
```

Users then install with `pip install git+https://github.com/MortezaPasha/GAMBIT-beam-model.git`
(or `...gambit.git@v0.1.0`), and the data files come with the package.
GitHub refuses files over 100 MB; if a data file is larger, use Git LFS or a
release asset instead.
=======
>>>>>>> 96509a75b8abf3d4bb0bd729290e523a9f5301f3
