# gambit

MeerKAT antenna-array average beam model.

There is **one base beam**, 128 × 128, with no frequency axis. It comes either

1. **from the original base beam posterior mean** (`source="mean"`, 128 × 128 only), or
2. **from the Zernike expansion of the base beam** (`source="zernike"`, any resolution).

Frequency enters only through the transformations: for each of the 900 L-band
channels (`FREQS = np.arange(856, 1712, 0.8359375)[:900]`, MHz),
`transformations.npy` holds the mean and std of the zoom factor and the x/y
shifts. The beam at a channel is the base beam zoomed and shifted with that
channel's parameters (JAX `map_coordinates` / `vmap`).


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
b1 = model.beam(freqs=1284.0, source="mean")                     # nearest channel to 1284 MHz
b2 = model.beam(freqs=1284.0, source="zernike", resolution=1024)
some = model.beam(channels=[0, 100, 899])                        # by channel index: (3, 128, 128)
cube = model.beam(source="zernike", resolution=256)              # all 900 channels: (900, 256, 256)

# One random beam per channel, parameters drawn from the stored means / stds
beams, (factor, shift_x, shift_y) = model.sample(channels=range(10), seed=0)

# Your own parameters, no channel involved
custom = model.transform(factor=1.02, shift_x=0.3, shift_y=-0.2, source="zernike", resolution=512)
```

### High resolution (e.g. for imaging)

Building a Zernike base beam at high resolution is the slow step, and a full cube
is large (900 × 1024² float64 ≈ 7.5 GB). So:

- The high-res base beam is **built once and saved** to `~/.cache/gambit/`
  (or `$GAMBIT_CACHE`). Later calls, also in new sessions, just load it.
- `iter_beams` then applies **only the geometric transform** per channel, one
  channel at a time, so memory stays at about two images.

```python
for ch, beam in model.iter_beams(freqs=[900.0, 1284.0, 1600.0], resolution=4096):
    ...   # use the (4096, 4096) beam for this channel, e.g. in imaging
```

On a test machine, building a 2048 × 2048 base beam took 48 s the first time and
0.01 s to load afterwards; each channel then took about 0.1 s.
