import numpy as np
import pytest

import gambit.meerkat
from gambit import FREQS, MeerkatModel, ZernikeDecomposer, zernike_image, zoom_and_shift_one_array

N = 48
NF = len(FREQS)


def make_beam(n=N):
    y, x = np.indices((n, n), float)
    c = n / 2
    r = np.hypot(y - c, (x - c) * 1.15) / (n / 8)
    return np.exp(-r**2 / 2) + 0.05 * np.cos(3 * np.arctan2(y - c, x - c)) * np.exp(-(r - 3)**2)


@pytest.fixture(scope="module")
def beam():
    return make_beam()


@pytest.fixture(scope="module")
def coeffs(beam):
    return ZernikeDecomposer(Nmodes=200).decompose(beam)


def transformations():
    return np.stack([np.linspace(0.95, 1.05, NF), np.full(NF, .01),
                     np.linspace(-0.5, 0.5, NF), np.full(NF, .1),
                     np.full(NF, 0.3), np.full(NF, .1)])


@pytest.fixture
def model(tmp_path, monkeypatch, beam, coeffs):
    np.save(tmp_path / "beam_mean.npy", beam)
    np.save(tmp_path / "transformations.npy", transformations())
    np.savez(tmp_path / "zernike_coeffs.npz", coeffs=coeffs)
    monkeypatch.setattr(gambit.meerkat, "DATA_DIR", tmp_path)
    return MeerkatModel()


def test_freqs():
    assert NF == 900 and FREQS[0] == 856 and np.isclose(FREQS[-1], 856 + 899 * 0.8359375)


def test_zernike_image_matches_reconstruct(beam):
    zd = ZernikeDecomposer(Nmodes=200)
    co = zd.decompose(beam)
    np.testing.assert_array_equal(zernike_image(co, N), zd.reconstruct(co))


def test_zernike_fit_is_close(beam, coeffs):
    mask = np.hypot(*(np.indices((N, N)) - N / 2)) <= N / 2
    assert np.abs(zernike_image(coeffs, N) - beam)[mask].max() < 1e-2 * beam.max()


def test_upsampled_grid_contains_native_pixels(coeffs):
    # With an odd factor, every third output pixel centre is a native pixel centre.
    hi = zernike_image(coeffs, N, 3 * N)
    np.testing.assert_allclose(hi[1::3, 1::3], zernike_image(coeffs, N), atol=1e-10)


def test_identity_and_integer_shift(beam):
    np.testing.assert_allclose(zoom_and_shift_one_array(beam, 1.0, 0.0, 0.0), beam, atol=1e-12)
    out = np.asarray(zoom_and_shift_one_array(beam, 1.0, 2.0, -1.0))
    # output(y, x) = input(y + shift_y, x + shift_x)
    np.testing.assert_allclose(out[1:, :-2], beam[:-1, 2:], atol=1e-12)


def test_channel_selection(model):
    assert model.beam().shape == (NF, N, N)
    assert model.beam(channels=10).shape == (N, N)
    assert model.beam(channels=[0, 5, 899], source="zernike", resolution=64).shape == (3, 64, 64)
    assert model.channels(freqs=1000.0) == np.argmin(np.abs(FREQS - 1000.0))
    np.testing.assert_array_equal(model.beam(freqs=FREQS[123]), model.beam(channels=123))


def test_channel_parameters_used(model):
    t = transformations()
    expected = model.transform(t[0, 7], t[2, 7], t[4, 7])
    np.testing.assert_allclose(model.beam(channels=7), expected)


def test_mean_and_zernike_paths_agree(model):
    a = np.asarray(model.beam(channels=450, source="mean"))
    b = np.asarray(model.beam(channels=450, source="zernike"))
    assert np.abs(a - b).max() < 2e-2 * a.max()


def test_resolution_consistency(model):
    lo = np.asarray(model.beam(channels=0, source="zernike"))
    hi = np.asarray(model.beam(channels=0, source="zernike", resolution=3 * N))
    assert np.abs(hi[1::3, 1::3] - lo).max() < 2e-2 * lo.max()


def test_mean_source_only_native(model):
    with pytest.raises(ValueError):
        model.beam(channels=0, source="mean", resolution=2 * N)


def test_sample(model):
    beams, (f, sx, sy) = model.sample(channels=np.arange(7), seed=1, source="zernike", resolution=64)
    assert beams.shape == (7, 64, 64) and f.shape == (7,)


def test_flattened_beam_is_reshaped(tmp_path, monkeypatch, beam, coeffs):
    np.save(tmp_path / "beam_mean.npy", beam.ravel()[None, :])
    np.save(tmp_path / "transformations.npy", transformations())
    np.savez(tmp_path / "zernike_coeffs.npz", coeffs=coeffs)
    monkeypatch.setattr(gambit.meerkat, "DATA_DIR", tmp_path)
    m = MeerkatModel()
    np.testing.assert_array_equal(m.beam_mean, beam)
    assert m.beam(freqs=1284.0).shape == (N, N)
