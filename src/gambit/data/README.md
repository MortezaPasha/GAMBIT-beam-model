Data folder shipped with the package. `MeerkatModel()` reads:

- `beam_mean.npy`: original base beam (square 2D array)
- `transformations.npy`: [scale_mean, scale_std, shift_l_mean, shift_l_std, shift_m_mean, shift_m_std], shape (6, 900), one column per channel of `gambit.FREQS`
- `zernike_coeffs.npz`: Zernike coefficients of the base beam, Noll order (key `coeffs`, or the only array in the file)
