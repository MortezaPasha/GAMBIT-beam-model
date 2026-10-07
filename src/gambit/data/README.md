Data folder shipped with the package. `MeerkatModel()` reads:

- `beam_mean.npy`: original base beam (square 2D array)
- `transformations.npy`: [factor_mean, factor_std, x_shift_mean, x_shift_std, y_shift_mean, y_shift_std], shape (6, 900), one column per channel of `gambit.FREQS`
- `zernike_coeffs.npz`: Zernike coefficients of the base beam, Noll order (key `coeffs`, or the only array in the file)
