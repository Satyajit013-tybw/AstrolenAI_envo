from .wavelet import wavelet_denoise
from .flatten import flatten_curve
import numpy as np


def preprocess(lc):

    lc = lc.remove_nans()

    flux = np.array(
        lc.flux.value
    )

    denoised_flux = wavelet_denoise(
        flux
    )

    lc_denoised = lc.copy()

    lc_denoised.flux = denoised_flux

    flat_lc = flatten_curve(
        lc_denoised
    )

    return flat_lc