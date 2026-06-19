from .wavelet import wavelet_denoise
from .flatten import flatten_curve


def preprocess(lc):

    flux = lc.flux.value

    denoised_flux = wavelet_denoise(
        flux
    )

    lc.flux = denoised_flux

    flat_lc = flatten_curve(lc)

    return flat_lc