import pywt
import numpy as np


def wavelet_denoise(signal):

    coeffs = pywt.wavedec(
        signal,
        wavelet="db4",
        level=4
    )

    sigma = np.median(
        np.abs(coeffs[-1])
    ) / 0.6745

    threshold = sigma * np.sqrt(
        2 * np.log(len(signal))
    )

    coeffs[1:] = [
        pywt.threshold(
            c,
            threshold,
            mode="soft"
        )
        for c in coeffs[1:]
    ]

    reconstructed = pywt.waverec(
        coeffs,
        "db4"
    )

    return reconstructed[:len(signal)]