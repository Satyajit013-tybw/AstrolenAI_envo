import pywt
import numpy as np


def wavelet_denoise(signal):

    coeff = pywt.wavedec(
        signal,
        'db4',
        level=3
    )

    threshold = np.std(coeff[-1])

    coeff[1:] = [
        pywt.threshold(
            c,
            threshold,
            mode='soft'
        )
        for c in coeff[1:]
    ]

    reconstructed = pywt.waverec(
        coeff,
        'db4'
    )

    return reconstructed[:len(signal)]