import numpy as np


def fold_curve(
    time,
    flux,
    period
):

    phase = (
        time % period
    ) / period

    idx = np.argsort(
        phase
    )

    return (
        phase[idx],
        flux[idx]
    )