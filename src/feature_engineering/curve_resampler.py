import numpy as np


def resample_curve(
    phase,
    flux,
    n_points=2000
):

    phase_grid = np.linspace(
        0,
        1,
        n_points
    )

    flux_interp = np.interp(
        phase_grid,
        phase,
        flux
    )

    return flux_interp