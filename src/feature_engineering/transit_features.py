# from transitleastsquares import transitleastsquares
# import numpy as np


# def extract_transit_features(
#     time,
#     flux
# ):

#     # Downsample for faster TLS
#     step = 5

#     time_ds = time[::step]
#     flux_ds = flux[::step]

#     model = transitleastsquares(
#         time_ds,
#         flux_ds
#     )

#     results = model.power(
#         period_min=0.5,
#         period_max=50.0,
#         use_threads=4
#     )

#     return {

#         "period":
#         float(results.period),

#         "duration":
#         float(results.duration),

#         "depth":
#         float(results.depth)
#     }

from transitleastsquares import transitleastsquares
import numpy as np


def extract_transit_features(
    time,
    flux
):

    # aggressive downsampling
    step = 20

    time_ds = time[::step]
    flux_ds = flux[::step]

    model = transitleastsquares(
        time_ds,
        flux_ds
    )

    results = model.power(
        period_min=1,
        period_max=10,
        oversampling_factor=1
    )

    return {

        "period":
        float(results.period),

        "duration":
        float(results.duration),

        "depth":
        float(results.depth)
    }