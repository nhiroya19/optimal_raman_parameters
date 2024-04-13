import numpy as np
from scipy.interpolate import pchip_interpolate

def interpolate(raman_shift, intensities, interval, start=None, end=None):
    """
    Performs PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation on 
    spectral intensity data at specified intervals.

    Parameters:
        raman_shift (np.array): Array of Raman shift values.
        intensities (np.array): Array of intensity values corresponding to Raman shifts.
        interval (int): Interval for the new Raman shift values.
        start (float, optional): Start of the Raman shift range for interpolation. 
                                 Defaults to the minimum of the original range if None.
        end (float, optional): End of the Raman shift range for interpolation. 
                               Defaults to the maximum of the original range if None.

    Returns:
        new_raman_shift (np.array): New Raman shift values.
        interpolated_results_np (np.array): Interpolated intensity values.
    """
    rounded_raman_shift = np.round(raman_shift)

    if start is None:
        start = rounded_raman_shift.min()
    if end is None:
        end = rounded_raman_shift.max()
    
    new_raman_shift = np.arange(start, end + interval, interval)

    if intensities.ndim == 1:
        interpolated_results_np = pchip_interpolate(
            rounded_raman_shift, intensities, new_raman_shift
        )
    else:
        interpolated_results = []
        for intensity_col_np in intensities.T:
            interpolated_col = pchip_interpolate(
                rounded_raman_shift, intensity_col_np, new_raman_shift
            )
            interpolated_results.append(interpolated_col)
        interpolated_results_np = np.array(interpolated_results).T

    return new_raman_shift, interpolated_results_np


def normalize(data):
    """
    Normalizes a numpy array to a range of [0, 1].

    Parameters:
        data (np.array): Input data array to be normalized.

    Returns:
        np.array: Normalized data array.
    """
    min_val = data.min()
    max_val = data.max()
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data
