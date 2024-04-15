import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate

def calculate_snr_532nm(raman_shift, intensities):
    """
    Calculates the Signal-to-Noise Ratio (SNR) for Raman spectra using a 532nm laser source.

    Parameters:
        raman_shift (np.array): Array of Raman shift values.
        intensities (np.array): Corresponding intensity values.

    Returns:
        float: Calculated SNR value.
    """
    signal_range = (2600, 3000)
    noise_range = (3000, 3400)
    signal = intensities[(raman_shift >= signal_range[0]) & (raman_shift <= signal_range[1])].mean()
    noise = intensities[(raman_shift >= noise_range[0]) & (raman_shift <= noise_range[1])].std()
    return signal / noise if noise != 0 else float('inf')

def calculate_snr_785nm(raman_shift, intensities):
    """
    Calculates the Signal-to-Noise Ratio (SNR) for Raman spectra using a 785nm laser source,
    taking the overall standard deviation of noise ranges combined.

    Parameters:
        raman_shift (np.array): Array of Raman shift values.
        intensities (np.array): Corresponding intensity values.

    Returns:
        float: Calculated SNR value.
    """
    signal_range = (750, 1500)
    noise_ranges = [(600, 750), (1500, 1800)]
    signal = intensities[(raman_shift >= signal_range[0]) & (raman_shift <= signal_range[1])].mean()

    # Combine all noise ranges and calculate the standard deviation once
    noise_data = []
    for nr in noise_ranges:
        mask = (raman_shift >= nr[0]) & (raman_shift <= nr[1])
        noise_data.extend(intensities[mask])
    
    if not noise_data:  # Check if noise data is empty
        return float('inf')  # Return infinity if no noise data

    noise_data = np.array(noise_data)  # Convert list to numpy array for operation
    noise_std = np.std(noise_data)  # Standard deviation of combined noise data

    return signal / noise_std if noise_std != 0 else float('inf')


def identify_outliers(group):
    """
    Identifies outliers based on the Signal-to-Noise Ratio (SNR) from a grouped DataFrame.

    Parameters:
        group (DataFrame): Grouped data containing SNR values.

    Returns:
        DataFrame: DataFrame containing only outliers.
    """
    mean_snr = group['SNR'].mean()
    std_snr = group['SNR'].std()
    lower_bound = mean_snr - 2 * std_snr
    upper_bound = mean_snr + 2 * std_snr
    return group[(group['SNR'] < lower_bound) | (group['SNR'] > upper_bound)]


def interpolate(raman_shift, intensities, interval, start=None, end=None):
    """
    Performs PCHIP interpolation on Raman spectral data at specified intervals.

    Parameters:
        raman_shift (np.array): Raman shift values.
        intensities (np.array): Intensity values corresponding to Raman shifts.
        interval (int): Interval for the new Raman shift values.
        start (float, optional): Start of the Raman shift range for interpolation.
        end (float, optional): End of the Raman shift range for interpolation.

    Returns:
        new_raman_shift (np.array): Interpolated Raman shift values.
        interpolated_intensities (np.array): Interpolated intensity values.
    """
    if start is None:
        start = np.min(raman_shift)
    if end is None:
        end = np.max(raman_shift)

    new_raman_shift = np.arange(start, end + interval, interval)
    interpolated_intensities = pchip_interpolate(raman_shift, intensities, new_raman_shift)

    return new_raman_shift, interpolated_intensities

def normalize(data):
    """
    Normalizes a numpy array to the range [0, 1].

    Parameters:
        data (np.array): Data array to be normalized.

    Returns:
        np.array: Normalized data.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data
