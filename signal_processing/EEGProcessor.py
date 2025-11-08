"""
This module provides classes for preprocessing EEG data.
It includes filtering and artifact removal techniques.
This module is setup to work with real-time EEG data streams.
The hardware specified is the Muse2 EEG headband.
"""

import numpy as np
from collections import deque
from scipy.signal import welch, butter, filtfilt


class EEGProcessor:
    """
    A class for processing EEG data in real-time.
    
    Attributes
    ----------
    buffer_size : int
        Maximum number of samples to buffer for real-time processing
    num_channels : int
        Number of EEG channels
    buffer : deque
        Buffer for storing incoming EEG samples
    """
    
    def __init__(self, buffer_size=256, num_channels=4):
        """
        Initialize the EEG processor.
        
        Parameters
        ----------
        buffer_size : int, optional
            Number of samples to buffer for real-time processing (default is 256)
        num_channels : int, optional
            Number of EEG channels (default is 4)
        """
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.buffer = deque(maxlen=buffer_size)
    
    def bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """
        Design a Butterworth bandpass filter.

        Parameters
        ----------
        data : array_like
            EEG data, shape: (samples, channels)
        lowcut : float
            Low cutoff frequency in Hz
        highcut : float
            High cutoff frequency in Hz
        fs : float
            Sampling frequency in Hz
        order : int, optional
            Order of the filter (default is 4)

        Returns
        -------
        filtered_data : array_like
            Filtered EEG data, shape: (samples, channels)
        """
        b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data

    def apply_filtering(self, data, lowcut=0.5, highcut=40.0, fs=256):
        """
        Apply bandpass filtering to the EEG data.

        Parameters
        ----------
        data : array_like
            EEG data, shape: (samples, channels)
        lowcut : float, optional
            Low cutoff frequency in Hz (default is 0.5)
        highcut : float, optional
            High cutoff frequency in Hz (default is 40.0)
        fs : float, optional
            Sampling frequency in Hz (default is 256)

        Returns
        -------
        filtered_data : array_like
            Filtered EEG data, shape: (samples, channels)
        """
        notch_filtered_data = self.notch_filter(data, fs=fs)
        filtered_data = self.bandpass_filter(notch_filtered_data, lowcut, highcut, fs)
        return filtered_data
    
    def notch_filter(self, data, notch_freq=60.0, fs=256, quality_factor=30.0):
        """
        Apply a notch filter to remove powerline noise.

        Parameters
        ----------
        data : array_like
            EEG data, shape: (samples, channels)
        notch_freq : float, optional
            Notch frequency in Hz (default is 60.0)
        fs : float, optional
            Sampling frequency in Hz (default is 256)
        quality_factor : float, optional
            Quality factor for the notch filter (default is 30.0)

        Returns
        -------
        filtered_data : array_like
            Notch filtered EEG data, shape: (samples, channels)
        """
        b, a = butter(2, [(notch_freq - 1)/(fs/2), (notch_freq + 1)/(fs/2)], btype='bandstop')
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data
    
    def process_sample(self, new_sample):
        """
        Process a new EEG sample.

        Parameters
        ----------
        new_sample : array_like
            A single EEG sample, shape: (channels,)

        Returns
        -------
        filtered_sample : array_like or None
            The most recently filtered sample, or None if buffer is not full
        """
        self.buffer.append(new_sample)

        if len(self.buffer) >= 256:
            data = np.array(self.buffer)  # (N, channels)

            # Common Average Reference (CAR)
            data_car = data - np.mean(data, axis=1, keepdims=True)

            # Apply filtering
            data_filtered = self.apply_filtering(data_car)

            # Return most recent filtered sample
            return data_filtered[-1]

        return None

    
    def convert_to_psd(self, data, fs=256):
        """
        Convert EEG data to Power Spectral Density (PSD) using Welch's method.

        Parameters
        ----------
        data : array_like
            EEG data, shape: (samples, channels)
        fs : float, optional
            Sampling frequency in Hz (default is 256)

        Returns
        -------
        psd : array_like
            Power Spectral Density of the EEG data, shape: (frequencies, channels)
        freqs : array_like
            Frequencies corresponding to the PSD values, shape: (frequencies,)
        """
        psd_list = []
        for ch in range(data.shape[1]):
            freqs, psd_ch = welch(data[:, ch], fs=fs, nperseg=128)
            psd_list.append(psd_ch)
        psd = np.array(psd_list).T  # shape: (frequencies, channels)
        return psd, freqs

    def extract_bands(self, psd, freqs, bands={'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}):
        """
        Extract power in specific frequency bands from the PSD.

        Parameters
        ----------
        psd : array_like
            Power Spectral Density of the EEG data, shape: (frequencies, channels)
        freqs : array_like
            Frequencies corresponding to the PSD values, shape: (frequencies,)
        bands : dict, optional
            Dictionary defining frequency bands (default includes delta, theta, alpha, beta)

        Returns
        -------
        band_powers : dict
            Dictionary with band names as keys and corresponding power values as arrays of shape (channels,)
        """
        band_powers = {}
        for band, (fmin, fmax) in bands.items():
            idx_band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
            band_power = psd[idx_band, :].sum(axis=0)  # sum power in the band for each channel
            band_powers[band] = band_power
        return band_powers
    
    def reset_buffer(self):
        """
        Clear the buffer.
        """
        self.buffer.clear()