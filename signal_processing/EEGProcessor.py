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
    
    def detect_emotional_state(self, data, fs=256, channel_names=['TP9', 'AF7', 'AF8', 'TP10']):
        """
        Detect emotional state using Frontal Alpha Asymmetry (FAA) and band power analysis.
        
        This method uses established neuroscience principles:
        - Frontal Alpha Asymmetry: Left frontal activation (lower alpha) = positive emotions
                                   Right frontal activation (lower alpha) = negative emotions
        - Beta band power: Higher beta = stress/anxiety (negative)
        - Theta/Alpha ratio: Higher ratio = relaxation/meditation (positive)
        
        Muse2 electrode positions:
        - AF7 (left frontal)
        - AF8 (right frontal)
        - TP9 (left temporal)
        - TP10 (right temporal)
        
        Parameters
        ----------
        data : array_like
            EEG data, shape: (samples, channels)
            Channels should be ordered as: [TP9, AF7, AF8, TP10]
        fs : float, optional
            Sampling frequency in Hz (default is 256)
        channel_names : list, optional
            List of channel names in order (default is ['TP9', 'AF7', 'AF8', 'TP10'])
        
        Returns
        -------
        emotion : str
            Detected emotional state: 'POSITIVE', 'NEUTRAL', or 'NEGATIVE'
        confidence : float
            Confidence score (0-1) indicating the strength of the emotion
        metrics : dict
            Dictionary containing the computed metrics used for classification:
            - 'faa_score': Frontal Alpha Asymmetry score
            - 'beta_power': Average beta band power
            - 'theta_alpha_ratio': Theta/Alpha ratio
            - 'alpha_left': Alpha power in left frontal (AF7)
            - 'alpha_right': Alpha power in right frontal (AF8)
        """
        # Ensure we have 4 channels
        if data.shape[1] != 4:
            raise ValueError(f"Expected 4 channels, got {data.shape[1]}")
        
        # Apply preprocessing
        filtered_data = self.apply_filtering(data, fs=fs)
        
        # Compute PSD
        psd, freqs = self.convert_to_psd(filtered_data, fs=fs)
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Extract band powers for all channels
        band_powers = self.extract_bands(psd, freqs, bands)
        
        # Channel indices (assuming order: TP9, AF7, AF8, TP10)
        idx_tp9 = 0   # Left temporal
        idx_af7 = 1   # Left frontal
        idx_af8 = 2   # Right frontal
        idx_tp10 = 3  # Right temporal
        
        # Extract alpha power from frontal channels
        alpha_left = band_powers['alpha'][idx_af7]   # AF7 (left frontal)
        alpha_right = band_powers['alpha'][idx_af8]  # AF8 (right frontal)
        
        # Compute Frontal Alpha Asymmetry (FAA)
        # FAA = ln(right alpha) - ln(left alpha)
        # Positive FAA = greater left activation = positive emotions
        # Negative FAA = greater right activation = negative emotions
        alpha_left_safe = max(alpha_left, 1e-10)  # Avoid log(0)
        alpha_right_safe = max(alpha_right, 1e-10)
        faa_score = np.log(alpha_right_safe) - np.log(alpha_left_safe)
        
        # Compute average beta power (across frontal channels)
        # Higher beta = stress, anxiety, negative emotions
        beta_frontal = (band_powers['beta'][idx_af7] + band_powers['beta'][idx_af8]) / 2
        
        # Compute theta/alpha ratio (relaxation indicator)
        # Higher ratio = more relaxed/meditative state
        theta_total = np.mean(band_powers['theta'])
        alpha_total = np.mean(band_powers['alpha'])
        theta_alpha_ratio = theta_total / max(alpha_total, 1e-10)
        
        # Normalize beta power (typical range: 0-100 µV²)
        beta_normalized = np.tanh(beta_frontal / 50.0)  # Sigmoid-like normalization
        
        # Decision thresholds (empirically determined)
        FAA_POSITIVE_THRESHOLD = 0.15   # Strong left activation
        FAA_NEGATIVE_THRESHOLD = -0.15  # Strong right activation
        BETA_HIGH_THRESHOLD = 0.5       # High stress/anxiety
        THETA_ALPHA_RELAXED = 1.2       # Relaxed state indicator
        
        # Classification logic
        metrics = {
            'faa_score': float(faa_score),
            'beta_power': float(beta_frontal),
            'theta_alpha_ratio': float(theta_alpha_ratio),
            'alpha_left': float(alpha_left),
            'alpha_right': float(alpha_right),
            'beta_normalized': float(beta_normalized)
        }
        
        # Score accumulation for confidence
        positive_score = 0
        negative_score = 0
        
        # FAA analysis (strongest indicator)
        if faa_score > FAA_POSITIVE_THRESHOLD:
            positive_score += 2.0
        elif faa_score < FAA_NEGATIVE_THRESHOLD:
            negative_score += 2.0
        else:
            # Mild asymmetry still contributes
            if faa_score > 0:
                positive_score += abs(faa_score) * 5
            else:
                negative_score += abs(faa_score) * 5
        
        # Beta power analysis
        if beta_normalized > BETA_HIGH_THRESHOLD:
            negative_score += 1.5  # High beta suggests stress/anxiety
        else:
            positive_score += 0.5  # Low beta suggests calm
        
        # Theta/Alpha ratio analysis
        if theta_alpha_ratio > THETA_ALPHA_RELAXED:
            positive_score += 1.0  # Relaxed/meditative state
        elif theta_alpha_ratio < 0.8:
            negative_score += 0.5  # Alert/tense state
        
        # Determine emotion based on scores
        total_score = positive_score + negative_score
        
        if total_score == 0:
            emotion = 'NEUTRAL'
            confidence = 0.5
        else:
            score_diff = abs(positive_score - negative_score)
            confidence = min(score_diff / (total_score + 1e-10), 1.0)
            
            if positive_score > negative_score:
                if score_diff > 1.0:
                    emotion = 'POSITIVE'
                else:
                    emotion = 'NEUTRAL'
                    confidence = max(0.3, 1.0 - confidence)  # Low confidence for near-neutral
            elif negative_score > positive_score:
                if score_diff > 1.0:
                    emotion = 'NEGATIVE'
                else:
                    emotion = 'NEUTRAL'
                    confidence = max(0.3, 1.0 - confidence)
            else:
                emotion = 'NEUTRAL'
                confidence = 0.5
        
        return emotion, confidence, metrics