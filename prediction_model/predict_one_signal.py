"""
Load the model using joblib and make predictions on a single EEG signal.
"""

import joblib
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import training module
sys.path.append(str(Path(__file__).parent.parent))
from prediction_model.training import EEGEmotionTrainer

def generate_test_signals():
    """
    Generate synthetic EEG signals for testing.
    Returns raw EEG data in the format: (channels, samples)
    """
    fs = 256               # Muse2 sampling rate
    duration = 10          # seconds
    t = np.arange(0, duration, 1/fs)

    def alpha_wave(t, freq=10, amp=1.0):
        return amp * np.sin(2*np.pi*freq*t)

    def beta_wave(t, freq=20, amp=0.3):
        return amp * np.sin(2*np.pi*freq*t)

    def noise(t, amp=0.2):
        return amp * np.random.randn(len(t))

    # ---------- 1) POSITIVE ----------
    # left frontal (AF7): lower alpha (more active), add a bit of beta
    AF7_pos  = alpha_wave(t, amp=0.5) + beta_wave(t, amp=0.25) + noise(t, 0.1)
    # right frontal (AF8): higher alpha (less active)
    AF8_pos  = alpha_wave(t, amp=1.1) + noise(t, 0.1)
    # temporals fairly neutral
    TP9_pos  = alpha_wave(t, amp=0.8) + noise(t, 0.1)
    TP10_pos = alpha_wave(t, amp=0.8) + noise(t, 0.1)

    positive = np.vstack([TP9_pos, AF7_pos, AF8_pos, TP10_pos])  # shape: (4, N)

    # ---------- 2) NEUTRAL ----------
    AF7_neu  = alpha_wave(t, amp=0.9) + noise(t, 0.1)
    AF8_neu  = alpha_wave(t, amp=0.9) + noise(t, 0.1)
    TP9_neu  = alpha_wave(t, amp=0.9) + noise(t, 0.1)
    TP10_neu = alpha_wave(t, amp=0.9) + noise(t, 0.1)

    neutral = np.vstack([TP9_neu, AF7_neu, AF8_neu, TP10_neu])

    # ---------- 3) NEGATIVE ----------
    # right frontal (AF8): lower alpha → more active
    AF8_neg  = alpha_wave(t, amp=0.5) + beta_wave(t, amp=0.2) + noise(t, 0.12)
    # left frontal (AF7): higher alpha → less active
    AF7_neg  = alpha_wave(t, amp=1.1) + noise(t, 0.12)
    TP9_neg  = alpha_wave(t, amp=0.8) + noise(t, 0.12)
    TP10_neg = alpha_wave(t, amp=0.8) + noise(t, 0.12)

    negative = np.vstack([TP9_neg, AF7_neg, AF8_neg, TP10_neg])

    return positive, neutral, negative

def predict_one_signal(model_path: str, eeg_signal: np.ndarray, label_encoder_path: str = None) -> tuple:
    """
    Load the trained model and predict the emotion class for a raw EEG signal.

    Parameters
    ----------
    model_path : str
        Path to the saved model file (pipeline).
    eeg_signal : np.ndarray
        Raw EEG signal with shape (channels, samples) or (samples, channels).
        For Muse2: 4 channels, typically 2560 samples for 10 seconds at 256 Hz.
    label_encoder_path : str, optional
        Path to the saved label encoder. If provided, returns the class name.

    Returns
    -------
    tuple
        (predicted_class_index, predicted_class_name) if label_encoder is provided,
        otherwise (predicted_class_index, None)
    """
    # Load the trained model 
    pipeline = joblib.load(model_path)
    
    # Load label encoder if provided
    label_encoder = None
    if label_encoder_path:
        label_encoder = joblib.load(label_encoder_path)
    
    # Create a trainer instance to use feature extraction
    trainer = EEGEmotionTrainer()
    trainer.model = pipeline
    
    # Ensure the signal is in the correct shape (samples, channels)
    signal = np.asarray(eeg_signal)
    if signal.shape[0] == 4:  # If shape is (channels, samples)
        signal = signal.T  # Transpose to (samples, channels)
    
    # Use the trainer's predict_emotion method which handles feature extraction
    prediction = trainer.predict_emotion(signal, fs=256)
    predicted_class = int(prediction[0])
    
    # Get class name if label encoder is available
    class_name = None
    if label_encoder is not None:
        class_name = label_encoder.inverse_transform([predicted_class])[0]
    
    return predicted_class, class_name

if __name__ == "__main__":
    print("=" * 60)
    print("EEG Emotion Prediction Test")
    print("=" * 60)
    
    model_file = "/Users/ishaanratanshi/Muse-ic/prediction_model/saved_models/eeg_emotion_classifier.pkl"
    label_encoder_file = "/Users/ishaanratanshi/Muse-ic/prediction_model/saved_models/label_encoder.pkl"
    
    # Generate test signals
    positive, neutral, negative = generate_test_signals()
    
    # Test positive signal
    print("\nTesting POSITIVE signal...")
    pred_class, pred_name = predict_one_signal(model_file, positive, label_encoder_file)
    print(f"  Predicted class index: {pred_class}")
    print(f"  Predicted class name: {pred_name}")

    # Test neutral signal
    print("\nTesting NEUTRAL signal...")
    pred_class, pred_name = predict_one_signal(model_file, neutral, label_encoder_file)
    print(f"  Predicted class index: {pred_class}")
    print(f"  Predicted class name: {pred_name}")

    # Test negative signal
    print("\nTesting NEGATIVE signal...")
    pred_class, pred_name = predict_one_signal(model_file, negative, label_encoder_file)
    print(f"  Predicted class index: {pred_class}")
    print(f"  Predicted class name: {pred_name}")
    
    print("\n" + "=" * 60)
    print("Prediction Complete!")
    print("=" * 60)
