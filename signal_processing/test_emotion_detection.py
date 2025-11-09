"""
Test script for the EEGProcessor emotional state detection method.
Generates synthetic EEG data with different characteristics and tests the detection algorithm.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_processing.EEGProcessor import EEGProcessor


def generate_test_eeg(emotion_type='neutral', duration_sec=4.0, fs=256):
    """
    Generate synthetic EEG data simulating different emotional states.
    
    Parameters:
    -----------
    emotion_type : str
        'positive', 'neutral', or 'negative'
    duration_sec : float
        Duration in seconds
    fs : int
        Sampling frequency
    
    Returns:
    --------
    np.ndarray : EEG data (samples, 4 channels: TP9, AF7, AF8, TP10)
    """
    num_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, num_samples)
    
    # Initialize channels: [TP9, AF7, AF8, TP10]
    eeg_data = np.zeros((num_samples, 4))
    
    if emotion_type == 'positive':
        # POSITIVE: Higher left frontal activity (AF7), lower alpha on left
        # More theta (relaxation), moderate beta
        
        for ch in range(4):
            delta = 10e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 20e-6 * np.sin(2 * np.pi * 6.0 * t)  # Higher theta
            
            if ch == 1:  # AF7 (left frontal)
                alpha = 15e-6 * np.sin(2 * np.pi * 10.0 * t)  # Lower alpha (more activation)
                beta = 18e-6 * np.sin(2 * np.pi * 20.0 * t)
            elif ch == 2:  # AF8 (right frontal)
                alpha = 30e-6 * np.sin(2 * np.pi * 10.0 * t)  # Higher alpha (less activation)
                beta = 15e-6 * np.sin(2 * np.pi * 20.0 * t)
            else:  # Temporal channels
                alpha = 25e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 16e-6 * np.sin(2 * np.pi * 20.0 * t)
            
            noise = 5e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
            
    elif emotion_type == 'negative':
        # NEGATIVE: Higher right frontal activity (AF8), lower alpha on right
        # Higher beta (stress/anxiety), lower theta
        
        for ch in range(4):
            delta = 10e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 12e-6 * np.sin(2 * np.pi * 6.0 * t)  # Lower theta
            
            if ch == 1:  # AF7 (left frontal)
                alpha = 30e-6 * np.sin(2 * np.pi * 10.0 * t)  # Higher alpha (less activation)
                beta = 18e-6 * np.sin(2 * np.pi * 22.0 * t)
            elif ch == 2:  # AF8 (right frontal)
                alpha = 15e-6 * np.sin(2 * np.pi * 10.0 * t)  # Lower alpha (more activation)
                beta = 35e-6 * np.sin(2 * np.pi * 22.0 * t)  # Higher beta (stress)
            else:  # Temporal channels
                alpha = 25e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 25e-6 * np.sin(2 * np.pi * 22.0 * t)
            
            noise = 5e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
            
    else:  # neutral
        # NEUTRAL: Balanced activity across both hemispheres
        # Moderate alpha, beta, and theta
        
        for ch in range(4):
            delta = 10e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 16e-6 * np.sin(2 * np.pi * 6.0 * t)
            alpha = 25e-6 * np.sin(2 * np.pi * 10.0 * t)  # Balanced alpha
            beta = 20e-6 * np.sin(2 * np.pi * 20.0 * t)   # Moderate beta
            
            noise = 5e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
    
    return eeg_data


def main():
    print("=" * 70)
    print("   EEG Emotional State Detection Test")
    print("=" * 70)
    print("\nTesting deterministic emotion detection using:")
    print("  - Frontal Alpha Asymmetry (FAA)")
    print("  - Beta band power (stress indicator)")
    print("  - Theta/Alpha ratio (relaxation indicator)")
    print("\nMuse2 Channels: TP9 (left temporal), AF7 (left frontal),")
    print("                AF8 (right frontal), TP10 (right temporal)")
    print("=" * 70)
    
    # Initialize processor
    processor = EEGProcessor()
    
    # Test each emotional state
    emotion_types = ['positive', 'neutral', 'negative']
    num_tests_per_type = 5
    
    results = {etype: {'correct': 0, 'total': 0, 'confidences': []} 
               for etype in emotion_types}
    
    for emotion_type in emotion_types:
        print(f"\n{'-' * 70}")
        print(f"Testing {emotion_type.upper()} emotional state:")
        print(f"{'-' * 70}")
        
        for test_num in range(1, num_tests_per_type + 1):
            # Generate test data
            eeg_data = generate_test_eeg(emotion_type, duration_sec=4.0, fs=256)
            
            # Detect emotion
            detected_emotion, confidence, metrics = processor.detect_emotional_state(eeg_data, fs=256)
            
            # Check if correct
            expected = emotion_type.upper()
            correct = detected_emotion == expected
            
            results[emotion_type]['total'] += 1
            if correct:
                results[emotion_type]['correct'] += 1
            results[emotion_type]['confidences'].append(confidence)
            
            status = "✓" if correct else "✗"
            
            print(f"\nTest {test_num}:")
            print(f"  Expected:  {expected}")
            print(f"  Detected:  {detected_emotion} {status}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Metrics:")
            print(f"    - FAA Score: {metrics['faa_score']:.4f}")
            print(f"    - Beta Power: {metrics['beta_power']:.2e}")
            print(f"    - Theta/Alpha Ratio: {metrics['theta_alpha_ratio']:.3f}")
            print(f"    - Alpha Left (AF7): {metrics['alpha_left']:.2e}")
            print(f"    - Alpha Right (AF8): {metrics['alpha_right']:.2e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)
    
    total_correct = 0
    total_tests = 0
    
    for emotion_type in emotion_types:
        correct = results[emotion_type]['correct']
        total = results[emotion_type]['total']
        accuracy = (correct / total * 100) if total > 0 else 0
        avg_confidence = np.mean(results[emotion_type]['confidences'])
        
        total_correct += correct
        total_tests += total
        
        print(f"\n{emotion_type.upper()}:")
        print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"  Avg Confidence: {avg_confidence:.3f}")
    
    overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    print(f"\n{'-' * 70}")
    print(f"Overall Accuracy: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")
    print("=" * 70)
    
    print("\nNote: This algorithm uses established neuroscience principles:")
    print("  • FAA (Frontal Alpha Asymmetry) for approach/withdrawal motivation")
    print("  • Beta power for stress/anxiety detection")
    print("  • Theta/Alpha ratio for relaxation state assessment")
    print("=" * 70)


if __name__ == "__main__":
    main()
