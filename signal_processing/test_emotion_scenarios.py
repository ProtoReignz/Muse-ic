"""
Extended test script for emotion detection with various EEG scenarios.
Tests how different brain states get classified into POSITIVE, NEUTRAL, or NEGATIVE.
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_processing.EEGProcessor import EEGProcessor


def generate_eeg_scenario(scenario_type='balanced', duration_sec=4.0, fs=256):
    """
    Generate various EEG scenarios to test classification robustness.
    
    Scenarios:
    - 'happy': Strong left frontal activation (should be POSITIVE)
    - 'calm_relaxed': High theta, moderate alpha (should be POSITIVE)
    - 'focused': Moderate beta, balanced alpha (could be NEUTRAL/POSITIVE)
    - 'anxious': Very high beta, right activation (should be NEGATIVE)
    - 'sad': Right frontal activation, low theta (should be NEGATIVE)
    - 'stressed': High beta everywhere, low alpha (should be NEGATIVE)
    - 'drowsy': Very high theta, high alpha (could be POSITIVE/NEUTRAL)
    - 'balanced': Equal activity everywhere (should be NEUTRAL)
    - 'left_dominant': Strong left bias (should be POSITIVE)
    - 'right_dominant': Strong right bias (should be NEGATIVE)
    """
    num_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, num_samples)
    eeg_data = np.zeros((num_samples, 4))
    
    # Channel indices: [TP9, AF7, AF8, TP10]
    
    if scenario_type == 'happy':
        # Very strong left frontal activation (low alpha left, high alpha right)
        for ch in range(4):
            delta = 8e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 22e-6 * np.sin(2 * np.pi * 6.0 * t)
            
            if ch == 1:  # AF7 (left frontal) - very activated
                alpha = 10e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 16e-6 * np.sin(2 * np.pi * 18.0 * t)
            elif ch == 2:  # AF8 (right frontal) - less activated
                alpha = 35e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 12e-6 * np.sin(2 * np.pi * 18.0 * t)
            else:
                alpha = 22e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 14e-6 * np.sin(2 * np.pi * 18.0 * t)
            
            noise = 4e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
    
    elif scenario_type == 'calm_relaxed':
        # High theta/alpha ratio, balanced hemispheres
        for ch in range(4):
            delta = 10e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 30e-6 * np.sin(2 * np.pi * 6.0 * t)  # High theta
            alpha = 20e-6 * np.sin(2 * np.pi * 10.0 * t)  # Moderate alpha
            beta = 12e-6 * np.sin(2 * np.pi * 18.0 * t)   # Low beta
            
            noise = 3e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
    
    elif scenario_type == 'focused':
        # Moderate beta, slightly left-biased
        for ch in range(4):
            delta = 8e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 15e-6 * np.sin(2 * np.pi * 6.0 * t)
            
            if ch == 1:  # AF7
                alpha = 18e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 22e-6 * np.sin(2 * np.pi * 20.0 * t)
            elif ch == 2:  # AF8
                alpha = 22e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 20e-6 * np.sin(2 * np.pi * 20.0 * t)
            else:
                alpha = 20e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 21e-6 * np.sin(2 * np.pi * 20.0 * t)
            
            noise = 4e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
    
    elif scenario_type == 'anxious':
        # Very high beta, right hemisphere activation
        for ch in range(4):
            delta = 8e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 10e-6 * np.sin(2 * np.pi * 6.0 * t)  # Low theta
            
            if ch == 1:  # AF7 (left frontal) - less activated
                alpha = 32e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 22e-6 * np.sin(2 * np.pi * 24.0 * t)
            elif ch == 2:  # AF8 (right frontal) - highly activated
                alpha = 12e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 40e-6 * np.sin(2 * np.pi * 24.0 * t)  # Very high beta
            else:
                alpha = 22e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 30e-6 * np.sin(2 * np.pi * 24.0 * t)
            
            noise = 6e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
    
    elif scenario_type == 'sad':
        # Right frontal activation, low theta
        for ch in range(4):
            delta = 12e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 8e-6 * np.sin(2 * np.pi * 6.0 * t)   # Low theta
            
            if ch == 1:  # AF7 (left frontal)
                alpha = 28e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 18e-6 * np.sin(2 * np.pi * 20.0 * t)
            elif ch == 2:  # AF8 (right frontal)
                alpha = 14e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 24e-6 * np.sin(2 * np.pi * 20.0 * t)
            else:
                alpha = 24e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 20e-6 * np.sin(2 * np.pi * 20.0 * t)
            
            noise = 5e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
    
    elif scenario_type == 'stressed':
        # High beta everywhere, low alpha everywhere
        for ch in range(4):
            delta = 10e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 10e-6 * np.sin(2 * np.pi * 6.0 * t)
            alpha = 12e-6 * np.sin(2 * np.pi * 10.0 * t)  # Low alpha
            beta = 38e-6 * np.sin(2 * np.pi * 22.0 * t)   # High beta
            
            noise = 7e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
    
    elif scenario_type == 'drowsy':
        # High theta and alpha (sleepy state)
        for ch in range(4):
            delta = 15e-6 * np.sin(2 * np.pi * 2.5 * t)   # Higher delta
            theta = 35e-6 * np.sin(2 * np.pi * 6.0 * t)   # High theta
            alpha = 32e-6 * np.sin(2 * np.pi * 10.0 * t)  # High alpha
            beta = 8e-6 * np.sin(2 * np.pi * 18.0 * t)    # Low beta
            
            noise = 3e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
    
    elif scenario_type == 'balanced':
        # Perfectly balanced across all channels and bands
        for ch in range(4):
            delta = 10e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 16e-6 * np.sin(2 * np.pi * 6.0 * t)
            alpha = 20e-6 * np.sin(2 * np.pi * 10.0 * t)
            beta = 18e-6 * np.sin(2 * np.pi * 20.0 * t)
            
            noise = 4e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
    
    elif scenario_type == 'left_dominant':
        # Strong left hemisphere dominance
        for ch in range(4):
            delta = 10e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 18e-6 * np.sin(2 * np.pi * 6.0 * t)
            
            if ch == 1:  # AF7 (left frontal)
                alpha = 12e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 20e-6 * np.sin(2 * np.pi * 20.0 * t)
            elif ch == 2:  # AF8 (right frontal)
                alpha = 32e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 14e-6 * np.sin(2 * np.pi * 20.0 * t)
            else:
                alpha = 20e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 16e-6 * np.sin(2 * np.pi * 20.0 * t)
            
            noise = 4e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
    
    elif scenario_type == 'right_dominant':
        # Strong right hemisphere dominance
        for ch in range(4):
            delta = 10e-6 * np.sin(2 * np.pi * 2.5 * t)
            theta = 14e-6 * np.sin(2 * np.pi * 6.0 * t)
            
            if ch == 1:  # AF7 (left frontal)
                alpha = 30e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 16e-6 * np.sin(2 * np.pi * 20.0 * t)
            elif ch == 2:  # AF8 (right frontal)
                alpha = 10e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 26e-6 * np.sin(2 * np.pi * 20.0 * t)
            else:
                alpha = 20e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 18e-6 * np.sin(2 * np.pi * 20.0 * t)
            
            noise = 4e-6 * np.random.randn(num_samples)
            eeg_data[:, ch] = delta + theta + alpha + beta + noise
    
    return eeg_data


def main():
    print("=" * 80)
    print("   Extended EEG Emotion Detection Test - Various Scenarios")
    print("=" * 80)
    print("\nClassification outputs: POSITIVE, NEUTRAL, or NEGATIVE")
    print("Testing various brain states to see how they map to these 3 categories")
    print("=" * 80)
    
    # Initialize processor
    processor = EEGProcessor()
    
    # Define test scenarios with expected classifications
    scenarios = [
        # (scenario_name, expected_class, description)
        ('happy', 'POSITIVE', 'Strong left frontal activation'),
        ('calm_relaxed', 'POSITIVE', 'High theta, meditation-like state'),
        ('focused', 'POSITIVE/NEUTRAL', 'Task engagement, moderate beta'),
        ('balanced', 'NEUTRAL', 'Equal activity across hemispheres'),
        ('drowsy', 'NEUTRAL/POSITIVE', 'High theta/alpha, sleepy state'),
        ('sad', 'NEGATIVE', 'Right frontal activation, low theta'),
        ('anxious', 'NEGATIVE', 'Very high beta, right activation'),
        ('stressed', 'NEGATIVE', 'High beta everywhere, stress response'),
        ('left_dominant', 'POSITIVE', 'Left hemisphere dominance'),
        ('right_dominant', 'NEGATIVE', 'Right hemisphere dominance'),
    ]
    
    results = []
    
    for scenario_name, expected, description in scenarios:
        print(f"\n{'=' * 80}")
        print(f"SCENARIO: {scenario_name.upper()}")
        print(f"Description: {description}")
        print(f"Expected: {expected}")
        print(f"{'-' * 80}")
        
        # Run multiple tests per scenario for consistency
        num_tests = 3
        detections = []
        confidences = []
        
        for test_num in range(1, num_tests + 1):
            # Generate EEG data
            eeg_data = generate_eeg_scenario(scenario_name, duration_sec=4.0, fs=256)
            
            # Detect emotion
            emotion, confidence, metrics = processor.detect_emotional_state(eeg_data, fs=256)
            
            detections.append(emotion)
            confidences.append(confidence)
            
            if test_num == 1:  # Show detailed metrics for first test
                print(f"\nTest 1 Details:")
                print(f"  Detected:   {emotion}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Metrics:")
                print(f"    - FAA Score:        {metrics['faa_score']:.4f}")
                print(f"    - Beta Power:       {metrics['beta_power']:.2e}")
                print(f"    - Theta/Alpha:      {metrics['theta_alpha_ratio']:.3f}")
                print(f"    - Alpha Left (AF7): {metrics['alpha_left']:.2e}")
                print(f"    - Alpha Right(AF8): {metrics['alpha_right']:.2e}")
        
        # Compute consistency
        most_common = max(set(detections), key=detections.count)
        consistency = detections.count(most_common) / len(detections) * 100
        avg_confidence = np.mean(confidences)
        
        print(f"\nSummary ({num_tests} tests):")
        print(f"  Most Common: {most_common} ({consistency:.0f}% consistent)")
        print(f"  Avg Confidence: {avg_confidence:.3f}")
        print(f"  All Detections: {detections}")
        
        # Check if matches expected
        expected_options = expected.split('/')
        match = most_common in expected_options
        status = "✓ AS EXPECTED" if match else f"✗ UNEXPECTED (expected {expected})"
        print(f"  Result: {status}")
        
        results.append({
            'scenario': scenario_name,
            'expected': expected,
            'detected': most_common,
            'consistency': consistency,
            'confidence': avg_confidence,
            'match': match
        })
    
    # Final Summary
    print("\n" + "=" * 80)
    print("   FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Scenario':<18} {'Expected':<20} {'Detected':<12} {'Confidence':<12} {'Status'}")
    print("-" * 80)
    
    matches = 0
    for r in results:
        status = "✓" if r['match'] else "✗"
        print(f"{r['scenario']:<18} {r['expected']:<20} {r['detected']:<12} {r['confidence']:.3f}        {status}")
        if r['match']:
            matches += 1
    
    accuracy = matches / len(results) * 100
    print("-" * 80)
    print(f"Overall Match Rate: {matches}/{len(results)} ({accuracy:.1f}%)")
    print("=" * 80)
    
    print("\n📊 Key Insights:")
    print("  • Left frontal activation (low alpha left) → POSITIVE")
    print("  • Right frontal activation (low alpha right) → NEGATIVE")
    print("  • High theta/alpha ratio → tends toward POSITIVE (relaxation)")
    print("  • High beta power → tends toward NEGATIVE (stress/anxiety)")
    print("  • Balanced hemispheres → NEUTRAL")
    print("=" * 80)


if __name__ == "__main__":
    main()
