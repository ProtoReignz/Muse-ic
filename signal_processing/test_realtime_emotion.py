"""
Test script demonstrating real-time emotion detection with streaming EEG data.
This simulates how the EEGProcessor would work with live Muse2 data.
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_processing.EEGProcessor import EEGProcessor


def simulate_muse2_stream(emotion_type='neutral', total_duration_sec=20.0, fs=256):
    """
    Generator that simulates real-time Muse2 EEG streaming.
    
    Parameters:
    -----------
    emotion_type : str
        'positive', 'neutral', 'negative', 'relaxed', 'stressed', 'focused', 'drowsy'
    total_duration_sec : float
        Total duration to stream
    fs : int
        Sampling frequency
    
    Yields:
    -------
    np.ndarray : Single EEG sample, shape (4,) [TP9, AF7, AF8, TP10]
    """
    total_samples = int(total_duration_sec * fs)
    t = 0
    dt = 1.0 / fs
    
    for sample_idx in range(total_samples):
        # Generate one sample at current time
        sample = np.zeros(4)
        
        if emotion_type == 'positive':
            # POSITIVE: Higher left frontal activity (AF7)
            for ch in range(4):
                delta = 10e-6 * np.sin(2 * np.pi * 2.5 * t)
                theta = 20e-6 * np.sin(2 * np.pi * 6.0 * t)
                
                if ch == 1:  # AF7 (left frontal)
                    alpha = 15e-6 * np.sin(2 * np.pi * 10.0 * t)
                    beta = 18e-6 * np.sin(2 * np.pi * 20.0 * t)
                elif ch == 2:  # AF8 (right frontal)
                    alpha = 30e-6 * np.sin(2 * np.pi * 10.0 * t)
                    beta = 15e-6 * np.sin(2 * np.pi * 20.0 * t)
                else:
                    alpha = 25e-6 * np.sin(2 * np.pi * 10.0 * t)
                    beta = 16e-6 * np.sin(2 * np.pi * 20.0 * t)
                
                noise = 5e-6 * np.random.randn()
                sample[ch] = delta + theta + alpha + beta + noise
                
        elif emotion_type == 'negative':
            # NEGATIVE: Higher right frontal activity (AF8)
            for ch in range(4):
                delta = 10e-6 * np.sin(2 * np.pi * 2.5 * t)
                theta = 12e-6 * np.sin(2 * np.pi * 6.0 * t)
                
                if ch == 1:  # AF7 (left frontal)
                    alpha = 30e-6 * np.sin(2 * np.pi * 10.0 * t)
                    beta = 18e-6 * np.sin(2 * np.pi * 22.0 * t)
                elif ch == 2:  # AF8 (right frontal)
                    alpha = 15e-6 * np.sin(2 * np.pi * 10.0 * t)
                    beta = 35e-6 * np.sin(2 * np.pi * 22.0 * t)
                else:
                    alpha = 25e-6 * np.sin(2 * np.pi * 10.0 * t)
                    beta = 25e-6 * np.sin(2 * np.pi * 22.0 * t)
                
                noise = 5e-6 * np.random.randn()
                sample[ch] = delta + theta + alpha + beta + noise
                
        else:  # neutral
            # NEUTRAL: Balanced activity
            for ch in range(4):
                delta = 10e-6 * np.sin(2 * np.pi * 2.5 * t)
                theta = 16e-6 * np.sin(2 * np.pi * 6.0 * t)
                alpha = 25e-6 * np.sin(2 * np.pi * 10.0 * t)
                beta = 20e-6 * np.sin(2 * np.pi * 20.0 * t)
                
                noise = 5e-6 * np.random.randn()
                sample[ch] = delta + theta + alpha + beta + noise
        
        yield sample
        t += dt


def main():
    print("=" * 70)
    print("   Real-Time EEG Emotion Detection Simulation")
    print("=" * 70)
    print("\nSimulating Muse2 streaming at 256 Hz")
    print("Emotion detection runs every 2 seconds with 4-second window")
    print("Channels: TP9, AF7, AF8, TP10")
    print("=" * 70)
    
    # Configuration
    FS = 256  # Sampling frequency
    BUFFER_SIZE = 1024  # 4 seconds of data
    DETECTION_INTERVAL = 2.0  # Detect every 2 seconds
    MIN_BUFFER_TIME = 3.0  # Need at least 3 seconds before first detection
    
    # Test scenarios
    scenarios = [
        ('positive', 12, "Simulating POSITIVE emotional state"),
        ('neutral', 12, "Simulating NEUTRAL emotional state"),
        ('negative', 12, "Simulating NEGATIVE emotional state"),
    ]
    
    for emotion_type, duration, description in scenarios:
        print(f"\n{'=' * 70}")
        print(f"SCENARIO: {description}")
        print(f"Duration: {duration} seconds")
        print(f"{'=' * 70}\n")
        
        # Initialize processor for this scenario
        processor = EEGProcessor(buffer_size=BUFFER_SIZE, num_channels=4)
        
        detection_count = 0
        sample_count = 0
        
        # Simulate streaming
        stream = simulate_muse2_stream(emotion_type, total_duration_sec=duration, fs=FS)
        
        print("Starting stream... (buffering initial data)")
        start_time = time.time()
        
        for sample in stream:
            sample_count += 1
            
            # Process sample and check for emotion detection
            result = processor.process_and_detect_emotion(
                sample,
                detection_interval_sec=DETECTION_INTERVAL,
                fs=FS,
                min_buffer_sec=MIN_BUFFER_TIME
            )
            
            if result:
                detection_count += 1
                elapsed = time.time() - start_time
                
                print(f"\n[Detection #{detection_count}] at {result['timestamp_sec']:.1f}s (elapsed: {elapsed:.1f}s)")
                print(f"  Emotion:     {result['emotion']}")
                print(f"  Confidence:  {result['confidence']:.3f}")
                print(f"  Buffer size: {result['buffer_size']} samples")
                print(f"  Metrics:")
                print(f"    - FAA Score:        {result['metrics']['faa_score']:.4f}")
                print(f"    - Beta Power:       {result['metrics']['beta_power']:.2e}")
                print(f"    - Theta/Alpha:      {result['metrics']['theta_alpha_ratio']:.3f}")
                print(f"    - Alpha Left (AF7): {result['metrics']['alpha_left']:.2e}")
                print(f"    - Alpha Right (AF8):{result['metrics']['alpha_right']:.2e}")
                
                # Verify it matches expected emotion
                expected = emotion_type.upper()
                status = "✓ CORRECT" if result['emotion'] == expected else "✗ INCORRECT"
                print(f"  Expected:    {expected} {status}")
            
            # Small delay to simulate real-time (optional, for demonstration)
            # In real Muse2 streaming, samples arrive at 256 Hz naturally
            # time.sleep(1.0 / FS)  # Uncomment to slow down for observation
        
        total_time = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"Scenario complete:")
        print(f"  Total samples:  {sample_count}")
        print(f"  Total detections: {detection_count}")
        print(f"  Expected detections: ~{int(duration / DETECTION_INTERVAL - 1)}")
        print(f"  Processing time: {total_time:.2f}s")
        print(f"{'=' * 70}")
    
    print("\n" + "=" * 70)
    print("   USAGE EXAMPLE FOR REAL MUSE2 STREAMING")
    print("=" * 70)
    print("""
# Initialize processor with 4-second buffer
processor = EEGProcessor(buffer_size=1024, num_channels=4)

# In your Muse2 streaming loop:
while streaming:
    # Get one sample from Muse2 (shape: 4, for [TP9, AF7, AF8, TP10])
    sample = muse2.get_next_sample()
    
    # Process and detect (returns None until conditions met)
    result = processor.process_and_detect_emotion(
        sample,
        detection_interval_sec=2.0,  # Detect every 2 seconds
        fs=256                        # Muse2 sampling rate
    )
    
    if result:
        print(f"Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        # Use result['metrics'] for detailed analysis
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
