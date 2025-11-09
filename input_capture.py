import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from signal_processing.EEGProcessor import EEGProcessor

class InputCapture:
    """
    Handles Muse 2 EEG data capture, buffering, and retrieval
    for use in real-time analysis (e.g. FAA computation).
    """

    def __init__(self, port="/dev/ttyACM1", window_seconds=3):
        # ---- BrainFlow setup ----
        self.params = BrainFlowInputParams()
        self.params.serial_port = port
        self.board_id = BoardIds.MUSE_2_BLED_BOARD.value
        self.board = BoardShim(self.board_id, self.params)

        self.fs = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.channel_names = ['TP9', 'AF7', 'AF8', 'TP10']

        # ---- Rolling buffer ----
        self.window_size = int(window_seconds * self.fs)
        self.num_channels = len(self.eeg_channels)
        self.data_buffer = np.zeros((self.num_channels, self.window_size))
        
        # ---- EEG Processor for emotion detection ----
        self.processor = EEGProcessor(buffer_size=1024, num_channels=4)
        self.sample_count = 0

    def start(self):
        """Start Muse 2 streaming session."""
        self.board.prepare_session()
        self.board.start_stream()
        print(f"Streaming from Muse 2 @ {self.fs} Hz")

    def stop(self):
        """Stop Muse 2 streaming session."""
        print("Stopping stream...")
        self.board.stop_stream()
        self.board.release_session()

    def update_buffer(self):
        """
        Fetch new EEG samples from Muse 2 and append to rolling buffer.
        Also processes samples for real-time emotion detection.
        """
        new_data = self.board.get_board_data()  # shape: (channels, n_samples)
        if new_data.shape[1] == 0:
            return None

        for i, ch in enumerate(self.eeg_channels):
            samples = new_data[ch]
            n = len(samples)
            # Roll old data left and insert new samples at the end
            self.data_buffer[i] = np.roll(self.data_buffer[i], -n)
            self.data_buffer[i, -n:] = samples
        
        # Process each new sample through emotion detector
        # new_data is in (channels, samples) format, need to transpose for sample-by-sample processing
        emotion_result = None
        for sample_idx in range(new_data.shape[1]):
            # Extract single sample across all EEG channels [TP9, AF7, AF8, TP10]
            sample = np.array([new_data[ch, sample_idx] for ch in self.eeg_channels])
            
            # Process through emotion detector
            result = self.processor.process_and_detect_emotion(
                sample,
                detection_interval_sec=2.0,
                fs=self.fs,
                min_buffer_sec=3.0,
                channel_names=self.channel_names
            )
            
            if result:
                emotion_result = result
                # Print emotion detection result
                print(f"\n{'='*60}")
                print(f"🧠 EMOTION DETECTED at {result['timestamp_sec']:.1f}s")
                print(f"{'='*60}")
                print(f"  Emotion:     {result['emotion']}")
                print(f"  Confidence:  {result['confidence']:.3f}")
                print(f"  FAA Score:   {result['metrics']['faa_score']:.4f}")
                print(f"  Beta Power:  {result['metrics']['beta_power']:.2e}")
                print(f"  Theta/Alpha: {result['metrics']['theta_alpha_ratio']:.3f}")
                print(f"{'='*60}\n")
        
        return emotion_result

    def get_latest(self):
        """
        Return the latest EEG buffer (channels × samples).
        """
        return self.data_buffer.copy()

    def get_channel_data(self, name):
        """
        Get buffer for a specific channel by name (TP9, AF7, AF8, TP10).
        """
        if name not in self.channel_names:
            raise ValueError(f"Invalid channel: {name}")
        idx = self.channel_names.index(name)
        return self.data_buffer[idx].copy()
    
    def detect_emotion_from_buffer(self):
        """
        Manually trigger emotion detection on the current buffer.
        Useful for on-demand emotion analysis.
        
        Returns
        -------
        dict or None
            Dictionary with emotion, confidence, and metrics if successful, None otherwise
        """
        # Transpose buffer from (channels, samples) to (samples, channels)
        data_transposed = self.data_buffer.T  # shape: (samples, 4)
        
        try:
            emotion, confidence, metrics = self.processor.detect_emotional_state(
                data_transposed,
                fs=self.fs,
                channel_names=self.channel_names
            )
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'metrics': metrics,
                'buffer_size': data_transposed.shape[0]
            }
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return None
