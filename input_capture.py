import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

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
        """
        new_data = self.board.get_board_data()  # shape: (channels, n_samples)
        if new_data.shape[1] == 0:
            return

        for i, ch in enumerate(self.eeg_channels):
            samples = new_data[ch]
            n = len(samples)
            # Roll old data left and insert new samples at the end
            self.data_buffer[i] = np.roll(self.data_buffer[i], -n)
            self.data_buffer[i, -n:] = samples

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
