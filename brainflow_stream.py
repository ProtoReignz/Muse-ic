import numpy as np
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# --------- BrainFlow Setup ----------
params = BrainFlowInputParams()
params.serial_port = "/dev/ttyACM1"   # your BLED112 port
board = BoardShim(BoardIds.MUSE_2_BLED_BOARD, params)
board.prepare_session()
board.start_stream()
print("Streaming from Muse 2...")

# --------- EEG Channel Info ----------
eeg_chs = BoardShim.get_eeg_channels(BoardIds.MUSE_2_BLED_BOARD.value)
eeg_names = ['TP9', 'AF7', 'AF8', 'TP10']
fs = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BLED_BOARD.value)

# --------- PyQtGraph Setup ----------
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Muse 2 Live EEG")
win.resize(1000, 600)
win.show()

plots = []
curves = []
buffer_len = fs * 5  # show last 5 seconds
data_buffer = np.zeros((len(eeg_chs), buffer_len))

for i, name in enumerate(eeg_names):
    p = win.addPlot(row=i, col=0)
    p.setLabel('left', name)
    p.setYRange(-200, 200)
    curve = p.plot(pen=pg.mkPen(width=1.5))
    plots.append(p)
    curves.append(curve)

# --------- Update Loop ----------
def update():
    global data_buffer
    # get newest data (non-blocking)
    new_data = board.get_board_data()
    if new_data.shape[1] == 0:
        return

    for i, ch in enumerate(eeg_chs):
        # append new data, maintain fixed window
        samples = new_data[ch]
        data_buffer[i] = np.roll(data_buffer[i], -len(samples))
        data_buffer[i, -len(samples):] = samples
        curves[i].setData(data_buffer[i])

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)   # update every 50 ms (~20 FPS)

# --------- Run GUI ----------
if __name__ == '__main__':
    try:
        app.exec()
    finally:
        print("Stopping stream...")
        board.stop_stream()
        board.release_session()
