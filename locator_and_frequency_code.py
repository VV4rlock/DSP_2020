import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from numpy import fft
from scipy.io import wavfile
import pickle


logging.basicConfig(format="[%(filename)s: %(funcName)s] %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
LOCATOR_DUMPFILE = r"../dsp_lab/lab_1/hw_1_examples/0/0_12_9_1221292.pickle"
RECV_SIGNAL = r"../dsp_lab/lab_1/hw_1_examples/1/0.wav"
#LOCATOR_DUMPFILE = r"../dsp_lab/lab_1/hw_1_examples/0/1_14_13_1179628.pickle"
#LOCATOR_DUMPFILE = r"../dsp_lab/lab_1/hw_1_examples/0/2_14_6_1424377.pickle"

def rangefinder():
    with open(LOCATOR_DUMPFILE, "rb") as f:
        data = pickle.load(f)
    c = 300*10**6
    M = 12
    recv_signal = np.array(data)
    samplerate = 192000
    t = np.linspace(0, M, M).astype(np.float64) - 0.5 * M
    sigma = 9
    signal = np.exp(-1/2 * (t / sigma) ** 2)
    plt.plot(t, signal)
    plt.show()
    cor = np.correlate(recv_signal, signal, mode='same')
    argmax = np.abs(cor).argmax()
    #print(np.abs(cor).max(), argmax)
    #plt.plot(t, signal)
    #plt.plot(recv_signal)
    plt.plot(np.abs(cor))
    plt.show()
    print(f"task 1:\n\ttime = {argmax / samplerate}s\n\tres={argmax / samplerate * c / 2 / 1000} km")

    #w(n)=\exp^{(-1/2(n/sigma)^2)

def extract_message_from_signal():
    samplerate, data = wavfile.read(RECV_SIGNAL)
    bands = np.array([[1, 3], [9, 12]]) * 1000
    T = 2000
    window_len = 30
    alpha = 0.5
    window = np.ones(window_len)
    pxx, freqs, bins, im = plt.specgram(data, NFFT=64, Fs=samplerate, noverlap=32)
    plt.show()
    out_words = []
    for low_freq, high_freq in bands:
        low = pxx[np.squeeze(np.where(freqs > low_freq))[0]]
        high = pxx[np.squeeze(np.where(freqs < high_freq))[-1]]

        low = np.where(low < T, 0, 1)
        low = np.correlate(window, low)
        low = np.where(low < alpha*window_len, 0, 1)

        high = np.where(high < T, 0, 1)
        high = np.correlate(window, high)
        high = np.where(high < alpha*window_len, 0, 1)
        tick = low + high
        measure_moments = np.squeeze(np.where(tick[1:] - tick[:-1] == 1)) + 1
        plt.plot(low)
        plt.plot(high)
        plt.show()
        res = []
        for i in measure_moments:
            if low[i] == 1:
                res.append('0')
            elif high[i] == 1:
                res.append('1')
            else:
                raise Exception(f"unknown tick {i}")

        #print(len(res))
        word = []
        for i in range(0, len(res), 8):
            word.append(chr(int(''.join(reversed(res[i:i+8])), 2)))
        out_words.append(''.join(reversed(word)))

    print(f"result: {' '.join(out_words)}")






if __name__=="__main__":
    rangefinder()
    extract_message_from_signal()


