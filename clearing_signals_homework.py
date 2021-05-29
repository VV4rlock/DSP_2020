
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from numpy import fft
from scipy.io import wavfile
from scipy import signal

SIGNAL1 = r"../dsp_lab/lab_3/0/0.wav"
SIGNAL2 = r"../dsp_lab/lab_3/1/0.wav"

def show_spec(freqs, fft):
    plt.figure(figsize=(14, 5))
    plt.plot(freqs, fft)
    plt.title('Spectrum')
    plt.ylabel("Level")
    plt.xlabel("Frequency")
    plt.show()


def show_filter_spec(b,a, fs):
    w, h = signal.freqz(b, a, fs=fs)
    h = np.abs(h)
    plt.plot(w,h)
    plt.show()


def clear_signal1():
    sample_rate, x = wavfile.read(SIGNAL1)
    #pxx, freqs, bins, im = plt.specgram(x, Fs=sample_rate)

    #print(sample_rate, len(x))
    x_fft = np.abs(fft.rfft(x))
    freqs = fft.fftfreq(len(x_fft)) * sample_rate / 2
    i = freqs > 0
    m = x_fft.max()
    w1_i = x_fft[i].argmax()
    w0_i = x_fft[i][:w1_i].argmax()
    w0,w1 = freqs[i][w0_i],freqs[i][w1_i]
    print(f"noise freq is {w0} and {w1}")
    #show_spec(freqs[i], x_fft[i])

    q=200
    b, a = signal.iirnotch(w0=w0, Q=q, fs=sample_rate)

    #y_1 = signal.lfilter(b, a, x)
    y_1 = signal.filtfilt(b, a, x)

    b, a = signal.iirnotch(w0=w1, Q=q, fs=sample_rate)
    y_1 = signal.filtfilt(b, a, y_1)

    plt.figure(figsize=(14, 5))
    plt.plot(freqs[i], x_fft[i])
    plt.plot(freqs[i], np.abs(fft.rfft(y_1))[i], alpha=0.5)
    plt.title('Spectrum')
    plt.ylabel("Level")
    plt.xlabel("Frequency")
    plt.show()


    wavfile.write("clear0.wav", sample_rate, y_1.astype(np.int16))


def apply_filter(b, a, x: np.ndarray, window_len=3):
    print(len(x), len(x) % window_len)
    pad = window_len - len(x) % window_len
    x = np.pad(x, (pad, 0), mode='constant')
    windows = np.split(x, window_len)

    def filter(i):
        v = i.var()**0.5
        m = i.mean()
        # не понятно что даст вычитание(
        #i = np.where(np.abs(i - m)>v, m)
        i -= v
        return signal.filtfilt(b, a, i)
    return np.concatenate(list(map(filter, windows)))[pad:]


def clear_signal2():
    sample_rate, x = wavfile.read(SIGNAL2)
    #pxx, freqs, bins, im = plt.specgram(x, Fs=sample_rate)

    #print(sample_rate, len(x))
    x_fft = np.abs(fft.rfft(x))
    freqs = fft.fftfreq(len(x_fft)) * sample_rate / 2
    i = freqs > 0

    VOICE_HIGH_FREQ = 4000
    VOICE_LOW_FREQ = 110
    q=5
    b,a = signal.butter(q, Wn=(VOICE_LOW_FREQ, VOICE_HIGH_FREQ), btype='bandpass', fs=sample_rate)

    show_filter_spec(b, a, sample_rate)
    #y_1 = signal.lfilter(b, a, x)
    y_1 = apply_filter(b, a, x)
    print(len(y_1), len(x))

    '''
    window_len = 551
    window = signal.gaussian(window_len, 3)
    window = window / window.sum()
    print(len(y_1), y_1.shape)
    y_1 = cv2.filter2D(x, -1, window).reshape(-1)
    print(len(y_1), y_1.shape)
    '''
    plt.figure(figsize=(14, 5))
    plt.plot(freqs[i], x_fft[i])
    plt.plot(freqs[i], np.abs(fft.rfft(y_1))[i], alpha=0.5)
    plt.title('Spectrum')
    plt.ylabel("Level")
    plt.xlabel("Frequency")
    plt.show()


    wavfile.write("clear1.wav", sample_rate, y_1.astype(np.int16))


if __name__=="__main__":

    #clear_signal1()
    clear_signal2()
