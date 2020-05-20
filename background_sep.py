import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from numpy import fft
from scipy.io import wavfile
from scipy import signal

SIGNAL = r"../dsp_lab/lab_4/0.wav"

def show_spec(freqs, fft):
    plt.figure(figsize=(14, 5))
    plt.plot(freqs, fft)
    plt.title('Spectrum')
    plt.ylabel("Level")
    plt.xlabel("Frequency")
    plt.show()


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

def sep():
    sample_rate, x = wavfile.read(SIGNAL)
    if False:
        plt.figure(figsize=(14, 5))
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(x)) / sample_rate, x)
        plt.subplot(2, 1, 2)
        pxx, freqs, bins, im = plt.specgram(x, Fs=sample_rate)
        plt.show()
    music_only = x[sample_rate*7:sample_rate*9]

    voice = x[int(sample_rate* 10.2):int(sample_rate*11.3)]

    music_fft = np.abs(fft.rfft(music_only))

    x_fft = fft.fft(x)
    x_imag = x_fft.imag*1j
    print(x_imag)
    m_fft = fft.fft(music_only)
    print(x_fft.shape, m_fft.shape)
    plt.figure(figsize=(14, 20))
    plt.subplot(2, 1, 1)
    #plt.plot(x_fft)

    new_real = x_fft.real
    new_real[:m_fft.shape[0]] -= 1*m_fft.real
    new_real[new_real < 0] = 0
    new = x_fft.real - new_real + x_imag
    plt.plot(x_fft)
    restores_data = fft.ifft(new).real
    wavfile.write("voice.wav", sample_rate, restores_data.astype(np.int16))
    voice_fft = np.abs(fft.rfft(voice))

    plt.subplot(2, 1, 1)
    plt.plot(m_fft)
    #plt.subplot(2, 1, 2)
    plt.plot(voice_fft)
    plt.show()

if __name__=="__main__":
    sep()