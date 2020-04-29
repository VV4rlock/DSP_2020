import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from functools import reduce
from scipy.signal import butter, lfilter

MSG = b"FINAL"

freq_array = {"C": 261.63, "D": 293.66, "E": 329.63, "F": 349.23, "G": 392.00, "A": 440.00, "B": 493.88, "_": 0}

# Частотная манипулиция
#mod_sequence = np.array(
#    [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1,
#     1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1,
#     1, 0, 1, 0, 0])
mod_sequence = np.array(list(map(int, reduce(lambda a, x: a+list(bin(x)[2:].zfill(8)), MSG, []))))


SAMPLERATE = 16000
CHR_PER_SEC = 4
AMPL = np.iinfo(np.int16).max
WIDTH = SAMPLERATE//CHR_PER_SEC

low_freq = 100
high_freq = 400
tick_freq = 800


def get_fsk_signal(mod_sequence, mod_1_freq=128, tick_freq=3000, mod_0_freq=None):
    tick_sample = np.zeros(WIDTH)
    tick_sample[WIDTH//2:] = 1
    tick_signal = np.tile(tick_sample, mod_sequence.size) * tick_freq
    mod_signal = np.repeat(mod_sequence, repeats=WIDTH)
    carirer_period = mod_signal.size

    t = np.arange(carirer_period) / SAMPLERATE
    mod_frequencies = np.zeros(carirer_period)
    if mod_0_freq is not None: # инвертированный сигнал)
        mod_frequencies[mod_signal == 0] = mod_0_freq
    mod_frequencies[mod_signal == 1] = mod_1_freq
    return AMPL // 2 * (np.sin(mod_frequencies * 2.0 * np.pi * t) + np.sin(tick_signal * 2.0 * np.pi * t))


def get_message_from_record(filename):
    fs, signal = wavfile.read(filename)
    print(fs)
    #signal = butter_bandpass_filter(signal, 1000, 4000, fs)
    pxx, freqs, bins, im = plt.specgram(signal, Fs=fs)
    plt.show()
    T = 15000
    print(freqs)
    high = pxx[np.abs(freqs - high_freq).argmin()]
    plt.plot(high)
    plt.show()
    high = np.where(high < T, 0, 1)

    tick = pxx[np.abs(freqs - tick_freq).argmin()]
    tick = np.where(tick < T, 0, 1)
    plt.plot(high)

    plt.plot(tick)
    plt.show()

    measure_moments = np.squeeze(np.where(tick[1:] - tick[:-1] == 1)) + 1
    res = (high[measure_moments] == 1).astype(np.int8)

    print((high[measure_moments] == 1).astype(np.int8))
    # print(len(res))
    word = []
    for i in range(0, res.size, 8):
        word.append(chr(int(''.join(map(str, res[i: i + 8])), 2)))
    return ''.join(word)

GENERATE = False
if __name__=="__main__":
    if GENERATE:
        fsk_signal = get_fsk_signal(mod_sequence, mod_1_freq=high_freq, tick_freq=tick_freq)
        pxx, freqs, bins, im = plt.specgram(fsk_signal, Fs=16000)
        plt.show()
        wavfile.write("fsk_signal.wav", SAMPLERATE, fsk_signal)
    print(f"result: {get_message_from_record('fsk_signal_message.wav')}")
