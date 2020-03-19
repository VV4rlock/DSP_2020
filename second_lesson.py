import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from numpy import fft
from scipy.io import wavfile


logging.basicConfig(format="[%(filename)s: %(funcName)s] %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
FILE_PATH = r"../dsp_lab/lab_0/hw_0_examples/0.wav"

def sample1():
    x_1 = np.arange(20)
    logger.info(f"positive {np.correlate(x_1,x_1)}")
    logger.info(f"negative {np.correlate(x_1, -1* x_1)}")

    np.random.seed(1)
    x_1 = np.random.randint(-10, 10, 20)
    x_2 = np.random.randint(-10, 10, 20)
    logger.info(f'zero correlation {np.correlate(x_1, x_2)}')
    plt.plot(x_1)
    #plt.plot(x_2)


def gen_signal(ampl, freq):
    samplerate = 32
    t = np.arange(samplerate)
    phase = 0
    return ampl * np.sin(2. * np.pi * freq * t / samplerate + phase)


def ft():
    samplerate = 32
    t = np.arange(samplerate)
    phase = 0
    data_1 = gen_signal(1,4)
    data_2 = gen_signal(1/2,8)
    data = data_1 + data_2
    samplerate = 32
    t = np.arange(samplerate)
    #plt.plot(data)
    #plt.show()

    shifted_data = data * np.exp(2j*np.pi*t/samplerate*10)
    fft_data = fft.fft(data)
    m_fft_data = np.abs(fft_data)
    m_shifted_fft = np.abs(fft.fft(shifted_data))
    p_fft_data = np.angle(fft_data)
    restores_data = fft.ifft(fft_data).real
    plt.plot(t, m_fft_data, '-pm')
    plt.plot(t, m_shifted_fft, "-go")

    plt.show()


def fft2():
    samplerate = 32
    t = np.arange(samplerate)/samplerate
    phase = 0
    freq = 4
    freq2 = 4.05
    ampl = 1
    data = ampl * np.sin(2. * np.pi * freq * t + phase)
    window = np.hanning(samplerate)
    data2 = ampl * np.sin(2. * np.pi * freq2 * t + phase)
    data = data2 * window
    m_abd1 = np.abs(fft.fft(data))
    m_abd2 = np.abs(fft.fft(data2))
    plt.figure(figsize=(14,5))
    plt.stem(t, m_abd1, '--',markerfmt='om')
    plt.stem(t, m_abd2, '--', markerfmt='^g')
    #plt.stem

    plt.show()


def noise():
    white = np.random.uniform(-1,1,1000)
    #мощность примерно одинаковая на всех частотах
    fft_x = fft.rfft(white)
    power = np.cumsum(fft_x**2)
    power /= power[-1]
    plt.plot(white)
    plt.show()
    plt.plot(np.abs(fft_x))
    plt.show()
    plt.plot(power)
    plt.show()


if __name__ == "__main__":
    #sample1()
    #fft2()
    noise()