import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from scipy.io import wavfile


logging.basicConfig(format="[%(filename)s: %(funcName)s] %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
FILE_PATH = r"../dsp_lab/lab_0/hw_0_examples/0.wav"

def dz1():
    F = 10
    samplerate = 1000
    ampl = np.iinfo(np.int16).max // 2
    logger.info(f"amplitude = {ampl}")
    t = np.linspace(0, 1, samplerate)
    S = ampl * np.sin(2 * np.pi * F * t)
    noise = ampl * (np.random.rand(samplerate) * 2 - 1)
    S_round = np.round(S)
    noise_round = np.round(noise)
    S_err = S - S_round
    noise_err = noise - noise_round

    S_sn = 10 * np.log10(np.var(S)/np.var(S_err))
    noise_sn = 10 * np.log10(np.var(noise) / np.var(noise_err))
    print(f"TASK 1:\n\ttheor = {6 * 16 - 7.2}, signal = {S_sn}, noise = {noise_sn}")


def dz2():
    samplerate, data = wavfile.read(FILE_PATH)
    data = data.astype(np.int64)
    window_time = 10 #ms
    T_Ex = 0.3
    T_sign = 0.3
    window_len = round(samplerate / 1000 * window_time)
    print(f"TASK 2:\n\tsamplerate = {samplerate}, windowtime = {window_time} ms, \
                 window_len = {window_len}, data_len={data.shape[0]}")
    Ex = np.array(sum([[(data[i:min(i + window_len, data.shape[0])]**2).sum()] * (window_len // 2)
              for i in range(0, data.shape[0], window_len // 2)], []))
    Ex = np.log10(Ex) # удобнее
    Ex = (Ex - Ex.min()) / (Ex.max() - Ex.min())
    Ex[ Ex < T_Ex] = 0
    Ex[ Ex != 0] = 0.2
    Ex = Ex - 0.2

    sign = np.sign(data).astype(np.float32)
    sign = np.abs(sign[1:] - sign[:-1])
    sign = np.array(sum([[(sign[i:min(i + window_len, sign.shape[0])]).sum() / (2 * window_len)] * (window_len // 2)
              for i in range(0, sign.shape[0], window_len // 2)], []))
    sign = sign / sign.max()
    sign[sign < T_sign] = 0
    sign[sign >= T_sign] = 0.2
    sign = sign + 1

    data = data - data.min()
    data = data / data.max()
    plt.plot(data / data.max(), label='voice')
    plt.plot(Ex, label='energy_detection')
    plt.plot(sign, label='zcr_detection')
    plt.legend()
    plt.show()






def gen_signal():
    samplerate = 2000
    ampl = np.iinfo(np.int16).max / 2
    t = np.linspace(0, 1, samplerate)
    F1 = 250  # frequent
    F2 = 500
    F3 = 750
    S1 = np.sin(2 * np.pi * F1 * t)
    S2 = np.sin(2 * np.pi * F2 * t)
    S3 = np.sin(2 * np.pi * F3 * t)
    plt.figure()
    plt.plot(t, S1, '-r', label='Freq=250')
    plt.plot(t, S2, '-g', label='Freq=500')
    plt.plot(t, S3, '-b', label='Freq=750')
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    wavfile.write('250.wav', samplerate, S1.astype(np.int16))

    plt.show()

if __name__=="__main__":

    #data = ampl * np.sin(2. * np.pi * freq * t/samplerate + phase)
    dz1()
    dz2()


