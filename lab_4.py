#!/usr/bin/env python
# coding: utf-8

# ### Многоскоростная обработка сигналов
# ![0.png](attachment:0.png)
# 
# #### Прореживание (Decimation)
# Уменьшение $f_d$ в целое число раз
# #### Интерполяция (Interpolation)
# Увеличение $f_d$ в целое число раз
# #### Передискретизация (Resampling)
# Изменение $f_d$ в произвольное число раз
# 
# 
# 

# In[190]:


import numpy as np
from numpy import fft
from scipy.io.wavfile import write
from scipy import signal 
from matplotlib import pyplot as plt


# ### Децимация

# In[192]:


# Условный пример появления ложных частот
samplerate = 10
number_of_samples = 32
t = np.arange(number_of_samples)/samplerate
amplitude = 1
phase = 0

data_1_d = amplitude * np.cos(2. * np.pi * 1 * t + phase)
data_1_a = amplitude * np.cos(2. * np.pi * 1 * np.arange(0,max(t),.01) + phase)

data_9_d = amplitude * np.cos(2. * np.pi * 9 * t + phase)
data_9_a = amplitude * np.cos(2. * np.pi * 9 * np.arange(0,max(t),.01) + phase)

data_10_d = amplitude * np.cos(2. * np.pi * 10 * t + phase)
data_10_a = amplitude * np.cos(2. * np.pi * 10 * np.arange(0,max(t),.01) + phase)

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlim((0,10))
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.stem(np.linspace(0, number_of_samples-1, data_9_a.shape[0]), data_9_a, basefmt='C0')
plt.plot(np.linspace(0, number_of_samples-1, data_9_a.shape[0]),data_9_a, linestyle='--',color='C0', alpha = 0.5)
plt.plot(data_9_d, marker='o', linestyle=' ', markersize=11, color='C1')
plt.plot(np.linspace(0, number_of_samples-1, data_1_a.shape[0]),data_1_a, linestyle='--',color='C1', alpha = 0.5)


# In[193]:


# Пример с чирпом появления ложных частот
t = np.linspace(0, 1, 10000)
chirp_signal = signal.chirp(t, f0=0, f1=5000, t1=1, method='linear')

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(chirp_signal)
plt.xlim(0,1000)
plt.figure(figsize=(14, 5))
pxx, freqs, bins, im = plt.specgram(chirp_signal, NFFT=128, Fs=10000, noverlap=0)


# In[194]:


# Пример децимации чирпа
"""
Функция decimate
Реализации децимации сигнала

Принимает на вход:
x -- сигнал для децимации 
q -- коэффициент децимации
n -- порядок фильтра нижних частот
ftype -- тип фильтра ‘iir’ или ‘fir’
zero_phase -- по умолчанию  True, использование filtfilt вместо lfilter
"""

decimated_signal = signal.decimate(chirp_signal, q=2, n=0)
plt.figure(figsize=(14, 5))
pxx, freqs, bins, im = plt.specgram(decimated_signal, NFFT=64, Fs=5000, noverlap=0)


# In[195]:


decimated_signal = signal.decimate(chirp_signal, q=2, n=10)
plt.figure(figsize=(14, 5))
pxx, freqs, bins, im = plt.specgram(decimated_signal, NFFT=64, Fs=5000, noverlap=0)


# ### Децимация
# 
# $ y(n) = x(nR), $ где $R$ -- коэффициент децимации, 
# 
# Учитывая фильтарцию:
# $ y(n) = \sum_{k=0}^{N-1}x(nR-k)h(k) $ ,
# 

# ### CIC-фильтр
# 
# Вид АЧХ:
# $$ H(f)=\left(\frac{sin(\pi RMf)}{sin(\pi f)}\right)^{N},$$ где 
# $M$ -- количество сэмлов на стадию, обычно 1; 
# 
# $N$ -- количество стадий.

# ### CIC-фильтр дециматор
# ![1.png](attachment:1.png)

# In[197]:


b =  [1,1,1,1,1,1,1,1]

w, h = signal.freqz(b,1)
h = np.abs(h)
log_h = 20*np.log10(h/np.max(h) + 10**(-15))

plt.figure(figsize=(14, 5))
plt.title('Frequency response')
plt.xlabel("Frequency")
plt.ylabel("Level")
plt.plot(w, h)

plt.figure(figsize=(14, 5))
plt.title('Frequency response')
plt.xlabel("Frequency")
plt.ylabel("Level, dB")
plt.plot(w,log_h)
plt.show()

plt.figure(figsize=(14, 5))
plt.title('Frequency response')
plt.xlabel("Frequency")
plt.ylabel("Level, dB")
plt.plot(w,log_h)
plt.ylim([-30, 0])
plt.show()


# In[198]:


def cic_decimator(x, r, n):
    """
    Функция cic_decimator
    CIC-фильтр дециматор

    Принимает на вход:
    x -- входной сигнал
    r -- коэффициент децимации
    n -- количество стадий
    """
    y = x
    # Integrator stages
    for i in range(n):
        y = np.cumsum(y)
    # Decimator
    y = y[::r]
    # COMB stages
    y = np.diff(y, n=n, prepend=np.zeros(n))
    return y


# In[203]:


# Пример децимации чирпа
decimated_signal = cic_decimator(chirp_signal, r=2, n=0)
plt.figure(figsize=(14, 5))
pxx, freqs, bins, im = plt.specgram(decimated_signal, NFFT=64, Fs=5000, noverlap=0)


# In[204]:


# Пример сигнала
sample_rate = 100
number_of_samples = 500
t = np.arange(number_of_samples)/sample_rate
t = 2*np.pi*t
x = np.sin(0.5 * t) + 0.3 * np.sin(1.5 * t + 0.1) + 0.2 * np.sin(5 * t) + 0.4 * np.sin(16.3 * t + 0.1)

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.xlim(0,int(x.shape[0]/2))
plt.plot(x)
plt.stem(x, basefmt='C0')

plt.figure(figsize=(14, 5))
plt.stem(np.abs(fft.rfft(x)), basefmt='C0')
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")


# In[206]:


y = cic_decimator(x, r=2, n=3)

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.xlim(0,int(y.shape[0]/2))
plt.plot(y)
plt.stem(y, basefmt='C0')

plt.figure(figsize=(14, 5))
plt.stem(np.abs(fft.rfft(y)), basefmt='C0')
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")


# ### Интерполяция

# In[38]:


# Условный пример интерполяции
samplerate = 64
number_of_samples = 64
t = np.arange(number_of_samples)/samplerate
amplitude = 1
phase = 0

data = amplitude * np.cos(2. * np.pi * 3 * t + phase)

plt.figure(figsize=(14, 5))
plt.title('Signal')
#plt.xlim((0,10))
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.stem(data, basefmt='C0')
plt.plot(data, linestyle='--',color='C0', alpha = 0.5)

up_data = np.array([i if j == 0 else 0 for i in data for j in range(2)])
    
plt.figure(figsize=(14, 5))
plt.title('Signal')
#plt.xlim((0,10))
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.stem(up_data, basefmt='C0')
plt.plot(up_data, linestyle='--',color='C0', alpha = 0.5)


plt.figure(figsize=(14, 5))
plt.stem(np.abs(fft.rfft(data)), basefmt='C0')
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")


plt.figure(figsize=(14, 5))
plt.stem(np.abs(fft.rfft(up_data)), basefmt='C0')
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")


# ###  Интерполяция
# 
# $
# y(n) = 
#  \begin{cases}
#    x(n/R) , n = 0, N, 2N, ...\\
#    0 , n \ne 0, N, 2N, ...
#  \end{cases}
# $, где $R$ -- коэффициент децимации, 
# 
# Учитывая фильтарцию:
# $ y(n) = \sum_{k=0}^{N-1}x(nR-k)h(k) $ ,

# ### CIC-фильтр интерполятор
# ![2.png](attachment:2.png)

# In[79]:


def cic_interpolator(x, r, n):
    """
    Функция cic_interpolator
    CIC-фильтр интерполятор

    Принимает на вход:
    x -- входной сигнал
    r -- коэффициент децимации
    n -- количество стадий
    """
    y = x
    # COMB stages
    y = np.diff(y, n=n, prepend=np.zeros(n), append=np.zeros(n))
    #y = np.diff(y, n=n)
    
    # Interpolator
    y = np.array([i if j == 0 else 0 for i in y for j in range(r)])

    # integrator
    for i in range(n):
        y = np.cumsum(y)

    y = y[r - 1:-n*r+r-1]
    return y


# In[83]:


# Пример сигнала
sample_rate = 100
number_of_samples = 500
t = np.arange(number_of_samples)/sample_rate
t = 2*np.pi*t
x = np.sin(0.5 * t) + 0.3 * np.sin(1.5 * t + 0.1) + 0.2 * np.sin(5 * t) + 0.4 * np.sin(16.3 * t + 0.1)

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.xlim(0,int(x.shape[0]/8))
plt.plot(x)
plt.stem(x, basefmt='C0')

plt.figure(figsize=(14, 5))
plt.stem(np.abs(fft.rfft(x)), basefmt='C0')
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")


# In[86]:


y = cic_interpolator(x, r=2, n=6)

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.xlim(0,int(y.shape[0]/8))
plt.plot(y)
plt.stem(y, basefmt='C0')

plt.figure(figsize=(14, 5))
plt.stem(np.abs(fft.rfft(y)), basefmt='C0')
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")


# In[92]:


# Пример интерполяции чирпа
"""
Функция resample
Реализации децимации сигнала

Принимает на вход:
x -- сигнал 
num -- количество отсчетов в выходном сигнала
window -- окно
"""
resampled_signal = signal.resample(chirp_signal, num=chirp_signal.shape[0]*2, window='hann')
plt.figure(figsize=(14, 5))
pxx, freqs, bins, im = plt.specgram(resampled_signal, NFFT=256, Fs=20000, noverlap=0)


# ### Передискретизация 
# 

# In[109]:


"""
Функция upfirdn
Реализации изменения частоты путем ее повышения, фильтрации и последующей децимации

Принимает на вход:
h -- коэффициенты КИХ-фильтра
x -- сигнал
up -- коэффициент интерполяции, по умолчанию 1
down -- коэффициент децимации, по умолчанию 1

"""
y = signal.upfirdn([1,1,1,1,1,1,1,1],x, up=2, down=3)

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.xlim(0,int(y.shape[0]/8))
plt.plot(y)
plt.stem(y, basefmt='C0')

plt.figure(figsize=(14, 5))
plt.stem(np.abs(fft.rfft(y)), basefmt='C0')
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")


# In[110]:


"""
Функция resample_poly
Реализации изменения частоты путем полифазной фильтрации

Принимает на вход:
x -- сигнал
up -- коэффициент интерполяции
down -- коэффициент децимации
window -- окно, по умолчанию Кайсера с B = 5
"""
y = signal.resample_poly(x, up=2, down=3)

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.xlim(0,int(y.shape[0]/8))
plt.plot(y)
plt.stem(y, basefmt='C0')

plt.figure(figsize=(14, 5))
plt.stem(np.abs(fft.rfft(y)), basefmt='C0')
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")


# ### Продолжение фильтрации 

# In[188]:


order = 5
b = np.ones(order)
a = 1

w, h = signal.freqz(b, a)
h = np.abs(h)

plt.figure(figsize=(14, 5))
plt.title('Frequency response')
plt.xlabel("Frequency")
plt.ylabel("Level")
plt.plot(w, h)

# Пример сигнала
sample_rate = 100
number_of_samples = 500
t = np.arange(number_of_samples)/sample_rate
t = 3*np.pi*t
x = np.sin(0.5 * t) + 5 * np.sin(1.5 * t + 0.1) + 2.2 * np.sin(10.2 * t) + 0.5 * np.sin(15.3 * t + 0.1)
x[np.random.randint(0, x.shape[0], 10)] = max(x) * 1.3


y = signal.lfilter(b,a,x)/order

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(x)
plt.plot(y)


# In[189]:


order = 20
b = np.ones(order)

# Пример сигнала
sample_rate = 100
number_of_samples = 500
t = np.arange(number_of_samples)/sample_rate
t = 3*np.pi*t
x = np.sin(0.5 * t) + 5 * np.sin(1.5 * t + 0.1) + 2.2 * np.sin(10.2 * t) + 0.5 * np.sin(15.3 * t + 0.1)
x[np.random.randint(0, x.shape[0], 10)] = max(x) * 1.3

y = np.convolve(x, b/order, mode='same')       

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(x)
plt.plot(y)


# In[ ]:





# In[ ]:




