
# coding: utf-8

# ### Постановка задачи
# ![0.png](attachment:0.png)

# ### Формулировка требований
# Фильтр нижних частот
# ![1.png](attachment:1.png)
# 
# * $A_{pass}$ -- неравномерность в полосе пропускания;
# * $A_{stop}$ -- уровень затухания в полосе подавления;
# * $f_{pass}$ -- граничная частота полосы пропускания;
# * $f_{stop}$ -- граничная частота полосы затухания.

# #### FIR (КИХ)-фильтры
# Фильтр с конечной импульсной характеристикой:
# $$y(n) = \sum_{k=L}^N h(k)x(n-k)$$
# Передаточная функция:
# $$H(w) = \sum_{k=L}^N h(k) \exp ^{(-2\pi ikw)}$$
# 

# In[1]:


import numpy as np
from numpy import fft
from scipy import signal 
import matplotlib.pyplot as plt


# In[2]:


# Пример сигнала
sample_rate = 100
number_of_samples = 500
t = np.arange(number_of_samples)/sample_rate
t = 2*np.pi*t
x = np.sin(0.5 * t) + 0.3 * np.sin(1.5 * t + 0.1) + 0.2 * np.sin(5 * t) + 0.4 * np.sin(16.3 * t + 0.1) + 0.4 * np.sin(20 * t + 0.8) + 0.2 * np.sin(21.5 * t + 0.8) + 0.3 * np.sin(24 * t + 0.2) + 0.2 * np.sin(25.3 * t + 0.3)

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(x)

plt.figure(figsize=(14, 5))
plt.stem(np.abs(fft.rfft(x)), basefmt='C0')
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")


# In[5]:


"""
Функция firwin
Реализация КИХ-фильтра N-го порядка оконным методом.

Принимает на вход:
numtaps -- количество коэфициентов фильтра,
           желательно использовать нечетное количество коээффициентов, для сохранения нулевого отклика на частоте Наквиста;
cutoff -- частота среза относительно частоты дискретизации, если фильтр полосовой, 
          то монотонно увеличивающаяся последовательность не включая 0 и частоту Найквиста;
width -- (необязательный параметр) примерная ширина перехода от полосы пропускания до полосы затухания;
window -- тип окна, по умолчанию окно Хэмминга
pass_zero -- Определение типа фильтра True, False, ‘bandpass’, ‘lowpass’, ‘highpass’, ‘bandstop’.
fs -- по умолчанию 2, можно установить частоту дискретизации и рабоать с реальными частотами, в противном случае 
      масштабировать значения частоты отсечения от 0 до 1
"""
number_of_coefficients = 51
cutoff = 10
b = signal.firwin(numtaps=number_of_coefficients, cutoff=cutoff, pass_zero='lowpass', fs=sample_rate)

"""
Функция freqz
Рассчет частотного отклика фильтра.

Принимает на вход 
b -- коэффициенты числитела
a -- коэффициенты знаменателя, по умолчанию 1 (для КИХ-фильтров)
fs -- частота дискретизации
"""

w, h = signal.freqz(b, fs=sample_rate)
p = np.unwrap(np.angle(h))
h = np.abs(h)
log_h = 20*np.log10(h/np.max(h) + 10**(-15))

plt.figure(figsize=(14, 5))
plt.title('Impulse responce')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(b)
plt.show()

plt.figure(figsize=(14, 5))
plt.title('Frequency response')
plt.xlabel("Frequency")
plt.ylabel("Level")
plt.plot(w,h)

# Upper inset plot.
ax1 = plt.axes([0.42, 0.6, .45, .25])
plt.plot(w, h)
plt.xlim(0, 10)
plt.ylim(0.995, 1.005)

# Lower inset plot
ax2 = plt.axes([0.42, 0.25, .45, .25])
plt.plot(w, h)
plt.xlim(10.0, 20.0)
plt.ylim(0, 0.005)
plt.show()

plt.figure(figsize=(14, 5))
plt.title('Frequency response')
plt.xlabel("Frequency")
plt.ylabel("Level, dB")
plt.plot(w,log_h)
plt.show()

plt.figure(figsize=(14, 5))
plt.title('Phase response')
plt.xlabel("Frequency")
plt.ylabel("Angle")
plt.plot(w,p)
plt.show()


# In[6]:


"""
Фильтрация сигнала
Функци lfilter

Принимает на вход:
    b -- коэффициенты числителя
    a -- коэффициенты знаменателя, для КИХ-фильтров = 1
    x -- сигнал для фильтрации

"""
y = signal.lfilter(b, 1, x)

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(y)
plt.show()

plt.figure(figsize=(14, 5))
plt.stem(np.abs(fft.rfft(y)))
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")
plt.show()


# In[7]:


plt.figure(figsize=(14, 5))
plt.title('Signal vs Filtered Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(x, alpha = 0.5)
plt.plot(y, linewidth=2)

delay = int(0.5 * (number_of_coefficients-1))
"""
Получение групповой задержки фазы для каждой из частот
"""
w, group_delay = signal.group_delay((b, 1))

print('delay', delay)
print('group_delay', int(group_delay.mean()))

plt.plot(y[delay:],   linewidth=3)
plt.show()


# In[8]:


"""
Устранение фазовой задержки с помощью двойной фильтрации
Функция filtfilt

Идея двойной фильтрации заключается в фильтрации сигнала справа налево, а потом наоборот. 
Также, данная функция позволяет обработать крайние значения сигнала 

Принимает на вход:
    b -- коэффициенты числителя
    a -- коэффициенты знаменателя, для КИХ-фильтров = 1
    x -- сигнал для фильтрации

"""
y_1 = signal.filtfilt(b,1,x)

plt.figure(figsize=(14, 5))
plt.title('Signal vs Filtered Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(x, alpha=0.5)
plt.plot(y[delay:])
plt.plot(y_1,  linewidth=3)
plt.show()


# In[9]:


# Полосовые фильтры и другие методы рассчета
number_of_coefficients = 151
b_firwin = signal.firwin(numtaps=number_of_coefficients, cutoff=[15,17], pass_zero='bandstop', fs=sample_rate)
"""
Примеры ниже условные. Разные фильтры требуют разных настроек. 
"""


"""
Фукнция firwin2
Задается набор частот и амплитуд на этих частотах.

Принимает на вход:
numtaps -- количество коэфициентов фильтра,
           желательно использовать нечетное количество коээффициентов, для сохранения нулевого отклика на частоте Наквиста;
freq -- Неубывающая последовательность частот. Последовательность начинается с 0 и заканчивается частотой Найквиста
gain - Сопоставляет усиление или ослабление для частот из freq.
fs -- по умолчанию 2, можно установить частоту дискретизации и рабоать с реальными частотами, в противном случае 
      масштабировать значения частоты отсечения от 0 до 1
"""
b_firwin2 = signal.firwin2(numtaps=number_of_coefficients, freq=[0, 14, 15, 17, 18, 50], gain=[1, 1, 0, 0, 1, 1], fs=sample_rate)
"""
Фукнция firls
Рассчет КИХ-фильтра методом наименьших квадратов

Принимает на вход:
numtaps -- количество коэфициентов фильтра,
           желательно использовать нечетное количество коээффициентов, для сохранения нулевого отклика на частоте Наквиста;
bands -- Неубывающая последовательность частот. Последовательность начинается с 0 и заканчивается частотой Найквиста
desired - Сопоставляет усиление или ослабление для частот из bands.
fs -- по умолчанию 2, можно установить частоту дискретизации и рабоать с реальными частотами, в противном случае 
      масштабировать значения частоты отсечения от 0 до 1
"""
b_firls = signal.firls(numtaps=number_of_coefficients, bands=[0, 14, 15, 17, 18, 50], desired=[1, 1, 0, 0, 1, 1], fs=sample_rate)
"""
Фукнция remez
Рассчет КИХ-фильтра методом Ремеза

Принимает на вход:
numtaps -- количество коэфициентов фильтра,
           желательно использовать нечетное количество коээффициентов, для сохранения нулевого отклика на частоте Наквиста;
bands -- Неубывающая последовательность частот. Последовательность начинается с 0 и заканчивается частотой Найквиста
desired - Сопоставляет усиление или ослабление для частот из bands, длина вдвое меньше чем bands.
fs -- по умолчанию 2, можно установить частоту дискретизации и рабоать с реальными частотами, в противном случае 
      масштабировать значения частоты отсечения от 0 до 1
"""

b_remez = signal.remez(numtaps=number_of_coefficients, bands=[0, 14, 15, 17, 18, 50], desired=[1, 0, 1], fs=sample_rate)

list_of_coefficients = [b_firwin, b_firwin2, b_firls, b_remez]
list_of_filter_names = ['firwin', 'firwin2', 'firls', 'remez']

plt.figure(figsize=(14, 5))
fig, ax = plt.subplots(2, figsize=(14,10))
for i in range(4):
    w, h = signal.freqz(list_of_coefficients[i], fs=sample_rate)
    p = np.unwrap(np.angle(h))
    h = np.abs(h)
    log_h = 20*np.log10(h/np.max(h) + 10**(-15))

    ax[0].set_title('Frequency response')
    ax[0].set_xlabel("Frequency")
    ax[0].set_ylabel("Level")
    ax[0].plot(w, h, label=list_of_filter_names[i])
    ax[0].legend()

    ax[1].set_title('Frequency response')
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Level, dB")
    ax[1].plot(w, log_h, label=list_of_filter_names[i])
    ax[1].legend()    
plt.show()


# In[10]:


# Фильтрация сигнала
y_firwin = signal.filtfilt(b_firwin,1,x)
y_firwin2 = signal.filtfilt(b_firwin2,1,x)
y_firls = signal.filtfilt(b_firls,1,x)
y_remez = signal.filtfilt(b_remez,1,x)


list_of_filtered_signals = [y_firwin, y_firwin2, y_firls, y_remez]
plt.figure(figsize=(14, 5))
for i in range(4):
    plt.plot(np.abs(fft.rfft(list_of_filtered_signals[i])), label=list_of_filter_names[i])
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")
plt.legend()
plt.show()


# In[11]:


# Полосовые фильтры
number_of_coefficients = 151
b_firwin = signal.firwin(numtaps=number_of_coefficients, cutoff=[19,27], pass_zero='bandpass', fs=sample_rate)
b_firwin2 = signal.firwin2(numtaps=number_of_coefficients, freq=[0, 18, 19, 27, 28, 50], gain=[0, 0, 1, 1, 0, 0], fs=sample_rate)
b_firls = signal.firls(numtaps=number_of_coefficients, bands=[0, 18, 19, 27, 28, 50], desired=[0, 0, 1, 1, 0, 0], fs=sample_rate)
b_remez = signal.remez(numtaps=number_of_coefficients, bands=[0, 18, 19, 27, 28, 50], desired=[0, 1, 0], fs=sample_rate)
list_of_coefficients = [b_firwin, b_firwin2, b_firls, b_remez]
list_of_filter_names = ['firwin', 'firwin2', 'firls', 'remez']

plt.figure(figsize=(14, 5))
fig, ax = plt.subplots(2, figsize=(14,10))
for i in range(4):
    w, h = signal.freqz(list_of_coefficients[i], fs=sample_rate)
    p = np.unwrap(np.angle(h))
    h = np.abs(h)
    log_h = 20*np.log10(h/np.max(h) + 10**(-15))

    ax[0].set_title('Frequency response')
    ax[0].set_xlabel("Frequency")
    ax[0].set_ylabel("Level")
    ax[0].plot(w, h, label=list_of_filter_names[i])
    ax[0].legend()

    ax[1].set_title('Frequency response')
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Level, dB")
    ax[1].plot(w, log_h, label=list_of_filter_names[i])
    ax[1].legend()   
plt.show()


# In[12]:


# Фильтрация сигнала
y_firwin = signal.filtfilt(b_firwin,1,x)
y_firwin2 = signal.filtfilt(b_firwin2,1,x)
y_firls = signal.filtfilt(b_firls,1,x)
y_remez = signal.filtfilt(b_remez,1,x)


list_of_filtered_signals = [y_firwin, y_firwin2, y_firls, y_remez]
plt.figure(figsize=(14, 5))
for i in range(4):
    plt.plot(np.abs(fft.rfft(list_of_filtered_signals[i])), label=list_of_filter_names[i])
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")
plt.legend()
plt.show()


# In[13]:


"""
Использование разных видов окон
Функция signal.get_window()

"""
number_of_coefficients = 51
cutoff = 10
b_boxcar = signal.firwin(numtaps=number_of_coefficients,cutoff=cutoff,window='boxcar', pass_zero='lowpass', fs=sample_rate)
b_hann = signal.firwin(numtaps=number_of_coefficients,   cutoff=cutoff,window='hann', pass_zero='lowpass', fs=sample_rate)
b_kaiser = signal.firwin(numtaps=number_of_coefficients,   cutoff=cutoff,window=('kaiser', 9), pass_zero='lowpass', fs=sample_rate)
boxcar = signal.get_window('boxcar', 51)
hann = signal.get_window('hann', 51)
kaiser = signal.get_window(('kaiser', 9), 51)

list_of_windows = [boxcar, hann, kaiser]
list_of_coefficients = [b_boxcar, b_hann, b_kaiser]
list_of_window_names = ['boxcar', 'hann', 'kaiser']

fig, ax = plt.subplots(3, figsize=(14,15))
for i in range(3):
    w, h = signal.freqz(list_of_coefficients[i], fs=sample_rate)
    p = np.unwrap(np.angle(h))
    h = np.abs(h)
    log_h = 20*np.log10(h/np.max(h) + 10**(-15))

    ax[0].set_title('Window')
    ax[0].set_xlabel("Sample")
    ax[0].set_ylabel("Amplitude")
    ax[0].plot(list_of_windows[i], label=list_of_window_names[i])
    ax[0].legend()
    
    ax[1].set_title('Frequency response')
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Level")
    ax[1].plot(w, h, label=list_of_window_names[i])
    ax[1].legend()

    ax[2].set_title('Frequency response')
    ax[2].set_xlabel("Frequency")
    ax[2].set_ylabel("Level, dB")
    ax[2].plot(w, log_h, label=list_of_window_names[i])
    ax[2].legend()    
plt.show()


# #### IIR (БИХ)-фильтры
# Фильтр определяется как:
# $$y(n)=-\sum_{k=1}^M a_k y(n-k)+\sum_{k=L}^N b_k x(n-k), n > n_0$$, где $y(n) = 0$ для все $n<= n_0$.
# Передаточная функция фильтра:
# $$H(w)=\frac{\sum_{k=L}^N b_k exp ^{(-2\pi ikw)}}{1 + \sum_{k=1}^M a_k exp ^{(-2\pi ikw)}}$$

# In[14]:


"""
Функция iirdesign
Реализация БИХ-фильтра минимального порядка для входных данных. 
Принимает на вход:
wp, ws -- граничные частоты полосы пропускания и затухания относительно частоты Найквиста (0,1) или реальные значения частоты;
gpass -- максимальная неравномерность в полосе пропускания в дБ;
gstop -- минимальное затухание в полосе подавления в дБ;
ftype -- тип реализации фильтра, по умолчанию ellip -- эллиптический фильтр Кауэра. Доступные варианты:
        Баттерворта : ‘butter’
        фильтр Чебышева 1 типа : ‘cheby1’
        фильтр Чебышева 2 типа : ‘cheby2’
        фильтр Кауэра : ‘ellip’
        фильтр Бесселя: ‘bessel’;
fs -- частота дискретизации.
"""
delta_function = signal.unit_impulse(number_of_samples)

b, a = signal.iirdesign(wp=10, ws=11, gpass=0.1, gstop=60, fs=sample_rate)
number_of_coefficients = a.shape[0]
print('Filter order:', number_of_coefficients)

w, h = signal.freqz(b, a, fs=sample_rate)
p = np.unwrap(np.angle(h))
h = np.abs(h)
log_h = 20*np.log10(h/np.max(h) + 10**(-15))

# Моделируем прохождение дельта-функции через фильтр
impulse_responce = signal.lfilter(b, a, delta_function)

plt.figure(figsize=(14, 5))
plt.title('Impulse responce')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(impulse_responce)
plt.show()

plt.figure(figsize=(14, 5))
plt.title('Frequency response')
plt.xlabel("Frequency")
plt.ylabel("Level")
plt.plot(w,h)

# Upper inset plot.
ax1 = plt.axes([0.42, 0.6, .45, .25])
plt.plot(w, h)
plt.xlim(0,10)
plt.ylim(0.98, 1.005)

# Lower inset plot
ax2 = plt.axes([0.42, 0.25, .45, .25])
plt.plot(w, h)
plt.xlim(10.0, 20.0)
plt.ylim(0, 0.005)
plt.show()

plt.figure(figsize=(14, 5))
plt.title('Frequency response')
plt.xlabel("Frequency")
plt.ylabel("Level, dB")
plt.plot(w,log_h)
plt.show()

plt.figure(figsize=(14, 5))
plt.title('Phase response')
plt.xlabel("Frequency")
plt.ylabel("Angle")
plt.plot(w,p)
plt.show()


# In[15]:


# Фильтрация сигнала
y = signal.lfilter(b, a, x)

plt.figure(figsize=(14, 5))
plt.title('Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(y)
plt.show()

plt.figure(figsize=(14, 5))
plt.stem(np.abs(fft.rfft(y)))
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")
plt.show()


# In[16]:


plt.figure(figsize=(14, 5))
plt.title('Signal vs Filtered Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(x, alpha = 0.5)
plt.plot(y, linewidth=2)

delay = int(0.5 * (number_of_coefficients-1))
"""
Получение групповой задержки фазы для каждой из частот
"""
w, group_delay = signal.group_delay((b, a))
print('delay', delay)
print('group_delay', int(group_delay.mean()))

plt.plot(y[delay:],   linewidth=3)
plt.show()


# In[17]:


y_1 = signal.filtfilt(b,a,x)

plt.figure(figsize=(14, 5))
plt.title('Signal vs Filtered Signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(x, alpha=0.5)
plt.plot(y_1,  linewidth=3)
plt.show()


# In[18]:


# Другие функции реализации БИХ-фильтров
"""
Функция iirfilter

Реализация БИХ-фильтра N-го порядка

Принимает на вход:
N -- порядок фильтра;
Wn -- частота среза или края дипазона частот;
rp -- максимальная неравномерность в полосе пропускания в дБ (только для cheby1 cheby2 и ellip); 
rs -- минимальное затухание в полосе подавления в дБ (только для cheby1 cheby2 и ellip);
btype -- тип фильтра 'bandpass', 'lowpass', 'highpass', 'bandstop';
ftype -- тип реализации фильтра, по умолчанию -- butter фильтр Баттерворта.  Доступные варианты:
        Баттерворта : ‘butter’
        фильтр Чебышева 1 типа : ‘cheby1’
        фильтр Чебышева 2 типа : ‘cheby2’
        фильтр Кауэра : ‘ellip’
        фильтр Бесселя: ‘bessel’
"""
b_iirfilter, a_iirfilter = signal.iirfilter(10, Wn=10, rp=0.1, rs=60, btype='lowpass', ftype='ellip', fs=sample_rate)

"""
Функция butter

Реализация БИХ-фильтра Баттерворта N-го порядка.
Фильтр Баттерворта имеет максимально плоскую АЧХ в полосе пропускания.

N -- порядок фильтра;
Wn -- частота среза или края дипазона частот;
btype -- тип фильтра 'bandpass', 'lowpass', 'highpass', 'bandstop';
fs -- частота дискретизации

Существует вспомогательная функция buttord, которая минимальный порядок фильтра по входным данным

Принимает на вход:
wp, ws -- граничные частоты полосы пропускания и затухания относительно частоты Найквиста (0,1) или реальные значения частоты;
gpass -- максимальная неравномерность в полосе пропускания в дБ;
gstop -- минимальное затухание в полосе подавления в дБ;
fs -- частота дискретизации
"""
b_butter, a_butter = signal.butter(10, Wn=10, btype='lowpass', fs=sample_rate) 

n, wn = signal.buttord(wp=10, ws=11, gpass=0.1, gstop=10, fs=sample_rate)
print('Lowest Butter order:', n, wn)
b_buttord, a_buttord = signal.butter(n, Wn=wn, btype='lowpass', fs=sample_rate) 

"""
Функция cheby1

Реализация БИХ-фильтра Чебышева N-го порядка.
Фильтр крутой спад АЧХ за счет пульсаций на частотах полос пропускания.

N -- порядок фильтра;
rp -- максимальная неравномерность в полосе пропускания в дБ;
Wn -- частота среза или края дипазона частот;
btype -- тип фильтра 'bandpass', 'lowpass', 'highpass', 'bandstop';
fs -- частота дискретизации

Существует вспомогательная функция cheb1ord, которая минимальный порядок фильтра по входным данным

Принимает на вход:
wp, ws -- граничные частоты полосы пропускания и затухания относительно частоты Найквиста (0,1) или реальные значения частоты;
gpass -- максимальная неравномерность в полосе пропускания в дБ;
gstop -- минимальное затухание в полосе подавления в дБ;
fs -- частота дискретизации
"""
b_cheby1, a_cheby1 = signal.cheby1(10, rp=0.1, Wn=10, btype='lowpass', fs=sample_rate) 

n, wn = signal.cheb1ord(wp=10, ws=11, gpass=0.1, gstop=60, fs=sample_rate)
print('Lowest Cheby1 order:', n, wn)
b_cheby1ord, a_cheby1ord = signal.cheby1(n, rp=0.1, Wn=wn, btype='lowpass', fs = sample_rate) 

"""
Функция cheby2

Реализация БИХ-фильтра Чебышева N-го порядка.
Фильтр крутой спад АЧХ за счет пульсаций на частотах полос затухания.

N -- порядок фильтра;
rs -- максимальная неравномерность в полосе пропускания в дБ;
Wn -- частота среза или края дипазона частот;
btype -- тип фильтра 'bandpass', 'lowpass', 'highpass', 'bandstop';
fs -- частота дискретизации

Существует вспомогательная функция cheb2ord, которая минимальный порядок фильтра по входным данным

Принимает на вход:
wp, ws -- граничные частоты полосы пропускания и затухания относительно частоты Найквиста (0,1) или реальные значения частоты;
gpass -- максимальная неравномерность в полосе пропускания в дБ;
gstop -- минимальное затухание в полосе подавления в дБ;
fs -- частота дискретизации
"""

b_cheby2, a_cheby2 = signal.cheby2(10, rs=60, Wn=10, btype='lowpass', fs=sample_rate) 

n, wn = signal.cheb2ord(wp=10, ws=11, gpass=0.1, gstop=60, fs=sample_rate)
print('Lowest Cheby2 order:', n, wn)
b_cheby2ord, a_cheby2ord = signal.cheby2(n, rs=60, Wn=wn, btype='lowpass', fs = sample_rate) 

list_of_coefficients = [(b_iirfilter, a_iirfilter), 
                        (b_butter, a_butter), 
                        (b_buttord, a_buttord),
                        (b_cheby1, a_cheby1),
                        (b_cheby1ord, a_cheby1ord),
                        (b_cheby2, a_cheby2),
                        (b_cheby2ord, a_cheby2ord)
                       ]
list_of_filter_names = ['iirfilter_ellip', 'butter', 'butter_ord', 'cheby1', 'cheby1_ord', 'cheby2', 'cheby2_ord']

plt.figure(figsize=(14, 5))
fig, ax = plt.subplots(2, figsize=(14,10))
for i in range(len(list_of_filter_names)):
    b = list_of_coefficients[i][0]
    a = list_of_coefficients[i][1]
    w, h = signal.freqz(b, a, fs=sample_rate)
    p = np.unwrap(np.angle(h))
    h = np.abs(h)
    log_h = 20*np.log10(h/np.max(h) + 10**(-15))

    ax[0].set_title('Frequency response')
    ax[0].set_xlabel("Frequency")
    ax[0].set_ylabel("Level")
    ax[0].plot(w, h, label=list_of_filter_names[i])
    ax[0].legend()

    ax[1].set_title('Frequency response')
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Level, dB")
    ax[1].plot(w, log_h, label=list_of_filter_names[i])
    ax[1].legend()    
plt.show()


# In[19]:


# list_of_filter_names = ['iirfilter_ellip', 'butter', 'butter_ord', 'cheby1', 'cheby1_ord', 'cheby2', 'cheby2_ord']
list_of_filtered_signals = []

for coefficients in list_of_coefficients:
    list_of_filtered_signals.append(signal.filtfilt(coefficients[0],coefficients[1],x))
    
plt.figure(figsize=(14, 5))
for i in range(len(list_of_filter_names)):
    plt.plot(np.abs(fft.rfft(list_of_filtered_signals[i])), label=list_of_filter_names[i])
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")
plt.legend()
plt.show()


# In[20]:


"""
Функция iirnotch
Реализация узкополосного режекторого БИХ-фильтра второго порядка

Принимает на вход:
w0 -- частоту отсечения 
Q -- добротность фильтра
fs -- частота дискретизации.

Аналогичный полосовой БИХ-фильтр второго порядка

Функция iirpeak
Реализация узкополосного режекторого БИХ-фильтра второго порядка

Принимает на вход:
w0 -- частоту пропускания 
Q -- добротность фильтра
fs -- частота дискретизации.
"""
b, a = signal.iirnotch(w0=20, Q=20, fs=sample_rate)


delta_function = signal.unit_impulse(number_of_samples)

number_of_coefficients = a.shape[0]
print('Filter order:', number_of_coefficients)

w, h = signal.freqz(b, a, fs=sample_rate)
p = np.unwrap(np.angle(h))
h = np.abs(h)
log_h = 20*np.log10(h/np.max(h) + 10**(-15))

# Моделируем прохождение дельта-функции через фильтр
impulse_responce = signal.lfilter(b, a, delta_function)

plt.figure(figsize=(14, 5))
plt.title('Impulse responce')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.plot(impulse_responce)
plt.show()

plt.figure(figsize=(14, 5))
plt.title('Frequency response')
plt.xlabel("Frequency")
plt.ylabel("Level")
plt.plot(w,h)

plt.figure(figsize=(14, 5))
plt.title('Frequency response')
plt.xlabel("Frequency")
plt.ylabel("Level, dB")
plt.plot(w,log_h)
plt.show()

plt.figure(figsize=(14, 5))
plt.title('Phase response')
plt.xlabel("Frequency")
plt.ylabel("Angle")
plt.plot(w,p)
plt.show()


# In[21]:


# Фильтрация сигнала
y = signal.lfilter(b, a, x)

plt.figure(figsize=(14, 5))
plt.plot(np.abs(fft.rfft(x)))
plt.plot(np.abs(fft.rfft(y)))
plt.title('Spectrum')
plt.ylabel("Level")
plt.xlabel("Frequency")
plt.show()

