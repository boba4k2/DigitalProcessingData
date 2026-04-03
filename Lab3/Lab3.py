import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

class Signal:
    """Класс для представления дискретного сигнала"""
    def __init__(self, np_fun, fs, duration, add_noise=False, noise_level=0.1):
        self.np_fun = np_fun
        self.fs = fs  # частота дискретизации
        self.duration = duration
        self.N = int(fs * duration)
        self.t = np.linspace(0, duration, self.N, endpoint=False)
        self.values = np_fun(self.t)
        
        if add_noise:
            self.noise = noise_level * np.random.normal(0, 1, self.N)
            self.values_noisy = self.values + self.noise
        else:
            self.noise = np.zeros(self.N)
            self.values_noisy = self.values

class ButterworthFilter:    
    def __init__(self, cutoff_freq, fs, order=2, filter_type='low'):
        self.cutoff = cutoff_freq
        self.fs = fs
        self.order = order
        self.filter_type = filter_type
        
        # Обработка частоты среза в зависимости от типа фильтра
        if filter_type in ['low', 'high']:
            self.wc = 2 * np.pi * float(cutoff_freq)
            self.wc_low = self.wc_high = self.wc
        elif filter_type in ['band', 'notch']:
            self.wc_low = 2 * np.pi * float(cutoff_freq[0])
            self.wc_high = 2 * np.pi * float(cutoff_freq[1])
            self.wc = (self.wc_low + self.wc_high) / 2 
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Вычисляем полюса для нормированного фильтра
        self.poles = self._compute_poles(order)
    
    def _compute_poles(self, order):
        poles = []
        for m in range(2 * order):
            angle = np.pi * ((2 * m + 1)/(2 * order) - 1 / 2)
            pole = np.exp(1j * angle)
            if np.real(pole) < 0: 
                poles.append(pole)
        return np.array(poles)
    
    def _h_lpf(self, w, wc):
        """АЧХ ФНЧ Баттерворта"""
        Hn = 1.0
        Omega = w / wc
        for pole in self.poles:
            Hn = Hn / (1j*(Omega) - pole)
        return Hn
    
    def _h_hpf(self, w, wc):
        """АЧХ ФВЧ Баттерворта"""
        Hn = 1.0

        w[np.where(w == 0)] = 0.1
        Omega = wc / w
        
        for pole in self.poles:
            Hn = Hn / (1j*(Omega) - pole)
        return Hn
    
    def get_frequency_response(self, frequencies):
        """Получение частотной характеристики фильтра"""
        w = 2 * np.pi * np.array(frequencies)
        
        if self.filter_type == 'low':
            return self._h_lpf(w, self.wc)
        elif self.filter_type == 'high':
            return self._h_hpf(w, self.wc)
        elif self.filter_type == 'band':
            h_lpf = self._h_lpf(w, self.wc_high)
            h_hpf = self._h_hpf(w, self.wc_low)
            return h_lpf * h_hpf
        elif self.filter_type == 'notch':
            h_lpf = self._h_lpf(w, self.wc_low)
            h_hpf = self._h_hpf(w, self.wc_high)
            return h_lpf + h_hpf
        else:
            return np.ones_like(w)
    
    def apply_filter(self, signal_values):
        """Применение фильтра к сигналу через БПФ"""
        X = np.fft.fft(signal_values)
        freqs = np.fft.fftfreq(len(signal_values), 1/self.fs)
        
        H = self.get_frequency_response(np.abs(freqs))
        
        Y = X * np.abs(H)
        
        # Обратное преобразование
        return np.fft.ifft(Y).real

class ChebyshevFilter:
    """Класс для реализации фильтра Чебышева I рода"""
    
    def __init__(self, cutoff_freq, fs, order=3):
        self.fs = fs
        self.cutoff = cutoff_freq
        self.order = order
        self.unev_transm = 0.5

    def _compute_polinomsChebyshev(self, x):
        previousT = [1, x]
        assert (self.order > 0)
        if (self.order<=1):
            return previousT[self.order]
        for i in range(self.order-1):
            T = 2*x*previousT[1] - previousT[0]
            previousT[0] = previousT[1]
            previousT[1] = T

        return T
    
    def get_frequency_response(self, freqs):
        Tn = self._compute_polinomsChebyshev(freqs/self.cutoff)
        return np.power((Tn*self.unev_transm)**2 + 1, -1/2)
    
    def apply_filter(self, signal_values):
        X = np.fft.fft(signal_values)
        freqs = np.fft.fftfreq(len(signal_values), 1/self.fs)

        H = self.get_frequency_response(np.abs(freqs))
        Y = X * np.abs(H)

        return np.fft.ifft(Y).real


def plot_spectrum(signal_obj, title='Spectrum', xlim=None):
    """Построение спектра сигнала"""
    X = np.fft.fft(signal_obj.values)
    freqs = np.fft.fftfreq(signal_obj.N, 1/signal_obj.fs)
    
    pos_idx = freqs >= 0
    freqs_pos = freqs[pos_idx]
    spectrum = np.abs(X[pos_idx]) * 2 / signal_obj.N
    
    plt.figure(figsize=(8, 4))
    plt.plot(freqs_pos, spectrum, linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.show()

def task1(sig):
    plot_spectrum(sig, 'Spectrum of signal with frequencies 50, 150, 450 Hz', [0, 500])
    return

def task2(sig, cutoff_freq=100):
    lpf = ButterworthFilter(cutoff_freq, sig.fs, order=2, filter_type='low')

    filtered = lpf.apply_filter(sig.values)
    
    freqs_plot = np.linspace(0, 500, 1000)
    H = lpf.get_frequency_response(freqs_plot)
    H_mag = np.abs(H)
    
    Y = np.fft.fft(filtered)
    freqs = np.fft.fftfreq(sig.N, 1/sig.fs)
    pos_idx = freqs >= 0
    spectrum_filtered = np.abs(Y[pos_idx]) * 2 / sig.N
    spectrum_orig = np.abs(np.fft.fft(sig.values)[pos_idx]) * 2 / sig.N
    freqs_pos = freqs[pos_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(sig.t[:200], sig.values[:200], 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Initial Signal')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(freqs_pos, spectrum_orig, 'b-', alpha=0.5)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Spectrum of the initial signal')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 500])
    
    axes[1, 0].plot(sig.t[:200], filtered[:200], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Filtred in function')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(freqs_pos, spectrum_filtered, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Spectrum after filtration')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 500])
    
    plt.tight_layout()
    plt.show()
    return

def task3(sig, cutoff_freq=100):
    hpf = ButterworthFilter(cutoff_freq, sig.fs, order=2, filter_type='high')

    filtered = hpf.apply_filter(sig.values)
    
    freqs_plot = np.linspace(0, 500, 1000)
    H = hpf.get_frequency_response(freqs_plot)
    H_mag = np.abs(H)
    
    Y = np.fft.fft(filtered)
    freqs = np.fft.fftfreq(sig.N, 1/sig.fs)
    pos_idx = freqs >= 0
    spectrum_filtered = np.abs(Y[pos_idx]) * 2 / sig.N
    spectrum_orig = np.abs(np.fft.fft(sig.values)[pos_idx]) * 2 / sig.N
    freqs_pos = freqs[pos_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(sig.t[:200], sig.values[:200], 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Initial Signal')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(freqs_pos, spectrum_orig, 'b-', alpha=0.5)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Spectrum of the initial signal')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 500])
    
    axes[1, 0].plot(sig.t[:200], filtered[:200], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Filtred in function')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(freqs_pos, spectrum_filtered, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Spectrum after filtration')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 500])
    
    plt.tight_layout()
    plt.show()
    return

def task4(sig):
    w_low, w_high = 100, 200  # Гц

    bpf = ButterworthFilter([w_low, w_high], sig.fs, order=2, filter_type='band')
    notch = ButterworthFilter([w_low, w_high], sig.fs, order=2, filter_type='notch')
    
    freqs_plot = np.linspace(0, 500, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    H_bp = bpf.get_frequency_response(freqs_plot)

    axes[0, 0].plot(freqs_plot, np.abs(H_bp), 'b-', linewidth=2)
    axes[0, 0].axvline(150, color='red', linestyle='--', label='Center: 150 Hz')
    axes[0, 0].axvspan(w_low, w_high, alpha=0.1, color='blue', label='Passband')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('|H(f)|')
    axes[0, 0].set_title('Band-pass Filter Response')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    
    H_notch = notch.get_frequency_response(freqs_plot)
    axes[0, 1].plot(freqs_plot, np.abs(H_notch), 'r-', linewidth=2)
    axes[0, 1].axvline(150, color='blue', linestyle='--', label='Notch: 150 Hz')
    axes[0, 1].axvspan(w_low, w_high, alpha=0.1, color='red', label='Stopband')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('|H(f)|')
    axes[0, 1].set_title('Notch Filter Response')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    


    filtered_bp = bpf.apply_filter(sig.values)

    Y_bp = np.fft.fft(filtered_bp)
    freqs = np.fft.fftfreq(sig.N, 1/sig.fs)
    pos_idx = freqs >= 0
    spectrum_bp = np.abs(Y_bp[pos_idx]) * 2 / sig.N
    freqs_pos = freqs[pos_idx]
    
    axes[1, 0].plot(freqs_pos, spectrum_bp, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Spectrum after Band-pass Filter')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 500])
    


    filtered_notch = notch.apply_filter(sig.values)
    Y_notch = np.fft.fft(filtered_notch)
    spectrum_notch = np.abs(Y_notch[pos_idx]) * 2 / sig.N
    
    axes[1, 1].plot(freqs_pos, spectrum_notch, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Spectrum after Notch Filter')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 500])
    
    plt.tight_layout()
    plt.show()

    return

def task5_6(sig):
    cutoff = 100
    orders = [2, 4, 5]
    
    freqs_plot = np.linspace(0, 500, 1000)
    
    plt.figure(figsize=(10, 5))
    
    for order in orders:
        bf = ButterworthFilter(cutoff, sig.fs, order=order, filter_type='low')
        H = bf.get_frequency_response(freqs_plot)
        plt.plot(freqs_plot, np.abs(H), 
                label=f'Order {order}', linewidth=2)
    
    plt.axvline(cutoff, color='black', linestyle='--', label=f'Cutoff: {cutoff} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|H|')
    plt.title('Butterworth LPF: Comparison of Different Orders')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 300])
    plt.tight_layout()
    plt.show()

    bf5 = ButterworthFilter(cutoff, sig.fs, order=5, filter_type='low')
    filtered_5th = bf5.apply_filter(sig.values)
    
    bf2 = ButterworthFilter(cutoff, sig.fs, order=2, filter_type='low')
    filtered_2nd = bf2.apply_filter(sig.values)
    
    plt.figure(figsize=(10, 4))
    plt.plot(sig.t[:200], sig.values[:200], 'k-', label='Original', alpha=0.5)
    plt.plot(sig.t[:200], filtered_2nd[:200], 'b--', label='2nd order', linewidth=1.5)
    plt.plot(sig.t[:200], filtered_5th[:200], 'r-', label='5th order', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Filtered Signals: 2nd vs 5th Order Butterworth LPF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return bf5, filtered_5th

def task7(sig, cutoff=100):
    order = 5
    
    my_filter = ButterworthFilter(cutoff, sig.fs, order=order, filter_type='low')
    filtered_my = my_filter.apply_filter(sig.values)

    b, a = signal.butter(order, cutoff, fs=sig.fs, btype='low')
    filtered_scipy = signal.filtfilt(b, a, sig.values)  # filtfilt для нулевой фазы

    freqs_plot = np.linspace(0, 500, 1000)
    H_my = my_filter.get_frequency_response(freqs_plot)
    w_scipy, H_scipy = signal.freqz(b, a, worN=freqs_plot, fs=sig.fs)
    
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_plot, np.abs(H_my), 'b-', label='My Implementation', linewidth=2)
    plt.plot(w_scipy, np.abs(H_scipy), 'r--', label='scipy.signal.butter', linewidth=2)
    plt.axvline(cutoff, color='black', linestyle=':', label=f'Cutoff: {cutoff} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|H|')
    plt.title(f'Butterworth LPF {order}th Order: My Implementation vs scipy.signal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 300])
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 4))
    plt.plot(sig.t[:200], sig.values[:200], 'k-', label='Original', alpha=0.4)
    plt.plot(sig.t[:200], filtered_my[:200], 'b-', label='My Filter', linewidth=1.5)
    plt.plot(sig.t[:200], filtered_scipy[:200], 'r--', label='scipy Filter', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def task8(sig, cutoff=100):
    # Создаём зашумленный сигнал
    sig_noisy = Signal(sig.np_fun, sig.fs, sig.duration, add_noise=True, noise_level=0.3)
    
    lpf = ButterworthFilter(cutoff, sig.fs, order=5, filter_type='low')
    filtered_noisy = lpf.apply_filter(sig_noisy.values_noisy)
    
    def get_spectrum(values, fs, N):
        X = np.fft.fft(values)
        freqs = np.fft.fftfreq(N, 1/fs)
        pos_idx = freqs >= 0
        return freqs[pos_idx], np.abs(X[pos_idx]) * 2 / N
    
    freqs_pos, spec_orig = get_spectrum(sig.values, sig.fs, sig.N)
    freqs_pos, spec_noisy = get_spectrum(sig_noisy.values_noisy, sig.fs, sig.N)
    freqs_pos, spec_filtered = get_spectrum(filtered_noisy, sig.fs, sig.N)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(sig.t[:200], sig.values[:200], 'b-', label='Clean', linewidth=1.5)
    axes[0].plot(sig.t[:200], sig_noisy.values_noisy[:200], 'r-', label='Noisy', alpha=0.7)
    axes[0].plot(sig.t[:200], filtered_noisy[:200], 'g-', label='Filtered', linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain: Noisy Signal Filtering')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(freqs_pos, spec_orig, 'b-', label='Clean', linewidth=1.5, alpha=0.7)
    axes[1].plot(freqs_pos, spec_noisy, 'r-', label='Noisy', linewidth=1, alpha=0.5)
    axes[1].plot(freqs_pos, spec_filtered, 'g-', label='Filtered', linewidth=2)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Spectrum: Noise Reduction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 500])
    
    plt.tight_layout()
    plt.show()

def task9(sig):

    cutoff = 100
    orders = [3, 5]
    
    freqs_plot = np.linspace(0, 500, 1000)
    
    plt.figure(figsize=(10, 5))
    
    # Сравнение с Баттервортом
    bf = ButterworthFilter(cutoff, sig.fs, order=5, filter_type='low')
    H_butter = bf.get_frequency_response(freqs_plot)
    plt.plot(freqs_plot, np.abs(H_butter), 
            'b-', label='Butterworth 5th', linewidth=2)
    
    for order in orders:
        cf = ChebyshevFilter(cutoff, sig.fs, order)
        H_cheb = cf.get_frequency_response(freqs_plot)
        plt.plot(freqs_plot, np.abs(H_cheb), 
                label=f'Chebyshev {order}th', linewidth=2)
    
    plt.axvline(cutoff, color='black', linestyle='--', label=f'Cutoff: {cutoff} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Chebyshev Type I vs Butterworth LPF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 300])
    plt.tight_layout()
    plt.show()
    
    # Применение фильтра Чебышева
    cf = ChebyshevFilter(cutoff, sig.fs, order=5)
    filtered_cheb = cf.apply_filter(sig.values)

    filtered_butter = bf.apply_filter(sig.values)
    
    plt.figure(figsize=(10, 4))
    plt.plot(sig.t[:200], sig.values[:200], 'k-', label='Original', alpha=0.4)
    plt.plot(sig.t[:200], filtered_butter[:200], 'b--', label='Butterworth', linewidth=1.5)
    plt.plot(sig.t[:200], filtered_cheb[:200], 'r-', label='Chebyshev', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain: Butterworth vs Chebyshev')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_signal():
    def signal_func(t):
        return (np.cos(2*np.pi*50*t) + 
                np.cos(2*np.pi*150*t) + 
                np.cos(2*np.pi*450*t))
    
    sig = Signal(signal_func, fs=2000, duration=0.1)
    return sig

def main():    
    sig = get_signal()

    task1(sig)

    task2(sig, cutoff_freq=100)
    
    task3(sig, cutoff_freq = 100)

    task4(sig)

    task5_6(sig)
    
    task7(sig)

    task8(sig)

    task9(sig)


if __name__ == "__main__":
    main()
