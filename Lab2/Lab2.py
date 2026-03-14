import numpy as np
import matplotlib.pyplot as plt
import time

def DFT_slow(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT_recursive(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    
    if N <= 1:
        return x
    
    # Check if N isn't power of 2
    if N & (N - 1) != 0:
        next_pow2 = 1 << N.bit_length()
        x = np.pad(x, (0, next_pow2 - N), 'constant')
        N = next_pow2
    
    even = FFT_recursive(x[0::2])
    odd = FFT_recursive(x[1::2])
    
    # Combine
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([
        even + factor[:N//2] * odd,
        even + factor[N//2:] * odd
    ])

class Signal:
    def __init__(self, np_fun, fs, duration):
        self.np_fun = np_fun
        self.fs = fs
        self.duration = duration
        self.N = int(fs * duration)
        self.t = np.linspace(0, duration, self.N, endpoint=False)
        self.values = np_fun(self.t)

def dirrConvCompareWith_fft(MySignal, myDFT):
    print("\nComparing computation times...")
    
    start_time = time.perf_counter()
    X_fft = np.fft.fft(MySignal.values)
    time_fft = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    X_dft_slow = myDFT(MySignal.values)
    time_dft_slow = time.perf_counter() - start_time
    
    print(f"FFT time: {time_fft:.6f} seconds")
    print(f"myDFT time: {time_dft_slow:.6f} seconds")
    
    freqs = np.fft.fftfreq(MySignal.N, 1/MySignal.fs)
    positive_freq_idx = freqs >= 0
    freqs_pos = freqs[positive_freq_idx]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(2, 2, 1)
    plt.plot(freqs_pos, np.abs(X_fft[positive_freq_idx]) * 2 / MySignal.N)
    plt.title('Spectrum using FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 300])
    
    plt.subplot(2, 2, 2)
    plt.plot(freqs_pos, np.abs(X_dft_slow[np.linspace(0, MySignal.N//2, MySignal.N//2, endpoint = False, dtype = int)]) * 2 / MySignal.N)
    plt.title('Spectrum using myDFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 300])

def verificationConvWith_fft(MySignal, myDFT):

    X_fft = np.fft.fft(MySignal.values)
    X_dft_slow = myDFT(MySignal.values)

    freqs = np.fft.fftfreq(MySignal.N, 1/MySignal.fs)
    positive_freq_idx = freqs >= 0
    freqs_pos = freqs[positive_freq_idx]


    signal_reconstructed = np.fft.ifft(X_fft).real
    
    plt.subplot(2, 2, 3)
    plt.plot(MySignal.t, MySignal.values, 'b-', label='Original signal', linewidth=2)
    plt.plot(MySignal.t, signal_reconstructed, 'r--', label='Reconstructed signal', linewidth=2)
    plt.title('Signal Reconstruction Verification')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n(c) Adding white noise and analyzing...")
    noise = np.random.normal(0, 1, MySignal.N)
    signal_noisy = MySignal.values + noise
    X_noisy = np.fft.fft(signal_noisy)
    signal_reconstructed_noisy = np.fft.ifft(X_noisy).real
    
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 2, 1)
    plt.plot(MySignal.t, MySignal.values, 'b-', linewidth=1.5)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    plt.plot(freqs_pos, np.abs(X_fft[positive_freq_idx]) * 2 / MySignal.N)
    plt.title('Spectrum of Original Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 300])
    
    plt.subplot(3, 2, 3)
    plt.plot(MySignal.t, signal_noisy, 'r-', linewidth=1.5, alpha=0.7)
    plt.title('Noisy Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 4)
    plt.plot(freqs_pos, np.abs(X_noisy[positive_freq_idx]) * 2 / MySignal.N)
    plt.title('Spectrum of Noisy Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 300])
    
    plt.subplot(3, 2, 5)
    plt.plot(MySignal.t, signal_reconstructed_noisy, 'g-', linewidth=1.5)
    plt.title('Reconstructed Signal from Noisy Spectrum')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 6)
    plt.plot(MySignal.t, MySignal.values, 'b-', label='Original', linewidth=2)
    plt.plot(MySignal.t, signal_reconstructed_noisy, 'g--', label='Reconstructed', linewidth=2)
    plt.title('Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def task1():

    def np_fun_TASK1(t):
        return np.cos(2 * np.pi * 50 * t) + np.cos(2 * np.pi * 150 * t)

    Signal_TASK1 = Signal(np_fun_TASK1, 1000, 0.1)
    dirrConvCompareWith_fft(Signal_TASK1, DFT_slow)
    verificationConvWith_fft(Signal_TASK1, DFT_slow)

def task2():
    
    def rectangular_pulse(t, A=2, T=2):
        t_mod = t % T
        return np.where(t_mod < T/2, A, -A)
    
    Signal_TASK2 = Signal(rectangular_pulse, 1000, 4)
    dirrConvCompareWith_fft(Signal_TASK2, DFT_slow)
    verificationConvWith_fft(Signal_TASK2, DFT_slow)


def task3():
    print("\n" + "="*60)
    print("TASK 3: Custom FFT Implementation")
    print("="*60)
    
    def np_fun_TASK3(t):
        return np.cos(2 * np.pi * 50 * t)
    
    Signal_TASK3 = Signal(np_fun_TASK3, 1024, 1)
    dirrConvCompareWith_fft(Signal_TASK3, FFT_recursive)
    verificationConvWith_fft(Signal_TASK3, FFT_recursive)

if __name__ == "__main__":    
    # task1()
    # task2()
    task3()
