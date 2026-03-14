import numpy as np
import matplotlib.pyplot as plt



class Signal:
    def __init__(self, fun, stPoint, endPoint, num_samples, addNoise = False):
        self.fun = fun
        self.stPoint = stPoint
        self.endPoint = endPoint
        self.num_samples = num_samples
        self.xS_values = np.linspace(self.stPoint, self.endPoint, num_samples, endpoint = False)
        self.yS_values = None
        if addNoise:
            self.noise = self._noise(A = 0.1, sigma = 1)
        else:
            self.noise = np.zeros(self.num_samples)

        self._sampling()
        
        self.dx = self.xS_values[1]-self.xS_values[0]

    def _noise(self, A = 1, sigma = 1):
        return A*np.random.normal(0, sigma, self.num_samples)

    def _sampling(self):
        self.yS_values = np.array([self.fun(x) for x in self.xS_values], dtype = float)
        self.yS_values += self.noise
        return self.yS_values

class FourierSeries:
    def __init__(self, Signal, maxHarmonic=10):
        self.maxHarmonic = maxHarmonic
        self.mainHarmPeriod = 1
        
        self._a_coeffs = np.zeros(self.maxHarmonic + 1)
        self._b_coeffs = np.zeros(self.maxHarmonic + 1)

        self.xA_values = Signal.xS_values
        self.yA_values = None
        
        self._compute_coefficients(Signal)
        self._approximating(Signal)
        
    def _compute_coefficients(self, Signal):
        self._a_coeffs[0] = (2 / (Signal.endPoint - Signal.stPoint)) * np.sum(Signal.yS_values) * Signal.dx
        
        for nHrm in range(1, self.maxHarmonic + 1):
            arg = (2 * np.pi * nHrm) * (Signal.xS_values - Signal.stPoint) / self.mainHarmPeriod
            
            cos_vals = np.cos(arg)
            sin_vals = np.sin(arg)
            
            self._a_coeffs[nHrm] = (2 / (Signal.endPoint - Signal.stPoint)) * np.sum(Signal.yS_values * cos_vals) * Signal.dx
            self._b_coeffs[nHrm] = (2 / (Signal.endPoint - Signal.stPoint)) * np.sum(Signal.yS_values * sin_vals) * Signal.dx

    def _approximating(self, Signal):
        summ = np.full_like(self.xA_values, self._a_coeffs[0] / 2.0, dtype=float)
        
        for nHrm in range(1, self.maxHarmonic + 1):
            arg = (2 * np.pi * nHrm) * (Signal.xS_values  - Signal.stPoint) / self.mainHarmPeriod
            summ += self._a_coeffs[nHrm] * np.cos(arg)
            summ += self._b_coeffs[nHrm] * np.sin(arg)
            
        self.yA_values = summ



def create_comparing_graph(Signal, FourierSeries, show_error = False, show_spectrum = False):
        plt.figure(figsize=(8, 8))

        plt.subplot(2, 2, 1)
        
        plt.plot(Signal.xS_values, Signal.yS_values, label='F(x)', color='blue', linewidth=2, linestyle='--')
        plt.plot(FourierSeries.xA_values, FourierSeries.yA_values, label='Fourier Approx', color='red', linewidth=2)
        
        plt.legend()
        plt.grid(True, alpha = 0.3)
        plt.xlabel('x')
        plt.ylabel('y')

        if (show_error):
            plt.subplot(2,2, 2)
            
            err = Signal.yS_values - FourierSeries.yA_values
            plt.plot(Signal.xS_values, err, label = 'error', color = 'green', linewidth = 1.5)
            
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('err')
            

        if (show_spectrum):
            plt.subplot(2, 2, 3)
            
            fft_vals = np.fft.fft(Signal.yS_values)
            fft_freqs = np.fft.fftfreq(Signal.num_samples, Signal.dx)
            
            positive_freq_idx = fft_freqs >= 0
            fft_freqs_pos = fft_freqs[positive_freq_idx]
            fft_spectrum = np.abs(fft_vals[positive_freq_idx]) * 2 / Signal.num_samples
            
            max_harmonic = FourierSeries.maxHarmonic
            fundamental_freq = 1.0
            
            harmonic_amplitudes = []
            
            for n in range(max_harmonic + 1):
                target_freq = n * fundamental_freq
                idx = np.argmin(np.abs(fft_freqs_pos - target_freq))
                harmonic_amplitudes.append(fft_spectrum[idx])
            
            x_pos = np.arange(max_harmonic + 1) - 0.2
            plt.bar(x_pos, harmonic_amplitudes, width=0.4, label='FFT spectrum', color='orange', alpha=0.7)
            
            my_spectrum = ((FourierSeries._a_coeffs ** 2) + (FourierSeries._b_coeffs ** 2))**(1/2)
            plt.bar(np.arange(max_harmonic + 1) + 0.2, my_spectrum, width=0.4, label='My DFT spectrum', color='blue', alpha=0.7)
            
            plt.grid()
            plt.xlabel('Harmonic number')
            plt.ylabel('Amplitude')
            plt.title('FFT vs My DFT Comparison')
            plt.legend()
                
            plt.title('Fourier Series')

        plt.tight_layout()
        plt.show()






    

def cosinus(x, A=1, freq=1):
    return A * np.cos(2 * np.pi * freq * x)

def meander(x, A=1, freq=1):
    period = 1 / freq
    x_mod = x % period
    return np.where(x_mod < period / 2, A, -A)

def triangular_pulse_simple(x, A = 1, frequency = 1):
    period = 1 / freq
    x_mod = x % period
    return np.where(x_mod < period / 2, A*x_mod, -A*x_mod)



signal = Signal(lambda x: cosinus(x, 1, 1), stPoint = 0, endPoint = 2, num_samples = 1000, addNoise = True)

fourier = FourierSeries(signal, maxHarmonic=5)

create_comparing_graph(signal, fourier, True, True)
