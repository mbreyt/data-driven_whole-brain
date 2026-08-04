import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(s):
    f = np.fft.rfftfreq(len(s))
    return plt.loglog(f, np.abs(np.fft.rfft(s)))[0]

def noise_psd(N, seed_p=0, psd = lambda f: 1, sigma=1):
        np.random.seed(seed_p)
        X_white = np.fft.rfft(np.random.randn(N)*sigma)
        S = psd(np.fft.rfftfreq(N))
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)

def PSDGenerator(f, exponent=None):
    return lambda N: noise_psd(N, psd=f)

@PSDGenerator
def white_noise(f):
    return 1

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)

@PSDGenerator
def violet_noise(f):
    return f

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

@PSDGenerator
def spectral_exponent_demo(f):
    return 1/np.where(f == 0, float('inf'), f**2)

def spectral_exponent(N, exponent, seed_p=0, sigma=1):
    return noise_psd(N, seed_p, psd= lambda f: 1/np.where(f == 0, float('inf'), f**exponent), sigma=sigma)


# plt.style.use('dark_background')
# plt.figure(figsize=(12, 8), tight_layout=True)
# for G, c in zip(
#         [brownian_noise, white_noise], 
#         ['brown', 'white']):
#     plot_spectrum(G(30*50000)).set(color=c, linewidth=3)
# plot_spectrum(spectral_exponent(30*50000, 1.1)).set(color='violet', linewidth=3)
# plot_spectrum(spectral_exponent(30*50000, 1)).set(color='red', linewidth=3)
# plt.legend(['brownian', 'white', 'exp=1.1', 'exp=1'])
# plt.suptitle("Colored Noise")
# plt.ylim([1e-3, None])
# plt.show()