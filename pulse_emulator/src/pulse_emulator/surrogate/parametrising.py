import numpy as np
from scipy.optimize import curve_fit

from pulse_emulator.surrogate.inference import abc_to_fourrier


def compute_thinning(fft_kxB, fs=2000, start_freq=500, threshold_quantile=0.99, security_offset=5):
    N = fft_kxB.shape[-1] * 2 - 2
    freqs = np.fft.rfftfreq(N, d=1/fs)
    index_10 = np.argmin(np.abs(freqs - 10))
    amplitude = np.abs(fft_kxB)
    thinning_amp = np.quantile(amplitude[freqs >= start_freq], threshold_quantile)
    below_thinning = amplitude[freqs >= index_10] < thinning_amp
    smallest_bellow_thinning = np.where(below_thinning)[0][0] + index_10
    thinning_freq = freqs[smallest_bellow_thinning] - security_offset
    return thinning_freq, thinning_amp


def _with_lstsq(freqs_shifted, y):
    log_amplitude = np.log(y)
    A = np.vstack([np.ones_like(freqs_shifted), freqs_shifted, freqs_shifted**2]).T
    coeffs = np.linalg.lstsq(A, log_amplitude.T, rcond=None)[0].T
    a,b,c = coeffs[:,0], coeffs[:,1], coeffs[:,2]
    return coeffs

def _with_polyfit(freqs_shifted, y):
    log_amplitude = np.log(y)
    c, b, a = np.polyfit(freqs_shifted, log_amplitude, 2)
    # c, b, a = np.polyfit(freqs_shifted, log_amplitude, 2, w=log_amplitude**2)[0]
    return a,b,c

def _with_curve_fit(freqs_shifted, y, b_lim=np.inf, c_lim=np.inf):
    def to_optimise(freqs_shifted, a, b, c):
        return abc_to_fourrier(a, b, c, freqs_shifted, f_0=0, gradient=False)[0]
    try:
        (a,b,c), pcov = curve_fit(
            to_optimise,
            freqs_shifted,
            y,
            p0=[1.0, -1e-2, -1e-4],
            maxfev=10000,
            bounds=([-np.inf, -np.inf, -np.inf], [np.inf, b_lim, c_lim]),

        )
    except RuntimeError:
        print("Optimal parameters not found: Number of calls to function has reached maxfev = 10000.")
        a,b,c = np.array([np.nan, np.nan, np.nan])
        pcov = np.full((3, 3), np.nan)
    return a,b,c

def compute_abc(fft_kxB, thinning_freq=80, fs=2000, f_0=30, low_fit_band=15, high_fit_band=250):
    N = 2*(fft_kxB.shape[-1] - 1)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    mask = (freqs >= low_fit_band) & (freqs <= high_fit_band) & (freqs <= thinning_freq)
    if thinning_freq < low_fit_band+10:  #less than a 40 MHz band to fit
        return np.nan, np.nan, np.nan
    freqs_shifted = freqs[mask] - f_0
    
    a,b,c = _with_polyfit(freqs_shifted, np.abs(fft_kxB)[mask])

    return a, b, c

def compute_phase_params(fft_kxB, thinning_freq=80, fs=2000, f_1=30, low_fit_band=10, high_fit_band=250):
    N = 2*(fft_kxB.shape[-1] - 1)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    phase = np.unwrap(np.angle(fft_kxB))

    mask = (freqs >= low_fit_band) & (freqs <= high_fit_band) & (freqs <= thinning_freq)
    if np.sum(mask)*freqs[1] < low_fit_band+10:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    freqs_shifted = freqs[mask] - f_1

    phase_q, phase_p, phase_offset_fit = np.polyfit(freqs_shifted, phase[mask], 2)
    phase_offset = phase[0]
    return phase_q, phase_p, phase_offset, phase_offset_fit

