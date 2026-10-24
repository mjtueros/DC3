import dataclasses
import os

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import differential_evolution
from scipy.signal import hilbert
from scipy.stats import gaussian_kde

from pulse_emulator.data.input_formating import (
    compute_kxB_kxkxB,
    event_swf_time,
    make_input_array,
)
from pulse_emulator.data.opening_rootsim import (
    _get_all_event_numbers,
    get_shower_properties,
    get_traces_single_event,
)
from pulse_emulator.recons.prior import log_xmax_prior_batch
from pulse_emulator.surrogate.inference import predict_voltage, to_voltage
from pulse_emulator.utils import (
    R2D,
    cart2sph,
    central_coverage,
    ns_to_us,
    sec_to_us,
    sph2cart,
    v2_per_hz_to_uv2_per_mhz,
)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)



@dataclasses.dataclass
class ReconsConfig:
    base_model_path: str
    model_subdir: str
    base_data_path: str
    base_root: str

    fs_input: float = 2000
    fs_ds: float = 500
    duration_us: float = 2.048
    threshold: float = 5
    pad_left: int = 256
    pad_right: int = 512
    slope_offset: float = 2 * np.pi * 30
    
    likelihood_type: str = "matched_filtering_marginalized"
    prior: bool = 'bricolage'  # 'informative', 'uninformative', or 'bricolage'
    smearing: float = 0.0
    jitter_time_us: float = 0.005
    swf_uncertainty: float = 0.002
    model_uncertainty: float = 0.0
    alpha=.5
    n_2=1
    quadrature_method: str = 'gh'  # 'mc', 'qmc', 'gh', or 'ut'
    n_gh_pts: int = 2              # 'gh' only: M = n_gh_pts ** n_outputs points
    n_samples: int = 32            # 'mc' / 'qmc' only ('qmc' rounds up to a power of 2)
    kappa: float = 1.0             # 'ut' only
    
    method: str = "de->emcee"
    vectorized: bool = True
    nwalkers: int = 15
    optim_steps: int = 2000
    burn_in: int = 1250
    max_events: int = np.inf
    continue_run: bool = False
    min_antennas: int = 5          # events with fewer triggered antennas are skipped
    n_jobs: int = 1                # split the event list across this many independent jobs
    job_id: int = 0                # this job's shard index (0-based, < n_jobs)

    plot: bool = False
    verbose: bool = False


    @property
    def recons_dir(self):
        return f"{self.base_model_path}/{self.model_subdir}/recons/optimizer_recons/"

# =============================================================================
# PREPROCESSING & SNR SELECTION
# =============================================================================

def preprocess_traces(traces_voltage: np.ndarray, noise_computer, smearing: float = 0., kill_noise: bool = False):
    """Preprocess traces with bandpass filtering and add noise."""
    if kill_noise:
        noise_voltage = np.zeros_like(traces_voltage)
        noise_std = 0.0
    else:
        noise_voltage = sample_galactic_noise(noise_computer, traces_voltage.shape[0], lst_hour=18)
        noise_std = np.mean(np.std(noise_voltage, axis=-1))
    amps = np.linalg.norm(traces_voltage, axis=1).max(axis=-1)
    measured_signal = traces_voltage + noise_voltage
    smearing_factor = 1 + np.random.normal(0, smearing, size=(measured_signal.shape[0], 1, 1))
    measured_signal *= smearing_factor
    return traces_voltage, measured_signal, amps, (smearing_factor-1, noise_std)


def pick_top_snr_indices(snr: np.ndarray, top_k: int):
    """Select indices of top-k SNR antennas."""
    strong_idx = np.isin(np.arange(len(snr)), np.argsort(snr)[-top_k:])
    return strong_idx


def pick_strong_antennas(amps: np.ndarray, threshold: int):
    """Select antennas above amplitude threshold."""
    strong_idx = amps > threshold
    return strong_idx


def sample_galactic_noise(noise_computer, n_antennas, lst_hour=18):

    noise_traces, noise_fft = noise_computer.noise_samples(
        lst_hour=lst_hour, n_samples=n_antennas, micro=True,
    )
    return noise_traces


def compute_std(traces):
    """Estimate per-polarization noise std from off-peak regions."""
    hilbert_traces = hilbert(traces, axis=-1)
    envelope = np.linalg.norm(np.abs(hilbert_traces), axis=1)
    loc_max = envelope.argmax(axis=1)
    med_pos = int(np.median(loc_max))
    mask = np.ones(traces.shape[-1], dtype=bool)
    mask[max(med_pos - 20, 0):min(med_pos + 40, len(mask))] = False
    std_polar = np.mean(np.std(traces[:, :, mask], axis=-1), axis=0)
    return std_polar

##############################################################################
###### Sampler helpers
##############################################################################

class EventContext:
    """
    Precomputes and caches all invariant quantities for a single event.
    
    This replaces the pattern of passing a DataFrame into the inner loop
    and calling .values, .copy(), column assignment, etc. 21,000 times.
    
    All arrays are stored as float32 for consistency with the model.
    """
    
    # Speed of light for SWF
    c = 299792458.0
    
    def __init__(self, df_ev, config_inputs, model, fs_ds,
                 t_SN, t_EW, t_Z, tf,
                 measured_signal, times_noisy, 
                 pad_left=0, pad_right=0,
                 noise_computer=None, smearing=None, sigma_t=None,
                 alpha=0.5, n_2=5,
                 k_guess=None, Xs_guess=None,
                 angle_conf=0.5, log_E_conf=1.5, Xs_conf=50e3,
                 amps=None, slope_offset=0, model_uncertainty=0.0):
        """
        Precompute everything that doesn't change across MCMC iterations.
        
        Parameters
        ----------
        df_ev : pd.DataFrame.     Event dataframe (used once, then discarded).
        config_inputs : list of str.     Input feature names for the model.
        model : MLP_metamodel.     The neural network model.
        fs_ds : float.     Sampling frequency.
        measured_signal : np.ndarray, shape (n_ant, 3, n_samples).     Noisy voltage traces.
        noise_std : float.     Noise standard deviation.
        smearing : float.     Smearing factor for heteroscedastic noise model.
        t_SN, t_EW, t_Z : objects.     Antenna response data tables.
        tf : np.ndarray.     Transfer function.
        times_noisy : np.ndarray.     Noisy arrival times for hybrid method.
        sigma_t : float, optional.     Time uncertainty for hybrid method.
        alpha : float.     Balance between shape and time likelihood.
        k_guess : np.ndarray, shape (3,), optional.     Initial guess shower direction (for amplitude prior).
        Xs_guess : np.ndarray, shape (3,), optional.     Initial guess Xmax position (for amplitude prior).
        angle_conf : float.     Angular prior width in degrees (for amplitude prior).
        log_E_conf : float.     Energy prior width in log10 decades (for amplitude prior).
        Xs_conf : float.     Xmax position prior width in meters (for amplitude prior).
        """
        
        # --- Extract numpy arrays from DataFrame ONCE ---
        self.du_pos = df_ev[['du_pos_x', 'du_pos_y', 'du_pos_z']].values.astype(np.float64)
        self.xmax_pos = df_ev[['xmax_pos_x', 'xmax_pos_y', 'xmax_pos_z']].values.astype(np.float64)
        self.n_ant = self.du_pos.shape[0]
        self.event_number = df_ev['event_number'].values[0]
        self.thinning_freqs = df_ev['thinning_freqs'].values if 'thinning_freqs' in df_ev.columns else None

        # Store all columns needed by make_input_array as a dict of numpy arrays
        self.base_values = {}
        for col in df_ev.columns:
            try:
                self.base_values[col] = df_ev[col].values.astype(np.float64).copy()
            except (ValueError, TypeError):
                self.base_values[col] = df_ev[col].values.copy()
        
        # Measured signal (keep as provided, typically float64)
        self.measured_signal = np.ascontiguousarray(measured_signal)
        
        # Store config
        self.config_inputs = config_inputs
        self.model = model

        self.fs_ds = fs_ds
        self.n_2 = n_2
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.alpha = alpha
        self.smearing = smearing
        self.model_uncertainty = model_uncertainty
        self.sigma_t = sigma_t
        dt = 1/fs_ds
        self.n_sigma = 5
        self.jitter_kernel = np.exp(-0.5 * (np.arange(-self.n_sigma*sigma_t, 
                                                      self.n_sigma*sigma_t+dt, 
                                                      dt) ** 2) / sigma_t**2)


        # --- Precompute antenna response data ---
        self.t_SN = t_SN
        self.t_EW = t_EW
        self.t_Z = t_Z
        self.tf = tf

        # Times for hybrid method
        if times_noisy is None:
            raise ValueError("times_noisy must be provided")
        
        self.times_noisy = times_noisy.astype(np.float64)
        self.times_noisy_centered = self.times_noisy - self.times_noisy.mean()
        
        
        # --- Pre-slice measured signal for shape comparison ---
        self.slope_offset = slope_offset
        self.pl = self.pad_left
        self.pr = self.pad_right if self.pad_right > 0 else None
        

        # Pre-extract the duration and frequency axis for pred_trace_kxB
        duration = 2.048
        self.trace_len = int(fs_ds * duration)
        self.phase_shift = 0
        if self.pad_right == self.trace_len//2:
            duration = duration / 2
            self.trace_len = int(fs_ds * duration)
        
        if self.pad_left == self.trace_len//2:
            duration = duration / 2
            self.trace_len = int(fs_ds * duration)
            self.phase_shift = -2 * np.pi * (self.pad_left / fs_ds)

        freqs_np = np.linspace(0, fs_ds / 2, int(fs_ds * duration / 2) + 1)
        self.duration = duration  
        df = 1 / self.duration  
        self.tf = np.stack([
            np.interp(freqs_np, np.linspace(0, fs_ds / 2, self.tf.shape[-1]), self.tf[p]) for p in range(3)
        ], axis=0)

        # Precompute the "delayed" (trimmed by n_2) version for _min_error_trace
        self.hilbert_delayed = np.linalg.norm(np.abs(hilbert(self.measured_delayed, axis=-1)), axis=1)
        if noise_computer is not None:
            nbins = self.measured_trimmed.shape[-1]
            freqs_mhz = np.fft.rfftfreq(nbins, d=1/self.fs_ds)
            target_freqs_mhz = noise_computer.target_freqs * 1e-6

            noise_traces, _ = noise_computer.noise_samples(
                lst_hour=18, n_samples=1000, micro=True,
            )
            # noise_traces already in uV
            noise_std = np.sqrt(np.var(noise_traces, axis=(0, -1)))
            self.noise_var_2 = noise_std[None, :, None] ** 2
            self.noise_std = noise_std

            psd = noise_computer.noise_psd(18) / 2
            
            psd = v2_per_hz_to_uv2_per_mhz(psd)  # V^2/Hz -> (uV)^2/MHz
            # psd *= (1 + self.smearing)**2
            self.psd = np.stack(
                [np.interp(freqs_mhz, target_freqs_mhz, psd[p]) for p in range(3)],
            )
            for i in range(3):
                self.psd[i, self.psd[i] < self.psd[i].max() / 1e6] = np.inf
            self.psd[:, (freqs_mhz <= 30) | (freqs_mhz >= 240)] = np.inf  # Ignore frequencies outside the model range 
            self.measured_fft = np.fft.rfft(self.measured_trimmed, axis=-1) * 1 / self.fs_ds
            self.measured_fft_over_psd = self.measured_fft / self.psd[None, :, :]

        if self.thinning_freqs is None:
            self.thinning_freqs = np.full_like(times_noisy, 240)
        self.thinning_idx = np.minimum(240, (self.thinning_freqs/df).astype(int))
        
        # Check if 'n_eff' is in config_inputs (for SWF with refraction)
        self.has_n_eff = 'n_eff' in config_inputs
        if self.has_n_eff:
            self.n_eff_idx = config_inputs.index('n_eff')
                
        
        
        # --- Precompute model normalizer on correct device ---
        self.device = getattr(self.model, 'device', 'cpu')

        
        # --- Amplitude-based reconstruction context ---
        # Alias for API compatibility with log_likelihood_amplitude
        
        # Measured amplitudes and per-polarization noise std
        self.measured_amps = get_amps(measured_signal)
        
        # Array center
        self.mean_pos = self.du_pos.mean(axis=0)
        
        # Prior knowledge from other reconstruction methods
        if k_guess is not None:
            self.k_guess = np.asarray(k_guess, dtype=np.float64)
            cos_theta = -self.k_guess[2]
            self.D_guess = Xmax_Distance_fit(cos_theta) * 1e3   # km → m
            self.D_conf = Xmax_Distance_conf(cos_theta) * 1e3
        else:
            self.k_guess = None
            self.D_guess = None
            self.D_conf = None
        
        if Xs_guess is not None:
            self.Xs_guess = np.asarray(Xs_guess, dtype=np.float64)
        else:
            self.Xs_guess = None
        
        self.angle_conf = angle_conf
        self.log_E_conf = log_E_conf
        self.Xs_conf = Xs_conf
        
        
        # self.Cov_traces = np.sum(self.noise_var_2)/9 * np.eye(self.hilbert_delayed.shape[-1], dtype=np.float64) +\
        #                     self.smearing**2 * self.hilbert_delayed[:,:,None] * self.hilbert_delayed[:,None,:]
        # self.det_Cov_traces = np.linalg.det(self.Cov_traces/np.linalg.norm(self.Cov_traces))*np.linalg.norm(self.Cov_traces)**self.Cov_traces.shape[0]
        # self.inv_Cov_traces = np.linalg.pinv(self.Cov_traces, rcond=1e-5)

    @property
    def measured_trimmed(self):
        if self.pr is not None:
            res = self.measured_signal[:, :, self.pl:-self.pr]
        else:
            res = self.measured_signal[:, :, self.pl:]
        return res
    
    @property
    def measured_delayed(self):
        return self.measured_trimmed[:, :, self.n_2:-self.n_2]

    @property
    def delays_range(self):
        return np.arange(-self.n_2, self.n_2 + 1)
    def copy(self):
        """Create a copy of the context for use in parallel sampling."""
        new_ctx = EventContext.__new__(EventContext)  # Create uninitialized instance
        new_ctx.__dict__.update(self.__dict__)  # Shallow copy of all attributes
        return new_ctx


def compute_swf_guess(times_noisy_bin0, du_pos, amps, n_walkers=10, sigma_t=0.007, method='DE', prior=True):
    """Compute prior guesses for informative reconstruction."""
    theta_pwf, phi_pwf = pwf_guess(times_noisy_bin0, du_pos)
    k_pwf = -sph2cart(theta_pwf, phi_pwf)
    barycenter = np.average(du_pos, axis=0, weights=amps)
    if method == 'DE':
        best_xmax = recons_swf_de(times_noisy_bin0, du_pos, k_guess=k_pwf, popsize=n_walkers, sigma_t=sigma_t, prior=prior)
    # elif method == 'MCMC':
    #     best_xmax = recons_swf(times_noisy_bin0, du_pos, k_guess=k_pwf, n_walkers=n_walkers, sigma_t=sigma_t)
    else:
        raise ValueError(f"Unknown method {method}")
    k_swf = barycenter - best_xmax
    k_swf /= np.linalg.norm(k_swf)
    return k_swf, best_xmax


def build_event_context(df_ev, measured_signal, times_noisy, noise_std, model, model_config,
                        config, t_SN, t_EW, t_Z, tf_ds, noise_computer=None, amps=None):
    """
    Build an EventContext using a unified configuration.

    Parameters
    ----------
    df_ev : pd.DataFrame
    measured_signal : np.ndarray
    times_noisy : np.ndarray
    noise_std : float
    model : MLP_metamodel
    model_config : dict
    config : ReconsConfig
    t_SN, t_EW, t_Z : objects
    tf_ds : np.ndarray
    noise_computer : optional
    amps : np.ndarray, optional

    Returns
    -------
    EventContext
    """
    measured_amps = get_amps(measured_signal) if amps is None else amps
    du_pos = df_ev[['du_pos_x', 'du_pos_y', 'du_pos_z']].values.astype(np.float64)

    k_guess = -sph2cart(df_ev['zenith'].iloc[0], df_ev['azimuth'].iloc[0])
    Xs_guess = np.array([df_ev['xmax_pos_x'].iloc[0], df_ev['xmax_pos_y'].iloc[0], df_ev['xmax_pos_z'].iloc[0]], dtype=np.float64)
    if getattr(config, 'informative', False):
        k_guess, Xs_guess = compute_swf_guess(times_noisy, du_pos, measured_amps)

    times_noisy_ctx = times_noisy
    sigma_t = np.sqrt(config.jitter_time_us ** 2 + config.swf_uncertainty ** 2)
    smearing = np.sqrt(config.smearing ** 2 + config.model_uncertainty ** 2)

    ctx = EventContext(
        df_ev=df_ev,
        config_inputs=model_config['data']['inputs'],
        model=model,
        fs_ds=config.fs_ds,
        t_SN=t_SN,
        t_EW=t_EW,
        t_Z=t_Z,
        tf=tf_ds,
        measured_signal=measured_signal,
        times_noisy=times_noisy_ctx,
        smearing=smearing,
        sigma_t=sigma_t,
        pad_left=config.pad_left,
        pad_right=config.pad_right,
        k_guess=k_guess,
        Xs_guess=Xs_guess,
        amps=measured_amps,
        n_2=config.n_2,
        alpha=config.alpha,
        noise_computer=noise_computer,
        slope_offset=config.slope_offset,
        model_uncertainty=config.model_uncertainty,
    )
    return ctx



# =============================================================================
# EVENT PREPARATION & FINALIZATION HELPERS
# =============================================================================
def load_event(root_dir: str, event_number: float, all_event_numbers: np.ndarray):
    """Load event data from ROOT file."""
    event_idx = np.where(all_event_numbers == event_number)[0][0]
    _, properties = get_shower_properties(root_dir, event_idx, event_idx + 1)
    all_traces, du_ids, theta, azimuth = get_traces_single_event(root_dir, event_number, all_event_numbers)
    k = -sph2cart(theta[0], azimuth[0])
    kxB, _ = compute_kxB_kxkxB(k)
    return event_idx, properties, all_traces, du_ids, theta, azimuth, kxB


def prepare_event(event_number, df_X_event, config, t_SN, t_EW, t_Z, tf, noise_computer, verbose=True, model=None, model_config=None, fixed_du_ids=None):
    """
    Shared event preparation: load ROOT data, add timing noise, preprocess
    traces, filter strong antennas.

    fixed_du_ids : array-like of du_id, optional
        Reuse this antenna selection instead of re-triggering on this call's
        amplitudes. Meant for comparing scenarios (e.g. emulated vs real
        signal, with/without noise) on the exact same antenna set: pass the
        `du_id`s from a reference call's `df_ev` so that only the trace and
        timing change between scenarios, never which antennas are kept.

    Returns a dict with all prepared data, or None if the event should be skipped.
    """
    prefix = f"[Worker {os.getpid()}] " if not config.verbose else ""

    # Necessary-condition guard, not the real cut. `strong_enough` below is a mask
    # over df_X_event's rows, so fewer rows than the minimum can never yield enough
    # antennas above threshold -- but the real cut is on amplitude, and only ~1% of
    # events are this sparse (median is ~125 antennas). This mainly avoids the
    # Xmax_pos [0]-index below blowing up on an empty event; the genuine saving on
    # a relaunch comes from the caller's skip list, not from here.
    min_antennas = getattr(config, 'min_antennas', 5)
    if len(df_X_event) < min_antennas:
        if config.verbose:
            print(f"{prefix}Skipping event {event_number}: only {len(df_X_event)} antennas "
                  f"in dataframe (< {min_antennas}), before any trace loading", flush=True)
        return None

    antenna_pos = df_X_event[['du_pos_x', 'du_pos_y', 'du_pos_z']].values
    Xmax_pos = df_X_event[['xmax_pos_x', 'xmax_pos_y', 'xmax_pos_z']].values[0]
    
    ratio_fs = config.fs_input / config.fs_ds
    tf_ds = tf[..., :int(tf.shape[-1] / ratio_fs) + 1]
    
    root_file = df_X_event['file_name'].values[0]
    root_dir = f"{config.base_root}/{root_file}"
    all_event_numbers = _get_all_event_numbers(root_dir)
    
    try:
        event_idx, properties, all_traces, du_ids, theta, azimuth, kxB = load_event(
            root_dir, event_number, all_event_numbers
        )
        # Keep only the antennas that are present in the event dataframe, but preserve
        # the ordering coming from the ROOT file (du_ids). Later we will reorder
        # the event dataframe to match this order so measured traces align with
        # the per-antenna rows in the dataframe.
        mask_antennas = np.isin(du_ids, df_X_event['du_id'].values)
        du_ids_sub = du_ids[mask_antennas]
        all_traces = all_traces[mask_antennas]
        all_traces, _ = to_voltage(all_traces, antenna_pos, Xmax_pos, config.fs_input, config.fs_ds, t_SN, t_EW, t_Z, tf, duration=2.048, to_adc=False)
        # Reorder df_X_event to follow the same du_id order as du_ids_sub so that
        # subsequent boolean masks (e.g., strong_enough) are correctly applied.
        try:
            df_X_event = df_X_event.set_index('du_id').loc[du_ids_sub].reset_index()
        except Exception:
            # Fallback: if du_id isn't present or indexing fails, keep original order
            pass



    except Exception as e:
        print(f"{prefix}Error loading event {event_number}: {e}")
        return None
    
    if hasattr(config, 'emulated_signal') and config.emulated_signal and model is not None and model_config is not None:
        values_dict = {col: df_X_event[col].values for col in df_X_event.columns}
        input_arr, (k, kxb, kxkxb) = make_input_array(values_dict, config_inputs=model_config['data']['inputs'])
        all_traces = predict_voltage(input_arr, kxB, antenna_pos, Xmax_pos, 
                                    model, config.fs_ds, 
                                    t_SN, t_EW, t_Z, tf_ds, duration=2.048)
        
    
    # --- Timing noise ---
    du_s = properties['du_s'][0][mask_antennas]
    du_ns = properties['du_ns'][0][mask_antennas]
    du_s = du_s - du_s.min()
    du_times = sec_to_us(du_s) + ns_to_us(du_ns)
    du_times -= du_times.min()
    # print(times_noisy_bin0)
    if hasattr(config, 'emulated_signal') and config.emulated_signal:
        du_times = event_swf_time(Xmax_pos, antenna_pos)
        du_times -= du_times.min()
    # print(times_noisy_bin0)
    time_resolution = 1 / config.fs_ds
    noise_samples = np.round(
        np.random.randn(*du_times.shape).flatten() * config.jitter_time_us / time_resolution
    ) * time_resolution
    times_noisy_bin0 = du_times + noise_samples
    times_noisy_bin0 = times_noisy_bin0
    
    
    # --- Preprocess traces ---
    traces_no_noise, measured_signal, amps, (smearing_factors, noise_std) = preprocess_traces(
        all_traces, noise_computer=noise_computer,
        smearing=config.smearing, kill_noise=getattr(config, 'kill_noise', False))
    threshold = config.threshold * max(noise_std, 7e2)
    snrs = amps / max(noise_std, 7e2)
    
    # --- Filter strong antennas ---
    if fixed_du_ids is not None:
        strong_enough = df_X_event['du_id'].isin(fixed_du_ids).values
    else:
        strong_enough = pick_strong_antennas(amps, threshold=threshold)
    if strong_enough.sum() < getattr(config, 'min_antennas', 5):
        if config.verbose:
            print(f"{prefix}Skipping event {event_number}: insufficient antennas "
                f"({strong_enough.sum()}), best SNR: {snrs.max():.1f}", flush=True)
        return None
    else:
        if config.verbose:
            print(f"{prefix}Using {strong_enough.sum()} strong antennas for event {event_number}", flush=True)
    
    df_ev = df_X_event.copy()
    df_ev = df_ev[strong_enough].reset_index(drop=True)
    measured_signal = measured_signal[strong_enough]
    times_noisy_bin0 = times_noisy_bin0[strong_enough]
    
    return {
        'df_ev': df_ev,
        'measured_signal': measured_signal,
        'times_noisy_bin0': times_noisy_bin0,
        'noise_std': noise_std,
        'noise_samples': noise_samples,
        'snrs': snrs,
        'tf_ds': tf_ds,
        'root_file': root_file,
        'theta': theta,
        'azimuth': azimuth,
        'kxB': kxB,
        'properties': properties,
    }



def min_error_trace(preds, measured_delayed, n_2):
    """
    Optimized delay search using sliding window view.
    
    Uses precomputed measured_delayed from EventContext.
    """
    l = preds.shape[-1]
    window_len = l - 2 * n_2
    preds_delayed = sliding_window_view(preds, window_len, axis=2)
    
    errors = ((preds_delayed - measured_delayed[:, :, None, :]) ** 2).sum(axis=(1, 3))
    
    delays = np.arange(-n_2, n_2 + 1)
    delay_best = delays[np.argmin(errors, axis=1)]
    return delay_best


def apply_shift(x, ctx):
    """
    Apply absolute reconstruction parameters using pure numpy arrays instead of DataFrame operations.
    
    Returns modified copies of the arrays that make_input_array needs.
    
    ~100x faster than the old DataFrame-based helper.
    
    The parameter vector is interpreted as:
    [xmax_x, xmax_y, xmax_z, log10(E), zenith, azimuth].
    
    OPTIMIZATION: Only copies arrays that actually change.
    Uses a shallow dict copy + targeted array copies.
    """
    xmax_x, xmax_y, xmax_z, log_E, zenith, azimuth = x
    energy = 10.0 ** float(log_E)
    
    # Shallow copy of dict (O(n_columns) pointer copies, no array copies)
    values = dict(ctx.base_values)
    
    # Only copy+modify arrays that change with the shift
    if 'energy_em' in values:
        values['energy_em'] = np.full_like(ctx.base_values['energy_em'], energy, dtype=float)
    if 'energy_primary' in values:
        values['energy_primary'] = np.full_like(ctx.base_values['energy_primary'], energy, dtype=float)
    
    values['xmax_pos_x'] = np.full_like(ctx.base_values['xmax_pos_x'], xmax_x, dtype=float)
    values['xmax_pos_y'] = np.full_like(ctx.base_values['xmax_pos_y'], xmax_y, dtype=float)
    values['xmax_pos_z'] = np.full_like(ctx.base_values['xmax_pos_z'], xmax_z, dtype=float)
    values['zenith'] = np.full_like(ctx.base_values['zenith'], zenith, dtype=float)
    values['azimuth'] = np.full_like(ctx.base_values['azimuth'], azimuth, dtype=float)
    
    # Absolute Xmax position (antenna positions don't change)
    xmax_shifted = np.array([[xmax_x, xmax_y, xmax_z]], dtype=float)
    return values, ctx.du_pos, xmax_shifted



#################################################################################
###### Other utilities
#################################################################################

def get_amps(traces):
    """Get peak Hilbert envelope amplitude per antenna."""
    hilbert_traces = hilbert(traces, axis=-1)
    envelope = np.linalg.norm(np.abs(hilbert_traces), axis=1)
    amps = envelope.max(axis=1)
    return amps


def Xmax_Distance_fit(cos_theta):
    """Empirical Xmax distance vs cos(theta) fit."""
    a, b, c = 12.72862024, -16.58390211, -20.95807057
    return a * 1 / cos_theta + b * np.log(cos_theta) + c


def Xmax_Distance_conf(cos_theta):
    """Empirical distance confidence vs cos(theta)."""
    return 20 + 1.0 / (cos_theta ** 1.5)




def pwf_guess(time_noisy, du_pos):
    """
    Get a PWF-based guess for the direction
    """
    try:
        from PWF_reconstruction.recons_PWF import PWF_semianalytical
    except ImportError as e:
        raise ImportError(
            "pwf_guess() needs the 'PWF_reconstruction' package: "
            "pip install git+https://github.com/arsenefer/PWF_reconstruction.git"
        ) from e
    theta_pwf, phi_pwf = PWF_semianalytical(du_pos, time_noisy*1e-6)
    return theta_pwf, phi_pwf


# def recons_swf(du_times, du_pos, k_guess, n_walkers=20, n_steps=2000, sigma_t=0.007):
#     cos_theta = -k_guess[2]
#     assert cos_theta > 0, "k_guess should point towards the sky (cos(theta) > 0)"
   
#     def _prior_bounds(X_cand):
#         mean_pos = du_pos.mean(axis=0)
#         k_swf_cands = mean_pos - X_cand
#         k_swf_cands /= np.linalg.norm(k_swf_cands, keepdims=True, axis=-1)
#         angle_with_guess = np.arccos((k_swf_cands * k_guess).sum(axis=-1)) * R2D
#         return -1e15*( (angle_with_guess > 2.) | (X_cand[2] < 1265) )
    
#     def log_posterior(X_cand):
#         swf_times = event_swf_time(X_cand, du_pos)
#         residuals = du_times - swf_times
#         residuals -= residuals[0]

#         likelihood = -np.square(residuals/sigma_t).sum() / 2

#         mean_pos = du_pos.mean(axis=0)
#         k_swf_cands = mean_pos - X_cand
#         k_swf_cands /= np.linalg.norm(k_swf_cands)
#         theta = np.arccos(-k_swf_cands[2]) 
#         xmax_dist = np.abs((X_cand[2]-1264)/k_swf_cands[2])
#         xmax_prior = log_xmax_prior_batch(np.atleast_1d(theta), np.atleast_1d(xmax_dist) )

#         bounds_penalty = _prior_bounds(X_cand)
#         # return likelihood
#         lp = xmax_prior + likelihood + bounds_penalty
#         if isinstance(lp, np.ndarray):
#             return lp.reshape(-1)[0]
#         return lp
    
#     ndim = 3
#     D0 = 100e3
#     X0 = du_pos.mean(axis=0) + D0 * (-k_guess)
#     pos_init = X0 + np.random.randn(n_walkers, ndim) * 1000
#     sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
#     sampler.run_mcmc(pos_init, n_steps)

#     samples = sampler.get_chain(discard=1000, flat=True, thin=10)
#     best_idx = np.argmax(sampler.get_log_prob(discard=1000, flat=True, thin=10))
#     best_xmax = samples[best_idx]

#     return best_xmax


def recons_swf_de(du_times, du_pos, k_guess, sigma_t=0.007, maxiter=2000, popsize=15, seed=None, prior=True):
    """
    Xmax reconstruction using differential evolution optimization.
    
    Minimizes the negative log-posterior to find the best-fit shower maximum position.
    Faster than MCMC for single-point estimation; returns best parameters directly.
    
    Parameters
    ----------
    du_times : np.ndarray, shape (n_ant,)
        Arrival times at antennas (micro-seconds).
    du_pos : np.ndarray, shape (n_ant, 3)
        Antenna positions (meters).
    k_guess : np.ndarray, shape (3,)
        Initial guess for shower direction (unit vector).
    sigma_t : float
        Time uncertainty for likelihood (micro-seconds).
    maxiter : int
        Maximum iterations for differential evolution.
    popsize : int
        Population size for DE (total_pop_size = popsize * n_dim).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    best_xmax : np.ndarray, shape (3,)
        Best-fit Xmax position (meters).
    """
    cos_theta = -k_guess[2]
    
    assert cos_theta > 0, "k_guess should point towards the sky (cos(theta) > 0)"
    
    def _prior_bounds(X_cand):
        mean_pos = du_pos.mean(axis=0)
        k_swf_cands = mean_pos - X_cand
        k_swf_cands /= np.linalg.norm(k_swf_cands, keepdims=True, axis=-1)
        ## compute angle guess and if runtime warning occurs, print k_swf_cands, k_guess, and their dot product
        dot_product = np.clip( (k_swf_cands * k_guess).sum(axis=-1),  -1, 1)
        angle_with_guess = np.arccos(dot_product) * R2D
        
        return -1e15*( (angle_with_guess > 2.) | (X_cand[2] < 1265) )
    
    def log_posterior(X_cand):
        swf_times = event_swf_time(X_cand, du_pos)
        residuals = du_times - swf_times
        residuals -= residuals.mean()

        likelihood = -np.square(residuals/sigma_t).sum() / 2

        mean_pos = du_pos.mean(axis=0)
        k_swf_cands = mean_pos - X_cand
        k_swf_cands /= np.linalg.norm(k_swf_cands)
        theta = np.arccos(-k_swf_cands[2]) 
        xmax_dist = np.abs((X_cand[2]-1264)/k_swf_cands[2])
        calib_factor = 1.5
        if prior:
            xmax_prior = log_xmax_prior_batch(np.atleast_1d(theta), np.atleast_1d(xmax_dist), calibration_factor=calib_factor)
            bounds_penalty = _prior_bounds(X_cand)
        # return likelihood
            lp = xmax_prior + likelihood + bounds_penalty
        else:
            lp = likelihood
        if isinstance(lp, np.ndarray):
            return lp.reshape(-1)[0]
        return lp
    
    # Define bounds: ±500 km from array center in each direction
    mean_pos = du_pos.mean(axis=0)
    bounds = [(mean_pos[0] - 500e3, mean_pos[0] + 500e3),
              (mean_pos[1] - 500e3, mean_pos[1] + 500e3),
              (mean_pos[2] + 10, mean_pos[2] + 100e3)]

    
    # Initial guess: along the shower direction
    D0 = 50e3
    X0 = mean_pos + D0 * (-k_guess)
    
    # Minimize negative log-posterior
    result = differential_evolution(
        lambda X: -log_posterior(X),
        bounds=bounds,
        x0=X0,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        workers=1,  # no multiprocessing by default
        updating='deferred',
        atol=1e-8,
        tol=1e-8,
    )
    return result.x



# =============================================================================
# Plots


def hdi_1d(samples, cred_mass=0.68):
    """Compute highest density interval (HDI) for a 1D sample."""
    x = np.sort(samples)
    n = len(x)
    interval = int(np.floor(cred_mass * n))
    widths = x[interval:] - x[:n - interval]
    i = np.argmin(widths)
    return x[i], x[i + interval]


def map_kde_1d(samples, grid_size=2000, subsample=20000):
    if samples.ndim == 1:
        samples = samples[:, None]

    N, D = samples.shape
    maps = np.zeros(D)

    for d in range(D):

        x = samples[:, d]

        # optional subsampling
        if len(x) > subsample:
            idx = np.random.choice(len(x), subsample, replace=False)
            x = x[idx]

        kde = gaussian_kde(x)

        xmin, xmax = x.min(), x.max()
        grid = np.linspace(xmin, xmax, grid_size)

        density = kde(grid)
        maps[d] = grid[np.argmax(density)]

    return maps



def hdi_1d(samples, cred_mass=0.68):
    """Compute highest density interval (HDI) for a 1D sample."""
    x = np.sort(samples)
    n = len(x)
    interval = int(np.floor(cred_mass * n))
    widths = x[interval:] - x[:n - interval]
    i = np.argmin(widths)
    return x[i], x[i + interval]


def map_kde_1d(samples, grid_size=2000, subsample=20000):
    if samples.ndim == 1:
        samples = samples[:, None]

    N, D = samples.shape
    maps = np.zeros(D)

    for d in range(D):

        x = samples[:, d]

        # optional subsampling
        if len(x) > subsample:
            idx = np.random.choice(len(x), subsample, replace=False)
            x = x[idx]

        kde = gaussian_kde(x)

        xmin, xmax = x.min(), x.max()
        grid = np.linspace(xmin, xmax, grid_size)

        density = kde(grid)
        maps[d] = grid[np.argmax(density)]

    return maps

def corner_sns(samples, labels=None, truths=None, title=None, show=True,
               show_titles=True, title_kwargs=None, recons=None, **kwargs):
    """Corner plot using seaborn pairplot with KDE contours, mimicking corner.corner syntax.
    
    Parameters
    ----------
    samples : array-like, shape (n_samples, n_dim)
        The samples to plot.
    labels : list of str, optional
        Labels for each dimension.
    truths : list of float, optional
        True values to mark on the plot.
    title : str, optional
        Super title for the figure.
    show : bool, optional
        Whether to call plt.show().
    show_titles : bool, optional
        Whether to show titles on diagonal subplots with median and quantiles.
    title_kwargs : dict, optional
        Keyword arguments passed to ax.set_title() for diagonal titles.
    **kwargs : dict
        Additional keyword arguments (ignored for compatibility).
    
    Returns
    -------
    g : sns.PairGrid
        The seaborn PairGrid object.
    """
    if title_kwargs is None:
        title_kwargs = {}
    
    samples = np.atleast_2d(samples)
    if samples.ndim == 3:
        samples = samples.reshape(-1, samples.shape[-1])
    
    n_dim = samples.shape[1]
    if labels is None:
        labels = [f'x_{i}' for i in range(n_dim)]
    
    df_samples = pd.DataFrame(samples, columns=labels)
    
    # Create PairGrid manually instead of pairplot
    g = sns.PairGrid(df_samples, corner=True, diag_sharey=False)
    
    # Diagonal: KDE plots
    g.map_diag(sns.kdeplot, color='#4C72B0', fill=True, alpha=0.4, linewidth=1.5)
    
    if recons is None:
        recons = df_samples.median()
        left_bound = df_samples.quantile(0.16) 
        right_bound = df_samples.quantile(0.84)
    else:
        recons = pd.Series(np.array(recons), index=labels)
        
        left_bound = {}
        right_bound = {}

        for label in labels:
            lo, hi = hdi_1d(df_samples[label].values, cred_mass=0.68)
            left_bound[label] = lo
            right_bound[label] = hi

        left_bound = pd.Series(left_bound)
        right_bound = pd.Series(right_bound)
    if truths is not None:
        truths = pd.Series(np.array(truths), index=labels)
    
    # Off-diagonal: Only 2D KDE contours
    for i in range(n_dim):
        for j in range(i):
            ax = g.axes[i, j]
            x = df_samples[labels[j]].values
            y = df_samples[labels[i]].values
            
            # Subsample for speed if necessary
            if len(x) > 10000:
                idx = np.random.choice(len(x), size=10000, replace=False)
                x_sub, y_sub = x[idx], y[idx]
            else:
                x_sub, y_sub = x, y
                
            kde = gaussian_kde(np.vstack([x_sub, y_sub]))
            
            # Use current limits or provide a slightly padded range
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            dx, dy = (xmax - xmin) * 0.1, (ymax - ymin) * 0.1
            xx, yy = np.mgrid[xmin-dx:xmax+dx:100j, ymin-dy:ymax+dy:100j]
            
            positions = np.vstack([xx.ravel(), yy.ravel()])
            zz = kde(positions).reshape(xx.shape)
            
            # Probability levels
            levels_frac = [0.989, 0.865, 0.393]
            zz_sorted = np.sort(zz.ravel())[::-1]
            cumsum = np.cumsum(zz_sorted) / np.sum(zz_sorted)
            contour_levels = [zz_sorted[np.searchsorted(cumsum, f)] for f in levels_frac]
            contour_levels = sorted(contour_levels)
            
            ax.contour(xx, yy, zz, levels=contour_levels, colors=['#C44E52', '#E8A838', '#4C72B0'],
                       linewidths=[1.0, 1.2, 1.5], alpha=0.8)
    
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(labelsize=12)
            ax.xaxis.label.set_size(1.2*BIGGER_SIZE)
            ax.yaxis.label.set_size(1.2*BIGGER_SIZE)
    
    # Add medians and quantiles on diagonals, crosshairs on off-diagonals
    for i, label in enumerate(labels):
        g.axes[i, i].axvline(recons[label], color='#C44E52', ls='--', lw=1.5, zorder=100)
        g.axes[i, i].axvspan(left_bound[label], right_bound[label], color='#C44E52', alpha=0.1, zorder=100)
        # if show_titles:
        #     plus = right_bound[label] - recons[label]
        #     minus = recons[label] - left_bound[label]
        #     t = f'{label} = {recons[label]:.2f}$^{{+{plus:.2f}}}_{{-{minus:.2f}}}$'
        #     default_title_kwargs = {'fontsize': BIGGER_SIZE}
        #     default_title_kwargs.update(title_kwargs)
        #     g.axes[i, i].set_title(t, **default_title_kwargs)
        if show_titles:
            plus = right_bound[label] - recons[label]
            minus = recons[label] - left_bound[label]
            # Extract the unit from the label if it exists
            if '[' in label and ']' in label:
                quantity, unit = label.split('[')
                unit = unit.strip(']')
                quantity = quantity.strip()
                t = f'{quantity} = {recons[label]:.2f}$^{{+{plus:.2f}}}_{{-{minus:.2f}}}$ {unit}'
            else:
                t = f'{label} = {recons[label]:.2f}$^{{+{plus:.2f}}}_{{-{minus:.2f}}}$'
            default_title_kwargs = {'fontsize': 1.2*BIGGER_SIZE}
            default_title_kwargs.update(title_kwargs)
            g.axes[i, i].set_title(t, **default_title_kwargs)

        for j in range(i):
            g.axes[i, j].scatter(recons[labels[j]], recons[label], color='#C44E52', marker='x', s=100, lw=5, zorder=100)
    
    # Mark truths if provided
    if truths is not None:
        for i, label in enumerate(labels):
            g.axes[i,i].axvline(truths[label], c='green', ls='-', lw=1.5)
            for j in range(i):
                g.axes[i, j].scatter(truths[labels[j]], truths[label], color='green', marker='o', s=100, edgecolor='none', lw=1.5)
    
    # Add legend
    handles = [
        Line2D([0], [0], color='#C44E52', lw=2, label='recons'),
        Line2D([0], [0], color='#C44E52', alpha=0.1, lw=24, label='68% CI')
    ]
    if truths is not None and any(t is not None for t in truths):
        handles.append(Line2D([0], [0], color='green', lw=1.5, label='truth'))
    g.figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.6, .85), frameon=False, fontsize=1.2*BIGGER_SIZE)
    
    suptitle = title if title is not None else 'MCMC Posterior Distributions'
    g.figure.suptitle(suptitle, y=1.02, fontsize=16)
    g.figure.tight_layout()
    if show:
        plt.show()
    g.figure.align_ylabels()
    
    return g

# === Coordinate conversion functions for xmax polar optimization ===
def cart2sph_xmax(x_cart_meters):
    """Convert Cartesian xmax [x, y, z] in meters to spherical [r, zen, az].
    
    r: distance in km
    zen: zenith angle from z-axis (0 if on z-axis, π/2 if on x-y plane)
    az: azimuth angle in [0, 2π], following phi convention (0 when y=0)
    """
    r, zen, az = cart2sph(x_cart_meters)
    r /= 1000.0  # convert to km
    return np.array([r, zen, az % (2 * np.pi)])

def sph2cart_xmax(r_km, zen, az):
    """Convert spherical xmax [r, zen, az] to Cartesian [x, y, z] in meters.
    
    r: distance in km
    zen: zenith angle from z-axis
    az: azimuth angle
    Returns: [x, y, z] in meters
    """
    r_m = r_km * 1000.0
    return sph2cart(zen, az, r_m)

def post_treatment_emcee(mc_res, config, ctx):
    # Determine MCMC settings
    n_steps = int(getattr(config, 'optim_steps', config.optim_steps))
    nwalkers = int(getattr(config, 'nwalkers', config.nwalkers))
    n_dim = 6

    samples = mc_res["details"]['samples']
    log_probas = mc_res["details"]['log_probas']
    sampler = mc_res["details"]['sampler']
    
    samples_post = samples[config.burn_in:, :, :]
    log_probas_post = log_probas[config.burn_in:, :]
    n_steps_after, n_walkers, n_dim = samples_post.shape

    # Convert to Cartesian/radians for reporting
    samples_cart = samples_post.copy()
    samples_cart[:, :, [1,2,4,5]] /= R2D
    r = samples_cart[:, :, 0]
    zen = samples_cart[:, :, 1]
    az = samples_cart[:, :, 2]

    x_cart, y_cart, z_cart = sph2cart_xmax(r, zen, az)
    samples_cart[:, :, 0] = x_cart
    samples_cart[:, :, 1] = y_cart
    samples_cart[:, :, 2] = z_cart
    samples_cart_flat = samples_cart.reshape(-1, n_dim)

    # basic posterior summaries
    post_mean = np.mean(samples_cart_flat, axis=0).tolist()
    post_median = np.median(samples_cart_flat, axis=0).tolist()
    cov = np.cov(samples_cart_flat, rowvar=False)
    # percentiles
    p16, p50, p84 = np.percentile(samples_cart_flat, [16,50,84], axis=0)
    p2_5, p97_5 = np.percentile(samples_cart_flat, [2.5,97.5], axis=0)
    # autocorr time and ESS
    try:
        tau = emcee.autocorr.integrated_time(samples_post, quiet=False)
        autocorr_time = np.asarray(tau).astype(float).tolist()
        ess = ((n_steps_after) * n_walkers) / np.asarray(tau)
        ess_list = ess.astype(float).tolist()
    except Exception:
        autocorr_time = None
        ess_list = None

    acc_frac = sampler.acceptance_fraction
    acc_mean = float(np.mean(acc_frac))


    # coverage: central (equal-tailed) credible level of the true value within
    # each marginal posterior. mean(coverage <= alpha) over many events gives
    # the observed coverage at credible level alpha directly (see
    # pulse_emulator.utils.central_coverage for why this replaces a plain
    # one-sided mean(samples <= true) rank).
    true_vals = np.array([ctx.base_values['xmax_pos_x'][0], ctx.base_values['xmax_pos_y'][0], ctx.base_values['xmax_pos_z'][0], np.log10(ctx.base_values['energy_em'][0]), ctx.base_values['zenith'][0], ctx.base_values['azimuth'][0]])
    coverage = central_coverage(samples_cart_flat, true_vals).tolist()

    # Compute marginal MAP (mode) per-parameter via KDE
    marginal_map = []
    for i in range(n_dim):
        arr = samples_cart_flat[:, i]
        if np.ptp(arr) == 0 or len(arr) < 3:
            marginal_map.append(float(np.median(arr)))
            continue
        kde = gaussian_kde(arr)
        grid = np.linspace(np.min(arr), np.max(arr), 200)
        dens = kde(grid)
        mode = float(grid[np.argmax(dens)])
        marginal_map.append(mode)

    
    # attach MCMC outputs
    res = {
        'n_steps': int(n_steps),
        'n_walkers': int(nwalkers),
        'burn_in': int(config.burn_in),
        'acceptance_fraction_mean': acc_mean,
        'autocorr_time': autocorr_time,
        'ESS_vector': ess_list,
        'posterior_mean': post_mean,
        'posterior_median': post_median,
        'marginal_map': marginal_map,
        'percentiles_16_50_84': np.vstack([p16, p50, p84]).tolist(),
        'percentiles_2_5_97_5': np.vstack([p2_5, p97_5]).tolist(),
        'true_coverage': coverage,
    }
    
    # Save raw MCMC outputs to per-event file for later inspection
    event_out_dir = f"{config.recons_dir}/{getattr(config, 'name', '')}" if getattr(config, 'name', None) else config.recons_dir
    os.makedirs(event_out_dir, exist_ok=True)
    np.savez(os.path.join(event_out_dir, f"mcmc_samples_{int(ctx.event_number)}.npz"),
                samples=samples, log_probas=log_probas, samples_cart_full=samples_cart,
                acceptance_fraction=sampler.acceptance_fraction, **res)

    return samples_cart
