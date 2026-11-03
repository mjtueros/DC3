import numpy as np
import torch

from pulse_emulator._rfchain import (
    efield_2_voltage,
    make_full_response_matrix,
    percieved_theta_phi,
)
from pulse_emulator.utils import trace_strong_filtering


class filters_tf:
    def __init__(self, filters, duration=2.048, fs=2000):
        self._tf = None
        self.filters = filters
        self.duration = duration
        self.fs = fs

    @property
    def tf(self):
        if self._tf is None:
            self._tf = self.create_tf()
        return self._tf

    @tf.setter
    def tf(self, value):
        self._tf = value

    def create_tf(self):
        dirac = torch.zeros(int(self.fs*self.duration))
        dirac[0] = 1.0
        for f in self.filters:
            dirac = f(dirac, fs=self.fs)
        fft_dirac = np.fft.rfft(dirac)
        return fft_dirac

def make_filter(fs_f: float, duration: float, low: float, high: float):
    """Create frequency domain filter."""
    f = filters_tf([lambda x, fs: trace_strong_filtering(x, low, high, fs)], fs=fs_f, duration=duration).tf
    return f
    
@torch.no_grad()
def get_preds_abc(model, params):
    """
    Optimized inference-only version of get_preds.
    
    Assumes model.eval() and requires_grad=False have already been set.
    Wraps everything in torch.no_grad() for speed.
    """
    if isinstance(params, np.ndarray):
        params = torch.tensor(params, dtype=torch.float32, device=model.device)
    elif params.device != model.device:
        params = params.to(model.device)
    preds = model(params)
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = model.normalizer.inverse(preds, outputs=True)
    return preds.cpu().numpy()

def abc_to_fourrier(a, b, c, freqs, f_0=30, gradient=False):
    if gradient:
        mod = torch
        assert isinstance(a, (torch.floating, torch.Tensor))
        assert isinstance(b, (torch.floating, torch.Tensor))
        assert isinstance(c, (torch.floating, torch.Tensor))
        if isinstance(a, (torch.floating)):
            a = torch.tensor(a, dtype=torch.float32).reshape(1)
            b = torch.tensor(b, dtype=torch.float32).reshape(1)
            c = torch.tensor(c, dtype=torch.float32).reshape(1)
        if type(freqs) is np.ndarray:
            freqs = torch.tensor(freqs, dtype=torch.float32, device=a.device)
        b_val, c_val = b.detach(), c.detach()
    else:
        mod = np
        assert type(freqs) is np.ndarray
        if isinstance(a, (np.floating, float)):
            a = np.array([a])
            b = np.array([b])
            c = np.array([c])
        b_val, c_val = b, c

    clipped = 250.
    minimums = - b_val/(2 * c_val + 1e-15) + f_0
    minimums[c_val<0] = clipped
    freqs_clip = mod.minimum(freqs[None,:], minimums[:, None])
    amp = mod.exp(a[:, None] + b[:, None]*(freqs_clip - f_0) + c[:, None]*(freqs_clip - f_0)**2)
    return amp


def params_to_phase(phase_q, phase_p, phase_offset, freqs, f_1=30, gradient=False):
    if gradient:
        mod = torch
        assert isinstance(phase_q, (torch.floating, torch.Tensor))
        assert isinstance(phase_p, (torch.floating, torch.Tensor))
        assert isinstance(phase_offset, (torch.floating, torch.Tensor))
        if isinstance(phase_q, (torch.floating)):
            phase_q = torch.tensor(phase_q, dtype=torch.float32).reshape(1)
            phase_p = torch.tensor(phase_p, dtype=torch.float32).reshape(1)
            phase_offset = torch.tensor(phase_offset, dtype=torch.float32).reshape(1)
        if type(freqs) is np.ndarray:
            freqs = torch.tensor(freqs, dtype=torch.float32, device=phase_q.device)
    else:
        mod = np
        assert type(freqs) is np.ndarray
        if isinstance(phase_q, (np.floating, float)):
            phase_q = np.array([phase_q])
            phase_p = np.array([phase_p])
            phase_offset = np.array([phase_offset])
    phase = phase_p[:, None]*(freqs[None,:] - f_1) + phase_q[:, None]*(freqs[None,:] - f_1)**2
    phase = phase + (phase_offset[:, None] - phase[:, [0]])
    return phase


def pred_trace_kxB(model, params, fs=2e3, duration=2.048, f_0=30, fourier_filter=1.0, phase=None):
    """
    Predict the time-domain signal from the model parameters.
    This function is optimized for inference speed. It avoids unnecessary conversions and function calls.
    It assumes that the model is already in eval mode and that requires_grad=False has been set.
    inputs:
        - model: the trained model to use for prediction
        - params: the input parameters for the model (numpy array/torch tensor)
        - fs: sampling frequency in MHz (default 2000 MHz)
        - duration: duration of the output signal in us (default 2.048 us)
        - f_0: reference frequency in MHz for the model's predictions (default 30 MHz)
        - fourier_filter: optional frequency domain filter to apply to the predicted Fourier coefficients (default 1.0, i.e. no filtering)
        - phase: optional pre-computed phase array to use instead of computing it from the model's predictions. 
                 If None, the phase will be computed from the model's predictions. (default None)
    Outputs:
        - signal: the predicted time-domain signal (numpy array)
        - fourrier_signal: the predicted Fourier signal (numpy array)
    """
    n_freq = int(fs * duration / 2) + 1
    n_time = int(fs * duration)
    freqs = np.linspace(0, fs / 2, n_freq)
    
    preds = get_preds_abc(model, params)
    a, b, c = preds[:, 0], preds[:, 1], preds[:, 2]
    
    if preds.shape[1] <= 3:
        raise ValueError("Phase must be provided or model must predict phase parameters.")
    if phase is None:
        phase_q, phase_p = preds[:, 3], preds[:, 4]
        phase_offset = preds[:, 5] if preds.shape[1] > 5 else np.zeros_like(phase_q) + np.pi
        
        # Inline params_to_phase (avoid function call overhead)
        phase = phase_p[:, None] * (freqs[None, :] - f_0) + phase_q[:, None] * (freqs[None, :] - f_0) ** 2
        phase = phase + (phase_offset[:, None] - phase[:, [0]])
    
    # Inline abc_to_fourrier (avoid function call overhead)
    clipped = 250.0
    minimums = -b / (2 * c + 1e-15) + f_0
    minimums[c < 0] = clipped
    freqs_clip = np.minimum(freqs[None, :], minimums[:, None])
    amps = np.exp(a[:, None] + b[:, None] * (freqs_clip - f_0) + c[:, None] * (freqs_clip - f_0) ** 2)
    
    fourrier_signal = amps * np.exp(1j * phase) * fourier_filter * (fs / 2e3)
    signal = np.fft.irfft(fourrier_signal, n=n_time)
    return signal, fourrier_signal
    

def predict_voltage_fast(input_arr, kxB, antenna_pos, Xs, ctx):
    """
    Optimized voltage prediction: model inference → E-field → voltage.
    
    This is the full pipeline equivalent of predict_voltage() in emcee_utils.py:
      1. Run the model to get the kxB E-field trace
      2. Expand to 3D via kxB polarization
      3. Apply the RF chain (antenna response) to convert E-field → voltage
    
    Key optimizations:
    - Uses pred_trace_kxB (torch.no_grad, single np↔torch conversion)
    - torch.compile on model forward pass (if available)
    - float32 throughout the torch path
    
    Parameters
    ----------
    input_arr : np.ndarray, shape (n_ant, n_features)
        Input features for the model.
    kxB : np.ndarray, shape (3,) or (n_ant, 3)
        Polarization direction(s) for the kxB component.
    antenna_pos : np.ndarray, shape (n_ant, 3)
        Antenna positions.
    Xs : np.ndarray, shape (3,)
        Shifted Xmax position (single source seen by all antennas).
    ctx : EventContext
        Precomputed event context.
    
    Returns
    -------
    preds_voltage : np.ndarray, shape (n_ant, 3, n_samples)
        Predicted voltage traces after RF chain application.
    """
    model = ctx.model
    fs = ctx.fs_ds
    
    # 1. Model inference: input features → kxB E-field trace
    preds_modif, _ = pred_trace_kxB(model, input_arr, fs=fs)
    
    # 2. Expand to 3D E-field via kxB polarization
    if kxB.ndim == 1:
        preds_modif_3d = preds_modif[:, None, :] * kxB[None, :, None]
    else:
        preds_modif_3d = preds_modif[:, None, :] * kxB[:, :, None]

    # 3. E-field → Voltage via RF chain (response matrix depends on viewing angles)
    theta_du, phi_du = percieved_theta_phi(antenna_pos, Xs)
    efield_fft = np.fft.rfft(preds_modif_3d, axis=-1)
    full_response_matrix = make_full_response_matrix(
        ctx.t_SN, ctx.t_EW, ctx.t_Z, theta_du, phi_du, ctx.tf,
        input_sampling_freq=fs * 1e6, duration=2.048e-6
    )
    vout, vout_f = efield_2_voltage(
        efield_fft, full_response_matrix,
        current_rate=fs * 1e6, target_rate=fs * 1e6
    )
    return vout