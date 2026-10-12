"""
Outlier analysis visualization functions for EAS reconstruction diagnostics.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import Normalize

warnings.filterwarnings('ignore')
from pulse_emulator.data.input_formating import compute_kxB_kxkxB, compute_l_omega_eta
from pulse_emulator.recons.recons_utils import R2D, get_amps


def _select_highest_error_antennas(measured_signal, predicted_traces_rec_td, n_plot):
    """Return antenna indices with the largest reconstruction error."""
    diff = np.nan_to_num(predicted_traces_rec_td - measured_signal, nan=0.0, posinf=0.0, neginf=0.0)
    errors = np.linalg.norm(diff, axis=(1, 2))
    n_plot = min(n_plot, len(errors))
    if n_plot <= 0:
        return np.array([], dtype=int), errors
    order = np.argsort(errors)
    return order[-n_plot:][::-1], errors


def plot_traces_and_model(measured_signal, predicted_traces_rec_td, predicted_traces_true_td, fs_ds, event_num, output_dir):
    """
    Plot measured vs predicted time-domain traces for all antennas/polarizations.
    
    Parameters
    ----------
    measured_signal : ndarray, shape (n_ant, 3, n_samples)
        Measured voltage traces
    predicted_traces_td : ndarray, shape (n_ant, 3, n_samples)
        Predicted voltage traces (time-domain)
    fs_ds : float
        Sampling frequency (Hz)
    event_num : int
        Event number
    output_dir : str
        Output directory path
    """
    n_ant = min(measured_signal.shape[0], 8)  # Limit to 8 antennas for clarity
    n_samples = measured_signal.shape[-1]
    dt = 1.0 / fs_ds
    time = np.arange(n_samples) * dt #in microseconds
    tmin, tmax = 0.55, 0.7
    mask_zoom = (time >= tmin) & (time <= tmax)
    selected_antennas, recon_errors = _select_highest_error_antennas(measured_signal, predicted_traces_rec_td, n_ant)
    # First figure: measured vs predicted (reconstructed parameters)
    fig, axes = plt.subplots(n_ant, 3, figsize=(15, 2*n_ant), sharex=True)
    if n_ant == 1:
        axes = axes.reshape(1, -1)
    for row, ant_idx in enumerate(selected_antennas):
        for p in range(3):
            ax = axes[row, p]
            ax.plot(time[mask_zoom], measured_signal[ant_idx, p, mask_zoom], 'b-', linewidth=1, label='Measured', alpha=0.7)
            ax.plot(time[mask_zoom], predicted_traces_rec_td[ant_idx, p, mask_zoom], 'r--', linewidth=1, label='Predicted (recon)', alpha=0.7)
            ax.set_ylabel(f'Ant {ant_idx}, Pol {p}')
            if row == 0:
                ax.set_title(f'Polarization {p}')
            if p == 0:
                ax.legend(loc='upper right', fontsize=8)
            if row == n_ant - 1:
                ax.set_xlabel('Time [μs]')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Event {event_num}: Measured vs Predicted Traces (Reconstructed params, highest-error antennas)')
    plt.tight_layout()
    fpath_full = os.path.join(output_dir, 'traces_and_model_recon.pdf')
    plt.savefig(fpath_full, dpi=100, bbox_inches='tight')
    plt.close()
    # Second figure: measured vs predicted (true parameters) with zoom
    fig, axes = plt.subplots(n_ant, 3, figsize=(15, 2*n_ant), sharex=True)
    if n_ant == 1:
        axes = axes.reshape(1, -1)

    for row, ant_idx in enumerate(selected_antennas):
        for p in range(3):
            ax = axes[row, p]
            ax.plot(time[mask_zoom], measured_signal[ant_idx, p, mask_zoom], 'b-', linewidth=1, label='Measured', alpha=0.7)
            ax.plot(time[mask_zoom], predicted_traces_true_td[ant_idx, p, mask_zoom], 'g--', linewidth=1, label='Predicted (true)', alpha=0.7)
            ax.set_ylabel(f'Ant {ant_idx}, Pol {p}')
            if row == 0:
                ax.set_title(f'Polarization {p}')
            if p == 0:
                ax.legend(loc='upper right', fontsize=8)
            if row == n_ant - 1:
                ax.set_xlabel('Time [μs]')
            ax.grid(True, alpha=0.3)


    plt.suptitle(f'Event {event_num}: Measured vs Predicted Traces (True params, same antennas)')
    plt.tight_layout()
    fpath_zoom = os.path.join(output_dir, 'traces_and_model_true.pdf')
    plt.savefig(fpath_zoom, dpi=100, bbox_inches='tight')
    plt.close()

    return fpath_full, fpath_zoom


def plot_spectra_and_psd(measured_signal, predicted_traces_td, fs_ds, duration, event_num, output_dir, fname_prefix='spectra_and_psd'):
    """
    Plot FFT magnitude and power spectral density.
    
    Parameters
    ----------
    measured_signal : ndarray, shape (n_ant, 3, n_samples)
        Measured voltage traces
    predicted_traces_td : ndarray, shape (n_ant, 3, n_samples)
        Predicted voltage traces
    fs_ds : float
        Sampling frequency (Hz)
    duration : float
        Duration in seconds
    event_num : int
        Event number
    output_dir : str
        Output directory path
    """
    n_samples = measured_signal.shape[-1]
    # Compute frequencies and convert to MHz (user requested multiplying by 1e6)
    freqs = np.fft.rfftfreq(n_samples, d=1.0/fs_ds) 
    
    # Select antennas with the largest reconstruction error
    n_ant_plot = min(4, measured_signal.shape[0])
    selected_antennas, recon_errors = _select_highest_error_antennas(measured_signal, predicted_traces_td, n_ant_plot)
    
    fig, axes = plt.subplots(n_ant_plot, 2, figsize=(12, 3*n_ant_plot))
    if n_ant_plot == 1:
        axes = axes.reshape(1, -1)
    
    for row, ant_idx in enumerate(selected_antennas):
        # FFT magnitude (sum over polarizations)
        fft_meas = np.abs(np.fft.rfft(measured_signal[ant_idx].sum(axis=0)))
        fft_pred = np.abs(np.fft.rfft(predicted_traces_td[ant_idx].sum(axis=0)))
        # Apply magnitude cut: values below 1e-5 set to nan
        fft_meas[fft_meas < 1e-5] = np.nan
        fft_pred[fft_pred < 1e-5] = np.nan
        psd_meas = fft_meas**2 / n_samples
        psd_pred = fft_pred**2 / n_samples
        
        # Spectra
        ax = axes[row, 0]
        # Start plotting at 30 MHz
        mask = freqs >= 30.0
        ax.semilogy(freqs[mask], fft_meas[mask], 'b-', label='Measured', linewidth=1, alpha=0.8)
        ax.semilogy(freqs[mask], fft_pred[mask], 'r--', label='Predicted', linewidth=1, alpha=0.8)
        ax.set_ylabel(f'Magnitude (Ant {ant_idx})')
        ax.set_xlabel('Frequency [MHz]')
        ax.set_title(f'Error rank {row + 1}: ||pred-meas||={recon_errors[ant_idx]:.2e}')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        # PSD
        ax = axes[row, 1]
        ax.semilogy(freqs[mask], psd_meas[mask], 'b-', label='Measured', linewidth=1, alpha=0.8)
        ax.semilogy(freqs[mask], psd_pred[mask], 'r--', label='Predicted', linewidth=1, alpha=0.8)
        ax.set_ylabel(f'PSD (Ant {ant_idx})')
        ax.set_xlabel('Frequency [MHz]')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
    
    plt.suptitle(f'Event {event_num}: Spectra and Power Spectral Density')
    plt.tight_layout()
    fpath = os.path.join(output_dir, f'{fname_prefix}.pdf')
    plt.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close()
    return fpath


def plot_snr_distribution(measured_signal, noise_std, event_num, output_dir):
    """
    Plot SNR per antenna and histogram.
    
    Parameters
    ----------
    measured_signal : ndarray, shape (n_ant, 3, n_samples)
        Measured voltage traces
    noise_std : float
        Estimated noise standard deviation
    event_num : int
        Event number
    output_dir : str
        Output directory path
    """
    # Compute SNR per antenna using hilbert-envelope peak (consistent with recons_utils)
    amps = get_amps(measured_signal)
    snr = amps / (noise_std + 1e-10)
    
    n_ant = len(snr)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar chart
    ax = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0, 1, n_ant))
    bars = ax.bar(np.arange(n_ant), snr, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(y=np.median(snr), color='k', linestyle='--', label='Median', linewidth=2)
    ax.set_xlabel('Antenna Index')
    ax.set_ylabel('SNR')
    ax.set_title(f'SNR per Antenna (Mean={snr.mean():.1f}, Std={snr.std():.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Histogram
    ax = axes[1]
    ax.hist(snr, bins=max(5, n_ant//2), edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=snr.mean(), color='r', linestyle='--', label=f'Mean={snr.mean():.1f}', linewidth=2)
    ax.axvline(x=np.median(snr), color='g', linestyle='--', label=f'Median={np.median(snr):.1f}', linewidth=2)
    ax.set_xlabel('SNR')
    ax.set_ylabel('Count')
    ax.set_title('SNR Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Event {event_num}: SNR Analysis')
    plt.tight_layout()
    fpath = os.path.join(output_dir, 'snr_distribution.pdf')
    plt.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close()
    return fpath


def plot_antenna_coverage(antenna_positions, measured_signal, noise_std, xmax_pos, event_num, output_dir):
    """
    Plot antenna array layout colored by SNR/signal strength.
    
    Parameters
    ----------
    antenna_positions : ndarray, shape (n_ant, 3)
        Antenna positions (x, y, z in meters)
    measured_signal : ndarray, shape (n_ant, 3, n_samples)
        Measured voltage traces
    noise_std : float
        Noise standard deviation
    xmax_pos : ndarray, shape (3,)
        Xmax position (x, y, z in meters)
    event_num : int
        Event number
    output_dir : str
        Output directory path
    """
    # Compute SNR per antenna using consistent amp estimator
    amps = get_amps(measured_signal)
    snr = amps / (noise_std + 1e-10)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2D XY view
    ax = axes[0]
    # Use smaller markers and avoid overlaying Xmax on the same scatter
    sc = ax.scatter(antenna_positions[:, 0], antenna_positions[:, 1], 
                    c=snr, s=60, cmap='RdYlGn', edgecolor='k', linewidth=0.6, alpha=0.9)
    
    # Add antenna labels
    for i, (x, y) in enumerate(antenna_positions[:, :2]):
        ax.text(x, y, str(i), ha='center', va='center', fontsize=7)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Array Layout (XY plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # ax.set_aspect('equal')
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('SNR')
    
    # 3D scatter (side view ZX)
    ax = axes[1]
    sc = ax.scatter(antenna_positions[:, 0], antenna_positions[:, 2], 
                    c=snr, s=60, cmap='RdYlGn', edgecolor='k', linewidth=0.6, alpha=0.9)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Array Layout (ZX plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('SNR')
    
    # Distance metrics
    dist_to_center = np.linalg.norm(antenna_positions[:, :2] - xmax_pos[:2], axis=1)
    
    plt.suptitle(f'Event {event_num}: Antenna Array Coverage (Mean distance to Xmax: {dist_to_center.mean():.1f} m)')
    plt.tight_layout()
    fpath = os.path.join(output_dir, 'antenna_coverage.pdf')
    plt.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return fpath


def plot_residuals_with_uncertainties(event_num, truth_dict, recon_dict, output_dir):
    """
    Plot reconstruction residuals with uncertainty estimates.
    
    Parameters
    ----------
    event_num : int
        Event number
    truth_dict : dict
        Truth values: 'energy_em', 'zenith', 'azimuth', 'xmax_pos'
    recon_dict : dict
        Reconstruction: absolute fields such as 'xmax_pos', 'zenith', 'azimuth', 'energy_em', 'logE'
        Optional: '*_std' for uncertainties
    output_dir : str
        Output directory path
    """
    # Compute residuals
    E_true = truth_dict.get('energy_em', 1.0)
    theta_true = truth_dict.get('zenith', 0.0)
    phi_true = truth_dict.get('azimuth', 0.0)
    xmax_true = truth_dict.get('xmax_pos', np.zeros(3))
    
    dE_pct = recon_dict.get('dE_pct', 0.0)
    dtheta_deg = recon_dict.get('dtheta_deg', 0.0)
    dphi_deg = recon_dict.get('dphi_deg', 0.0)
    dxmax_g = recon_dict.get('dxmax_g', 0.0)
    
    # Uncertainties (if available)
    dE_std = recon_dict.get('dE_std', None)
    dtheta_std = recon_dict.get('dtheta_std', None) 
    dphi_std = recon_dict.get('dphi_std', None)
    dxmax_std = recon_dict.get('dxmax_std', None)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Energy
    ax = axes[0, 0]
    ax.bar([0], [dE_pct], yerr=[[dE_std] if dE_std else [0]], 
           color='steelblue', alpha=0.7, edgecolor='black', capsize=5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.set_ylabel('ΔE/E [%]')
    ax.set_title('Energy Residual')
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Theta
    ax = axes[0, 1]
    ax.bar([0], [dtheta_deg], yerr=[[dtheta_std] if dtheta_std else [0]], 
           color='orange', alpha=0.7, edgecolor='black', capsize=5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.set_ylabel('Δθ [°]')
    ax.set_title('Zenith Angle Residual')
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Phi
    ax = axes[1, 0]
    ax.bar([0], [dphi_deg], yerr=[[dphi_std] if dphi_std else [0]], 
           color='green', alpha=0.7, edgecolor='black', capsize=5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.set_ylabel('Δφ [°]')
    ax.set_title('Azimuth Angle Residual')
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Xmax
    ax = axes[1, 1]
    ax.bar([0], [dxmax_g], yerr=[[dxmax_std] if dxmax_std else [0]], 
           color='red', alpha=0.7, edgecolor='black', capsize=5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.set_ylabel('ΔXmax [g/cm²]')
    ax.set_title('Xmax Residual')
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Event {event_num}: Reconstruction Residuals')
    plt.tight_layout()
    fpath = os.path.join(output_dir, 'residuals_with_uncertainties.pdf')
    plt.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return fpath


def plot_timing_residuals(times_measured, times_swf, sigma_t, antenna_positions, xmax_pos, event_num, output_dir):
    """
    Plot timing residuals and SWF comparison.
    
    Parameters
    ----------
    times_measured : ndarray, shape (n_ant,)
        Measured event times
    times_swf : ndarray, shape (n_ant,)
        Shower wavefront times (prediction)
    sigma_t : float
        Timing uncertainty
    antenna_positions : ndarray, shape (n_ant, 3)
        Antenna positions
    xmax_pos : ndarray, shape (3,)
        Xmax position
    event_num : int
        Event number
    output_dir : str
        Output directory path
    """
    times_measured_centered = times_measured - times_measured.mean()
    times_swf_centered = times_swf - times_swf.mean()
    residuals = times_swf_centered - times_measured_centered
    
    chi2_red = np.sum((residuals / (sigma_t + 1e-10))**2) / (len(residuals) - 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Times comparison
    ax = axes[0, 0]
    x = np.arange(len(times_measured))
    ax.errorbar(x, times_measured_centered * 1e9, yerr=sigma_t * 1e9, 
                fmt='o-', label='Measured', capsize=3, alpha=0.7)
    ax.plot(x, times_swf_centered * 1e9, 's--', label='SWF prediction', alpha=0.7)
    ax.set_xlabel('Antenna Index')
    ax.set_ylabel('Time [ns]')
    ax.set_title('SWF vs Measured Times')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residuals
    ax = axes[0, 1]
    colors = np.where(np.abs(residuals) > 2*sigma_t, 'red', 'steelblue')
    ax.bar(x, residuals * 1e9, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.axhline(y=2*sigma_t * 1e9, color='r', linestyle='--', alpha=0.5, label='±2σ')
    ax.axhline(y=-2*sigma_t * 1e9, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Antenna Index')
    ax.set_ylabel('Residual [ns]')
    ax.set_title(f'Timing Residuals (χ²_red={chi2_red:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Distance vs residual
    ax = axes[1, 0]
    dist = np.linalg.norm(antenna_positions - xmax_pos, axis=1)
    scatter = ax.scatter(dist, residuals * 1e9, c=np.abs(residuals) / sigma_t, 
                        cmap='RdYlGn_r', s=100, edgecolor='black', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.set_xlabel('Distance to Xmax [m]')
    ax.set_ylabel('Residual [ns]')
    ax.set_title('Residual vs Distance')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('|Residual|/σ')
    ax.grid(True, alpha=0.3)
    
    # Histogram
    ax = axes[1, 1]
    ax.hist(residuals * 1e9, bins=max(5, len(residuals)//2), 
            edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='r', linestyle='--', label='Expected', linewidth=2)
    ax.set_xlabel('Residual [ns]')
    ax.set_ylabel('Count')
    ax.set_title(f'Residual Distribution (σ={residuals.std() * 1e9:.2f} ns)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Event {event_num}: Timing Residual Analysis')
    plt.tight_layout()
    fpath = os.path.join(output_dir, 'timing_residuals_detailed.pdf')
    plt.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return fpath


def plot_amplitude_distribution(measured_signal, antenna_positions, xmax_pos, event_num, output_dir):
    """
    Plot amplitude distribution across antenna array.
    
    Parameters
    ----------
    measured_signal : ndarray, shape (n_ant, 3, n_samples)
        Measured voltage traces
    antenna_positions : ndarray, shape (n_ant, 3)
        Antenna positions
    xmax_pos : ndarray, shape (3,)
        Xmax position
    event_num : int
        Event number
    output_dir : str
        Output directory path
    """
    # Peak amplitude per antenna (use hilbert envelope estimator)
    amps = get_amps(measured_signal)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Bar chart
    ax = axes[0, 0]
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(amps)))
    ax.bar(np.arange(len(amps)), amps, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Antenna Index')
    ax.set_ylabel('Peak Amplitude')
    ax.set_title('Peak Amplitude per Antenna')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Replace amplitude-vs-distance scatter (not useful) with simple stats text
    ax = axes[0, 1]
    dist = np.linalg.norm(antenna_positions[:, :2] - xmax_pos[:2], axis=1)
    stats_text = f"Mean amp = {amps.mean():.3e}\nMedian amp = {np.median(amps):.3e}\nMean dist = {dist.mean():.1f} m"
    ax.axis('off')
    ax.text(0.05, 0.5, stats_text, fontsize=11, family='monospace', va='center')
    
    # Histogram
    ax = axes[1, 0]
    ax.hist(amps, bins=max(5, len(amps)//2), edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=amps.mean(), color='r', linestyle='--', 
              label=f'Mean={amps.mean():.2f}', linewidth=2)
    ax.axvline(x=np.median(amps), color='g', linestyle='--', 
              label=f'Median={np.median(amps):.2f}', linewidth=2)
    ax.set_xlabel('Peak Amplitude')
    ax.set_ylabel('Count')
    ax.set_title(f'Amplitude Distribution (Std={amps.std():.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2D map
    ax = axes[1, 1]
    scatter = ax.scatter(antenna_positions[:, 0], antenna_positions[:, 1], 
                        c=amps, s=90, cmap='RdYlGn', edgecolor='k', 
                        linewidth=0.6, alpha=0.9)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Amplitude Distribution on Array')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # ax.set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Amplitude')
    
    plt.suptitle(f'Event {event_num}: Amplitude Distribution')
    plt.tight_layout()
    fpath = os.path.join(output_dir, 'amplitude_distribution.pdf')
    plt.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return fpath


def _compute_shower_core(xmax_pos, k):
    """Project Xmax onto the ground along the shower axis."""
    xmax_pos = np.asarray(xmax_pos, dtype=float)
    k = np.asarray(k, dtype=float)
    if k[2] == 0:
        return xmax_pos
    return xmax_pos - k * ((xmax_pos[2]-1265) / k[2])


def plot_omega_scatter(true_ant_pos, true_xmax_pos, true_k,
                       recon_ant_pos, recon_xmax_pos, recon_k,
                       event_num, output_dir):
    """Scatter antenna positions colored by omega for true and reconstructed geometry."""
    true_kxB, true_kxkxB = compute_kxB_kxkxB(true_k)
    recon_kxB, recon_kxkxB = compute_kxB_kxkxB(recon_k)

    true_omega, _, _ = compute_l_omega_eta(true_ant_pos, true_xmax_pos, true_k, true_kxB, true_kxkxB)
    recon_omega, _, _ = compute_l_omega_eta(recon_ant_pos, recon_xmax_pos, recon_k, recon_kxB, recon_kxkxB)

    true_core = _compute_shower_core(true_xmax_pos, true_k)
    recon_core = _compute_shower_core(recon_xmax_pos, recon_k)

    omega_min = float(np.nanmin([np.nanmin(true_omega), np.nanmin(recon_omega)]))
    omega_max = float(np.nanmax([np.nanmax(true_omega), np.nanmax(recon_omega)]))
    norm = Normalize(vmin=omega_min*R2D, vmax=omega_max*R2D)
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    panels = [
        (axes[0], true_ant_pos, true_omega, true_core, 'True geometry'),
        (axes[1], recon_ant_pos, recon_omega, recon_core, 'Reconstructed geometry'),
    ]

    for ax, ant_pos, omega, core, title in panels:
        order = np.argsort(omega)
        sc = ax.scatter(
            ant_pos[order, 0], ant_pos[order, 1],
            c=omega[order]*R2D, cmap=cmap, norm=norm,
            s=42, edgecolor='k', linewidth=0.4, alpha=0.95,
        )
        ax.scatter(core[0], core[1], marker='x', s=160, c='crimson', linewidths=2.5, label='Shower core')
        ax.set_title(title)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(r'$\omega$ [rad]')

    plt.suptitle(f'Event {event_num}: Antenna position colored by $\\omega$')
    plt.tight_layout()
    fpath = os.path.join(output_dir, 'omega_scatter_true_recon.pdf')
    plt.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close()
    return fpath


def plot_quality_summary(event_num, truth_dict, recon_dict, measured_signal,
                        noise_std, antenna_positions, xmax_pos,
                        outlier_reason, log_prob, output_dir):
    """Create a summary figure focused on reconstructed values and amplitude diagnostics."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    amps = get_amps(measured_signal)

    ax = fig.add_subplot(gs[0:2, 0:2])
    ax.axis('off')
    residuals_text = f"""
Residuals

ΔE = {recon_dict.get('dE_pct', np.nan):.2f} %
Δθ = {recon_dict.get('dtheta_deg', np.nan):.2f} °
Δφ = {recon_dict.get('dphi_deg', np.nan):.2f} °
ΔXmax = {recon_dict.get('dxmax_g', np.nan):.2f} g/cm²
"""
    ax.text(0.05, 0.95, residuals_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    ax.text(0.5, 0.5, 'Removed duplicate panel', ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

    ax = fig.add_subplot(gs[1, 2])
    n_ant = len(antenna_positions)
    ax.bar(['N Ant'], [n_ant], color=['orange'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax.set_title('Array Configuration', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(1, n_ant * 1.1))
    ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[2, 0:2])
    order = np.lexsort((antenna_positions[:, 1], antenna_positions[:, 0]))
    scatter = ax.scatter(antenna_positions[order, 0], antenna_positions[order, 1],
                         c=amps[order], s=70, cmap='viridis', edgecolor='k', alpha=0.9)
    ax.set_xlabel('X [m]', fontsize=10)
    ax.set_ylabel('Y [m]', fontsize=10)
    ax.set_title('Antenna Array (colored by amplitude)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Amplitude')

    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    metadata_text = f"""
Event: {event_num}

Outlier Type:
  {outlier_reason}

Quality Metrics:
  log P = {log_prob:.1f}
  N Ant = {n_ant}
  amp μ = {amps.mean():.1f}

Reconstructed:
  E_rec = {recon_dict.get('energy_em', np.nan):.2e} GeV
  θ_rec = {np.degrees(recon_dict.get('zenith', 0)):.1f}°
  Xmax_rec = {recon_dict.get('xmax_pos', [0,0,0])[2]:.0f} m
"""
    ax.text(0.05, 0.95, metadata_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Event {event_num}: Quality Summary Dashboard',
                 fontsize=13, fontweight='bold', y=0.995)

    fpath = os.path.join(output_dir, 'quality_summary.pdf')
    plt.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close()

    return fpath


def plot_mcmc_diagnostics(samples, burn_in=100, param_names=None, event_num=None, output_dir=None):
    """
    Plot MCMC sample diagnostics: posteriors, correlation, traces.
    
    Parameters
    ----------
    samples : ndarray, shape (n_steps, n_walkers, n_params)
        MCMC samples
    burn_in : int
        Burn-in steps
    param_names : list of str
        Parameter names
    event_num : int
        Event number (for title)
    output_dir : str
        Output directory path
    """
    if param_names is None:
        param_names = ['xmax_x', 'xmax_y', 'xmax_z', 'zenith', 'azimuth', 'logE']
    
    n_params = samples.shape[-1]
    
    # Burn-in and reshape
    samples_burned = samples[burn_in:].reshape(-1, n_params)
    
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 2*n_params))
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_params):
        # Posterior histogram
        ax = axes[i, 0]
        ax.hist(samples_burned[:, i], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(samples_burned[:, i]), color='r', linestyle='--', 
                  label=f'μ={np.mean(samples_burned[:, i]):.2e}', linewidth=2)
        ax.set_ylabel('Count')
        ax.set_title(f'{param_names[i]} Posterior')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Trace plot
        ax = axes[i, 1]
        trace = samples[:, :, i]
        for walker in range(min(trace.shape[1], 10)):  # Limit to 10 walkers for clarity
            ax.plot(trace[:, walker], alpha=0.5, linewidth=0.5)
        ax.axvline(x=burn_in, color='r', linestyle='--', label='Burn-in', linewidth=2)
        ax.set_ylabel(f'{param_names[i]}')
        ax.set_xlabel('Step')
        ax.set_title(f'{param_names[i]} Trace')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Event {event_num}: MCMC Diagnostics' if event_num else 'MCMC Diagnostics')
    plt.tight_layout()
    
    if output_dir:
        fpath = os.path.join(output_dir, 'mcmc_diagnostics.pdf')
        plt.savefig(fpath, dpi=100, bbox_inches='tight')
        plt.close()
        return fpath
    else:
        plt.show()
        return None
