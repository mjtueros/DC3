import matplotlib.pyplot as plt
import numpy as np

from pulse_emulator.recons.prob import (
    event_swf_time,
    make_input_array,
    predict_voltage,
)


def plot_true_event(ctx, Xs=None, theta=None, phi=None, E_em=None, show=True, save_path=None):
    """Generate the matched-filter diagnostic figure for a given parameter set.

    Parameters
    ----------
    ctx : EventContext
        Precomputed event context.
    Xs : array-like, shape (3,), optional
        Xmax position in Cartesian coordinates (meters). If None, uses
        the event's Xmax from `ctx.base_values`.
    theta : float, optional
        Zenith angle in radians. If None, uses value from `ctx.base_values`.
    phi : float, optional
        Azimuth angle in radians. If None, uses value from `ctx.base_values`.
    E_em : float, optional
        Electromagnetic energy. If None, uses value from `ctx.base_values`.
    show : bool
        Whether to call `plt.show()`.
    save_path : str or None
        If provided, the figure will be saved to this path.
    """
    params = np.zeros(6)

    # Build a `values` dict overriding base values when explicit inputs are provided
    values = dict(ctx.base_values)
    n_ant = ctx.du_pos.shape[0]
    if Xs is None:
        Xs = np.array([ctx.base_values['xmax_pos_x'][0], ctx.base_values['xmax_pos_y'][0], ctx.base_values['xmax_pos_z'][0]])
    if theta is None:
        theta = ctx.base_values['zenith'][0]
    if phi is None:
        phi = ctx.base_values['azimuth'][0]
    if E_em is None and 'energy_em' in ctx.base_values:
        E_em = ctx.base_values['energy_em'][0]

    # Overwrite base_values arrays with the provided scalars (repeated per antenna)
    values['xmax_pos_x'] = np.full(n_ant, Xs[0])
    values['xmax_pos_y'] = np.full(n_ant, Xs[1])
    values['xmax_pos_z'] = np.full(n_ant, Xs[2])
    if 'energy_em' in values:
        values['energy_em'] = np.full(n_ant, E_em)
    values['zenith'] = np.full(n_ant, theta)
    values['azimuth'] = np.full(n_ant, phi)

    dt = 1 / ctx.fs_ds
    N = int(ctx.duration * ctx.fs_ds)
    df = 1 / ctx.duration

    # Prepare input arrays and predicted voltage using the (possibly) overridden values
    antenna_pos = ctx.du_pos
    Xs_first = Xs
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    voltage_traces_f = predict_voltage(
        input_arr,
        kxB_loc,
        antenna_pos,
        Xs_first,
        ctx.model,
        ctx.fs_ds,
        ctx.t_SN,
        ctx.t_EW,
        ctx.t_Z,
        ctx.tf,
        duration=ctx.duration,
        compute_td=False,
        slope_offset=ctx.slope_offset,
    )

    # timing and scaling
    t_swf = event_swf_time(Xs_first, antenna_pos)
    t_swf -= t_swf.mean()
    voltage_traces_f = voltage_traces_f * dt

    # correlations (same formula as in prob.log_likelihood_matched_filtering_interf)
    correlations = 2 * np.sum(
        np.fft.irfft(ctx.measured_fft_over_psd * np.conjugate(voltage_traces_f) * df * N),
        axis=1,
    )
    correlations -= np.sum(2 * np.abs(voltage_traces_f) ** 2 / ctx.psd * df, axis=(1, 2))[:, None]
    correlations -= np.sum(2 * np.abs(ctx.measured_fft) ** 2 / ctx.psd * df, axis=(1, 2))[:, None]

    maxes = np.max(correlations, axis=-1)
    probas = np.exp(correlations - maxes[:, None])

    # jitter convolution
    g = ctx.jitter_kernel
    from scipy.signal import convolve

    v = np.maximum(convolve(probas, g[None, :], mode="same"), 1e-300)
    out = maxes[:, None] + np.log(v)

    # align traces using event times
    delta_ts = t_swf - ctx.times_noisy_centered
    shifts = np.rint(delta_ts * ctx.fs_ds).astype(int)
    idx = (np.arange(out.shape[-1])[None, :] - shifts[:, None]) % out.shape[-1]
    out = np.take_along_axis(out, idx, axis=-1)
    aligned_trace = np.sum(out, axis=0)

    # Build the figure (replicates the original diagnostic)
    fig, ax = plt.subplots(4, 1, figsize=(10, 8))
    t = np.arange(N) * dt
    ax[0].plot(t, ctx.measured_trimmed[0, 0], label="Measured")
    ax[0].plot(t, np.fft.irfft(voltage_traces_f / dt)[0, 0], label="Predicted")

    ax[1].plot(correlations[0] - correlations[0].max(), label="correlation")
    ax[1].set_xlabel("Time bins")
    ax[1].set_title("Correlation and jitter-convolved traces (antenna 0)")
    ax[1].legend()

    ax[2].plot(probas[0], label="raw probas")
    ax[2].plot(v[0], label="convolved probas")
    ax[2].plot(g, label="jitter kernel")
    ax[2].set_xlabel("Time bins")
    ax[2].set_title("Raw probas, jitter-convolved probas, and jitter kernel (antenna 0)")
    ax[2].legend()

    for i in range(len(out)):
        ax[3].plot(np.exp(out[i]-out[i].max()), c="gray", lw=1, alpha=0.5)
    ax[3].plot(np.exp(aligned_trace-aligned_trace.max()), c="C1", lw=2)
    ax[3].set_xlabel("Time bins")
    ax[3].set_title("Aligned traces across antennas and their sum")

    plt.tight_layout()
    fig.suptitle(f"Matched-filter diagnostic:\nLikelihood: {aligned_trace.max()}", fontsize=12, fontweight='bold')
    plt.subplots_adjust(top=0.88)
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return voltage_traces_f, (fig, ax)


def plot_swf(X_s, X_ants, times_noisy, sigma_t, ax = None, label='Time residuals'):
    """Plot SWF times vs noisy times with uncertainty."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    swf_times = event_swf_time(X_s, X_ants)
    swf_times -= swf_times.mean()  # Align the average time to zero
    times_noisy_centered = times_noisy - times_noisy.mean()
    ax.errorbar(times_noisy_centered, swf_times-times_noisy_centered, yerr=sigma_t, fmt='o', label=label)
    ax.plot([times_noisy_centered.min(), times_noisy_centered.max()], [0, 0], 'r--', label='Ideal SWF = Noisy times')
    ax.set_xlabel('Noisy times (centered) (µs)')
    ax.set_ylabel('SWF times (µs)')
    residuals = swf_times - times_noisy_centered
    # plt.title(f'SWF vs Noisy Times with σ={sigma_t*1e3:.1f} ns, Log prob={-np.sum(( residuals / sigma_t )**2)/2:.3f}')
    ax.set_title(f'SWF vs Noisy Times with σ={sigma_t*1e3:.1f} ns, $\\chi_{{red}}^2$={np.sum(( residuals / sigma_t )**2)/2 / (len(times_noisy)-3):.2f}')
    ax.legend()
    ax.grid()
    return ax
    


def plot_interactive_du_traces(ctx, voltage_traces_f):
    """Create an interactive scatter plot of DU positions colored by correlation max index.
    
    Clicking on any DU point opens a figure with 3 subplots showing the measured
    vs predicted voltage traces for that DU across all 3 axes, with correlation
    displayed on a secondary axis.
    
    Parameters
    ----------
    ctx : EventContext
        Context object containing DU positions, measured traces, times, and sampling info.
    voltage_traces_f : ndarray
        Predicted voltage traces in frequency domain from predict_voltage().
    """
    dt = 1 / ctx.fs_ds
    N = int(ctx.duration * ctx.fs_ds)
    df = 1 / ctx.duration
    
    # Compute correlations (same formula as in plot_true_event)
    correlations = 2 * np.fft.irfft(ctx.measured_fft_over_psd * np.conjugate(voltage_traces_f) * df * N)
    correlations -= np.sum(2 * np.abs(voltage_traces_f) ** 2 / ctx.psd * df, axis=(2))[:, :, None]
    # Apply Tukey window to suppress boundary oscillations
    # from scipy.signal.windows import tukey
    # window = tukey(correlations.shape[-1], alpha=0.1)
    # correlations = correlations * window[None, :]
    
    # Get index of maximum correlation for coloring
    Xs = np.array([ctx.base_values['xmax_pos_x'][0], ctx.base_values['xmax_pos_y'][0], ctx.base_values['xmax_pos_z'][0]])
    swf_times = event_swf_time(Xs, ctx.du_pos)
    delta_ts = ctx.times_noisy_centered - swf_times
    delta_ts -= delta_ts.mean()
    max_corr_indices = np.argmax(np.sum(correlations, axis=1), axis=-1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with colors from max correlation index
    scatter = ax.scatter(
        ctx.du_pos[:, 0],
        ctx.du_pos[:, 1],
        c=delta_ts,
        s=150,
        cmap='viridis',
        picker=5,  # tolerance in points
        edgecolors='black',
        linewidth=0.5,
    )
    
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('DU Positions\n(Click on a DU to view measured vs predicted traces)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Max correlation index')
    
    # Convert voltage traces from frequency domain to time domain
    voltage_traces_t = np.fft.irfft(voltage_traces_f / dt)
    
    # Keep explicit references to detail figures so they are not garbage-collected
    # too early by some backends.
    detail_figures = []

    def on_pick(event):
        """Handle pick events on the scatter plot."""
        if event.artist != scatter:
            return
        
        du_indices = event.ind
        
        # Open a figure for each picked DU
        for du_i in du_indices:
            fig_detail, axes_detail = plt.subplots(3, 1, figsize=(12, 9))
            
            # Plot measured vs predicted for each axis
            for ax_i in range(3):
                ax_main = axes_detail[ax_i]
                ax_main.plot(
                    ctx.measured_trimmed[du_i, ax_i],
                    label='Measured',
                    linewidth=2,
                    color='C0',
                )
                ax_main.plot(
                    voltage_traces_t[du_i, ax_i],
                    label='Predicted',
                    linewidth=2,
                    alpha=0.8,
                    color='C1',
                )
                ax_main.set_xlabel('Time bin')
                ax_main.set_ylabel(f'Voltage (axis {ax_i})', color='C0')
                ax_main.legend(loc='upper right')
                ax_main.grid(alpha=0.3)
                
                # Add correlation on twin axis
                ax_corr = ax_main.twinx()
                ax_corr.plot(
                    np.exp(correlations[du_i, ax_i] - correlations[du_i, ax_i].max()),
                    label='Correlation',
                    linewidth=2,
                    alpha=0.7,
                    color='C2',
                    linestyle='--',
                )
                ax_corr.set_ylabel('Correlation', color='C2')
                ax_corr.tick_params(axis='y', labelcolor='C2')
            
            fig_detail.suptitle(
                f'DU {du_i}: Max correlation index = {max_corr_indices[du_i]}',
                fontsize=12,
                fontweight='bold'
            )
            fig_detail.tight_layout()
            detail_figures.append(fig_detail)

            # Force immediate popup rendering instead of waiting for a later
            # draw triggered by another figure.
            try:
                fig_detail.show()
                fig_detail.canvas.draw_idle()
                try:
                    fig_detail.canvas.flush_events()
                except Exception:
                    pass
                plt.pause(0.001)
            except Exception:
                try:
                    plt.show(block=False)
                    fig_detail.canvas.draw_idle()
                    try:
                        fig_detail.canvas.flush_events()
                    except Exception:
                        pass
                    try:
                        plt.pause(0.001)
                    except Exception:
                        pass
                except Exception:
                    pass
            
    
    # Connect the pick event handler
    fig.canvas.mpl_connect('pick_event', on_pick)
    # Show the interactive figure non-blocking without changing global interactive
    # mode so that subsequent `plt.show()` calls remain blocking.
    try:
        # Use figure-level show which is non-blocking in most GUI backends
        fig.show()
        fig.canvas.draw_idle()
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
        # Small pause to let the GUI process initial draw events
        try:
            plt.pause(0.001)
        except Exception:
            pass
    except Exception:
        # If the backend doesn't support fig.show(), fall back to non-blocking
        # plt.show but do not enable interactive mode globally.
        try:
            plt.show(block=False)
            fig.canvas.draw_idle()
            try:
                fig.canvas.flush_events()
            except Exception:
                pass
            try:
                plt.pause(0.001)
            except Exception:
                pass
        except Exception:
            # If all fails, silently continue — the plot may appear on the next blocking show()
            pass
    

