import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import psutil

from pulse_emulator.recons.optimizers import maximize_log_prob
from pulse_emulator.recons.prob import log_prob
from pulse_emulator.recons.recons_utils import (
    build_event_context,
    cart2sph_xmax,
    compute_swf_guess,
    corner_sns,
    post_treatment_emcee,
    prepare_event,
    sph2cart_xmax,
)
from pulse_emulator.utils import R2D, cart2sph, convert_to_grams, sph2cart
from pulse_emulator.viz.event_viewer import (
    plot_interactive_du_traces,
    plot_swf,
    plot_true_event,
)


def print_memory(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2  # MB
    print(f"[PID {os.getpid()}] {tag}: {mem:.1f} MB")
def _skip_record(event_number, reason, n_antennas=None):
    """Record for an event that cannot be reconstructed.

    Written to the skip list (not the results summary) so that --continue knows
    the event was already attempted and never retries it, while the summary CSV
    the analysis scripts read stays free of empty rows.
    """
    return {
        '__skipped__': True,
        'event_number': int(event_number),
        'reason': reason,
        'n_antennas': int(n_antennas) if n_antennas is not None else -1,
    }



def prepare_initial_guess(k_guess, Xs_guess):
    _, theta_guess, phi_guess = cart2sph(-k_guess[None,:])
    theta_guess = theta_guess[0]
    phi_guess = phi_guess[0]
    theta_guess = float(np.clip(theta_guess, 0.0, np.pi / 2))
    # Ensure angular conventions match priors: `log_prior_uninformative` expects
    # azimuth in [0, 2*pi]. Convert phi_guess into that range.
    phi_guess_mod = float(phi_guess % (2 * np.pi))

    # Convert Xs_guess to polar coordinates for optimization
    xmax_polar_guess = cart2sph_xmax(Xs_guess)
    r_guess_km = xmax_polar_guess[0][0]
    zen_guess = xmax_polar_guess[1][0]
    az_guess = xmax_polar_guess[2][0]
    
    # Initial guess in polar coordinates
    log_E_guess = np.clip(np.random.randn()*0.2+8.5, 7.6, 9.5)
    x0 = np.array([r_guess_km,
                   zen_guess*R2D,
                   az_guess*R2D,
                   log_E_guess,
                   theta_guess*R2D,
                   phi_guess_mod*R2D])
    bounds = [
        (max(10, r_guess_km - 50), r_guess_km + 50),        # r in km: ±50 km (clipped at 10 km min)
        (max(10, zen_guess*R2D - 1), min(89.5, zen_guess*R2D + 1)),  # zenith ±5°
        (az_guess*R2D - 1, az_guess*R2D + 1),  # azimuth ±5° (sin/cos handle any angle)
        (7.3, 9.6),                      # log10(E)
        (max(0, theta_guess*R2D - 6), min(89, theta_guess*R2D + 6)),  # zenith ±1°
        (phi_guess_mod*R2D - 6, phi_guess_mod*R2D + 6)  # azimuth ±1°
    ]
    return x0, bounds

def compute_residuals(res, k_true, true_xmaxgr, truxmaxpos, true_logE, suffix=""):
    _, theta_true, phi_true = cart2sph(-k_true[None,:])

    x_cart = np.array([
        res[f'x_best_0{suffix}'],
        res[f'x_best_1{suffix}'],
        res[f'x_best_2{suffix}'],
        res[f'x_best_3{suffix}'],
        res[f'x_best_4{suffix}'],
        res[f'x_best_5{suffix}'],
    ])
    if x_cart is None:
        raise ValueError(f"Result dict does not contain 'x_best{suffix}' for residual computation.")
    xmax_rec_cart, logE_rec, theta_rec, phi_rec = x_cart[:3], x_cart[3], x_cart[4], x_cart[5]
    k_rec = -sph2cart(theta_rec, phi_rec).flatten()
    # Compute residuals
    spatial_err = np.linalg.norm(xmax_rec_cart - truxmaxpos)
    xmaxgr_rec = res[f'xmaxgr{suffix}']
    xmaxgr_err = xmaxgr_rec - true_xmaxgr
    logE_err = logE_rec - true_logE

    # Attach to result dict
    res[f'spatial_error{suffix}'] = spatial_err
    res[f'xmaxgr_error{suffix}'] = xmaxgr_err
    res[f'logE_error{suffix}'] = logE_err
    res[f'E_rel_error{suffix}'] = 10**logE_rec / 10**true_logE - 1
    res[f'theta_error{suffix}'] = (theta_rec - theta_true[0]) * R2D
    res[f'phi_error{suffix}'] = ((phi_rec - phi_true[0]) * R2D) % 360 
    res[f"angular_error{suffix}"] = np.arccos(np.clip(np.dot(k_rec, k_true), -1.0, 1.0)) * R2D

    return res


def prepare_output(all_res, ctx, config, k_guess, Xs_guess):


    true_zenith, true_azimuth = ctx.base_values['zenith'][0], ctx.base_values['azimuth'][0]
    k_true = -sph2cart(true_zenith, true_azimuth).flatten()  # shape (3,)
    true_xmax = np.array([ctx.base_values['xmax_pos_x'][0], 
                          ctx.base_values['xmax_pos_y'][0], 
                          ctx.base_values['xmax_pos_z'][0]])
    r_true_km, zen_true, az_true = cart2sph_xmax(true_xmax)
    true_log_E = np.log10(ctx.base_values['energy_em'][0])
    true_log_prob = log_prob(np.array([true_xmax[0], true_xmax[1], true_xmax[2], true_log_E, true_zenith, true_azimuth]), ctx, vectorized=False, likelihood_type=config.likelihood_type, prior=config.prior,
                             quadrature_method=config.quadrature_method, n_gh_pts=config.n_gh_pts, kappa=config.kappa, n_samples=config.n_samples)
    true_xmaxgr = convert_to_grams(true_xmax, true_zenith, true_azimuth)


    guess_angular_error = np.arccos(np.clip(np.dot(k_guess, k_true), -1.0, 1.0))
    _, theta_guess, phi_guess = cart2sph(-k_guess[None,:])
    guess_xmaxgr = convert_to_grams(Xs_guess, theta_guess[0], phi_guess[0])
    r_guess_km = np.linalg.norm(Xs_guess)/1e3

    # If user requested an MCMC follow-up (method ending with ->emcee), run emcee
    method_lower = config.method.lower()
    # Convert result from polar back to Cartesian
    Final_res = dict({})
    for i, res in enumerate(all_res):
        _suffix = f"_stage{i+1}" if i != len(all_res)-1 else ""
        current_method = res['method']
        best_x_polar = res['x'].reshape(-1)
        r_km_recon, zen_recon, az_recon = best_x_polar[:3]
        xmax_cart_recon = sph2cart_xmax(r_km_recon, zen_recon/R2D, az_recon/R2D)
        # Construct full Cartesian result
        best_x_cart = np.array([xmax_cart_recon[0], xmax_cart_recon[1], xmax_cart_recon[2], 
                    best_x_polar[3], best_x_polar[4]/R2D, best_x_polar[5]/R2D])
        Final_res[f'method{_suffix}'] = current_method
        Final_res[f'log_prob{_suffix}'] = res['log_prob']
        Final_res[f'success{_suffix}'] = res['success']
        Final_res[f'uncertainties{_suffix}'] = res.get('uncertainties', None)

        for i in range(6):
            Final_res[f'x_best_{i}{_suffix}'] = best_x_cart[i]
            Final_res[f'x_best_polar_{i}{_suffix}'] = best_x_polar[i]

        Final_res[f'xmax_r_km{_suffix}'] = r_km_recon
        Final_res[f'xmax_zenith{_suffix}'] = zen_recon/R2D
        Final_res[f'xmax_azimuth{_suffix}'] = az_recon/R2D
        Final_res[f'logE{_suffix}'] = best_x_polar[3]
        Final_res[f'theta{_suffix}'] = best_x_polar[4]/R2D
        Final_res[f'phi{_suffix}'] = best_x_polar[5]/R2D
        
        Final_res[f'xmax_pos_x{_suffix}'] = best_x_cart[0]
        Final_res[f'xmax_pos_y{_suffix}'] = best_x_cart[1]
        Final_res[f'xmax_pos_z{_suffix}'] = best_x_cart[2]

        Final_res[f'xmaxgr{_suffix}'] = convert_to_grams(best_x_cart[:3], best_x_cart[4], best_x_cart[5])

        Final_res = compute_residuals(Final_res, k_true, true_xmaxgr, true_xmax, true_log_E, _suffix)
        ## Getting into details

        if current_method.lower() == 'emcee':
            samples_cart = post_treatment_emcee(res, config, ctx)
    
    Final_res['event_number'] = int(ctx.event_number)
    Final_res['method'] = config.method.lower()

    if config.verbose:
        # print(true_xmax, true_zenith, true_azimuth, true_xmaxgr)
        print(f"Recons values: r={Final_res['xmax_r_km']:.2f} km, zen={Final_res['xmax_zenith']*R2D:.2f}°, az={Final_res['xmax_azimuth']*R2D:.2f}°, xmaxgr={Final_res['xmaxgr']:.1f} g/cm^2")
        print(f"Recons error: spatial={Final_res['spatial_error']:.2f} km, xmaxgr={Final_res['xmaxgr_error']:.1f} g/cm^2, logE={Final_res['logE_error']:.3f}, E_rel={Final_res['E_rel_error']*100:.1f}%, theta={Final_res['theta_error']:.2f}°, phi={Final_res['phi_error']:.2f}°")


    if config.plot:
        plot_title = f"event {int(ctx.event_number)} | method={config.method} | success={bool(Final_res.get('success', True))}"

        Xs_true = np.array((ctx.base_values['xmax_pos_x'][0], ctx.base_values['xmax_pos_y'][0], ctx.base_values['xmax_pos_z'][0]))
        theta_true = ctx.base_values['zenith'][0]
        phi_true = ctx.base_values['azimuth'][0]
        E_em_true = ctx.base_values['energy_em'][0]

        # true event traces
        true_voltages_f, (fig, ax) = plot_true_event(ctx, Xs_true, theta_true, phi_true, E_em_true, show=False)
        fig_title = fig.get_suptitle() if fig.get_suptitle() else ""
        fig.suptitle('True Parameters ' + fig_title, fontsize=10, fontweight='bold')
        plot_interactive_du_traces(ctx, true_voltages_f)

        # reconstructed traces using MAP or posterior mean
            # use MAP sample from MCMC (cartesian)
        Xs_rec_plot = np.array((Final_res['x_best_0'], Final_res['x_best_1'], Final_res['x_best_2']))
        E_rec_plot = 10 ** Final_res['x_best_3']
        theta_rec_plot = Final_res['x_best_4']
        phi_rec_plot = Final_res['x_best_5']

        rec_voltages_f, (fig, ax) = plot_true_event(ctx, Xs_rec_plot, theta_rec_plot, phi_rec_plot, E_rec_plot, show=False)
        fig_title = fig.get_suptitle() if fig.get_suptitle() else ""
        fig.suptitle('Reconstructed Parameters ' + fig_title, fontsize=10, fontweight='bold')
        plot_interactive_du_traces(ctx, rec_voltages_f)

        # SWF plot
        plt.show()

        # Corner plot from MCMC samples
        if 'emcee' in method_lower:
            try:
                labels = [r"$x_{\max,x}$", r"$x_{\max,y}$", r"$x_{\max,z}$", r"$\log_{10}(E)$", r"$\theta$", r"$\phi$"]
                samples_for_corner = samples_cart   # (N,6) in cart/rad units
                truths = [float(Xs_true[0]), float(Xs_true[1]), float(Xs_true[2]), np.log10(float(E_em_true)), float(theta_true), float(phi_true)]
                g = corner_sns(samples_for_corner, labels=labels, truths=truths, show=True, show_titles=True)
                # save corner
                # g.figure.savefig(os.path.join(event_out_dir, f"mcmc_corner_{int(ctx.event_number)}.png"), dpi=200)
            except Exception as e:
                print('Corner plotting failed:', e)
        
    return Final_res





def process_event(event_number, df_X_event, config, model, model_config,
                  t_SN, t_EW, t_Z, tf, noise_computer):
    """Run optimizer-based reconstruction for a single event.

    Returns a dict with best parameters and diagnostic plotting data, or a skip
    record (see `_skip_record`) if the event cannot be reconstructed.
    """
    if config.verbose:
        print_memory("start")

    # Cheap guard only: too few rows here can never yield enough antennas above
    # threshold, but the real cut is on amplitude and needs the traces. Rejecting
    # an event costs a trace load; that is small next to the DE+emcee fit, and on a
    # relaunch the skip list means it is not paid at all.
    if len(df_X_event) < config.min_antennas:
        return _skip_record(event_number, 'too_few_antennas_in_dataframe', len(df_X_event))

    prepared = prepare_event(event_number, df_X_event, config, t_SN, t_EW, t_Z, tf, noise_computer, model=model, model_config=model_config)
    if prepared is None:
        # prepare_event skips on insufficient triggered antennas or an unreadable
        # event. Record it so --continue does not retry it on every relaunch.
        return _skip_record(event_number, 'prepare_event_rejected', len(df_X_event))
    df_ev = prepared['df_ev']
    measured_signal = prepared['measured_signal']
    times_noisy_bin0 = prepared['times_noisy_bin0']
    noise_std = prepared['noise_std']
    tf_ds = prepared['tf_ds']
    del prepared

    ctx = build_event_context(
        df_ev=df_ev,
        measured_signal=measured_signal,
        times_noisy=times_noisy_bin0,
        noise_std=noise_std,
        model=model,
        model_config=model_config,
        config=config,
        t_SN=t_SN,
        t_EW=t_EW,
        t_Z=t_Z,
        tf_ds=tf_ds,
        noise_computer=noise_computer,
    )
    del df_ev, measured_signal, times_noisy_bin0, noise_std, tf_ds
    # print(f"Size : {sys.getsizeof(ctx)}MB")
    ## initial guess: zeros (no shift)
    # x0 = np.zeros(6)
    # initial guess: small random shift around zero
    # x0 = np.random.normal(loc=0.0, scale=[10000.0, 10000.0, 5000.0, 0.4, 0.5, 0.5], size=6)
    # initial guess: from pwf and swf
    k_guess, Xs_guess = compute_swf_guess(ctx.times_noisy_centered, ctx.du_pos, 
                                            amps=ctx.measured_amps, n_walkers=20, sigma_t=ctx.sigma_t)
    if config.plot:
        # plot_swf opens a figure it never closes, so this must stay behind the
        # flag: unconditionally it leaked two figures per event across a run.
        print('True xmax vs SWF guess:', ctx.xmax_pos[0], Xs_guess)
        plot_swf(ctx.xmax_pos[0], ctx.du_pos, ctx.times_noisy_centered, sigma_t=ctx.sigma_t)
        plot_swf(Xs_guess, ctx.du_pos, ctx.times_noisy_centered, sigma_t=ctx.sigma_t)
    true_theta, true_phi = ctx.base_values['zenith'][0], ctx.base_values['azimuth'][0]
    true_k= -sph2cart(true_theta, true_phi).flatten()
    true_xmax_pos = np.array([ctx.base_values['xmax_pos_x'][0], ctx.base_values['xmax_pos_y'][0], ctx.base_values['xmax_pos_z'][0]])
    x0, bounds = prepare_initial_guess(k_guess, Xs_guess)
    x0[3] = np.log10(ctx.base_values['energy_em'][0])  # set logE guess to true value for now (can randomize if desired)
    # concise startup log
    if config.verbose:
        print(f"Event {int(event_number)}: starting optimizer '{config.method}'")
        true_xmax_pos = np.array([ctx.base_values['xmax_pos_x'][0], ctx.base_values['xmax_pos_y'][0], ctx.base_values['xmax_pos_z'][0]])
        r_true, zen_true, az_true = cart2sph_xmax(true_xmax_pos)[:,0]
        true_theta, true_phi = ctx.base_values['zenith'][0], ctx.base_values['azimuth'][0]
        true_xmaxgr = convert_to_grams(true_xmax_pos, true_theta, true_phi)
        true_k= -sph2cart(true_theta, true_phi).flatten()

        print('True values (polar): '
              f"r={r_true:.2f} km, zen={zen_true*R2D:.2f}°, az={az_true*R2D:.2f}°, "
              f"theta={true_theta*R2D:.2f}°, phi={true_phi*R2D:.2f}°, "
              f"xmaxgr={true_xmaxgr:.2f} g/cm^2, logE={np.log10(ctx.base_values['energy_em'][0]):.2f}"
              )
        guess_xmaxgr = convert_to_grams(Xs_guess, true_theta, true_phi)
        guess_k = -sph2cart(x0[4]/R2D, x0[5]/R2D).flatten()
        print(f"Initial guess (polar): r={x0[0]:.2f} km, zen={x0[1]:.2f}°, az={x0[2]:.2f}°, "
              f"theta={x0[4]:.2f}°, phi={x0[5]:.2f}°, xmaxgr={guess_xmaxgr:.2f} g/cm^2")

        guess_angular_error = np.arccos(np.clip(np.dot(guess_k, true_k), -1.0, 1.0))
        print(f"Guess errors: "
              f"spatial={np.linalg.norm(Xs_guess-true_xmax_pos)/1e3:.2f} km, xmaxgr={guess_xmaxgr-true_xmaxgr:.1f} g/cm^2, "
              f"theta={x0[4]-true_theta*R2D:.2f}°, phi={(x0[5]-true_phi*R2D)%360:.2f}°, angular error={guess_angular_error*R2D:.2f}°")

    

    # Prepare options depending on chosen method. For composite methods
    # (de->minuit, de->l-bfgs-b) we rely on pulse_emulator.optimizers to run two stages.
    options = {}
    if 'de' in config.method or "differential_evolution" in config.method:
        options['de'] = {'popsize': config.nwalkers, 'maxiter': config.optim_steps, 'tol': 1e-4, 'atol': 1e-4}
    if "minuit" in config.method:
        options['minuit'] = {'edm_goal': 0.1, 'errordef': 0.5}
    if 'emcee' in config.method:
        options['emcee'] = {"n_walkers": config.nwalkers, 'n_steps': config.optim_steps, 'progress': config.verbose}
    

    t0 = time()
    # Wrapper that converts from polar to Cartesian before calling log_prob
    def logp_fn_polar(x_polar):
        # print('called')
        """Wrapper: accepts parameters in polar coords [r_km, zen, az, log_E, theta, phi]
        and converts xmax to Cartesian before calling log_prob."""
        x_polar = np.atleast_1d(x_polar)
        # Handle both 1D and 2D (vectorized) inputs
        if x_polar.ndim == 2 and x_polar.shape[0] > 1 and config.vectorized:
            n_samples = x_polar.shape[0]
            # Unpack polar coordinates (degrees) and convert all at once
            r_km, zen_deg, az_deg, log_E, theta_deg, phi_deg = x_polar[:, :].T
            
            # Convert angles from degrees to radians
            zen_rad, az_rad, theta_rad, phi_rad = zen_deg/R2D, az_deg/R2D, theta_deg/R2D, phi_deg/R2D
            
            X = sph2cart_xmax(r_km, zen_rad, az_rad)
            xmax_x, xmax_y, xmax_z = sph2cart_xmax(r_km, zen_rad, az_rad)
            x_cartesian = np.column_stack([xmax_x, xmax_y, xmax_z, log_E, theta_rad, phi_rad])
            return log_prob(x_cartesian, ctx, vectorized=True, likelihood_type=config.likelihood_type, prior=config.prior,
                            quadrature_method=config.quadrature_method, n_gh_pts=config.n_gh_pts, kappa=config.kappa, n_samples=config.n_samples)
        else:
            # Scalar: single evaluation
            r_km, zen, az, log_E, theta, phi = x_polar.reshape(-1)
            xmax_cart = sph2cart_xmax(r_km, zen/R2D, az/R2D)
            x_cart = np.array([xmax_cart[0], xmax_cart[1], xmax_cart[2], log_E, theta/R2D, phi/R2D])
            return log_prob(x_cart, ctx, vectorized=False, likelihood_type=config.likelihood_type, prior=config.prior,
                            quadrature_method=config.quadrature_method, n_gh_pts=config.n_gh_pts, kappa=config.kappa, n_samples=config.n_samples)
    
    print('Starting optimization...')
    all_res = maximize_log_prob(logp_fn_polar,
                                x0=x0, 
                                method=config.method, 
                                vectorized=config.vectorized, 
                                bounds=bounds, 
                                options=options)
    out = prepare_output(all_res, ctx, config, k_guess, Xs_guess)
    del ctx, k_guess, Xs_guess, all_res
    if config.verbose:
        print_memory("end")

    return out
