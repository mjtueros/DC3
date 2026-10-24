"""
Selectable optimizers for maximizing a log-probability function.

Provides wrappers for different optimizers (Minuit via `iminuit` and
Differential Evolution via `scipy.optimize`) with automatic handling of
vectorized and non-vectorized `log_prob` call signatures.

Functions
---------
maximize_log_prob(log_prob, x0, bounds=None, method='differential_evolution',
                  vectorized=None, options=None)
    Maximize `log_prob` using the chosen optimizer and return standardized
    result information.

Notes
-----
- The optimizers call a *minimizer*, so we internally minimize `-log_prob`.
- If `vectorized` is None, the routine will try to autodetect whether
  `log_prob` accepts a batch (2D) input and returns an array of values.
- `iminuit` and `scipy` are optional dependencies; informative ImportError
  messages are raised if they're missing when a backend is selected.

Example
-------
>>> from pulse_emulator.optimizers import maximize_log_prob

>>> def logp(x):
...     # simple gaussian log-prob centered at zero
...     return -0.5 * (x**2).sum()

>>> out = maximize_log_prob(logp, x0=[1.0, -0.5], method='differential_evolution')
>>> print(out['x'], out['log_prob'])
"""

from collections.abc import Callable, Sequence
from typing import Any

import emcee
import numpy as np
from iminuit import Minuit
from scipy.optimize import differential_evolution, minimize

R2D = 180.0 / np.pi

SAFE_COST = 1e100


def _strip_method_prefix(method: str, *names: str) -> str:
    """Remove the leading stage name from a chained method string.

    'de->nelder-mead' -> '->nelder-mead'. Only the prefix is removed (a plain
    str.replace would also eat the 'de' inside 'nelder-mead'); the longest
    matching name wins, so 'differential_evolution' is not cut as 'di...'.
    """
    for name in sorted(names, key=len, reverse=True):
        if method.startswith(name):
            return method[len(name):]
    return method

def _ensure_1d_x(x0: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x0, dtype=float)
    if arr.ndim == 0:
        return arr.reshape((1,))
    if arr.ndim > 1:
        # If user passed a 2D array (e.g. walkers), take the first row as initial guess
        return arr.reshape(-1)[: arr.shape[-1]]
    return arr




def _make_neg_logprob(log_prob: Callable, vectorized: bool):
    """Return a callable f(x) -> -log_prob(x) that accepts 1D x arrays.

    The returned function always accepts a 1-D array-like `x` for
    optimizers that call the function with a single parameter vector.
    """

    def neg_f(x: Sequence[float]) -> float:
        x = np.asarray(x, dtype=float)
        # Ensure a 2D array for vectorized log_prob
        if vectorized:
            val = log_prob(np.atleast_2d(x))
            # Expecting an array-like; take the first element
            val0 = float(np.asarray(val).reshape(-1)[0])
        else:
            # Non-vectorized: assume log_prob accepts 1D arrays
            val0 = float(log_prob(x))
        # If result is not finite, return large positive cost
        if not np.isfinite(val0):
            return SAFE_COST
        return -val0

    return neg_f


def maximize_log_prob(
    log_prob: Callable,
    x0: Sequence[float],
    vectorized: bool | None = False,
    bounds: Sequence[tuple[float, float]] | None = None,
    method: str = "differential_evolution",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Maximize `log_prob` using a selectable optimizer.

    Parameters
    ----------
    log_prob : callable
        Function returning the log-probability. Can be vectorized: accept a
        2D array (N, D) and return array (N,) of log-probs, or non-vectorized
        that accepts a 1D array and returns scalar.
    x0 : sequence
        Initial guess for parameters (1D sequence) or an array of walkers
        (2D) — the first row is used as initial guess.
    bounds : sequence of (low, high) pairs, optional
        Bounds for parameters (required for `differential_evolution`). If not
        provided for DE a default wide window around `x0` is used.
    method : str
        One of: 'differential_evolution' (global), 'minuit' (gradient-based),
        'l-bfgs-b' (gradient-based, bounded), 'nelder-mead' (simplex),
        'powell' (quadratic interpolation, robust), 'trust-constr' (trust-region),
        'cobyla' (derivative-free, robust), or two-stage methods like
        'de->powell', 'de->trust-constr', 'de->cobyla', etc.
    vectorized : bool or None
        If None, will attempt to autodetect vectorization. Otherwise force.
    options : dict, optional
        Extra options passed to the chosen optimiser.

    Returns
    -------
    result : dict
        Dictionary containing at least: 'x' (best-fit array), 'log_prob'
        (maximized log-probability value), 'success' (bool) and
        'details' (optimizer-specific raw result).
    """

    x0 = _ensure_1d_x(x0)
    ndim = x0.shape[0]
    if options is None:
        options = {}

    
    # Note: Optimizers don't need vectorized evaluation:
    # - differential_evolution: can use vectorized=True as an optimization, but works fine without it
    # - minuit, l-bfgs-b, nelder-mead: always call with single points
    # Our function wrappers aren't designed for DE's vectorized interface (which requires
    # (popsize, ndim) -> (popsize,) signature). If the caller did not specify whether
    # `log_prob` is vectorized, attempt a safe autodetection using `x0`. Fall back to
    # non-vectorized mode on any exception to avoid raising within optimizers.    
    method = method.lower()  # Normalize method name for comparison
    
    neg_logprob = _make_neg_logprob(log_prob, vectorized)


    neg_for_opt = neg_logprob

    if method.startswith("differential_evolution") or method.startswith("de"):
        options_here = options.get('de', {})
        neg_for_opt_de = neg_logprob
        if bounds is None:
            # If no bounds are provided, use broad physical ranges for absolute parameters.
            raise ValueError("Bounds must be provided for differential evolution.")
        elif len(bounds) != ndim:
            raise ValueError(f"Bounds length {len(bounds)} does not match number of parameters {ndim}")
        res = differential_evolution(neg_for_opt_de, bounds=bounds, **options_here)
        best_x = np.asarray(res.x, dtype=float)
        best_logp = -float(res.fun)
        meta = {
            "nfev": getattr(res, "nfev", None),
            "nit": getattr(res, "nit", None),
            "message": getattr(res, "message", None),
        }
        method_left = _strip_method_prefix(method, "differential_evolution", "de")
        res = {'method':'de', 
               "x": best_x,
               "log_prob": best_logp,
               "success": bool(res.success),
               "uncertainties": np.std(getattr(res, "population", np.empty((0, ndim))), axis=0).tolist() if hasattr(res, "population") else None,
               "details": res,
               "meta": meta,
            }

        if len(method_left)==0:
            return (res, )
        else:
            # If method is something like 'de->powell', run the second stage refinement
            refine_method = method_left[2:]
            return (res, *maximize_log_prob(log_prob, x0=best_x, bounds=bounds, method=refine_method, vectorized=vectorized, options=options))

    elif method.startswith("minuit"):
        options_here = options.get("minuit", {})
        # Minuit expects named parameters; build parameter names p0..p{D-1}
        names = [f"p{i}" for i in range(ndim)]

        def f_minuit(*params):
            # Minuit passes positional parameters to the cost function.
            x = np.asarray(params, dtype=float)
            return neg_logprob(x)

        m = Minuit(f_minuit, *map(float, x0), name=names)

        # Apply bounds if provided
        if bounds is not None:
            for i, b in enumerate(bounds):
                if b is not None:
                    m.limits[names[i]] = b

        # Apply user options to Minuit (like tol, errordef, print_level)
        for k, v in options_here.get("minuit_kwargs", {}).items():
            setattr(m, k, v)

        # Run migrad (default) and Hesse to estimate errors
        m.migrad()
        try:
            m.hesse()
        except Exception:
            # Hesse may fail; ignore but keep minimization
            pass

        best_x = np.array([m.values[n] for n in names], dtype=float)
        best_logp = -float(m.fval)
        uncertainties = np.array([float(m.errors[n]) for n in names], dtype=float)
        uncertainties = np.where(np.isfinite(uncertainties), uncertainties, np.nan)
        meta = {
            "nfcn": getattr(m, "nfcn", None),
            "edm": getattr(m.fmin, "edm", None),
            "is_valid": bool(m.fmin.is_valid),
            "has_valid_parameters": bool(m.fmin.has_valid_parameters),
            "has_accurate_covar": bool(m.fmin.has_accurate_covar),
            "has_made_posdef_covar": bool(m.fmin.has_made_posdef_covar),
            "has_reached_call_limit": bool(m.fmin.has_reached_call_limit),
            "message": str(m.fmin),
        }
        details = {"minuit": m}
        success = bool(m.fmin.has_valid_parameters) and not bool(m.fmin.has_reached_call_limit)
        res = {'method':'minuit',
               "x": best_x,
               "log_prob": best_logp,
               "success": success,
               "uncertainties": uncertainties,
               "details": details,
               "meta": meta,
            }
        
        method_left = _strip_method_prefix(method, "minuit")
        if len(method_left) == 0:
            return (res, )
        else:
            # If method is something like 'minuit->powell', run the second stage refinement
            refine_method = method_left[2:]
            return (res, *maximize_log_prob(log_prob, x0=best_x, bounds=bounds, method=refine_method, vectorized=vectorized, options=options))

    elif method.startswith("nelder-mead") or method.startswith("nm"):
        options_here = options.get('nm', {})
        scipy_options = {k: v for k, v in options_here.items() if k not in {"popsize", "tol", "atol", "minuit_kwargs"}}
        res = minimize(lambda x: neg_logprob(x), x0, method="Nelder-Mead", options=scipy_options)
        best_x = np.asarray(res.x, dtype=float)
        best_logp = -float(res.fun)
        meta = {
            "nfev": getattr(res, "nfev", None),
            "nit": getattr(res, "nit", None),
            "message": getattr(res, "message", None),
        }
        method_left = _strip_method_prefix(method, "nelder-mead", "nm")
        res = {'method':'nm',
                "x": best_x,
                "log_prob": best_logp,
                "success": bool(res.success),
                "uncertainties": None,
                "details": res,
                "meta": meta,
            }
        if len(method_left) == 0:
            return (res, )
        else:
            # If method is something like 'nm->powell', run the second stage refinement
            refine_method = method_left[2:]
            return (res, *maximize_log_prob(log_prob, x0=best_x, bounds=bounds, method=refine_method, vectorized=vectorized, options=options))

    elif method.startswith("l-bfgs-b") or method.startswith("l_bfgs_b") or method.startswith("lbfgsb"):
        options_here = options.get('l-bfgs-b', {})
        # L-BFGS-B via scipy minimize with bounds support

        scipy_options = {k: v for k, v in options_here.items() if k not in {"popsize", "tol", "atol", "minuit_kwargs"}}
        res = minimize(lambda x: neg_logprob(x), x0, method="L-BFGS-B", bounds=bounds, options=scipy_options)
        best_x = np.asarray(res.x, dtype=float)
        best_logp = -float(res.fun)
        meta = {
            "nfev": getattr(res, "nfev", None),
            "nit": getattr(res, "nit", None),
            "message": getattr(res, "message", None),
        }
        method_left = _strip_method_prefix(method, "l-bfgs-b", "l_bfgs_b", "lbfgsb")
        res = {'method':'l-bfgs-b',
                "x": best_x,
                "log_prob": best_logp,
                "success": bool(res.success),
                "uncertainties": None,
                "details": res,
                "meta": meta,
            }
        if len(method_left) == 0:
            return (res, ) 
        else:   
            # If method is something like 'l-bfgs-b->powell', run the second stage refinement
            refine_method = method_left[2:]
            return (res, *maximize_log_prob(log_prob, x0=best_x, bounds=bounds, method=refine_method, vectorized=vectorized, options=options))

    elif method.startswith("powell") or method.startswith("pw"):
        options_here = options.get('powell', {})
        # Powell: quadratic interpolation, robust to noisy objectives
        scipy_options = {k: v for k, v in options_here.items() if k not in {"popsize", "tol", "atol", "minuit_kwargs"}}
        res = minimize(lambda x: neg_logprob(x), x0, method="Powell", options=scipy_options)
        best_x = np.asarray(res.x, dtype=float)
        best_logp = -float(res.fun)
        meta = {
            "nfev": getattr(res, "nfev", None),
            "nit": getattr(res, "nit", None),
            "message": getattr(res, "message", None),
        }
        method_left = _strip_method_prefix(method, "powell", "pw")
        res = {'method':'powell',
                "x": best_x,
                "log_prob": best_logp,
                "success": bool(res.success),
                "uncertainties": None,
                "details": res,
                "meta": meta,
            }
        if len(method_left) == 0:
            return (res, )
        else:
            # If method is something like 'powell->l-bfgs-b', run the second stage refinement
            refine_method = method_left[2:]
            return (res, *maximize_log_prob(log_prob, x0=best_x, bounds=bounds, method=refine_method, vectorized=vectorized, options=options))

    elif method.startswith("trust-constr") or method.startswith("trust_constr") or method.startswith("tc"):
        options_here = options.get('tc', {})
        # Trust-region with constraints: very stable for rough objectives and bounds
        scipy_options = {k: v for k, v in options_here.items() if k not in {"popsize", "tol", "atol", "minuit_kwargs"}}
        res = minimize(lambda x: neg_logprob(x), x0, method="trust-constr", bounds=bounds, options=scipy_options)
        best_x = np.asarray(res.x, dtype=float)
        best_logp = -float(res.fun)
        meta = {
            "nfev": getattr(res, "nfev", None),
            "nit": getattr(res, "nit", None),
            "message": getattr(res, "message", None),
        }
        method_left = _strip_method_prefix(method, "trust-constr", "trust_constr", "tc")
        res = {'method':'trust-constr',
                "x": best_x,
                "log_prob": best_logp,
                "success": bool(res.success),
                "uncertainties": None,
                "details": res,
                "meta": meta,
            }
        if len(method_left) == 0:  # No refinement stage
            return (res, )
        else:
            # If method is something like 'trust-constr->powell', run the second stage refinement
            refine_method = method_left[2:]
            return (res, *maximize_log_prob(log_prob, x0=best_x, bounds=bounds, method=refine_method, vectorized=vectorized, options=options))

    elif method.startswith("cobyla") or method.startswith("cb"):
        # COBYLA: derivative-free, constraint-aware, robust to non-smooth objectives
        options_here = options.get('cobyla', {})
        scipy_options = {k: v for k, v in options_here.items() if k not in {"popsize", "tol", "atol", "minuit_kwargs"}}
        res = minimize(lambda x: neg_logprob(x), x0, method="COBYLA", options=scipy_options)
        best_x = np.asarray(res.x, dtype=float)
        best_logp = -float(res.fun)
        meta = {
            "nfev": getattr(res, "nfev", None),
            "message": getattr(res, "message", None),
        }
        method_left = _strip_method_prefix(method, "cobyla", "cb")
        res = {'method':'cobyla',
                "x": best_x,
                "log_prob": best_logp,
                "success": bool(res.success),
                "uncertainties": None,
                "details": res,
                "meta": meta,
            }
        if len(method_left) == 0:
            return (res, )
        else:
            # If method is something like 'cobyla->powell', run the second stage refinement
            refine_method = method_left[2:]
            return (res, *maximize_log_prob(log_prob, x0=best_x, bounds=bounds, method=refine_method, vectorized=vectorized, options=options))

    elif method.startswith("emcee"):
        options_here = options.get('emcee', {})
        x_center = x0
        scales = np.array([0.3, 0.1, 0.1, 0.2, 0.1, 0.1], dtype=float)
        rng = np.random.RandomState(42)
        x0_walkers = x_center[None, :] + rng.randn(options_here['n_walkers'], ndim) * scales[None, :]


        sampler = emcee.EnsembleSampler(
            x0_walkers.shape[0], ndim,
            log_prob,
            vectorize=vectorized,
        )
        
        sampler.run_mcmc(x0_walkers, options_here['n_steps'], progress=options_here.get('progress', True))
        samples = sampler.get_chain()
        log_probas = sampler.get_log_prob()

        burn_in = int(options_here.get('burn_in', max(1, options_here['n_steps'] // 2)))
        samples_post = samples[burn_in:, :, :]
        n_steps_after, n_walkers, n_dim = samples_post.shape
        samples_flat = samples_post.reshape(-1, n_dim)
        log_probas_flat = log_probas[burn_in:, :].reshape(-1)
        best_idx = np.argmax(log_probas_flat)
        best_x = samples_flat[best_idx]
        best_logp = log_probas_flat[best_idx]
        meta = {
            "n_steps": options_here['n_steps'],
            "burn_in": burn_in,
            "n_walkers": options_here['n_walkers'],
            "acceptance_fraction": np.mean(log_probas[1:, :] > log_probas[:-1, :]),
        }
        return ({
            "method": "emcee",
            "x": best_x,
            "log_prob": best_logp,
            "success": True,  # MCMC doesn't have a traditional success flag
            "uncertainties": None,
            "details": {
                "samples": samples,
                "log_probas": log_probas,
                'sampler':sampler
            },
            "meta": meta,
        }, )

    else:
        raise ValueError(f"Unknown optimization method: {method}")



def mcmc_emcee_optimized(
        log_prob: Callable,
        x0: Sequence[float],
        vectorized: bool | None = None,
        options: dict[str, Any] | None = None,
    ):
        
    """
    Run MCMC using emcee with vectorized log_prob.
    
    Uses vectorize=True in emcee.EnsembleSampler for batch evaluation.
    """
    # x0, ctx, n_steps=10000, seed=None, progress=True,
    # amplitude_based=False, matched_filtering=False,
    # likelihood_type=None, vectorized=True
    
    nwalkers = x0.shape[0]
    ndim = x0.shape[-1]
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim,
        log_prob,
        vectorize=vectorized,
    )
    
    sampler.run_mcmc(x0, options['maxiter'], progress=options.get('progress', True))
    samples = sampler.get_chain()
    log_probas = sampler.get_log_prob()
    return samples, log_probas

# def mcmc_emcee_smearing(x0, ctx, n_steps=10000, seed=None, progress=True, amplitude_based=False):
#     """
#     Run MCMC using emcee with vectorized log_prob.
    
#     Uses vectorize=True in emcee.EnsembleSampler for batch evaluation.
#     """
    
#     nwalkers = x0.shape[0]
#     ndim = x0.shape[-1]
    
#     sampler = emcee.EnsembleSampler(
#         nwalkers, ndim, 
#         _log_prob_pulse_smearing,
#         args=(ctx,),
#         vectorize=True  # Key optimization: emcee passes all walkers at once
#     )
    
#     sampler.run_mcmc(x0, n_steps, progress=progress)
#     samples = sampler.get_chain()
#     log_probas = sampler.get_log_prob()
#     return samples, log_probas



