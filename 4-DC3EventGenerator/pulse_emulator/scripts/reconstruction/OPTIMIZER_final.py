import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from make_input import load_input_params_from_dict
from noise import compute_noise
from tqdm import tqdm

from pulse_emulator.recons.recons_func import process_event
from pulse_emulator.recons.recons_utils import ReconsConfig
from pulse_emulator.utils import load_datasets, load_model

torch.set_num_threads(1)  # main process too, not just pool workers -- see thread-cap note above
import multiprocessing as py_mp

sys.path.append('scripts/')  
from local_paths import (
    BASE_DATA_PATH,
    BASE_MODEL_PATH,
    BASE_ROOT,
    LFMAP_DIR,
    RF_PARAMS_CONFIG_FILE,
)


def _save_summary(results, path):
    for entry in results:
        if 'x_best_1' not in entry:
            continue
        # Ensure all values are JSON-serializable (e.g., convert numpy types to native Python types)
        for key, value in entry.items():
            if isinstance(value, np.ndarray) and value.size > 1:
                entry[key] = value.tolist()
            elif isinstance(value, np.ndarray) and value.size == 1:
                entry[key] = value.item()
            elif isinstance(value, (np.float32, np.float64)):
                entry[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                entry[key] = int(value)
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(path, index=False)

_WORKER_STATE = {}


def _init_worker_shared_state(df_X, config, model, model_config, t_SN, t_EW, t_Z, tf, noise_computer):
    # Keep heavy objects in worker globals so each task only sends event IDs.
    global _WORKER_STATE
    _WORKER_STATE = {
        'df_X': df_X,
        'config': config,
        'model': model,
        'model_config': model_config,
        't_SN': t_SN,
        't_EW': t_EW,
        't_Z': t_Z,
        'tf': tf,
        'noise_computer': noise_computer,
    }
    # Limit thread thrashing in parallel workers.
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'


def _process_event(event_number):
    st = _WORKER_STATE
    df_X_event = st['df_X'][st['df_X'].event_number == event_number]
    return process_event(
        event_number,
        df_X_event,
        st['config'],
        st['model'],
        st['model_config'],
        st['t_SN'],
        st['t_EW'],
        st['t_Z'],
        st['tf'],
        st['noise_computer'],
    )

def _save_skipped(skipped, path):
    if not skipped:
        return
    pd.DataFrame(skipped).to_csv(path, index=False)


def recons_all(df_X, event_ids, config, n_workers, model, model_config, t_SN, t_EW, t_Z, tf, noise_computer,
               csv_path, skipped_path, prev_results=(), prev_skipped=()):
    # Seed with the rows this shard already wrote (--continue) so that every periodic
    # save rewrites the full shard CSV, not just this run's rows: a job killed
    # mid-run must not lose the results of the previous runs.
    results = list(prev_results)
    skipped = list(prev_skipped)
    n_prev_results = len(prev_results)
    if n_workers == 1:
        # Sequential processing (allows plotting); load model once

        for event_number in event_ids[:]:
            df_X_event = df_X[df_X.event_number == event_number]
            out = process_event(event_number, df_X_event, config, model, model_config, t_SN, t_EW, t_Z, tf, noise_computer)
            if out is None:
                continue
            if out.pop('__skipped__', False):
                skipped.append(out)
                if len(skipped) % 20 == 0:
                    _save_skipped(skipped, skipped_path)
                continue
            results.append(out)
            # plotting (if enabled) happens inside process_event
            # Periodic save
            if len(results) % 10 == 0 and len(results) > 0:
                _save_summary(results, csv_path)
            if (len(results) - n_prev_results) >= config.max_events:
                break
    else:
        # Parallel processing using shared worker state to avoid per-task copies.
        if config.plot:
            print('Warning: Plotting is disabled in parallel mode.', flush=True)
            config.plot = False
            config.verbose = False

        mp_ctx = py_mp.get_context('fork')

        with mp_ctx.Pool(
            processes=n_workers,
            initializer=_init_worker_shared_state,
            initargs=(df_X, config, model, model_config, t_SN, t_EW, t_Z, tf, noise_computer),
        ) as pool:
            for i, out in enumerate(tqdm(pool.imap_unordered(_process_event, event_ids, chunksize=1), total=len(event_ids))):
                if out is None:
                    continue
                if out.pop('__skipped__', False):
                    skipped.append(out)
                    if len(skipped) % 20 == 0:
                        _save_skipped(skipped, skipped_path)
                    continue
                results.append(out)
                # Periodic save every 10 results or at the end
                if (len(results) % 10) == 0 and len(results) > 0:
                    _save_summary(results, csv_path)
                if (len(results) - n_prev_results) >= config.max_events:
                    break

    return results, skipped


def main(config, n_workers=4):
    # load datasets
    # lambda_skip = None
    lambda_skip = lambda x: x > 0 and x < 700071 or x > 938303
        
    df_X, df_Y, df_quality, df_trace = load_datasets(config.base_data_path, 
                                                     include_val=False, include_test=True, include_train=False,
                                                     lambda_skip=lambda_skip)
    all_events = df_X['event_number'].unique()
    with open(RF_PARAMS_CONFIG_FILE, 'r') as f:
        params_RF = json.load(f)

    _, latitude, altitude, input_sampling_freq, out_sampling_freq, \
    N_samples, sampling_period, freqs, \
    out_N_samples, out_sampling_period, out_freqs, \
    LST_radians, tf, t_SN, t_EW, t_Z = load_input_params_from_dict(params_RF)

    base_LFmap_path = LFMAP_DIR

    noise_computer = compute_noise(18., latitude, 
                              [f"{base_LFmap_path}/LFmapshort{i}.npy" for i in range(20, 251)], 
                              np.arange(20,251)*1e6, 
                              np.fft.rfftfreq(1024, 1/500e6), 
                              tf, leff_x=t_SN, leff_y=t_EW, leff_z=t_Z, duration=config.duration_us * 1e-6)
    noise_computer.noise_psd(18)

    


    # Prepare arguments for each event
    model, model_config = load_model(config.base_model_path, config.model_subdir, inference_only=True)
    model.eval()
    # Event numbers are lightweight task payloads for parallel workers.
    event_ids = list(all_events)
    if config.n_jobs > 1:
        event_ids = [e for e in event_ids if int(e) % config.n_jobs == config.job_id]
        print(f"Sharding: job {config.job_id}/{config.n_jobs} -> {len(event_ids)}/{len(all_events)} events (event_number % {config.n_jobs} == {config.job_id})", flush=True)


    del df_Y, df_quality, df_trace

    out_dir = f'{config.recons_dir}/{config.name}'
    os.makedirs(out_dir, exist_ok=True)
    shard_suffix = '' if config.n_jobs <= 1 else f'_job{config.job_id}of{config.n_jobs}'
    summary_name = f'optimizer_recons_summary{shard_suffix}.csv'
    skipped_name = f'optimizer_recons_skipped{shard_suffix}.csv'
    csv_path = os.path.join(out_dir, summary_name)
    skipped_path = os.path.join(out_dir, skipped_name)

    prev_results, prev_skipped = [], []
    if config.continue_run:
        # An event is "already handled" if any previous run reconstructed it or
        # recorded it as skipped, under any --n-jobs split. Scan every summary and
        # skip CSV in out_dir to build that exclusion set.
        #
        # But seed `results`/`skipped` only from THIS shard's own files: each shard
        # rewrites its own CSV in full, so seeding them from the merged set would
        # make every shard write every other shard's rows, leaving n_jobs copies of
        # each event once the shard CSVs are concatenated for analysis.
        done_events = set()
        n_files = 0
        for fname in sorted(os.listdir(out_dir)):
            if not (fname.endswith('.csv') and fname.startswith(('optimizer_recons_summary', 'optimizer_recons_skipped'))):
                continue
            try:
                prev_df = pd.read_csv(os.path.join(out_dir, fname))
            except pd.errors.EmptyDataError:
                continue
            n_files += 1
            if 'event_number' in prev_df.columns:
                done_events.update(prev_df['event_number'].astype(int).tolist())
            if fname == summary_name:
                prev_results = prev_df.to_dict('records')
            elif fname == skipped_name:
                prev_skipped = prev_df.to_dict('records')

        n_before = len(event_ids)
        event_ids = [e for e in event_ids if int(e) not in done_events]
        print(f"--continue: scanned {n_files} CSV(s) in {out_dir}; {len(done_events)} events already handled ({len(prev_results)} reconstructed + {len(prev_skipped)} skipped in this shard's own files). {len(event_ids)}/{n_before} events left for this shard.", flush=True)

    print('Results will be saved to', csv_path)
    print('Skipped events will be recorded in', skipped_path)
    # The returned lists already contain prev_results/prev_skipped; max_events caps
    # the events processed in *this* run, not the cumulative total.
    results, skipped = recons_all(df_X, event_ids, config, n_workers, model, model_config, t_SN, t_EW, t_Z, tf, noise_computer,
                                  csv_path, skipped_path, prev_results=prev_results, prev_skipped=prev_skipped)
    if len(results) > 0:
        _save_summary(results, csv_path)
    if len(skipped) > 0:
        _save_skipped(skipped, skipped_path)
    print(f"Done: {len(results)} reconstructed, {len(skipped)} skipped for this shard.", flush=True)

if __name__ == '__main__':
    # set_start_method('spawn')

    def parse_args_to_config():
        cfg = ReconsConfig(
            base_model_path=BASE_MODEL_PATH,
            model_subdir="moriond/last_train_betercov",
            base_data_path=BASE_DATA_PATH,
            base_root=BASE_ROOT,
        )

        p = argparse.ArgumentParser()
        p.add_argument('--method', type=str, default=cfg.method, help=f'Optimization method to use (default: {cfg.method})')
        p.add_argument('--max-events', type=int, default=cfg.max_events, help='Max events to process when not using --all')
        p.add_argument('--jitter-time-us', type=float, default=cfg.jitter_time_us, help='Amount of jitter (in microseconds) to add to the input times for each event (default: 0.000 us)')
        p.add_argument('--kill-noise', action='store_true')
        p.add_argument('--emulated-signal', action='store_true')
        p.add_argument('--name', type=str, default='', help='Name for this reconstruction run (used in output directory)')
        p.add_argument('--plot', action='store_true')
        p.add_argument('--alpha', type=float, default=cfg.alpha, help='Alpha parameter for correlation scaling (default: 0.5)')
        p.add_argument('--prior', choices=['informative', 'uninformative', 'bricolage'], default=cfg.prior, help='Prior type')
        p.add_argument('--verbose', action='store_true', help='Enable verbose logging')
        p.add_argument('--continue', dest='continue_run', action='store_true', help='Continue a previous run: auto-detect events already present in the output summary CSV for this --name/--method and only process the rest')
        p.add_argument('--n-jobs', type=int, default=1,help='Split the event list across this many independent jobs (default: 1, no sharding)')
        p.add_argument('--job-id', type=int, default=0,help='This job\'s shard index, 0-based, must be < --n-jobs (default: 0)')
        args = p.parse_args()

        if args.n_jobs < 1:
            raise ValueError(f"--n-jobs must be >= 1, got {args.n_jobs}")
        if not (0 <= args.job_id < args.n_jobs):
            raise ValueError(f"--job-id must satisfy 0 <= job_id < n_jobs, got job_id={args.job_id}, n_jobs={args.n_jobs}")

        for met in args.method.split('->'):
            if met.lower() not in ("de", "differential_evolution", 
                                   "minuit", 
                                   "l-bfgs-b", "l_bfgs_b", "lbfgsb", 
                                   'powell', 'pw',
                                   'nelder-mead', 'nm', 
                                   'trust-constr', 'tc', 'trust_constr',
                                   'cobyla', 'cb',
                                   "emcee",):
                raise ValueError(f"Unsupported method: {met}. Supported methods are DE, Minuit, L-BFGS-B, Powell, Nelder-Mead, Trust-Constr, COBYLA, and EMCEE (or combinations like 'de->emcee').")
        cfg.method = args.method
        cfg.max_events = args.max_events
        cfg.plot = args.plot
        cfg.name = args.name if len(args.name) else cfg.method
        cfg.jitter_time_us = args.jitter_time_us
        cfg.alpha = args.alpha
        cfg.prior = args.prior
        cfg.continue_run = args.continue_run
        cfg.n_jobs = args.n_jobs
        cfg.job_id = args.job_id
        if args.kill_noise:
            cfg.kill_noise = True
        if args.emulated_signal:
            cfg.emulated_signal = True
        if args.verbose:
            cfg.verbose = True
        return cfg

    config = parse_args_to_config()

    for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[_var] = '1'
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))


    print(f"System total CPUs: {os.cpu_count()}", flush=True)
    print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'Not set')}", flush=True)
    try:
        main(config, n_workers=n_workers)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting gracefully.", flush=True)
        sys.exit(0)

