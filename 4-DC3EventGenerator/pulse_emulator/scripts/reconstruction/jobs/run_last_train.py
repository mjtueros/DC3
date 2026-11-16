import subprocess
import sys


def submit(args, sbatch_args=None):
    cmd = ["sbatch", *(sbatch_args or []), "scripts/reconstruction/jobs/job_optim.sh", *args]
    subprocess.run(cmd, check=True)


def submit_sharded(base_args, n_jobs=1, mem_mb=None, cpus_per_task=None):
    """Submit `n_jobs` SLURM jobs that split the event list between them.

    Each job gets --n-jobs/--job-id so OPTIMIZER_final.py partitions events by
    event_number % n_jobs == job_id: jobs never touch the same event and each
    writes to its own summary CSV, so they can run concurrently without
    colliding. mem_mb/cpus_per_task (if given) override the #SBATCH defaults
    in job_optim.sh, e.g. to request less per job when running many shards.
    """
    sbatch_overrides = []
    if mem_mb is not None:
        sbatch_overrides += [f"--mem={mem_mb}"]
    if cpus_per_task is not None:
        sbatch_overrides += [f"--cpus-per-task={cpus_per_task}"]
    for job_id in range(n_jobs):
        submit([*base_args, "--n-jobs", str(n_jobs), "--job-id", str(job_id)], sbatch_args=sbatch_overrides)


# Usage: python scripts/reconstruction/jobs/run_last_train.py [n_jobs]
# e.g. `python scripts/reconstruction/jobs/run_last_train.py 4` splits the run into 4 non-overlapping jobs.
n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# submit_sharded([
#     "--max-events", "500",
#     "--continue",
#     "--method", "de->emcee",
#     "--emulated-signal",
#     "--kill-noise",
#     "--jitter-time-us", "0.000",
#     "--name", f"emulated_perfect_bricolage",
#     "--prior",  'bricolage'
# ], n_jobs=n_jobs)

# submit_sharded([
#     "--max-events", "500",
#     "--continue",
#     "--method", "de->emcee",
#     "--jitter-time-us", "0.002",
#     "--name", f"realistic_2ns_bricolage",
#     "--prior",  'bricolage'
# ], n_jobs=n_jobs)

submit_sharded([
    "--continue",
    "--method", "de->emcee",
    "--jitter-time-us", "0.005",
    "--name", "realistic_5ns_bricolage",
    "--prior",  'bricolage'
], n_jobs=n_jobs, mem_mb=(120000 // n_jobs) if n_jobs > 1 else None,
   cpus_per_task=(80 // n_jobs) if n_jobs > 1 else None)
