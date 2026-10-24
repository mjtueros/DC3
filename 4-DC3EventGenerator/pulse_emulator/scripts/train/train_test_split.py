"""
Stratified event-level train / val / test split.

Events are kept exclusive to one set (no leakage).
Stratification is done on per-event physics features so that the
distribution of all output variables — especially phase_p — is
balanced across sets.

Stratification keys (binned into quantile bins):
  1. per-event phase_p standard deviation  (the main culprit)
  2. zenith angle                          (correlated with phase_p spread)
  3. log10(energy_em)                      (overall energy scale)

The training set is further split, within the same strata, into two folds:

  mean_fold == 1  MEAN_FRAC of the train events — fits the trunk and the mean head
  mean_fold == 0  the rest of the train events  — fits the covariance head
                  (and, trivially, every val/test event)

The covariance head has to learn the spread of the residuals the mean *will make
on unseen data*. Fitted on the same events as the mean, it sees residuals shrunk
by memorisation and learns a sigma that is too narrow. Holding a fold out keeps
those residuals out-of-sample and the sigma honest.

This second split uses its own RNG so that the train/val/test assignment above is
bit-for-bit unchanged from before mean_fold existed.

Two modes:
  (default)     rebuild the whole split from input_dataset.csv
  --fold-only   read the existing input_dataset_strat.csv and add/refresh only
                mean_fold, leaving every other column exactly as it is

Use --fold-only on an existing dataset. A full rebuild re-derives every column
from input_dataset.csv, and the two files are NOT currently interchangeable: the
strat file that the released models were trained on carries a +1264 m (site
altitude) offset on du_pos_z / core_pos_z / xmax_pos_z that input_dataset.csv does
not. xmax_pos_z is a model input, so a full rebuild would silently move every
model's input distribution. The guard in `save()` refuses that overwrite unless
--force is given.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append('scripts/')
from local_paths import BASE_DATA_PATH

# ── settings ────────────────────────────────────────────────────────
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15          # test = 1 - train - val = 0.15
MEAN_FRAC  = 0.80          # share of the *train* events used to fit the mean;
                           # the remaining 0.20 is reserved for the covariance head
N_BINS     = 10             # quantile bins per feature for stratification

SAVE_DF_DIR = BASE_DATA_PATH
STRAT_PATH = os.path.join(SAVE_DF_DIR, "input_dataset_strat.csv")


def build_strata(df_X, df_Y, n_bins=N_BINS):
    """Per-event stratification key from phase_p spread, zenith and log energy."""
    ev = pd.DataFrame({
        "phase_p_std": df_Y.groupby(df_X["event_number"])["phase_p"].std(),
        "zenith":      df_X.groupby("event_number")["zenith"].first(),
        "log_energy":  np.log10(df_X.groupby("event_number")["energy_em"].first().clip(lower=1.0)),
    }).dropna()
    for col in ["phase_p_std", "zenith", "log_energy"]:
        ev[f"{col}_bin"] = pd.qcut(ev[col], q=n_bins, labels=False, duplicates="drop")
    ev["stratum"] = (ev["phase_p_std_bin"].astype(str) + "_" +
                     ev["zenith_bin"].astype(str) + "_" +
                     ev["log_energy_bin"].astype(str))
    return ev


def split_mean_fold(ev, train_set, mean_frac=MEAN_FRAC, seed=SEED + 1):
    """Split the train events into the mean fold and the covariance-head fold.

    Stratum by stratum, so the two folds match in phase_p spread, zenith and
    energy. A dedicated RNG keeps the train/val/test draws untouched.
    """
    rng_fold = np.random.RandomState(seed)
    mean_events = []
    for _, grp in ev.groupby("stratum"):
        events = np.array([e for e in grp.index.values if e in train_set])
        if len(events) == 0:
            continue
        rng_fold.shuffle(events)
        n_mean = max(1, int(round(mean_frac * len(events))))   # at least 1 in the mean fold
        mean_events.append(events[:n_mean])
    mean_set = set(np.concatenate(mean_events)) if mean_events else set()
    head_set = train_set - mean_set
    assert mean_set <= train_set, "mean fold leaked outside the train set!"
    assert len(mean_set & head_set) == 0, "mean/head fold overlap!"
    assert mean_set | head_set == train_set, "some train event landed in neither fold!"
    return mean_set, head_set


def assign_mean_fold(df_X, mean_set):
    """1 only for train events reserved for the main (trunk + mean) training."""
    df_X["mean_fold"] = df_X["event_number"].isin(mean_set).astype(int)
    # Folds are assigned per event, so every antenna of an event shares one fold.
    assert df_X.groupby("event_number")["mean_fold"].nunique().max() == 1, \
        "an event was split across the mean/head folds!"
    assert df_X.loc[~df_X.is_train.astype(bool), "mean_fold"].max() == 0, \
        "mean_fold set outside the train split!"
    return df_X


def report_folds(df_X, mean_set, head_set):
    tr = df_X.is_train.astype(bool)
    print(f"Train folds — mean (mean_fold=1): {len(mean_set)} events / "
          f"{int((df_X.mean_fold == 1).sum())} samples  |  "
          f"head (mean_fold=0): {len(head_set)} events / "
          f"{int(((df_X.mean_fold == 0) & tr).sum())} samples")


def save(df_X, force=False):
    """Write the strat file, refusing to silently rewrite unrelated columns.

    A full rebuild re-derives every column from input_dataset.csv. If that
    disagrees with the file already on disk anywhere outside the split flags, the
    models trained on the old file would no longer see the same inputs — so stop
    and make the caller say so explicitly.
    """
    flags = {"is_train", "is_val", "is_test", "mean_fold"}
    if os.path.exists(STRAT_PATH) and not force:
        existing = pd.read_csv(STRAT_PATH)
        shared = [c for c in existing.columns if c in df_X.columns and c not in flags]
        changed = [c for c in shared if not existing[c].equals(df_X[c].reset_index(drop=True))]
        if len(existing) != len(df_X) or changed:
            raise SystemExit(
                f"Refusing to overwrite {STRAT_PATH}: it would change "
                f"{'the row count and ' if len(existing) != len(df_X) else ''}"
                f"columns {changed}, which existing trained models depend on.\n"
                f"Use --fold-only to add mean_fold without touching anything else, "
                f"or --force if you really mean to rebuild the dataset.")
    df_X.to_csv(STRAT_PATH, index=False)
    print(f"\nSaved to {STRAT_PATH}")


def fold_only(force=False):
    """Add/refresh mean_fold on the existing strat file, changing nothing else."""
    df_X = pd.read_csv(STRAT_PATH)
    df_Y = pd.read_csv(os.path.join(SAVE_DF_DIR, "output_dataset.csv"))
    assert len(df_X) == len(df_Y), "X and Y must have the same number of rows"
    ev = build_strata(df_X, df_Y)
    train_set = set(df_X.loc[df_X.is_train.astype(bool), "event_number"].unique())
    mean_set, head_set = split_mean_fold(ev, train_set)
    df_X = assign_mean_fold(df_X, mean_set)
    print(f"Events  — train: {len(train_set)}  "
          f"val: {int(df_X.is_val.astype(bool).groupby(df_X.event_number).first().sum())}  "
          f"test: {int(df_X.is_test.astype(bool).groupby(df_X.event_number).first().sum())}")
    report_folds(df_X, mean_set, head_set)
    save(df_X, force=force)


_args = argparse.ArgumentParser(description=__doc__.split("\n")[1])
_args.add_argument("--fold-only", action="store_true",
                   help="Only add/refresh mean_fold on the existing strat file")
_args.add_argument("--force", action="store_true",
                   help="Allow a rebuild to overwrite columns the models depend on")
ARGS = _args.parse_args()

if ARGS.fold_only:
    fold_only(force=ARGS.force)
    raise SystemExit(0)

# ── load data ───────────────────────────────────────────────────────
df_X = pd.read_csv(os.path.join(SAVE_DF_DIR, "input_dataset.csv"))
df_Y = pd.read_csv(os.path.join(SAVE_DF_DIR, "output_dataset.csv"))

assert len(df_X) == len(df_Y), "X and Y must have the same number of rows"

# ── build event-level features for stratification ───────────────────
ev = build_strata(df_X, df_Y)

# ── stratified split ────────────────────────────────────────────────
rng = np.random.RandomState(SEED)

train_events, val_events, test_events = [], [], []

for _, grp in ev.groupby("stratum"):
    events = grp.index.values.copy()
    rng.shuffle(events)
    n = len(events)
    n_train = max(1, int(round(TRAIN_FRAC * n)))  # at least 1 in train
    n_val   = max(0, int(round(VAL_FRAC * n)))
    # ensure we don't exceed total
    if n_train + n_val > n:
        n_val = n - n_train
    train_events.append(events[:n_train])
    val_events.append(events[n_train:n_train + n_val])
    test_events.append(events[n_train + n_val:])

train_events = np.concatenate(train_events)
val_events   = np.concatenate(val_events)
test_events  = np.concatenate(test_events)

# ── sanity checks ───────────────────────────────────────────────────
train_set, val_set, test_set = set(train_events), set(val_events), set(test_events)
assert len(train_set & val_set)  == 0, "Train/val overlap!"
assert len(train_set & test_set) == 0, "Train/test overlap!"
assert len(val_set & test_set)   == 0, "Val/test overlap!"
assert len(train_set) + len(val_set) + len(test_set) == len(ev), "Missing events!"

# ── split the train events into mean / head folds ───────────────────
mean_set, head_set = split_mean_fold(ev, train_set)

# ── assign flags ────────────────────────────────────────────────────
df_X["is_train"] = df_X["event_number"].isin(train_set)
df_X["is_val"]   = df_X["event_number"].isin(val_set)
df_X["is_test"]  = df_X["event_number"].isin(test_set)
df_X = assign_mean_fold(df_X, mean_set)

# ── report ──────────────────────────────────────────────────────────
print(f"Events  — train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")
print(f"Samples — train: {df_X.is_train.sum()}  val: {df_X.is_val.sum()}  test: {df_X.is_test.sum()}")
report_folds(df_X, mean_set, head_set)

# Per-output distribution check
for col in ["a", "b", "c", "phase_q", "phase_p"]:
    t_std = df_Y.loc[df_X.is_train, col].std()
    v_std = df_Y.loc[df_X.is_val,   col].std()
    ratio = v_std / t_std if t_std > 0 else float("nan")
    print(f"  {col}: train_std={t_std:.6f}  val_std={v_std:.6f}  ratio={ratio:.3f}")

# ── save ────────────────────────────────────────────────────────────
save(df_X, force=ARGS.force)